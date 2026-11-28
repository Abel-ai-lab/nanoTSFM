import math
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from nanotsfm._cdf import pool_quantiles
from nanotsfm.data import MAX_VARIATES, pack_windows

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MIN_SCALE = 1e-5
POLYPHASE_MIN_PATCHES = 2  # a stride joins a series' forecast if each phase has this much history
PERIOD_TARGET = 24  # the points of its main cycle a period-locked stride leaves a series
PERIOD_STRIDES = (2, 12)  # the least and most such a stride can be
PERIOD_SPAN = 4096  # points read for a period: what a stride-4 forecast sees
PERIOD_MIN_STRENGTH = 0.3  # autocorrelation at the period below which a series has no clear cycle
PERIOD_MIN_POINTS = 64  # each phase keeps this many, two patches of the base model
PHASE_SLOTS = 4  # TinyCast's K: a series can hold a daily and a weekly period together
PHASE_ALPHA = 0.05  # TinyCast's Fisher test level, Bonferroni-corrected across bins
LIST_FIELDS = ("polyphase_forecast_strides",)
LOCKED_HISTORY_BATCH_ROWS = 8


@dataclass(frozen=True)
class ModelConfig:
    context_length: int = 512
    prediction_length: int = 256
    patch_length: int = 32
    width: int = 256
    layers: int = 3  # causal time layers; one variate layer follows them
    heads: int = 4
    dropout: float = 0.0
    # Inference only; the defaults leave forecasts unchanged. See `forecast`.
    infer_clamp: float = 0.0  # keep quantiles within this many context ranges of the context
    # The interlaced forecast; see Polyphase. Forecast only, so it can re-score a checkpoint.
    polyphase_forecast_strides: tuple = ()  # strides whose phases are forecast and interlaced
    polyphase_forecast_weight: float = 0.25  # the interlaced forecasts' share of the blend
    polyphase_forecast_period: bool = False  # a series with a clear cycle uses its locked stride
    # Periods found in the context as coordinates; see detect_periods.
    phase_tokens: bool = False  # each token gets sin and cos of its phase in each period
    infer_tail_floor: bool = False  # past one pass, bands stay as wide as a cycle before
    infer_views: bool = False  # median of the x, -x and, if never negative, square-root views
    infer_sign_clip: bool = False  # a one-signed history keeps its forecasts on that side of 0
    infer_integer_snap: bool = False  # round forecasts of a never-negative whole-number history
    infer_locked_history: bool = False  # pool CDFs and extend period-locked phases on the first block
    # Forecast-time options; they change forecast(), never the weights or training.
    flip_symmetric: bool = False  # average f(x) with -f(-x), quantile order reversed
    clamp_positive: bool = False  # no quantile below 0 when the observed history is >= 0

    def __post_init__(self):
        # Lists, as config.json holds them, so that asdict(config) equals the recorded config.
        for key in LIST_FIELDS:
            object.__setattr__(self, key, list(getattr(self, key)))
        if self.infer_clamp < 0:
            raise ValueError("infer_clamp must be >= 0")
        for key in (
            "context_length",
            "prediction_length",
            "patch_length",
            "width",
            "layers",
            "heads",
        ):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.context_length % self.patch_length:
            raise ValueError("context_length must divide into patches")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width must split into heads of even size")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        for key in ("infer_views", "infer_sign_clip", "infer_integer_snap", "infer_locked_history"):
            if type(getattr(self, key)) is not bool:
                raise ValueError(f"{key} must be true or false")
        if type(self.infer_tail_floor) is not bool:
            raise ValueError("infer_tail_floor must be true or false")
        strides = self.polyphase_forecast_strides
        if any(type(stride) is not int or stride < 2 for stride in strides):
            raise ValueError("polyphase_forecast_strides must be integers of at least 2")
        if not 0 <= self.polyphase_forecast_weight <= 1:
            raise ValueError("polyphase_forecast_weight must be from 0 to 1")
        if type(self.polyphase_forecast_period) is not bool:
            raise ValueError("polyphase_forecast_period must be true or false")
        if self.polyphase_forecast_period and not strides:
            raise ValueError("polyphase_forecast_period falls back on polyphase_forecast_strides")
        if self.infer_locked_history and not (self.polyphase_forecast_period and self.infer_views):
            raise ValueError("infer_locked_history requires period-locked strides and inference views")
        if type(self.phase_tokens) is not bool:
            raise ValueError("phase_tokens must be true or false")


def network_config(config: ModelConfig) -> ModelConfig:
    """The network's own config. With polyphase_forecast_strides, context_length is the history the
    forecast reads, and the network reads that divided by the longest stride: each phase of the
    longest stride then gets the network's full context. See Polyphase."""
    if not config.polyphase_forecast_strides:
        return config
    stride = max(config.polyphase_forecast_strides)
    if config.context_length % (stride * config.patch_length):
        raise ValueError(
            "With polyphase_forecast_strides, context_length must divide into "
            "patches after division by the longest stride"
        )
    return replace(config, context_length=config.context_length // stride)


def causal_scale(values: torch.Tensor, patch: int):
    observed = torch.isfinite(values)
    x = torch.where(observed, values, 0).double()
    count = observed.cumsum(-1).clamp_min(1)
    loc = x.cumsum(-1) / count
    previous = F.pad(loc[..., :-1], (1, 0))
    # Welford update: sum of (x - old mean) * (x - new mean) over observed values.
    m2 = ((x - previous) * (x - loc) * observed).cumsum(-1)
    std = (m2 / (count - 1).clamp_min(1)).clamp_min(MIN_SCALE**2).sqrt()
    end = slice(patch - 1, None, patch)
    return loc[..., end], std[..., end]


def detect_periods(context: torch.Tensor) -> torch.Tensor:
    """Up to PHASE_SLOTS periods of each [B,V,C] context, [B,V,PHASE_SLOTS], strongest first; a slot
    that fails is 0. Pass only the context: a period read from later points would leak them.

    As TinyCast's section 3.1 and appendix A: one FFT of the context with its mean removed, gaps
    filled from the nearest point and padded to a power of two. A bin's share of the non-DC power
    must top ln(bins / alpha) / bins, the large-N Bonferroni form of Fisher's test, and be a strict
    local maximum, with a period from 2 to half the context, rounded to whole points. A period must
    also fit twice into the series' observed points, or a short series could fill slots from its
    padding. The test screens out white noise, not every aperiodic series: a strong AR(1) can pass.
    """
    length = context.shape[-1]
    size = 2 ** math.ceil(math.log2(length))
    with torch.autocast(context.device.type, enabled=False):
        seen = context.isfinite().sum(-1, keepdim=True)
        filled = forward_fill(forward_fill(context.double()).flip(-1)).flip(-1).nan_to_num(0.0)
        power = torch.fft.rfft(filled - filled.mean(-1, keepdim=True), size).abs().square()[..., 1:]
        # The floor is the power of a spread of MIN_SCALE, so a flat series can't pass on rounding.
        share = power / power.sum(-1, keepdim=True).clamp_min((size * MIN_SCALE) ** 2)
        bins = share.shape[-1]
        periods = size / torch.arange(1, bins + 1, device=context.device, dtype=torch.float64)
        peak = torch.zeros_like(share, dtype=torch.bool)
        peak[..., 1:-1] = (share[..., 1:-1] > share[..., :-2]) & (share[..., 1:-1] > share[..., 2:])
        peak &= share > math.log(bins / PHASE_ALPHA) / bins
        peak &= (periods >= 2) & (periods <= length / 2) & (periods <= seen / 2)
        top, place = torch.where(peak, share, 0).topk(PHASE_SLOTS, dim=-1)
        return torch.where(top > 0, periods[place].round(), 0).long()


def rotate(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    frequency = 10000 ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    angle = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)[:, None] * frequency
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    first, second = x[..., :half], x[..., half:]
    return torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1)


class Block(nn.Module):
    def __init__(self, config: ModelConfig, causal: bool):
        super().__init__()
        self.heads, self.causal, self.dropout = config.heads, causal, config.dropout
        self.attention_norm = nn.RMSNorm(config.width)
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.out = nn.Linear(config.width, config.width, bias=False)
        self.mlp_norm = nn.RMSNorm(config.width)
        self.mlp = nn.Sequential(
            nn.Linear(config.width, 4 * config.width),
            nn.GELU(),
            nn.Linear(4 * config.width, config.width),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        n, s, width = x.shape
        q, k, v = self.qkv(self.attention_norm(x)).view(n, s, 3, self.heads, -1).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if self.causal:
            q, k = rotate(q), rotate(k)
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal,
        )
        x = x + self.out(attended.transpose(1, 2).reshape(n, s, width))
        return x + self.mlp(self.mlp_norm(x))


class NanoTSFM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        patch = config.patch_length
        self.embed = nn.Linear(2 * patch, config.width)
        self.embed_mlp = nn.Sequential(
            nn.Linear(2 * patch, config.width), nn.SiLU(), nn.Linear(config.width, config.width)
        )
        self.time_blocks = nn.ModuleList(Block(config, causal=True) for _ in range(config.layers))
        self.variate_block = Block(config, causal=False)
        self.norm = nn.RMSNorm(config.width)
        self.head = nn.Linear(config.width, patch * len(QUANTILES))
        if config.phase_tokens:  # starts at zero and draws no random numbers
            self.phase_link = nn.Parameter(torch.zeros(2 * PHASE_SLOTS, config.width))

    def next_patch(
        self, values: torch.Tensor, series_ids: torch.Tensor, *, phase_context: int | None = None
    ):
        """Predict the next patch from [B,V,T] values; equal IDs share variate attention."""
        batch, variates, length = values.shape
        patch = self.config.patch_length
        if length % patch:
            raise ValueError("Sequence length must divide into patches")
        patches = length // patch
        loc, scale = causal_scale(values, patch)
        observed = torch.isfinite(values)
        # Zero-fill before arithmetic so NaN never reaches the gradient.
        clean = torch.where(observed, values, 0).double()
        scaled = (clean - loc.repeat_interleave(patch, -1)) / scale.repeat_interleave(patch, -1)
        scaled = torch.where(observed, scaled, 0).asinh().float()
        tokens = torch.cat(
            [scaled.unflatten(-1, (patches, patch)), observed.float().unflatten(-1, (-1, patch))],
            dim=-1,
        )
        x = self.embed(tokens) + self.embed_mlp(tokens)
        if self.config.phase_tokens:
            # The window's first point is t = 0, and a token's phase is its patch center's.
            context = self.config.context_length if phase_context is None else phase_context
            periods = detect_periods(values[..., :context])[..., None, :]
            center = torch.arange(patches, device=x.device, dtype=torch.float64)
            center = (center * patch + (patch - 1) / 2)[:, None]
            turns, found = center / periods.clamp_min(1), periods > 0
            angle = 2 * math.pi * turns
            phase = torch.cat([angle.sin() * found, angle.cos() * found], -1).float()
            x = x + phase @ self.phase_link

        x = x.reshape(batch * variates, patches, -1)
        for block in self.time_blocks:
            x = block(x)
        x = (
            x.reshape(batch, variates, patches, -1)
            .transpose(1, 2)
            .reshape(batch * patches, variates, -1)
        )
        related = series_ids[:, :, None] == series_ids[:, None, :]
        related = related[:, None, None].expand(batch, patches, 1, variates, variates)
        x = self.variate_block(x, related.reshape(batch * patches, 1, variates, variates))
        x = x.reshape(batch, patches, variates, -1).transpose(1, 2)

        quantiles = self.head(self.norm(x)).float().unflatten(-1, (patch, len(QUANTILES)))
        return quantiles.sort(dim=-1).values, loc, scale

    def forward(self, history: torch.Tensor, series_ids: torch.Tensor) -> torch.Tensor:
        patch, horizon = self.config.patch_length, self.config.prediction_length
        future = math.ceil(horizon / patch)
        values = F.pad(history, (0, future * patch), value=float("nan"))
        quantiles, loc, scale = self.next_patch(values, series_ids)
        # Position i predicts patch i + 1; the last history patch predicts the first future one.
        first = history.shape[-1] // patch - 1
        window = slice(first, first + future)
        location, spread = loc[..., window, None, None], scale[..., window, None, None]
        forecast = quantiles[:, :, window].double().sinh() * spread + location
        return forecast.flatten(2, 3)[:, :, :horizon].float()


class Polyphase(nn.Module):
    """Blend a model's normal forecast with interlaced forecasts of the history's polyphase parts.

    For stride s the history splits into s subsequences of every s-th real point. Each one is
    forecast alone, and its forecast fills every s-th future position. The config asks for s times
    the base model's context, so the longest stride still gets a full context of real points.
    Every forecast here is the trained model's.

    With polyphase_forecast_period, a series with a clear cycle uses only its period-locked stride,
    and the others use the configured strides. A batch row can hold several series, so each is
    read and forecast apart from its neighbors, by its series id. The context stays as configured,
    so a stride above the longest configured one gets a shorter phase.
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.strides = base.config.polyphase_forecast_strides
        self.weight = base.config.polyphase_forecast_weight
        self.by_period = base.config.polyphase_forecast_period
        self.recent_context = base.config.context_length * max(self.strides)
        context = self.recent_context
        if base.config.infer_locked_history:
            context = max(context, base.config.context_length * PERIOD_STRIDES[1])
        self.config = replace(base.config, context_length=context)

    def interlaced(self, history: torch.Tensor, series_ids: torch.Tensor, stride: int):
        """Return the interlaced forecast and which series have enough history in every phase."""
        context = self.base.config.context_length
        points = self.base.config.patch_length * POLYPHASE_MIN_PATCHES
        length = history.shape[-1]
        part = math.ceil(self.base.config.prediction_length / stride)
        forecast = torch.empty(*history.shape[:2], part * stride, len(QUANTILES)).to(history)
        usable = torch.ones(history.shape[:2], dtype=torch.bool, device=history.device)
        for lead in range(stride):
            # This phase's last point sits `stride` steps before future position `lead`.
            phase = history[..., (length + lead) % stride :: stride][..., -context:]
            usable &= torch.isfinite(phase).sum(-1) >= points
            phase = F.pad(phase, (context - phase.shape[-1], 0), value=float("nan"))
            forecast[:, :, lead::stride] = self.base(phase, series_ids)[:, :, :part]
        return forecast, usable

    def period_plan(self, history: torch.Tensor, series_ids: torch.Tensor) -> dict:
        """Map each stride to the series using it, as (row, its channels) pairs."""
        strides, strength = channel_strides(history)
        plan = {}
        for row, ids in enumerate(series_ids):
            for series in ids.unique().tolist():
                if series < 0:  # padding
                    continue
                channels = (ids == series).nonzero()[:, 0]
                locked = int(strides[row, channels][strength[row, channels].argmax()])
                for stride in (locked,) if locked else self.strides:
                    plan.setdefault(stride, []).append((row, channels))
        return plan

    def interlaced_series(self, history: torch.Tensor, stride: int, chosen: list):
        """interlaced for the chosen series only, placed back in their rows; the rest are NaN."""
        samples = [(history[row, channels], history[row, channels, :0]) for row, channels in chosen]
        packed, _, ids = pack_windows(samples)
        ids = ids.to(history.device)
        forecast, usable = self.interlaced(packed.to(history), ids, stride)
        placed = forecast.new_full((*history.shape[:2], *forecast.shape[2:]), float("nan"))
        reached = torch.zeros(history.shape[:2], dtype=torch.bool, device=history.device)
        for index, (row, channels) in enumerate(chosen):
            placed[row, channels] = forecast[ids == index]
            reached[row, channels] = usable[ids == index]
        return placed, reached

    def forecasts(self, history: torch.Tensor, series_ids: torch.Tensor):
        if not self.by_period:
            for stride in self.strides:
                yield self.interlaced(history, series_ids, stride)
            return
        for stride, chosen in sorted(self.period_plan(history, series_ids).items()):
            yield self.interlaced_series(history, stride, chosen)

    def locked_first_forecasts(self, recent, older, series_ids):
        """Give a high-stride phase enough observations to fill its original model context."""
        limit = self.base.config.context_length
        minimum = self.base.config.patch_length * POLYPHASE_MIN_PATCHES
        has_older = older.isfinite().any(-1)
        for stride, groups in sorted(self.period_plan(recent, series_ids).items()):
            if stride <= max(self.strides):
                yield self.interlaced_series(recent, stride, groups)
                continue
            counts = [
                recent[..., (recent.shape[-1] + lead) % stride :: stride][..., -limit:]
                .isfinite().sum(-1)
                for lead in range(stride)
            ]
            eligible = torch.stack(counts).amin(0) >= minimum
            on_route = torch.zeros_like(eligible)
            for row, channels in groups:
                on_route[row, channels] = True
            eligible &= on_route
            extend = eligible & has_older
            if extend.any():
                prefix = older.masked_fill(~extend[..., None], torch.nan)
                values = torch.cat((prefix, recent), dim=-1)
                prediction, usable = self.interlaced_series(values, stride, groups)
                yield prediction, usable & eligible
            else:
                yield self.interlaced_series(recent, stride, groups)

    @staticmethod
    def blend(normal: torch.Tensor, forecasts, share: float) -> torch.Tensor:
        """normal, with `share` of it replaced by the mean of the forecasts each series can use."""
        total, count = torch.zeros_like(normal), torch.zeros_like(normal[..., :1])
        for forecast, usable in forecasts:
            forecast = forecast[:, :, : normal.shape[2]].double()
            valid = torch.isfinite(forecast).all(-1, keepdim=True) & usable[:, :, None, None]
            total += torch.where(valid, forecast, 0.0)
            count += valid
        weight = share * (count > 0)
        return (1 - weight) * normal + weight * total / count.clamp_min(1)

    def forward(self, history: torch.Tensor, series_ids: torch.Tensor) -> torch.Tensor:
        recent = history[..., -self.base.config.context_length :]
        normal = self.base(recent, series_ids).double()
        blended = self.blend(normal, self.forecasts(history, series_ids), self.weight)
        return blended.float().sort(dim=-1).values


def device_for(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("Use cpu, cuda, or auto")
    return device


def load_checkpoint(path: Path, device="cpu", *, locked_history=None):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 3:
        raise ValueError("Unsupported checkpoint version")
    config = ModelConfig(**checkpoint["model_config"])
    if locked_history is not None:
        config = replace(config, infer_locked_history=locked_history)
    model = NanoTSFM(config)
    model.load_state_dict(checkpoint["model"])
    if model.config.polyphase_forecast_strides:
        model = Polyphase(model)
    model.to(device_for(str(device))).eval()
    return model, checkpoint


def rollout(model, history: torch.Tensor, horizon: int, series_ids: torch.Tensor) -> torch.Tensor:
    """Feed medians back past one model pass."""
    context = model.config.context_length
    blocks = []
    remaining = horizon
    while remaining:
        prediction = model(history[..., -context:], series_ids)
        take = min(remaining, prediction.shape[-2])
        blocks.append(prediction[..., :take, :])
        history = torch.cat([history, prediction[..., 4]], dim=-1)[..., -context:]
        remaining -= take
    return torch.cat(blocks, dim=-2)


def locked_history_rollout(model: Polyphase, history, horizon, series_ids):
    """Emit CDF quantiles while retaining the original blend for median feedback."""
    context = model.recent_context
    older, recent = history[..., :-context], history[..., -context:]
    blocks, remaining = [], horizon
    while remaining:
        raw = model.base(recent[..., -model.base.config.context_length :], series_ids).double()
        if older.shape[-1]:
            members = tuple(model.locked_first_forecasts(recent, older, series_ids))
            older = older[..., :0]
        else:
            members = tuple(model.forecasts(recent, series_ids))
        take = min(remaining, raw.shape[-2])
        phase = model.blend(raw[..., :take, :], members, 1.0)
        pooled = pool_quantiles(raw[..., :take, :], phase, model.weight)
        blocks.append(pooled.float().sort(dim=-1).values)
        remaining -= take
        if remaining:
            feedback = model.blend(raw, members, model.weight).float().sort(dim=-1).values[..., 4]
            recent = torch.cat((recent, feedback), dim=-1)[..., -context:]
    return torch.cat(blocks, dim=-2)


def forward_fill(history: torch.Tensor) -> torch.Tensor:
    """Fill gaps from the last observation; points before the first one stay NaN."""
    positions = torch.arange(history.shape[-1], device=history.device).expand_as(history)
    last = torch.where(history.isfinite(), positions, 0).cummax(-1).values
    return history.gather(-1, last)


def infer_period(filled: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Each series' period and its autocorrelation there: the strongest cycle seen at least
    twice, refined on the autocorrelation. Series without one get period 1 and strength 0."""
    length = filled.shape[-1]
    x = filled - filled.nanmean(-1, keepdim=True)
    x = x.nan_to_num(0.0).double()
    power = torch.fft.rfft(x).abs().square()
    frequencies = torch.arange(power.shape[-1], device=x.device)
    cycles = torch.where(frequencies * lengths[..., None] >= 2 * length, power, 0)
    cycle = cycles.argmax(-1, keepdim=True).clamp_min(2)
    autocorrelation = torch.fft.irfft(torch.fft.rfft(x, 2 * length).abs().square())
    lags = torch.arange(length // 2 + 1, device=x.device)
    autocorrelation = autocorrelation[..., lags]
    autocorrelation = autocorrelation / (lengths[..., None] - lags).clamp_min(1)
    near = (lags >= 2) & (lags >= length // (cycle + 1)) & (lags <= (length // (cycle - 1)) + 1)
    near &= lags <= lengths[..., None] // 2
    best = torch.where(near, autocorrelation, -torch.inf).amax(-1, keepdim=True)
    period = torch.where(near, autocorrelation, -torch.inf).argmax(-1)
    period = torch.where(cycles.amax(-1) > 0, period.clamp_min(1), 1)
    strength = (best / autocorrelation[..., :1].clamp_min(1e-12))[..., 0]
    return period, torch.where(period > 1, strength.float(), 0.0)


def channel_strides(history: torch.Tensor):
    """Each [...,T] channel's period-locked stride and its period's strength, read from its own
    points. The stride puts the main cycle near PERIOD_TARGET points and leaves PERIOD_MIN_POINTS in
    each phase; a channel with no clear cycle, or too few points, gets 0."""
    filled = forward_fill(history[..., -PERIOD_SPAN:])
    lengths = filled.isfinite().sum(-1)
    period, strength = infer_period(filled, lengths)
    stride = (period / PERIOD_TARGET).round().long().clamp(*PERIOD_STRIDES)
    room = lengths // PERIOD_MIN_POINTS
    clear = (strength >= PERIOD_MIN_STRENGTH) & (room >= PERIOD_STRIDES[0])
    return torch.where(clear, torch.minimum(stride, room), 0), strength


def signed_forecast(model, history, horizon: int, series_ids, *, predict=rollout) -> torch.Tensor:
    """The forecast of -x, mapped back: q_t(x) = -q_{1-t}(-x)."""
    return -predict(model, -history, horizon, series_ids).flip(-1)


def views_forecast(
    model, history, horizon: int, series_ids, *, predict=rollout, full=None, native=None
) -> torch.Tensor:
    """Per quantile, the mean of the middle two of four forecasts: of x, of -x, and for a
    never-negative series of sqrt(x) and -sqrt(x), each mapped back. Any other series counts its
    first two views twice. Every view is a forecast of this model from the series' own history."""
    values = history if full is None else full
    if native is None:
        native = predict(model, values, horizon, series_ids)
    flipped = signed_forecast(model, values, horizon, series_ids, predict=predict)
    observed = history.isfinite()
    nonneg = observed.any(-1) & torch.where(observed, history >= 0, True).all(-1)  # [B,V]
    passes = [native, flipped]
    rows = nonneg.any(-1)
    if rows.any():
        root = torch.where(observed[rows], history[rows].clamp_min(0).sqrt(), torch.nan)
        if full is not None:
            older = full[rows, :, : -history.shape[-1]]
            root = torch.cat((torch.where(older >= 0, older.sqrt(), torch.nan), root), dim=-1)
        keep = nonneg[rows][..., None, None]
        views = (
            (predict(model, root, horizon, series_ids[rows]), native),
            (signed_forecast(model, root, horizon, series_ids[rows], predict=predict), flipped),
        )
        for view, fallback in views:
            full = fallback.clone()
            full[rows] = torch.where(keep, view.sign() * view.square(), fallback[rows])
            passes.append(full)
    else:
        passes += [native, flipped]
    ordered = torch.stack([each.sort(-1).values for each in passes]).sort(0).values
    return ordered[1:3].mean(0)


def model_forecast(model, history, horizon: int, series_ids, *, full=None) -> torch.Tensor:
    config = model.config
    if config.infer_views:  # views over the first pass; past it, the plain rollout
        first = config.prediction_length
        if full is None:
            native = rollout(model, history, horizon, series_ids)
            near = views_forecast(model, history, min(horizon, first), series_ids)
        else:
            native = locked_history_rollout(model, full, horizon, series_ids)
            near = views_forecast(
                model, history, min(horizon, first), series_ids,
                predict=locked_history_rollout, full=full, native=native[..., :first, :],
            )
        return torch.cat([near, native[..., first:, :]], dim=-2).sort(-1).values
    return rollout(model, history, horizon, series_ids)


def floor_tail(result: torch.Tensor, cycle: torch.Tensor, first: int) -> torch.Tensor:
    """result [...,H,9] with its bands past the first model pass held at least as wide as they were
    at the same phase of the pass's last cycle; `cycle` [...] is each series' whole period, or 1.

    A rollout feeds its own median back, so what it reads past one pass looks surer than any real
    history, and its bands narrow where the truth can only be as uncertain or more. Each quantile's
    distance from the median, at a step past the pass, is kept no smaller than at the latest step
    inside the pass a whole number of cycles earlier. Phase by phase, so a series that is quiet at
    night stays quiet at night. The median is untouched."""
    horizon = result.shape[-2]
    cycle = torch.where(cycle <= first, cycle, torch.ones_like(cycle)).clamp_min(1)[..., None]
    steps = torch.arange(first, horizon, device=result.device)
    back = ((steps - first) // cycle + 1) * cycle  # whole cycles back, into the first pass
    source = (steps - back)[..., None].expand(*result.shape[:-2], horizon - first, 9)
    spread = result - result[..., 4:5]
    held, tail = spread.gather(-2, source), spread[..., first:, :]
    low = torch.arange(9, device=result.device) < 4
    wider = torch.where(low, torch.minimum(tail, held), torch.maximum(tail, held))
    return torch.cat([result[..., :first, :], result[..., first:, 4:5] + wider], dim=-2)


def infer_forecast(model, history, horizon: int, series_ids, *, full=None) -> torch.Tensor:
    """The model's forecast with the infer_* steps: views, tail floor, clamp, sign clip, snap."""
    config = model.config
    result = model_forecast(model, history, horizon, series_ids, full=full)
    if config.infer_tail_floor and horizon > config.prediction_length:
        filled = forward_fill(history)
        cycle, clear = infer_period(filled, filled.isfinite().sum(-1))
        cycle = torch.where(clear >= PERIOD_MIN_STRENGTH, cycle, torch.ones_like(cycle))
        result = floor_tail(result, cycle, config.prediction_length)
    if config.infer_clamp:
        observed = history.isfinite()
        low = torch.where(observed, history, torch.inf).amin(-1)
        high = torch.where(observed, history, -torch.inf).amax(-1)
        room = ((high - low) * config.infer_clamp).nan_to_num(0.0, posinf=0.0, neginf=0.0)
        low, high = (low - room)[..., None, None], (high + room)[..., None, None]
        clamped = torch.minimum(torch.maximum(result, low), high)
        result = torch.where(observed.any(-1)[..., None, None], clamped, result)
    if config.infer_sign_clip or config.infer_integer_snap:
        observed = history.isfinite()
        low = torch.where(observed, history, torch.inf).amin(-1)
        high = torch.where(observed, history, -torch.inf).amax(-1)
        seen = observed.any(-1)
        nonneg = (seen & (low >= 0))[..., None, None]
        if config.infer_sign_clip:
            nonpos = (seen & (high <= 0) & (low < 0))[..., None, None]
            result = torch.where(nonneg, result.clamp_min(0), result)
            result = torch.where(nonpos, result.clamp_max(0), result)
        if config.infer_integer_snap:
            whole = torch.where(observed, history == history.round(), True).all(-1)
            result = torch.where(nonneg & whole[..., None, None], result.round(), result)
    return result


@torch.inference_mode()
def forecast(model, history: torch.Tensor, horizon: int, series_ids=None) -> torch.Tensor:
    """Forecast [B,T] or [B,V,T] history, rolling out from median predictions if needed.

    The infer_* config fields switch on test-time tricks that use only this history.
    With config.flip_symmetric the forecast is (f(x) - f(-x) with its quantiles reversed) / 2,
    so negating the input exactly negates the forecast. With config.clamp_positive, series whose
    observed history is nonnegative get no negative quantiles.
    """
    config = model.config
    result = _forecast(model, history, horizon, series_ids)
    if getattr(config, "flip_symmetric", False):
        flipped = _forecast(model, -history, horizon, series_ids)
        result = (result - flipped.flip(-1)) / 2
    if getattr(config, "clamp_positive", False):
        values = history.to(result.device, torch.float32)
        observed = torch.isfinite(values)
        lowest = torch.where(observed, values, float("inf")).amin(-1)
        positive = (lowest >= 0) & observed.any(-1)
        result = torch.where(positive[..., None, None], result.clamp_min(0), result)
    return result


def _forecast(model, history: torch.Tensor, horizon: int, series_ids=None) -> torch.Tensor:
    if history.ndim not in (2, 3) or history.shape[-1] == 0 or horizon <= 0:
        raise ValueError("Provide nonempty [B,T] or [B,V,T] history and a positive horizon")
    univariate = history.ndim == 2
    if univariate:
        history = history[:, None]
    if history.shape[1] > MAX_VARIATES:
        raise ValueError("Split related variates into consecutive groups of at most 32")
    if series_ids is not None and series_ids.shape != history.shape[:2]:
        raise ValueError("series_ids must match batch and variate dimensions")
    was_training = model.training
    model.eval()
    try:
        if isinstance(model, Polyphase) and model.config.infer_locked_history:
            parts = []
            for start in range(0, len(history), LOCKED_HISTORY_BATCH_ROWS):
                rows = slice(start, start + LOCKED_HISTORY_BATCH_ROWS)
                ids = None if series_ids is None else series_ids[rows]
                parts.append(_forecast_rows(model, history[rows], horizon, ids, locked_history=True))
            result = torch.cat(parts)
        else:
            result = _forecast_rows(model, history, horizon, series_ids)
        return result[:, 0] if univariate else result
    finally:
        model.train(was_training)


def _forecast_rows(model, history, horizon, series_ids, *, locked_history=False):
    device = next(model.parameters()).device
    history = history.to(device=device, dtype=torch.float32)
    if series_ids is None:
        series_ids = torch.zeros(history.shape[:2], dtype=torch.long, device=device)
    else:
        series_ids = series_ids.to(device)
    context = model.config.context_length
    if locked_history:
        history = history[..., -context:]
        recent = model.recent_context
        history = F.pad(history, (max(0, recent - history.shape[-1]), 0), value=torch.nan)
        return infer_forecast(model, history[..., -recent:], horizon, series_ids, full=history)
    history = F.pad(history, (max(0, context - history.shape[-1]), 0), value=float("nan"))
    config = model.config
    tricks = (
        config.infer_clamp
        or config.infer_tail_floor
        or config.infer_views
        or config.infer_sign_clip
        or config.infer_integer_snap
    )
    predict = infer_forecast if tricks else rollout
    return predict(model, history, horizon, series_ids)
