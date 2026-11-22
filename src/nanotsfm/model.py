import math
from dataclasses import dataclass, replace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from nanotsfm.data import MAX_VARIATES, pack_windows

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MIN_SCALE = 1e-5
INFER_CANDIDATES = ("deseason",)
INFER_TEMPERATURE = 0.1  # softmax over the candidates' relative backtest loss
INFER_BACKTEST = 0.25  # least share of the context the backtest holds back
POLYPHASE_MIN_PATCHES = 2  # a stride joins a series' forecast if each phase has this much history
PERIOD_TARGET = 24  # the points of its main cycle a period-locked stride leaves a series
PERIOD_STRIDES = (2, 12)  # the least and most such a stride can be
PERIOD_SPAN = 4096  # points read for a period: what a stride-4 forecast sees
PERIOD_MIN_STRENGTH = 0.3  # autocorrelation at the period below which a series has no clear cycle
PERIOD_MIN_POINTS = 64  # each phase keeps this many, two patches of the base model
PHASE_SLOTS = 4  # TinyCast's K: a series can hold a daily and a weekly period together
PHASE_ALPHA = 0.05  # TinyCast's Fisher test level, Bonferroni-corrected across bins
# Fields that checkpoints from before the submission's trim may hold, at the only values this code
# reproduces: each was off there.
LEGACY_FIELDS = {
    "infer_sign_flip": False,
    "infer_temperature": INFER_TEMPERATURE,
    "infer_backtest": INFER_BACKTEST,
    "forecast_clamp": None,
    "combiner": "mean",
}


LIST_FIELDS = ("infer_candidates", "polyphase_forecast_strides")


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
    infer_candidates: tuple = ()  # from INFER_CANDIDATES, gated by a backtest on the context
    infer_margin: float = 0.1  # relative backtest gain a candidate needs to beat the model
    infer_min_season: float = 0.0  # autocorrelation at the period that opens the candidates
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
    # Forecast-time options; they change forecast(), never the weights or training.
    flip_symmetric: bool = False  # average f(x) with -f(-x), quantile order reversed
    clamp_positive: bool = False  # no quantile below 0 when the observed history is >= 0

    def __post_init__(self):
        # Lists, as config.json holds them, so that asdict(config) equals the recorded config.
        for key in LIST_FIELDS:
            object.__setattr__(self, key, list(getattr(self, key)))
        if set(self.infer_candidates) - set(INFER_CANDIDATES):
            raise ValueError(f"infer_candidates must come from {INFER_CANDIDATES}")
        if self.infer_margin < 0 or self.infer_clamp < 0:
            raise ValueError("infer_margin and infer_clamp must be >= 0")
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
        for key in ("infer_views", "infer_sign_clip", "infer_integer_snap"):
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
        if type(self.phase_tokens) is not bool:
            raise ValueError("phase_tokens must be true or false")


def read_config(fields: dict) -> ModelConfig:
    """The ModelConfig of a checkpoint's fields, dropping LEGACY_FIELDS at their old values."""
    fields = dict(fields)
    for key, value in LEGACY_FIELDS.items():
        if key in fields and fields.pop(key) != value:
            raise ValueError(f"This code forecasts only with {key} = {value!r}")
    return ModelConfig(**fields)


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

    def next_patch(self, values: torch.Tensor, series_ids: torch.Tensor):
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
            periods = detect_periods(values[..., : self.config.context_length])[..., None, :]
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
        context = base.config.context_length * max(self.strides)
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


def load_checkpoint(path: Path, device="cpu"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 3:
        raise ValueError("Unsupported checkpoint version")
    model = NanoTSFM(read_config(checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    if model.config.polyphase_forecast_strides:
        model = Polyphase(model)
    model.to(device_for(str(device))).eval()
    return model, checkpoint


def rollout(model, history: torch.Tensor, horizon: int, series_ids: torch.Tensor) -> torch.Tensor:
    """Feed medians back past one model pass."""
    context, patch = model.config.context_length, model.config.patch_length
    blocks = []
    remaining = horizon
    while remaining:
        prediction = model(history[..., -context:], series_ids)
        take = min(remaining, prediction.shape[-2])
        blocks.append(prediction[..., :take, :])
        history = torch.cat([history, prediction[..., 4]], dim=-1)
        history = history[..., -(min(context, history.shape[-1]) // patch * patch) :]
        remaining -= take
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


def seasonal_profile(filled: torch.Tensor, period: torch.Tensor, horizon: int):
    """Each series' mean cycle, zero-centered, over the history and over the horizon."""
    length = filled.shape[-1]
    steps = torch.arange(length + horizon, device=filled.device)
    phase = (length - 1 - steps) % period[..., None]
    past = phase[..., :length]
    observed = filled.isfinite().to(filled.dtype)
    width = int(period.max())
    total = torch.zeros(*filled.shape[:-1], width, device=filled.device, dtype=filled.dtype)
    count = total.clone()
    total.scatter_add_(-1, past, filled.nan_to_num(0.0) * observed)
    count.scatter_add_(-1, past, observed)
    mean = total / count.clamp_min(1)
    mean = mean - mean.sum(-1, keepdim=True) / (count > 0).sum(-1, keepdim=True).clamp_min(1)
    profile = mean.gather(-1, phase)
    return profile[..., :length], profile[..., length:]


def signed_forecast(model, history, horizon: int, series_ids) -> torch.Tensor:
    """The forecast of -x, mapped back: q_t(x) = -q_{1-t}(-x)."""
    return -rollout(model, -history, horizon, series_ids).flip(-1)


def views_forecast(model, history, horizon: int, series_ids) -> torch.Tensor:
    """Per quantile, the mean of the middle two of four forecasts: of x, of -x, and for a
    never-negative series of sqrt(x) and -sqrt(x), each mapped back. Any other series counts its
    first two views twice. Every view is a forecast of this model from the series' own history."""
    native = rollout(model, history, horizon, series_ids)
    flipped = signed_forecast(model, history, horizon, series_ids)
    observed = history.isfinite()
    nonneg = observed.any(-1) & torch.where(observed, history >= 0, True).all(-1)  # [B,V]
    members = [native, flipped]
    rows = nonneg.any(-1)
    if rows.any():
        root = torch.where(observed[rows], history[rows].clamp_min(0).sqrt(), torch.nan)
        keep = nonneg[rows][..., None, None]
        views = (
            (rollout(model, root, horizon, series_ids[rows]), native),
            (signed_forecast(model, root, horizon, series_ids[rows]), flipped),
        )
        for view, fallback in views:
            full = fallback.clone()
            full[rows] = torch.where(keep, view.sign() * view.square(), fallback[rows])
            members.append(full)
    else:
        members += [native, flipped]
    ordered = torch.stack([member.sort(-1).values for member in members]).sort(0).values
    return ordered[1:3].mean(0)


def model_forecast(model, history, horizon: int, series_ids) -> torch.Tensor:
    config = model.config
    if config.infer_views:  # views over the first pass; past it, the plain rollout
        first = config.prediction_length
        native = rollout(model, history, horizon, series_ids)
        near = views_forecast(model, history, min(horizon, first), series_ids)
        return torch.cat([near, native[..., first:, :]], dim=-2).sort(-1).values
    return rollout(model, history, horizon, series_ids)


def candidate_forecasts(model, history, horizon: int, series_ids, period, base=None):
    """Every configured forecast, [candidates,B,V,H,9]; the model itself comes first."""
    if base is None:
        base = model_forecast(model, history, horizon, series_ids)
    forecasts = [base]
    filled = forward_fill(history)
    for _ in model.config.infer_candidates:  # "deseason", the only candidate
        past, future = seasonal_profile(filled, period, horizon)
        residual = rollout(model, history - past, horizon, series_ids)
        forecasts.append(residual + future[..., None])
    return torch.stack(forecasts)


def backtest_scores(model, history, horizon: int, series_ids, period) -> torch.Tensor:
    """Pinball loss of each candidate on the end of each series' own context, [candidates,B,V].

    Each series holds back min(max(horizon, C/4), C/2, half its observed length) points. A series
    with fewer than one patch to spare scores 0 everywhere, which keeps the model."""
    context, patch = model.config.context_length, model.config.patch_length
    lengths = history.isfinite().cumsum(-1).clamp_max(1).sum(-1)
    held = min(max(horizon, int(context * INFER_BACKTEST)), context // 2)
    held = (lengths // 2).clamp_max(held)
    held = torch.where(held >= patch, held, 0)
    longest = int(held.max())
    if not longest:
        return torch.zeros(1 + len(model.config.infer_candidates), *history.shape[:2])
    length = history.shape[-1]
    steps = torch.arange(length, device=history.device)
    origin = steps - held[..., None]
    past = history.gather(-1, origin.clamp_min(0))
    past = torch.where(origin >= 0, past, torch.nan)
    target_steps = length - held[..., None] + torch.arange(longest, device=history.device)
    target = history.gather(-1, target_steps.clamp_max(length - 1))
    target = torch.where(target_steps < length, target, torch.nan)
    predictions = candidate_forecasts(model, past, longest, series_ids, period)
    levels = torch.tensor(QUANTILES, device=history.device, dtype=torch.float64)
    # Double, so that series near the float32 limit cannot overflow the loss into NaN weights.
    error = target[None, ..., None].double() - predictions.double()
    loss = torch.maximum(levels * error, (levels - 1) * error).nan_to_num(0.0).sum(-1)
    return loss.sum(-1) / target.isfinite().sum(-1).clamp_min(1)


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


def gated_forecast(model, history, horizon: int, series_ids) -> torch.Tensor:
    """Weight the candidates per series by their backtest loss, then clamp to the context."""
    config = model.config
    result = model_forecast(model, history, horizon, series_ids)
    if config.infer_candidates:
        filled = forward_fill(history)
        period, strength = infer_period(filled, filled.isfinite().sum(-1))
        # Candidates cost model passes and add noise, so only clearly seasonal series get them.
        allowed = strength >= config.infer_min_season
        rows = allowed.any(-1)
        if rows.any():
            arguments = history[rows], horizon, series_ids[rows], period[rows]
            forecasts = candidate_forecasts(model, *arguments, base=result[rows])
            scores = backtest_scores(model, *arguments).to(forecasts.device)
            relative = scores / scores.amin(0).clamp_min(MIN_SCALE)
            relative[1:] += config.infer_margin
            shut = (scores == 0).all(0) | ~allowed[rows]  # no backtest, or no clear season
            relative[1:] = torch.where(shut, torch.inf, relative[1:])
            weights = (-relative / INFER_TEMPERATURE).softmax(0)
            forecasts = torch.where(forecasts.isfinite(), forecasts, forecasts[:1])
            blend = weights[..., None, None].to(forecasts.dtype) * forecasts
            blend = blend.sum(0).sort(-1).values
            result = result.clone()
            result[rows] = blend
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
    device = next(model.parameters()).device
    history = history.to(device=device, dtype=torch.float32)
    if series_ids is None:
        series_ids = torch.zeros(history.shape[:2], dtype=torch.long, device=device)
    else:
        series_ids = series_ids.to(device)
    if series_ids.shape != history.shape[:2]:
        raise ValueError("series_ids must match batch and variate dimensions")
    context = model.config.context_length
    history = F.pad(history, (max(0, context - history.shape[-1]), 0), value=float("nan"))
    was_training = model.training
    model.eval()
    config = model.config
    tricks = (
        config.infer_candidates
        or config.infer_clamp
        or config.infer_tail_floor
        or config.infer_views
        or config.infer_sign_clip
        or config.infer_integer_snap
    )
    try:
        predict = gated_forecast if tricks else rollout
        result = predict(model, history, horizon, series_ids)
        return result[:, 0] if univariate else result
    finally:
        model.train(was_training)
