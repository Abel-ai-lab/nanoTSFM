import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from nanotsfm.data import MAX_VARIATES, pack_windows

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
MIN_SCALE = 1e-5
INFER_CANDIDATES = ("rate2", "rate4", "seasonal", "deseason", "flip")
COMBINERS = ("mean", "median", "asinh_mean")
SWITCHES = (
    "far_tiers",
    "scale_halflife",
    "mask_padded",
    "quantile_levels",
    "coarse_blocks",
    "last_patch_anchor",
    "shape_codes",
    "future_query",
    "shifted_block",
    "gap_embedding",
    "scale_features",
    "infer_shifts",
    "markov_path",
    "zero_init",
    "drop_path",
    "stat_pool",
    "lhuc",
    "seed_bias",
    "learned_level_weights",
    "rv_aux",
    "next_next",
    "tied_blocks",
    "block_heads",
    "polyphase_forecast_strides",
    "polyphase_forecast_period",
    "polyphase_joint",
    "residual_target",
    "phase_tokens",
    "phase_bins",
    "rope_periods",
    "refine_masked_stats",
    "infer_contexts",
    "rate_embedding",
    "head_hidden",
)
RESIDUAL_TARGETS = ("", "increment")
# The shape vocabulary; see ShapeVocabulary.
SHAPE_LATENT = 32
SHAPE_HIDDEN = 128
SHAPE_COMMITMENT = 0.25  # the VQ-VAE paper's value
SHAPE_MIN_SEEN = 0.5  # share of a patch that must be observed for it to have a shape
SHAPE_FLAT_RELATIVE = 1e-6  # spread below this share of the level is float32 rounding
SHAPE_MAX_LOG_SPREAD = 6.0  # bound on the log ratio of a patch's spread to the running scale
SHAPE_REVIVE_EVERY = 50  # steps between revivals
SHAPE_DEAD_SHARE = 1 / 16  # a code below this share of a fair split of the wins is dead
SHAPE_SPLIT_NOISE = 0.1  # how far a revived code starts from the code it splits
# log scale and asinh(loc / scale) span about +-20; this brings them near 1.
SCALE_FEATURE_GAIN = 0.1
# Means of 5 stretches of a patch give 4 rises or falls, so 16 words: chance alone matches 1 pair
# in 16, which keeps seed hits above seed_bias's kill line of 5% of pairs.
WORD_SEGMENTS = 5
POOL_FLOOR = 1e-4  # keeps the square root of a token's running variance smooth at zero
POLYPHASE_MIN_PATCHES = 2  # a stride joins a series' forecast if each phase has this much history
PERIOD_TARGET = 24  # the points of its main cycle a period-locked stride leaves a series
PERIOD_STRIDES = (2, 12)  # the least and most such a stride can be
PERIOD_SPAN = 4096  # points read for a period: what a stride-4 forecast sees, and training too
PERIOD_MIN_STRENGTH = 0.3  # autocorrelation at the period below which a series has no clear cycle
PERIOD_MIN_POINTS = 64  # each phase keeps this many, two patches of the base model
PHASE_SLOTS = 4  # TinyCast's K: a series can hold a daily and a weekly period together
PHASE_ALPHA = 0.05  # TinyCast's Fisher test level, Bonferroni-corrected across bins
REFINE_MAX_SCALED = 6.0  # bound on a masked patch's median in asinh units, as with shape spreads


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
    infer_sign_flip: bool = False  # average forecasts of x and -x
    infer_temperature: float = 0.1  # softmax over relative backtest loss; 0 picks the best
    infer_margin: float = 0.1  # relative backtest gain a candidate needs to beat the model
    infer_min_season: float = 0.0  # autocorrelation at the period that opens the candidates
    infer_backtest: float = 0.25  # least share of the context the backtest holds back
    infer_clamp: float = 0.0  # keep quantiles within this many context ranges of the context
    # More forecast-time options; neither changes training, and the defaults change nothing.
    forecast_clamp: float | None = None  # K in forecast_range; None leaves forecasts unbounded
    combiner: str = "mean"  # how an ensemble joins its members: one of COMBINERS
    # Older history as coarse tokens; the default is the plain model. See `far_embed`.
    far_tiers: tuple = ()  # (rate, tokens) pairs, nearest first, within context_length
    # Structural switches; each default is the plain model.
    scale_halflife: int = 0  # points over which causal_scale halves a point's weight; 0 never does
    mask_padded: bool = False  # no attention to all-missing tokens in the time blocks
    quantile_levels: tuple = ()  # levels the head trains on, holding QUANTILES; () is QUANTILES
    coarse_blocks: tuple = ()  # training-only heads for means of this many future patches
    last_patch_anchor: bool = False  # add the last patch's scaled mean through a learned scale
    # Learned patch shapes added to the tokens; 0 is the plain model. See `ShapeVocabulary`.
    shape_codes: int = 0

    # A joint decoder of every future patch in place of rollout; see `future_quantiles`.
    future_query: bool = False
    future_query_max_horizon: int = 900
    # Small switches from the audit's Final 40; each default is the plain model.
    shifted_block: bool = False  # one more time block, reading the token after each match
    gap_embedding: bool = False  # tell each token its patches past the last observed one
    scale_features: bool = False  # give each token its log scale, loc over scale and zero share
    infer_shifts: tuple = ()  # also forecast without the newest n points, and pool; forecast only
    # Switches from the verified audit order; each default is the plain model.
    markov_path: int = 0  # a learned linear map of this many latest patches joins the output
    zero_init: bool = False  # residual branches and the head start at zero
    drop_path: float = 0.0  # chance the last time block's branches drop, rising to it by block
    stat_pool: bool = False  # each token gets its series' running token mean and std
    lhuc: bool = False  # that pool sets a gain on each time block's hidden units; needs stat_pool
    seed_bias: bool = False  # a learned attention bonus to earlier tokens with the same word
    learned_level_weights: bool = False  # the loss learns one log scale per quantile level
    rv_aux: float = 0.0  # weight of a training-only head for the next patch's log variance
    next_next: float = 0.0  # weight of a training-only head for the patch after next
    # Switches from the winners audit; each default is the plain model.
    variate_blocks: int = 1  # 0 reads every channel on its own, with no variate block
    tied_blocks: int = 0  # one time block called this many times in place of layers blocks
    block_heads: float = 0.0  # weight of training-only heads on each time block's tokens
    # The interlaced forecast; see Polyphase. Forecast only, so it can re-score a checkpoint.
    polyphase_forecast_strides: tuple = ()  # strides whose phases are forecast and interlaced
    polyphase_forecast_weight: float = 0.25  # the interlaced forecasts' share of the blend
    polyphase_forecast_period: bool = False  # a series with a clear cycle uses its locked stride
    polyphase_joint: bool = False  # a series' phases are forecast together, as its channels
    # The head's frame; "increment" is the change from the last value in mean absolute steps.
    residual_target: str = ""
    # Periods found in the context as coordinates; see detect_periods.
    phase_tokens: bool = False  # each token gets sin and cos of its phase in each period
    phase_bins: int = 0  # bins per period of the template that future positions read; 0 is none
    # Switches from the unused-papers audit; each default is the plain model.
    rope_periods: bool = False  # one head's first rotations turn with the context's periods
    refine_masked_stats: bool = False  # output frames follow the median through masked patches
    infer_contexts: tuple = ()  # lengths of the context pool; forecast only, see Polyphase
    infer_context_weight: float = 0.25  # the shorter contexts' share of the normal forecast
    rate_embedding: tuple = ()  # rising strides from 1, with a token each; see rate_token
    sort_in_training: bool = True  # False scores the head's raw quantiles; forecasts still sort
    head_hidden: int = 0  # hidden size of a residual MLP ahead of the head; 0 is none

    @property
    def far_length(self) -> int:
        """How many of the context's oldest points enter through far_tiers."""
        return sum(rate * tokens for rate, tokens in self.far_tiers) * self.patch_length

    def __post_init__(self):
        object.__setattr__(self, "infer_candidates", tuple(self.infer_candidates))
        object.__setattr__(self, "far_tiers", tuple(tuple(tier) for tier in self.far_tiers))
        object.__setattr__(self, "quantile_levels", tuple(self.quantile_levels))
        object.__setattr__(self, "coarse_blocks", tuple(self.coarse_blocks))
        object.__setattr__(self, "infer_shifts", tuple(self.infer_shifts))
        object.__setattr__(self, "infer_contexts", tuple(self.infer_contexts))
        object.__setattr__(self, "rate_embedding", tuple(self.rate_embedding))
        strides = tuple(self.polyphase_forecast_strides)
        object.__setattr__(self, "polyphase_forecast_strides", strides)
        if set(self.infer_candidates) - set(INFER_CANDIDATES):
            raise ValueError(f"infer_candidates must come from {INFER_CANDIDATES}")
        if self.infer_temperature < 0 or self.infer_margin < 0 or self.infer_clamp < 0:
            raise ValueError("infer_temperature, infer_margin and infer_clamp must be >= 0")
        for key in (
            "context_length",
            "prediction_length",
            "patch_length",
            "width",
            "layers",
            "heads",
            "future_query_max_horizon",
        ):
            if getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.context_length % self.patch_length:
            raise ValueError("context_length must divide into patches")
        tiers = self.far_tiers
        if any(len(tier) != 2 or any(type(v) is not int for v in tier) for tier in tiers):
            raise ValueError("far_tiers must be (rate, tokens) integer pairs")
        rates = [1, *(rate for rate, _ in tiers)]
        if rates != sorted(set(rates)) or any(tokens <= 0 for _, tokens in tiers):
            raise ValueError("far_tiers rates must rise from above 1, with positive token counts")
        if self.far_length >= self.context_length:
            raise ValueError("far_tiers must leave recent patches in context_length")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width must split into heads of even size")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        clamp = self.forecast_clamp
        if clamp is not None and (type(clamp) not in (int, float) or not 0 < clamp < float("inf")):
            raise ValueError("forecast_clamp must be finite and positive, or None")
        if self.combiner not in COMBINERS:
            raise ValueError(f"combiner must be one of {COMBINERS}")
        halflife = self.scale_halflife
        if type(halflife) is not int or (halflife and halflife < self.patch_length):
            raise ValueError("scale_halflife must be 0 or an integer of at least patch_length")
        levels = self.quantile_levels
        if levels and (
            set(QUANTILES) - set(levels)
            or any(not 0 < level < 1 for level in levels)
            or list(levels) != sorted(set(levels))
        ):
            raise ValueError("quantile_levels must rise within (0, 1) and hold QUANTILES")
        patches = self.context_length // self.patch_length
        patches += math.ceil(self.prediction_length / self.patch_length)
        if any(type(block) is not int or not 2 <= block < patches for block in self.coarse_blocks):
            raise ValueError(
                "coarse_blocks must be integers from 2 to under the sequence's patches"
            )
        if type(self.shape_codes) is not int or self.shape_codes < 0 or self.shape_codes == 1:
            raise ValueError("shape_codes must be 0, for none, or an integer of at least 2")
        if any(type(shift) is not int or shift <= 0 for shift in self.infer_shifts):
            raise ValueError("infer_shifts must be positive integers")
        if type(self.markov_path) is not int or self.markov_path < 0:
            raise ValueError("markov_path must be a nonnegative integer")
        if not 0 <= self.drop_path < 1:
            raise ValueError("drop_path must be in [0, 1)")
        if self.lhuc and not self.stat_pool:
            raise ValueError("lhuc reads the statistics pool, so it needs stat_pool")
        if self.seed_bias and self.far_tiers:
            raise ValueError("seed_bias reads recent patches only, so it excludes far_tiers")
        if self.seed_bias and self.patch_length < WORD_SEGMENTS:
            raise ValueError(f"seed_bias needs patch_length of at least {WORD_SEGMENTS}")
        if self.rv_aux < 0 or self.next_next < 0 or self.block_heads < 0:
            raise ValueError("rv_aux, next_next and block_heads must be nonnegative")
        if self.variate_blocks not in (0, 1) or type(self.variate_blocks) is not int:
            raise ValueError("variate_blocks must be 0 or 1")
        if type(self.tied_blocks) is not int or self.tied_blocks < 0:
            raise ValueError("tied_blocks must be a nonnegative integer")
        if self.shape_codes and self.far_tiers:
            raise ValueError("shape_codes reads recent patches only, so it excludes far_tiers")
        if any(type(stride) is not int or stride < 2 for stride in strides):
            raise ValueError("polyphase_forecast_strides must be integers of at least 2")
        if not 0 <= self.polyphase_forecast_weight <= 1:
            raise ValueError("polyphase_forecast_weight must be from 0 to 1")
        if type(self.polyphase_forecast_period) is not bool:
            raise ValueError("polyphase_forecast_period must be true or false")
        if self.polyphase_forecast_period and not strides:
            raise ValueError("polyphase_forecast_period falls back on polyphase_forecast_strides")
        if type(self.polyphase_joint) is not bool:
            raise ValueError("polyphase_joint must be true or false")
        if self.polyphase_joint and not (strides and self.variate_blocks):
            raise ValueError("polyphase_joint needs polyphase_forecast_strides and a variate block")
        if self.residual_target not in RESIDUAL_TARGETS:
            raise ValueError(f"residual_target must be one of {RESIDUAL_TARGETS}")
        if self.residual_target and self.last_patch_anchor:
            raise ValueError("last_patch_anchor adds to the input's frame, not residual_target's")
        if type(self.phase_tokens) is not bool:
            raise ValueError("phase_tokens must be true or false")
        if type(self.phase_bins) is not int or self.phase_bins < 0 or self.phase_bins == 1:
            raise ValueError("phase_bins must be 0, for none, or an integer of at least 2")
        if (self.phase_tokens or self.phase_bins) and self.far_tiers:
            raise ValueError("phase_tokens and phase_bins read recent patches, not far_tiers")
        if type(self.rope_periods) is not bool or type(self.refine_masked_stats) is not bool:
            raise ValueError("rope_periods and refine_masked_stats must be true or false")
        if self.rope_periods and self.far_tiers:
            raise ValueError("rope_periods reads recent patches, not far_tiers")
        if self.rope_periods and self.width // self.heads // 2 < PHASE_SLOTS:
            raise ValueError(f"rope_periods needs {PHASE_SLOTS} pairs in a head")
        frames = (
            self.far_tiers,
            self.scale_halflife,
            self.residual_target,
            self.last_patch_anchor,
            self.markov_path,
        )
        if self.refine_masked_stats and any(frames):
            raise ValueError(
                "refine_masked_stats walks causal_scale's frame over recent patches, so it "
                "excludes far_tiers, scale_halflife, residual_target, last_patch_anchor and "
                "markov_path"
            )
        contexts = self.infer_contexts
        if any(
            type(length) is not int
            or length <= 0
            or length % self.patch_length
            or length > self.context_length
            for length in contexts
        ):
            raise ValueError("infer_contexts must be whole patches within context_length")
        if contexts and not strides:
            raise ValueError("infer_contexts pools in Polyphase, so it needs its strides")
        if not 0 <= self.infer_context_weight <= 1:
            raise ValueError("infer_context_weight must be from 0 to 1")
        tagged = self.rate_embedding
        if tagged and (
            any(type(rate) is not int for rate in tagged)
            or tagged[0] != 1
            or list(tagged) != sorted(set(tagged))
        ):
            raise ValueError("rate_embedding must be integers rising from 1")
        if tagged and any(name.startswith("rate") for name in self.infer_candidates):
            raise ValueError("rate_embedding doesn't reach the rate candidates' forecasts")
        if type(self.sort_in_training) is not bool:
            raise ValueError("sort_in_training must be true or false")
        if type(self.head_hidden) is not int or self.head_hidden < 0:
            raise ValueError("head_hidden must be a nonnegative integer")
        replaced = (
            self.far_tiers
            or self.quantile_levels
            or self.coarse_blocks
            or self.last_patch_anchor
            or self.shape_codes
            or self.markov_path
            or self.learned_level_weights
            or self.rv_aux
            or self.next_next
            or self.block_heads
            or strides
            or self.residual_target
            or self.phase_tokens
            or self.phase_bins
            or self.rope_periods
            or self.refine_masked_stats
            or tagged
            or self.head_hidden
        )
        if self.future_query and replaced:
            raise ValueError(
                "future_query has its own head and loss, so it excludes far_tiers, "
                "quantile_levels, coarse_blocks, last_patch_anchor, shape_codes, markov_path, "
                "learned_level_weights, rv_aux, next_next, block_heads, "
                "polyphase_forecast_strides, residual_target, phase_tokens, phase_bins, "
                "rope_periods, refine_masked_stats, rate_embedding and head_hidden"
            )


def config_fields(config: ModelConfig) -> dict:
    """The fields a run records; a plain model records what it did before these fields existed."""
    fields = asdict(config)
    for key in SWITCHES:
        if not fields[key]:
            del fields[key]
    if not config.future_query:
        del fields["future_query_max_horizon"]
    if config.variate_blocks == 1:
        del fields["variate_blocks"]
    if not config.polyphase_forecast_strides:
        del fields["polyphase_forecast_weight"]
    if not config.infer_contexts:
        del fields["infer_context_weight"]
    if config.sort_in_training:
        del fields["sort_in_training"]
    return fields


def forgetting_scale(values: torch.Tensor, patch: int, halflife: int):
    """causal_scale with weights that halve every `halflife` points back from the latest one.

    Weights grow along time rather than fall, so running sums forget without a loop: only their
    ratios matter. As halflife grows this becomes causal_scale, and the same Welford update holds.
    """
    observed = torch.isfinite(values)
    x = torch.where(observed, values, 0).double()
    steps = torch.arange(values.shape[-1], device=values.device, dtype=torch.float64)
    weight = 2 ** (steps / halflife) * observed
    total = weight.cumsum(-1).clamp_min(1)  # the first weight is 1, so this only guards no data
    loc = (weight * x).cumsum(-1) / total
    previous = F.pad(loc[..., :-1], (1, 0))
    m2 = (weight * (x - previous) * (x - loc)).cumsum(-1)
    # The reliability form of count - 1, and at least one point's weight, as causal_scale has.
    room = (total - weight.square().cumsum(-1) / total).maximum(weight.cummax(-1).values)
    std = (m2 / room.clamp_min(1)).clamp_min(MIN_SCALE**2).sqrt()
    end = slice(patch - 1, None, patch)
    return loc[..., end], std[..., end]


def patch_gaps(padded: torch.Tensor) -> torch.Tensor:
    """Patches since the latest observed one at or before each position, 0 on an observed patch.
    Before the first observed patch this is the position + 1, as if one lay just before it."""
    positions = torch.arange(padded.shape[-1], device=padded.device)
    latest = torch.where(padded, -1, positions).cummax(-1).values
    return torch.where(latest >= 0, positions - latest, positions + 1)


def increment_frame(values: torch.Tensor, patch: int, scale: torch.Tensor):
    """residual_target's loc and scale at each patch end: the last observed value, and the mean
    absolute step between neighboring observed points so far. A gap's step counts once. With no
    step yet, scale stands in, which is at its floor there."""
    last = forward_fill(values.double())
    steps = last.diff(dim=-1).abs()
    usable = steps.isfinite() & values[..., 1:].isfinite()
    total = F.pad(torch.where(usable, steps, 0).cumsum(-1), (1, 0))
    count = F.pad(usable.cumsum(-1), (1, 0))
    end = slice(patch - 1, None, patch)
    unit = (total[..., end] / count[..., end].clamp_min(1)).clamp_min(MIN_SCALE)
    return last[..., end].nan_to_num(0.0), torch.where(count[..., end] > 0, unit, scale)


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


def forecast_range(history: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor, strength: float):
    """Lower and upper bounds [B,V] on predicted quantiles, in the asinh-scaled space of the output:
    the range of the [B,V,T] history in that space, widened by strength * n ** -0.25 on each side,
    where n counts observed points. The fewer the points, the less the range says about the future.

    A history with no spread (constant, or under two points) has no scale to bound by, and its
    future often does move, so it stays unbounded.
    """
    observed = torch.isfinite(history)
    scaled = torch.where(observed, history, 0).double().sub(loc[..., None]).div(scale[..., None])
    # 0 is the history's mean, which lies inside its range, so empty histories need no special case.
    scaled = torch.where(observed, scaled.asinh(), 0)
    margin = strength * observed.sum(-1).double() ** -0.25
    flat = scale <= 2 * MIN_SCALE
    lower = (scaled.amin(-1) - margin).masked_fill(flat, -math.inf)
    upper = (scaled.amax(-1) + margin).masked_fill(flat, math.inf)
    return lower, upper


def patch_words(patches: torch.Tensor) -> torch.Tensor:
    """Each [...,L] patch's word: whether its stretches' means rise or fall in turn, as an integer.
    A patch with an empty stretch or a tie has no word, and gets -1."""
    size = patches.shape[-1] // WORD_SEGMENTS
    stretches = patches[..., : size * WORD_SEGMENTS].unflatten(-1, (WORD_SEGMENTS, size))
    observed = stretches.isfinite()
    means = torch.where(observed, stretches, 0).double().sum(-1) / observed.sum(-1).clamp_min(1)
    steps = means.diff(dim=-1)
    place = 2 ** torch.arange(WORD_SEGMENTS - 1, device=patches.device)
    readable = observed.any(-1).all(-1) & (steps != 0).all(-1)
    return torch.where(readable, ((steps > 0) * place).sum(-1), -1)


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


def running_stats(x: torch.Tensor) -> torch.Tensor:
    """The mean and std of each [N,S,W] token and those before it, [N,S,2W]; never later ones."""
    x = x.float()
    count = torch.arange(1, x.shape[1] + 1, device=x.device, dtype=x.dtype)[:, None]
    mean = x.cumsum(1) / count
    variance = (x.square().cumsum(1) / count - mean.square()).clamp_min(0)
    return torch.cat([mean, (variance + POOL_FLOOR).sqrt()], dim=-1)


def zero_linear(inputs: int, outputs: int) -> nn.Linear:
    """A Linear at zero that draws no random numbers, so a seed's other weights stay paired."""
    layer = nn.utils.skip_init(nn.Linear, inputs, outputs)
    nn.init.zeros_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


def rotate(x: torch.Tensor, cycles: torch.Tensor | None = None) -> torch.Tensor:
    """RoPE on [N,H,S,D]. With rope_periods' [N,PHASE_SLOTS] cycles per token, the first head's
    first pairs turn with each sequence's own periods, and a slot at 0 keeps its fixed frequency."""
    half = x.shape[-1] // 2
    frequency = 10000 ** (-torch.arange(half, device=x.device, dtype=torch.float32) / half)
    place = torch.arange(x.shape[-2], device=x.device, dtype=torch.float32)[:, None]
    angle = place * frequency
    if cycles is not None:
        fixed = angle[:, :PHASE_SLOTS]
        locked = 2 * math.pi * place * cycles[:, None]
        angle = angle.expand(*x.shape[:2], -1, -1).clone()
        angle[:, 0, :, :PHASE_SLOTS] = torch.where(cycles[:, None] > 0, locked, fixed)
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    first, second = x[..., :half], x[..., half:]
    return torch.cat([first * cos - second * sin, first * sin + second * cos], dim=-1)


class Block(nn.Module):
    def __init__(
        self, config: ModelConfig, causal: bool, shifted: bool = False, drop_path: float = 0.0
    ):
        """With shifted, a position that matches token j reads the value of token j + 1: what
        followed the match, rather than the match. Only earlier tokens can be read, so the first
        position's one key is the last token, whose value is zero."""
        super().__init__()
        self.heads, self.causal, self.dropout = config.heads, causal, config.dropout
        self.shifted, self.drop_path = shifted, drop_path
        self.drop_live = True  # training's drop_window turns drop_path off between steps
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

    def dropped(self, branch: torch.Tensor) -> torch.Tensor:
        """In training, drop this residual branch for a share of sequences, keeping its mean."""
        if not self.training or not self.drop_path or not self.drop_live:
            return branch
        keep = 1 - self.drop_path
        return branch * (torch.rand(len(branch), 1, 1, device=branch.device) < keep) / keep

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
        gains: torch.Tensor | None = None,
        cycles: torch.Tensor | None = None,
    ) -> torch.Tensor:
        n, s, width = x.shape
        q, k, v = self.qkv(self.attention_norm(x)).view(n, s, 3, self.heads, -1).unbind(2)
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))
        if self.causal:
            q, k = rotate(q, cycles), rotate(k, cycles)
        if self.shifted:
            v = torch.cat([v[:, :, 1:], torch.zeros_like(v[:, :, :1])], dim=2)
            mask = torch.ones(s, s, dtype=torch.bool, device=x.device).tril(-1)
            mask[0, -1] = True
        attended = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=self.causal and mask is None,  # a given mask already holds causality
        )
        x = x + self.dropped(self.out(attended.transpose(1, 2).reshape(n, s, width)))
        if gains is None:
            return x + self.dropped(self.mlp(self.mlp_norm(x)))
        hidden = self.mlp[1](self.mlp[0](self.mlp_norm(x))) * gains
        return x + self.dropped(self.mlp[3](self.mlp[2](hidden)))


class ShapeVocabulary(nn.Module):
    """A learned alphabet of patch shapes, so one shape reads the same in every series.

    Each patch, centered and divided by its own spread, is encoded and snapped to the nearest of
    shape_codes unit vectors. That letter's embedding and the patch's level and spread, relative to
    the running loc and scale, join the numeric token; both start at zero, so a new model forecasts
    as the plain one does. Two more letters stand for patches with no shape: flat ones, and ones
    too sparse to read, which include every masked or future patch.

    Only the letter embeddings and the level link feel the forecast loss. The encoder, decoder and
    codes train on reconstructing the shape, with the usual VQ terms: the codes move to the
    encodings, whose gradient is stopped, and the commitment term moves the encodings to the codes.
    Reconstruction is what keeps the encoder from mapping everything to one code. Its gradient
    skips the snap straight through.

    A code that stops winning splits the busiest one instead of copying a training patch, so the
    weights hold no training data.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        patch, codes = config.patch_length, config.shape_codes
        self.encoder = nn.Sequential(
            nn.Linear(2 * patch, SHAPE_HIDDEN), nn.GELU(), nn.Linear(SHAPE_HIDDEN, SHAPE_LATENT)
        )
        self.decoder = nn.Sequential(
            nn.Linear(SHAPE_LATENT, SHAPE_HIDDEN), nn.GELU(), nn.Linear(SHAPE_HIDDEN, patch)
        )
        self.codes = nn.Parameter(torch.randn(codes, SHAPE_LATENT))
        self.letters = nn.Parameter(torch.zeros(codes + 2, config.width))
        self.link = nn.Linear(2, config.width)
        nn.init.zeros_(self.link.weight)
        nn.init.zeros_(self.link.bias)
        self.flat, self.blank = codes, codes + 1
        self.register_buffer("wins", torch.zeros(codes, dtype=torch.long))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))

    def read(self, values: torch.Tensor):
        """Split [B,V,P,L] patches into the shape of each and its own mean, spread and masks."""
        observed = values.isfinite()
        seen = observed.sum(-1)
        clean = torch.where(observed, values, 0).double()
        mean = clean.sum(-1) / seen.clamp_min(1)
        centered = (clean - mean[..., None]) * observed
        spread = (centered.square().sum(-1) / seen.clamp_min(1)).sqrt()
        flat = spread <= MIN_SCALE + SHAPE_FLAT_RELATIVE * mean.abs()
        sparse = seen < SHAPE_MIN_SEEN * values.shape[-1]
        shape = (centered / spread.clamp_min(MIN_SCALE)[..., None]).float()
        return shape, observed, mean, spread, flat, sparse

    def nearest(self, shape: torch.Tensor, observed: torch.Tensor):
        """Unit encodings of [N,L] shapes, the unit codes, and each shape's nearest code."""
        encoded = F.normalize(self.encoder(torch.cat([shape, observed.float()], dim=-1)), dim=-1)
        codes = F.normalize(self.codes, dim=-1)
        return encoded, codes, (encoded @ codes.T).argmax(-1)

    def forward(self, values, loc, scale, aux: bool = False):
        """Add-on tokens [B,V,P,W] for [B,V,P,L] patches, and with aux the vocabulary's own loss."""
        # Bfloat16 distances between these patches would blur nearby codes.
        with torch.autocast(values.device.type, enabled=False):
            shape, observed, mean, spread, flat, sparse = self.read(values)
            seen = observed.sum(-1)
            letter = torch.where(sparse, self.blank, self.flat)
            readable = ~flat & ~sparse
            shape, observed = shape[readable], observed[readable]
            encoded, codes, chosen = self.nearest(shape, observed)
            letter = letter.masked_scatter(readable, chosen)

            level = ((mean - loc) / scale).asinh()
            bound = math.exp(SHAPE_MAX_LOG_SPREAD)
            ratio = (spread / scale).clamp(1 / bound, bound).log()
            features = torch.stack([level * (seen > 0), ratio * (seen > 1)], -1).float()
            tokens = self.letters[letter] + self.link(features)
            if not aux:
                return tokens, None
            self.wins += chosen.bincount(minlength=len(self.wins))
            if not len(chosen):
                return tokens, tokens.new_zeros(())
            code = codes[chosen]
            decoded = self.decoder(encoded + (code - encoded).detach())
            reconstruction = (decoded - shape).square().mul(observed).sum() / observed.sum()
            codebook = (code - encoded.detach()).square().sum(-1).mean()
            commitment = (encoded - code.detach()).square().sum(-1).mean()
            return tokens, reconstruction + codebook + SHAPE_COMMITMENT * commitment

    @torch.no_grad()
    def revive(self) -> int:
        """Every SHAPE_REVIVE_EVERY steps, split the busiest codes into the dead ones."""
        self.steps += 1
        if self.steps % SHAPE_REVIVE_EVERY:
            return 0
        wins, count = self.wins, len(self.wins)
        dead = (wins * count < SHAPE_DEAD_SHARE * wins.sum()).nonzero()[:, 0]
        busy = wins.topk(min(len(dead), count - len(dead))).indices
        dead = dead[: len(busy)]
        codes = F.normalize(self.codes[busy], dim=-1)
        noise = F.normalize(torch.randn_like(codes), dim=-1)
        self.codes[dead] = F.normalize(codes + SHAPE_SPLIT_NOISE * noise, dim=-1)
        self.letters[dead] = self.letters[busy]
        wins.zero_()
        return len(dead)


class NanoTSFM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        patch = config.patch_length
        self.embed = nn.Linear(2 * patch, config.width)
        self.embed_mlp = nn.Sequential(
            nn.Linear(2 * patch, config.width), nn.SiLU(), nn.Linear(config.width, config.width)
        )
        blocks = [
            Block(config, causal=True, drop_path=config.drop_path * (index + 1) / config.layers)
            for index in range(config.layers)
        ]
        variate_block = Block(config, causal=False)
        # Blocks left out are still built, so a seed's other weights and masks stay paired.
        self.time_blocks = nn.ModuleList(blocks[:1] if config.tied_blocks else blocks)
        if config.variate_blocks:
            self.variate_block = variate_block
        self.norm = nn.RMSNorm(config.width)
        self.levels = config.quantile_levels or QUANTILES
        self.head = nn.Linear(config.width, patch * len(self.levels))
        # These two start at zero, so they draw no random numbers and a new model is the plain one.
        if config.gap_embedding:
            rows = config.context_length // patch + math.ceil(config.prediction_length / patch) + 1
            self.gap_tokens = nn.Parameter(torch.zeros(rows, config.width))
        if config.scale_features:
            self.scale_link = nn.Parameter(torch.zeros(3, config.width))
        if config.shifted_block:
            self.shifted_block = Block(config, causal=True, shifted=True)
        if config.future_query:
            self.query_embed = nn.Sequential(
                nn.Linear(4, config.width), nn.SiLU(), nn.Linear(config.width, config.width)
            )
            self.query_norm = nn.RMSNorm(config.width)
            self.context_norm = nn.RMSNorm(config.width)
            self.query_attention = nn.MultiheadAttention(
                config.width, config.heads, dropout=config.dropout, batch_first=True
            )
            self.query_mixer = nn.Sequential(
                nn.Conv1d(config.width, config.width, 3, padding=1, groups=config.width),
                nn.SiLU(),
                nn.Conv1d(config.width, config.width, 1),
            )
        if config.far_tiers:
            # One row per resolution; the first is the recent patches'.
            self.resolution = nn.Parameter(torch.zeros(1 + len(config.far_tiers), config.width))
            self.far_link = nn.Linear(2, config.width)
        if config.coarse_blocks:  # trained alongside the head; forecasts never read them
            self.coarse_heads = nn.ModuleList(
                nn.Linear(config.width, len(QUANTILES)) for _ in config.coarse_blocks
            )
        if config.last_patch_anchor:
            # Starts at 0, so the plain forecast is what training moves away from, if at all.
            self.anchor_gain = nn.Parameter(torch.zeros(()))
        if config.shape_codes:
            # On its own random stream, so the rest of the model and the training masks of a seed
            # stay those of the plain model, which keeps paired runs paired.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed())
                self.vocabulary = ShapeVocabulary(config)
        # All of these start at zero and draw no random numbers, so a new model is the plain one.
        if config.markov_path:
            self.markov_link = zero_linear(config.markov_path * patch, patch)
        if config.stat_pool:
            self.pool_link = zero_linear(2 * config.width, config.width)
        if config.lhuc:  # 2 * sigmoid(0) is a gain of 1
            self.lhuc_link = zero_linear(2 * config.width, 4 * config.width)
        if config.seed_bias:
            self.seed_gain = nn.Parameter(torch.zeros(config.heads))
        if config.learned_level_weights:  # trained alongside the head; forecasts never read it
            self.level_log_scales = nn.Parameter(torch.zeros(len(self.levels)))
        if config.rv_aux:
            self.rv_head = zero_linear(config.width, 1)
        if config.next_next:
            self.next_head = zero_linear(config.width, patch * len(self.levels))
        if config.phase_tokens:
            self.phase_link = nn.Parameter(torch.zeros(2 * PHASE_SLOTS, config.width))
        if config.phase_bins:  # TinyCast's W_phase: mixes the slots' readouts into the tokens
            self.template_link = zero_linear(PHASE_SLOTS * config.width, config.width)
        if config.rate_embedding:
            self.rate_tokens = nn.Parameter(torch.zeros(len(config.rate_embedding), config.width))
        if config.head_hidden:
            # On its own stream, as the vocabulary is, so a seed's training masks stay paired. The
            # head itself stays the plain Linear, which is what head_refit and zero_init look for.
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(torch.initial_seed())
                self.head_residual = nn.Sequential(
                    nn.Linear(config.width, config.head_hidden),
                    nn.SiLU(),
                    zero_linear(config.head_hidden, config.width),
                )
        if config.block_heads:  # trained alongside the head; forecasts never read them
            outputs = patch * len(self.levels)
            self.block_heads = nn.ModuleList(
                zero_linear(config.width, outputs)
                for _ in range(config.tied_blocks or config.layers)
            )
        if config.zero_init:
            blocks = [*self.time_blocks]
            if config.variate_blocks:
                blocks.append(self.variate_block)
            if config.shifted_block:
                blocks.append(self.shifted_block)
            for layer in [self.head, *(b.out for b in blocks), *(b.mlp[2] for b in blocks)]:
                nn.init.zeros_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def set_drop_live(self, live: bool):
        for block in self.time_blocks:
            block.drop_live = live

    def far_embed(self, far: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor):
        """Tokens for [B,V,F] older history, oldest first, a [B,V,P,W] link for recent patches, and
        which far tokens are all missing.

        A tier's token holds patch_length averages of `rate` points each, so it spans rate *
        patch_length points through the same embedding as a recent patch, and its observed channel
        holds the share of points seen. Recent patches keep the scale they have without far tiers,
        running from the first recent point, so a far token must not read it: that scale moves with
        the target. All far tokens instead share one scale from the far points alone, and the link
        tells each recent patch where its own loc and scale sit on it. Without far points it is 0.
        """
        patch = self.config.patch_length
        far_loc, far_scale = causal_scale(far, far.shape[-1])
        tokens, padded, end = [], [], far.shape[-1]
        for index, (rate, count) in enumerate(self.config.far_tiers, 1):  # nearest first
            start = end - rate * count * patch
            group = far[..., start:end].unflatten(-1, (count * patch, rate))
            observed = group.isfinite()
            seen = observed.sum(-1)
            mean = torch.where(observed, group, 0).double().sum(-1) / seen.clamp_min(1)
            scaled = torch.where(seen > 0, (mean - far_loc) / far_scale, 0).asinh().float()
            shares = (seen / rate).float()
            token = torch.cat(
                [part.unflatten(-1, (count, patch)) for part in (scaled, shares)], dim=-1
            )
            tokens.insert(0, self.embed(token) + self.embed_mlp(token) + self.resolution[index])
            padded.insert(0, seen.unflatten(-1, (count, patch)).sum(-1) == 0)
            end = start
        link = torch.stack([((loc - far_loc) / far_scale).asinh(), (scale / far_scale).log()], -1)
        link = link.float() * far.isfinite().any(-1)[..., None, None]
        return torch.cat(tokens, dim=-2), self.far_link(link), torch.cat(padded, dim=-1)

    def phase_readout(self, x, turns, found, padded):
        """What each position from the last context token on reads, [N,F,KW]: per slot, the mean of
        the context's [N,P,W] tokens in its own phase's bin. An empty bin or a failed slot reads
        the whole-context mean, as in TinyCast. Only context tokens enter, and never all-missing
        ones. The result joins the tokens ahead of the variate block and the head, so it is only
        ever seen through learned layers: no template is decoded directly."""
        bins, width = self.config.phase_bins, x.shape[-1]
        origin = self.config.context_length // self.config.patch_length
        place = (turns.frac() * bins).long().clamp_max(bins - 1)
        tokens, mass = x[:, :origin], (~padded[:, :origin, None]).to(x.dtype)
        total = tokens * mass
        whole = total.sum(1, keepdim=True) / mass.sum(1, keepdim=True).clamp_min(1)
        reads = []
        for slot in range(PHASE_SLOTS):
            index = place[:, :origin, slot, None]
            sums = x.new_zeros(len(x), bins, width)
            sums.scatter_add_(1, index.expand(-1, -1, width), total)
            count = x.new_zeros(len(x), bins, 1).scatter_add_(1, index, mass)
            filled = (count > 0) & found[:, :1, slot, None]
            template = torch.where(filled, sums / count.clamp_min(1), whole)
            at = place[:, origin - 1 :, slot, None].expand(-1, -1, width)
            reads.append(template.gather(1, at))
        return torch.cat(reads, dim=-1)

    def next_patch(self, values: torch.Tensor, series_ids: torch.Tensor, rates=None):
        """Predict the next patch from [B,V,T] values; equal IDs share variate attention.

        With far_tiers, the first far_length points are older history: they enter as coarse tokens
        ahead of the rest, and the outputs, loc and scale cover the recent patches only."""
        quantiles, _, loc, scale = self.predict(values, series_ids, rates=rates)
        return quantiles, loc, scale

    def rate_token(self, rates, shape: tuple, device) -> torch.Tensor:
        """rate_embedding's [B,V,W] tokens. rates are the supplied values' strides relative to the
        series as given, one integer or [B,V]; None is 1. A stride between listed ones, or past
        them, takes the last listed one below it: forecast strides reach 12, and training 4."""
        table = torch.tensor(self.config.rate_embedding, device=device)
        rates = torch.as_tensor(1 if rates is None else rates, device=device)
        rates = rates.expand(shape).contiguous()
        return self.rate_tokens[torch.searchsorted(table, rates, right=True) - 1]

    def encode(
        self,
        values: torch.Tensor,
        series_ids: torch.Tensor,
        aux: bool = False,
        rates=None,
        lead=None,
        periods=None,
        frame=None,
    ):
        """The final [B,V,P,W] tokens of the supplied values, with what the heads read of them.

        lead, from training's length buckets, is how many leading tokens of nothing but padding
        were dropped, so tokens keep their phases. periods are then those of the [B,V] windows
        whole: a trimmed window's first points would reach past its forecast origin.

        frame, training's frame_gain only, is a pair of [B,V] gains and shifts: each window's loc
        moves by shift times its scale, then the scale is multiplied by the gain, so tokens, outputs
        and the target all read the changed frame. None, as at forecast, is the plain frame."""
        far = self.config.far_length
        older, values = values[..., :far], values[..., far:]
        batch, variates, length = values.shape
        patch = self.config.patch_length
        if length % patch:
            raise ValueError("Sequence length must divide into patches")
        patches = length // patch
        halflife = self.config.scale_halflife
        loc, scale = (
            forgetting_scale(values, patch, halflife) if halflife else causal_scale(values, patch)
        )
        if frame is not None:
            gains, shifts = (part.to(scale.dtype)[..., None] for part in frame)
            loc, scale = loc + shifts * scale, scale * gains
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
        padded = ~observed.unflatten(-1, (patches, patch)).any(-1)
        if self.config.gap_embedding:
            x = x + self.gap_tokens[patch_gaps(padded).clamp_max(len(self.gap_tokens) - 1)]
        if self.config.scale_features:
            zeros = (values == 0).unflatten(-1, (patches, patch)).float().mean(-1)
            level = (loc / scale).asinh() * SCALE_FEATURE_GAIN
            features = [scale.log() * SCALE_FEATURE_GAIN, level, zeros.double()]
            x = x + torch.stack(features, -1).float() @ self.scale_link
        if self.config.rate_embedding:
            x = x + self.rate_token(rates, (batch, variates), x.device)[:, :, None].to(x.dtype)
        turns, found, cycles = None, None, None
        if self.config.phase_tokens or self.config.phase_bins or self.config.rope_periods:
            # The window's first point is t = 0, and a token's phase is its patch center's.
            if periods is None:
                periods = detect_periods(values[..., : self.config.context_length])
            periods = periods[..., None, :]
            if self.config.rope_periods:
                cycles = (patch / periods.clamp_min(1)).float() * (periods > 0)
                cycles = cycles.reshape(batch * variates, PHASE_SLOTS)
            center = torch.arange(patches, device=x.device, dtype=torch.float64) + (lead or 0)
            center = (center * patch + (patch - 1) / 2)[:, None]
            turns, found = center / periods.clamp_min(1), periods > 0
        if self.config.phase_tokens:
            angle = 2 * math.pi * turns
            phase = torch.cat([angle.sin() * found, angle.cos() * found], -1).float()
            x = x + phase @ self.phase_link
        shape_loss = None
        if self.config.shape_codes:
            letters, shape_loss = self.vocabulary(
                values.unflatten(-1, (patches, patch)), loc, scale, aux
            )
            x = x + letters.to(x.dtype)
        if far:
            far_tokens, link, far_padded = self.far_embed(older, loc, scale)
            x = torch.cat([far_tokens, x + self.resolution[0] + link], dim=-2)
            padded = torch.cat([far_padded, padded], dim=-1)

        size = x.shape[-2]
        x = x.reshape(batch * variates, size, -1)
        mask = None
        if self.config.mask_padded and (lead is None or padded.any()):
            # Every token keeps itself, so a fully padded row still has something to attend to.
            mine = torch.eye(size, dtype=torch.bool, device=x.device)
            mask = (~padded.reshape(batch * variates, 1, 1, size) | mine).tril()
        if self.config.seed_bias:
            allowed = torch.ones(size, size, dtype=torch.bool, device=x.device).tril()
            allowed = allowed if mask is None else mask
            words = patch_words(values.unflatten(-1, (patches, patch))).reshape(-1, size)
            hits = (words[:, None] == words[:, :, None]) & (words[:, None] >= 0)
            bonus = hits.tril(-1)[:, None] * self.seed_gain[:, None, None]
            mask = bonus.masked_fill(~allowed, -torch.inf).to(x.dtype)
        layers = []  # each time block's recent tokens, for block_heads
        for block in [*self.time_blocks] * (self.config.tied_blocks or 1):
            gains = None
            if self.config.stat_pool:
                stats = running_stats(x)
                x = x + self.pool_link(stats).to(x.dtype)
                if self.config.lhuc:
                    gains = 2 * self.lhuc_link(stats).sigmoid().to(x.dtype)
            x = block(x, mask, gains, cycles)
            if self.config.block_heads:
                layers.append(x[:, -patches:].reshape(batch, variates, patches, -1))
        if self.config.shifted_block:
            x = self.shifted_block(x)
        if self.config.phase_bins:
            first = self.config.context_length // patch - 1
            reads = self.phase_readout(
                x, turns.flatten(0, 1), found.flatten(0, 1), padded.reshape(batch * variates, -1)
            )
            x = torch.cat([x[:, :first], x[:, first:] + self.template_link(reads).to(x.dtype)], 1)
        x = x[:, -patches:]  # far tokens inform later ones and predict nothing
        x = x.reshape(batch, variates, patches, -1)
        if self.config.variate_blocks:
            x = x.transpose(1, 2).reshape(batch * patches, variates, -1)
            related = series_ids[:, :, None] == series_ids[:, None, :]
            related = related[:, None, None].expand(batch, patches, 1, variates, variates)
            x = self.variate_block(x, related.reshape(batch * patches, 1, variates, variates))
            x = x.reshape(batch, patches, variates, -1).transpose(1, 2)
        if self.config.residual_target:  # only the output's frame moves; tokens keep the plain one
            loc, scale = increment_frame(values, patch, scale)
        return x, loc, scale, scaled, observed, shape_loss, layers

    def predict(
        self, values, series_ids, aux: bool = False, rates=None, lead=None, periods=None, frame=None
    ):
        """What next_patch returns, with the final [B,V,P,W] features after the quantiles. With
        aux, which trains the shape vocabulary, its loss and the time blocks' tokens come back as
        a fifth and sixth value. frame is training's frame_gain; see encode."""
        patch = self.config.patch_length
        encoded = self.encode(values, series_ids, aux, rates, lead, periods, frame)
        x, loc, scale, scaled, observed, shape_loss, layers = encoded
        features = self.norm(x)
        reading = features
        if self.config.head_hidden:
            reading = features + self.head_residual(features)
        quantiles = self.head(reading).float().unflatten(-1, (patch, len(self.levels)))
        if self.config.last_patch_anchor:
            quantiles = quantiles + self.anchor_gain * self.anchor(scaled, observed, patch)
        if self.config.markov_path:
            quantiles = quantiles + self.markov(values, loc, scale)
        ordered = quantiles.sort(dim=-1).values
        if self.config.refine_masked_stats:
            loc, scale = self.refined_frame(values, ordered, loc, scale)
        if self.config.sort_in_training or not self.training:
            quantiles = ordered
        result = quantiles, features, loc, scale
        return (*result, shape_loss, layers) if aux else result

    @torch.no_grad()
    def refined_frame(self, values, quantiles, loc: torch.Tensor, scale: torch.Tensor):
        """The [B,V,P] loc and scale each position's output reads through, with refine_masked_stats.

        causal_scale freezes over a patch with no observed point. Here such a patch, once the series
        has begun, joins the running stats as the median path the position before it predicted, read
        through that position's own frame, so the walk goes patch by patch. Later observed patches
        join on top of it. A sequence with no such patch keeps causal_scale's numbers exactly.
        Nothing here carries a gradient, and tokens keep causal_scale's frame."""
        patch = self.config.patch_length
        level = self.levels.index(0.5)
        median = quantiles[..., level].double().clamp(-REFINE_MAX_SCALED, REFINE_MAX_SCALED).sinh()
        groups = values.double().unflatten(-1, (-1, patch))
        observed = groups.isfinite()
        sizes = observed.sum(-1).double()
        means = torch.where(observed, groups, 0).sum(-1) / sizes.clamp_min(1)
        moments = torch.where(observed, groups - means[..., None], 0).square().sum(-1)
        loc, scale = loc.clone(), scale.clone()
        count, mean, m2 = torch.zeros((3, *loc.shape[:2]), dtype=loc.dtype, device=loc.device)
        refined = torch.zeros(loc.shape[:2], dtype=torch.bool, device=loc.device)
        for index in range(loc.shape[-1]):
            size, center, moment = sizes[..., index], means[..., index], moments[..., index]
            masked = (size == 0) & (count > 0)
            if masked.any():
                path = median[:, :, index - 1] * scale[:, :, index - 1, None]
                path = path + loc[:, :, index - 1, None]
                middle = path.mean(-1)
                size = torch.where(masked, patch, size)
                center = torch.where(masked, middle, center)
                spread = (path - middle[..., None]).square().sum(-1)
                moment = torch.where(masked, spread, moment)
                refined |= masked
            total = (count + size).clamp_min(1)
            gap = center - mean
            m2 = m2 + moment + gap.square() * count * size / total
            mean = mean + gap * size / total
            count = count + size
            std = (m2 / (count - 1).clamp_min(1)).clamp_min(MIN_SCALE**2).sqrt()
            loc[..., index] = torch.where(refined, mean, loc[..., index])
            scale[..., index] = torch.where(refined, std, scale[..., index])
        return loc, scale

    def anchor(self, scaled: torch.Tensor, observed: torch.Tensor, patch: int) -> torch.Tensor:
        """The mean of the latest observed patch at or before each position, [B,V,P,1,1], in that
        position's scaled space: loc and scale stay still over patches with no observed point."""
        seen = observed.unflatten(-1, (-1, patch)).sum(-1)
        mean = scaled.unflatten(-1, (-1, patch)).sum(-1) / seen.clamp_min(1)
        mean = torch.where(seen > 0, mean, torch.nan)
        return forward_fill(mean).nan_to_num(0.0)[..., None, None]

    def markov(self, values: torch.Tensor, loc: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        """The learned linear path, [B,V,P,L,1]: each position's latest markov_path patches, moved
        to that position's own loc and scale, where its next patch is scored. Missing points read 0.
        """
        patch, count = self.config.patch_length, self.config.markov_path
        recent = values[..., self.config.far_length :]
        recent = F.pad(recent, ((count - 1) * patch, 0), value=float("nan"))
        lags = recent.unfold(-1, count * patch, patch).double()
        lags = (lags - loc[..., None]) / scale[..., None]
        lags = torch.where(lags.isfinite(), lags, 0).asinh().float()
        return self.markov_link(lags).float()[..., None]

    def future_quantiles(self, history: torch.Tensor, series_ids: torch.Tensor, horizon: int):
        """Decode all future patches from latent prefix tokens, in prefix-normalized space."""
        patch, width = self.config.patch_length, self.config.width
        batch, variates, _ = history.shape
        memory, loc, scale, *_ = self.encode(history, series_ids)
        memory = self.context_norm(memory).reshape(batch * variates, -1, width)
        offset = torch.arange(0, horizon, patch, device=history.device).float() + 1
        offset = offset / self.config.context_length
        span = torch.full_like(offset, horizon / self.config.context_length)
        coordinates = torch.stack([offset, offset.log1p(), span, span.log1p()], dim=-1)
        queries = self.query_embed(coordinates).expand(batch * variates, -1, -1)
        attended, _ = self.query_attention(
            self.query_norm(queries), memory, memory, need_weights=False
        )
        queries = queries + attended
        queries = queries + self.query_mixer(queries.transpose(1, 2)).transpose(1, 2)
        raw = self.head(self.norm(queries)).float().unflatten(-1, (patch, len(QUANTILES)))
        median = raw[..., 4:5]
        lower = median - F.softplus(raw[..., :4]).flip(-1).cumsum(-1).flip(-1)
        upper = median + F.softplus(raw[..., 5:]).cumsum(-1)
        quantiles = torch.cat([lower, median, upper], dim=-1)
        quantiles = quantiles.reshape(batch, variates, -1, len(QUANTILES))[:, :, :horizon]
        return quantiles, loc[..., -1], scale[..., -1]

    def forward(
        self,
        history: torch.Tensor,
        series_ids: torch.Tensor,
        horizon: int | None = None,
        rates=None,
    ) -> torch.Tensor:
        if self.config.future_query:
            horizon = self.config.prediction_length if horizon is None else horizon
            quantiles, loc, scale = self.future_quantiles(history, series_ids, horizon)
            quantiles = quantiles.double()
            if self.config.forecast_clamp is not None:
                lower, upper = forecast_range(history, loc, scale, self.config.forecast_clamp)
                quantiles = quantiles.clamp(lower[..., None, None], upper[..., None, None])
            largest = torch.finfo(torch.float32).max
            return (
                (quantiles.sinh() * scale[..., None, None] + loc[..., None, None])
                .clamp(-largest, largest)
                .float()
            )
        patch, horizon = self.config.patch_length, self.config.prediction_length
        future = math.ceil(horizon / patch)
        far = self.config.far_length
        if far:  # tiers sit at fixed places, measured from the start of the full context
            missing = self.config.context_length - history.shape[-1]
            history = F.pad(history, (missing, 0), value=float("nan"))
        values = F.pad(history, (0, future * patch), value=float("nan"))
        quantiles, loc, scale = self.next_patch(values, series_ids, rates)
        # Position i predicts patch i + 1; the last history patch predicts the first future one.
        first = (history.shape[-1] - far) // patch - 1
        window = slice(first, first + future)
        location, spread = loc[..., window, None, None], scale[..., window, None, None]
        quantiles = quantiles[:, :, window].double()
        if self.config.quantile_levels:
            quantiles = quantiles[..., [self.levels.index(level) for level in QUANTILES]]
        if self.config.forecast_clamp is None:
            forecast = quantiles.sinh() * spread + location
        else:
            # The same bound on every quantile keeps them sorted.
            lower, upper = forecast_range(
                history, loc[..., first], scale[..., first], self.config.forecast_clamp
            )
            quantiles = quantiles.clamp(
                lower[:, :, None, None, None], upper[:, :, None, None, None]
            )
            largest = torch.finfo(torch.float32).max
            forecast = (quantiles.sinh() * spread + location).clamp(-largest, largest)
        return forecast.flatten(2, 3)[:, :, :horizon].float()


class Ensemble(nn.Module):
    """Independently trained members forecasting together: the forecast at each quantile level
    joins the members' forecasts at that level. config.combiner chooses how: their mean (quantile
    averaging, or Vincentization), their median, or their mean in the asinh-scaled space of the
    history, which one exploded member moves far less than a mean in raw units. Each keeps sorted
    quantiles sorted, because each is nondecreasing in every member's forecast."""

    def __init__(self, members: list[NanoTSFM], config: ModelConfig):
        super().__init__()
        if not members:
            raise ValueError("An ensemble needs at least one member")
        for member in members:
            if member.config.future_query != config.future_query:
                raise ValueError("Ensemble members must share the future-query setting")
            if member.config.prediction_length != config.prediction_length:
                raise ValueError("Ensemble members must share the prediction length")
            if member.config.context_length > config.context_length:
                raise ValueError("No ensemble member may read more context than the ensemble")
        self.members = nn.ModuleList(members)
        # The training config's model section: forecast() and the evaluator read its context and
        # prediction lengths, and each member reads the most recent part of that context it uses.
        self.config = config

    def forward(
        self,
        history: torch.Tensor,
        series_ids: torch.Tensor,
        horizon: int | None = None,
        rates=None,
    ) -> torch.Tensor:
        predictions = []
        for member in self.members:
            view = history[..., -member.config.context_length :]
            # A member's patches may be longer than the ensemble's, which sets rollout's lengths.
            short = -view.shape[-1] % member.config.patch_length
            view = F.pad(view, (short, 0), value=float("nan")) if short else view
            predictions.append(member(view, series_ids, horizon, rates))
        predictions = torch.stack(predictions).double()
        if self.config.combiner != "mean":  # an infinite member must not reach the arithmetic
            largest = torch.finfo(torch.float32).max
            predictions = predictions.clamp(-largest, largest)
        if self.config.combiner == "median":
            ordered = predictions.sort(dim=0).values
            count = len(ordered)  # an even count averages the middle two
            joined = (ordered[(count - 1) // 2] + ordered[count // 2]) / 2
        elif self.config.combiner == "asinh_mean":
            loc, scale = causal_scale(history, history.shape[-1])
            loc, scale = loc[..., None], scale[..., None]
            joined = ((predictions - loc) / scale).asinh().mean(dim=0).sinh() * scale + loc
        else:
            joined = predictions.mean(dim=0)
        # Sorting guards rounding.
        return joined.float().sort(dim=-1).values


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

    infer_contexts pools the normal forecast too: each shorter length forecasts with everything
    older blanked, as a series of that length would, so its causal scale restarts there. Every
    length keeps the newest points, and a series with too little left in one sits that one out.

    With polyphase_joint, one pass forecasts all of a series' phases as its channels, so the
    variate block reads each phase beside the others, and the outputs go back in time order. A
    series whose channels times the stride pass MAX_VARIATES goes phase by phase.
    """

    def __init__(self, base: nn.Module):
        super().__init__()
        self.base = base
        self.strides = base.config.polyphase_forecast_strides
        self.weight = base.config.polyphase_forecast_weight
        self.by_period = base.config.polyphase_forecast_period
        self.joint = base.config.polyphase_joint
        self.contexts = [n for n in base.config.infer_contexts if n < base.config.context_length]
        context = base.config.context_length * max(self.strides)
        self.config = replace(base.config, context_length=context)

    def pass_at(self, phase: torch.Tensor, series_ids: torch.Tensor, stride: int) -> torch.Tensor:
        """The base's forecast of one phase, told its stride if the base takes that."""
        if self.base.config.rate_embedding:
            return self.base(phase, series_ids, rates=stride)
        return self.base(phase, series_ids)

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
            forecast[:, :, lead::stride] = self.pass_at(phase, series_ids, stride)[:, :, :part]
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

    def joint_series(self, history: torch.Tensor, stride: int, chosen: list):
        """interlaced_series with each series' phases as channels of one pass; see the class."""
        context = self.base.config.context_length
        points = self.base.config.patch_length * POLYPHASE_MIN_PATCHES
        length = history.shape[-1]
        part = math.ceil(self.base.config.prediction_length / stride)
        fits = [item for item in chosen if len(item[1]) * stride <= MAX_VARIATES]
        rest = [item for item in chosen if len(item[1]) * stride > MAX_VARIATES]
        placed = history.new_full((*history.shape[:2], part * stride, len(QUANTILES)), float("nan"))
        reached = torch.zeros(history.shape[:2], dtype=torch.bool, device=history.device)
        if rest:
            placed, reached = self.interlaced_series(history, stride, rest)
        if not fits:
            return placed, reached
        samples = []
        for row, channels in fits:
            phases = [
                history[row, channels][..., (length + lead) % stride :: stride][..., -context:]
                for lead in range(stride)
            ]
            samples.append(torch.cat([
                F.pad(phase, (context - phase.shape[-1], 0), value=float("nan")) for phase in phases
            ]))
        packed, _, ids = pack_windows([(sample, sample[:, :0]) for sample in samples])
        ids = ids.to(history.device)
        forecast = self.pass_at(packed.to(history), ids, stride)[:, :, :part]
        for index, (row, channels) in enumerate(fits):
            # [lead,V,part,Q], and lead l of a channel holds future positions l, l + s and so on.
            grid = forecast[ids == index].unflatten(0, (stride, len(channels))).transpose(0, 1)
            placed[row, channels] = grid.transpose(1, 2).flatten(1, 2)
            usable = torch.isfinite(samples[index]).sum(-1) >= points
            reached[row, channels] = usable.unflatten(0, (stride, -1)).all(0).to(reached)
        return placed, reached

    def forecasts(self, history: torch.Tensor, series_ids: torch.Tensor):
        if not self.by_period and not self.joint:
            for stride in self.strides:
                yield self.interlaced(history, series_ids, stride)
            return
        if self.by_period:
            plan = sorted(self.period_plan(history, series_ids).items())
        else:
            every = [(row, (ids == series).nonzero()[:, 0]) for row, ids in enumerate(series_ids)
                     for series in ids.unique().tolist() if series >= 0]
            plan = [(stride, every) for stride in self.strides]
        through = self.joint_series if self.joint else self.interlaced_series
        for stride, chosen in plan:
            yield through(history, stride, chosen)

    def shorter(self, recent: torch.Tensor, series_ids: torch.Tensor):
        """Forecasts of the newest points at each shorter context, with who has enough of them."""
        points = self.base.config.patch_length * POLYPHASE_MIN_PATCHES
        for length in self.contexts:
            view = recent.clone()
            view[..., :-length] = float("nan")
            yield self.base(view, series_ids), torch.isfinite(view).sum(-1) >= points

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

    def forward(
        self, history: torch.Tensor, series_ids: torch.Tensor, horizon: int | None = None
    ) -> torch.Tensor:
        recent = history[..., -self.base.config.context_length :]
        normal = self.base(recent, series_ids).double()
        if self.contexts:
            pooled = self.shorter(recent, series_ids)
            normal = self.blend(normal, pooled, self.base.config.infer_context_weight)
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
    if "ensemble" in checkpoint:
        members = [NanoTSFM(ModelConfig(**member)) for member in checkpoint["ensemble"]]
        model = Ensemble(members, ModelConfig(**checkpoint["model_config"]))
    else:
        model = NanoTSFM(ModelConfig(**checkpoint["model_config"]))
    model.load_state_dict(checkpoint["model"])
    if model.config.polyphase_forecast_strides:
        model = Polyphase(model)
    model.to(device_for(str(device))).eval()
    return model, checkpoint


def rollout(model, history: torch.Tensor, horizon: int, series_ids: torch.Tensor) -> torch.Tensor:
    """Decode jointly when enabled; otherwise feed medians back past one model pass."""
    context, patch = model.config.context_length, model.config.patch_length
    if model.config.future_query:
        return model(history[..., -context:], series_ids, horizon)
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


def error_quantiles(error: torch.Tensor) -> torch.Tensor:
    """The nine quantile levels of each series' errors, ignoring NaN; [...,9]."""
    levels = torch.tensor(QUANTILES, device=error.device, dtype=error.dtype)
    # Sorting puts NaN last; torch.quantile refuses tensors this large.
    count = error.isfinite().sum(-1, keepdim=True)
    ranks = ((count - 1).clamp_min(0) * levels).round().long()
    return error.sort(-1).values.gather(-1, ranks).nan_to_num(0.0)


def seasonal_forecast(filled: torch.Tensor, period: torch.Tensor, horizon: int) -> torch.Tensor:
    """Seasonal Naive, with quantiles from the context's own errors at that period."""
    length = filled.shape[-1]
    steps = torch.arange(length, device=filled.device)
    shifted = filled.gather(-1, (steps - period[..., None]).clamp_min(0))
    error = filled - shifted
    error = torch.where(steps >= period[..., None], error, torch.nan)
    future = torch.arange(horizon, device=filled.device)
    point = filled.gather(-1, length - period[..., None] + future % period[..., None])
    return point.nan_to_num(0.0)[..., None] + error_quantiles(error)[..., None, :]


def rate_forecast(model, history, horizon: int, series_ids, rate: int) -> torch.Tensor:
    """Forecast the series averaged over blocks of `rate` steps, then interpolate it back."""
    patch = model.config.patch_length
    blocks = history.shape[-1] // rate
    grouped = history[..., history.shape[-1] - blocks * rate :].unflatten(-1, (blocks, rate))
    observed = grouped.isfinite()
    total = torch.where(observed, grouped, 0).sum(-1)
    coarse = total / observed.sum(-1).clamp_min(1)
    coarse = torch.where(observed.any(-1), coarse, torch.nan)
    if blocks % patch:
        coarse = F.pad(coarse, (patch - blocks % patch, 0), value=float("nan"))
    # Block centers sit (rate - 1) / 2 steps before the end of their block.
    coarse_horizon = math.ceil(horizon / rate) + 1
    prediction = rollout(model, coarse, coarse_horizon, series_ids)
    anchor = forward_fill(coarse)[..., -1:, None].nan_to_num(0.0).expand_as(prediction[..., :1, :])
    knots = torch.cat([anchor, prediction], dim=-2)
    position = (torch.arange(1, horizon + 1, device=history.device) + (rate - 1) / 2) / rate
    lower = position.floor().long().clamp_max(coarse_horizon - 1)
    weight = (position - lower).to(knots.dtype)[:, None]
    return (knots[..., lower, :] * (1 - weight) + knots[..., lower + 1, :] * weight).sort(-1).values


def signed_forecast(model, history, horizon: int, series_ids) -> torch.Tensor:
    """The forecast of -x, mapped back: q_t(x) = -q_{1-t}(-x)."""
    return -rollout(model, -history, horizon, series_ids).flip(-1)


def model_forecast(model, history, horizon: int, series_ids) -> torch.Tensor:
    base = rollout(model, history, horizon, series_ids)
    if model.config.infer_sign_flip:
        base = (base + signed_forecast(model, history, horizon, series_ids)) / 2
    shifts = model.config.infer_shifts
    if shifts:  # each grid forecasts the points it dropped too, and those are thrown away
        pooled = [base]
        for shift in shifts:
            older = F.pad(history[..., :-shift], (shift, 0), value=float("nan"))
            pooled.append(rollout(model, older, horizon + shift, series_ids)[..., shift:, :])
        base = torch.stack(pooled).mean(0)
    return base


def candidate_forecasts(model, history, horizon: int, series_ids, period, base=None):
    """Every configured forecast, [candidates,B,V,H,9]; the model itself comes first."""
    config = model.config
    if base is None:
        base = model_forecast(model, history, horizon, series_ids)
    forecasts = [base]
    filled = forward_fill(history)
    for name in config.infer_candidates:
        if name == "flip":
            forecasts.append(signed_forecast(model, history, horizon, series_ids))
        elif name == "seasonal":
            forecasts.append(seasonal_forecast(filled, period, horizon))
        elif name == "deseason":
            past, future = seasonal_profile(filled, period, horizon)
            residual = rollout(model, history - past, horizon, series_ids)
            forecasts.append(residual + future[..., None])
        else:
            rate = int(name.removeprefix("rate"))
            forecasts.append(rate_forecast(model, history, horizon, series_ids, rate))
    return torch.stack(forecasts)


def backtest_scores(model, history, horizon: int, series_ids, period) -> torch.Tensor:
    """Pinball loss of each candidate on the end of each series' own context, [candidates,B,V].

    Each series holds back min(max(horizon, C/4), C/2, half its observed length) points. A series
    with fewer than one patch to spare scores 0 everywhere, which keeps the model."""
    context, patch = model.config.context_length, model.config.patch_length
    lengths = history.isfinite().cumsum(-1).clamp_max(1).sum(-1)
    held = min(max(horizon, int(context * model.config.infer_backtest)), context // 2)
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


def gated_forecast(model, history, horizon: int, series_ids) -> torch.Tensor:
    """Weight the candidates per series by their backtest loss, then clamp to the context.

    For an Ensemble every model pass here is the combined forecast: the model candidate, its
    backtest and the deseasonalized rollout all go through Ensemble.forward, and rollouts feed the
    combined median back in. So the gate judges the ensemble, not one member, against the other
    candidates, and picks one weighting per series; members are never gated separately."""
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
            if config.infer_temperature:
                weights = (-relative / config.infer_temperature).softmax(0)
            else:
                weights = F.one_hot(relative.argmin(0), len(forecasts)).movedim(-1, 0)
            forecasts = torch.where(forecasts.isfinite(), forecasts, forecasts[:1])
            blend = weights[..., None, None].to(forecasts.dtype) * forecasts
            blend = blend.sum(0).sort(-1).values
            result = result.clone()
            result[rows] = blend
    if config.infer_clamp:
        observed = history.isfinite()
        low = torch.where(observed, history, torch.inf).amin(-1)
        high = torch.where(observed, history, -torch.inf).amax(-1)
        room = ((high - low) * config.infer_clamp).nan_to_num(0.0, posinf=0.0, neginf=0.0)
        low, high = (low - room)[..., None, None], (high + room)[..., None, None]
        clamped = torch.minimum(torch.maximum(result, low), high)
        result = torch.where(observed.any(-1)[..., None, None], clamped, result)
    return result


@torch.inference_mode()
def forecast(model, history: torch.Tensor, horizon: int, series_ids=None) -> torch.Tensor:
    """Forecast [B,T] or [B,V,T] history, jointly or with the legacy median rollout.

    The infer_* config fields switch on test-time tricks that use only this history."""
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
        or config.infer_sign_flip
        or config.infer_clamp
        or config.infer_shifts
    )
    try:
        predict = gated_forecast if tricks else rollout
        result = predict(model, history, horizon, series_ids)
        return result[:, 0] if univariate else result
    finally:
        model.train(was_training)
