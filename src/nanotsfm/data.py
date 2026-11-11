"""Pretraining data and window sampling."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

MAX_VARIATES = 32
REPOSITORY = "abel-lab/nanoTSFM-pretrain"
REVISION = "b8eee2fc1f7d71cf4d40001c39ac69e1af0d8cf7"
TOY = "toy"
SPECTRUM_POINTS = 2048  # the tail of a series that predictability reads
SPECTRUM_CHANNELS = 8
# Training transforms of one real series; see apply_augment. stride, block_mean and peak_detect
# exclude each other, so a window is never resampled twice.
AUGMENTATIONS = (
    "sign_flip",
    "time_reversal",
    "stride",
    "block_mean",
    "offset",
    "rescale",
    "censor",
    "peak_detect",
)
COARSENINGS = ("stride", "block_mean", "peak_detect")
PEAK = "peak"  # coarse_points' mode that keeps each block's extremes; False strides, True averages
COARSE_RATES = (2, 3, 4)
OFFSET_RANGE = 3.0  # the largest offset, in the channel's standard deviations
RESCALE_RANGE = 2.0  # factors are log-uniform from 1 / RESCALE_RANGE to RESCALE_RANGE
# A rescale may not take a channel near model.MIN_SCALE, where the causal scale stops following it.
RESCALE_MIN_STD = 1e-3
HOLD_OUT_STREAM = 7919  # keeps the hold_out draw apart from any window's
SHORT_STREAM = 104729  # keeps short_window's draws apart from any window's
PERIOD_FALLBACK = (2, 4)  # strides of a window whose series shows no clear cycle
MANIFEST = "manifest.json"  # a built slice: a GEP config plus Arrow shards of added train rows


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load(name: str = "GEP-M", split: str = "train"):
    from datasets import load_dataset

    if name == TOY:
        return toy(split)
    if (Path(name) / MANIFEST).is_file():
        return load_built(Path(name), split)
    if Path(name).is_dir():
        return load_dataset(str(name), split=split)
    return load_dataset(REPOSITORY, name, split=split, revision=REVISION)


def load_built(folder: Path, split: str):
    """A built slice: its base config, with the added Arrow shards on the train split.

    Every shard is memory-mapped where it lies, so the base is not copied, and held-out splits are
    the base's own.
    """
    from datasets import Dataset as HFDataset
    from datasets import concatenate_datasets

    manifest = json.loads((folder / MANIFEST).read_text())
    base = load(manifest["base"], split)
    if split != "train":
        return base
    shards = [HFDataset.from_file(str(folder / name)) for name in manifest["shards"]]
    return concatenate_datasets([base, *shards]) if shards else base


def describe(name: str, dataset) -> dict:
    local = name != TOY and Path(name).is_dir()
    return {
        "name": str(Path(name).resolve()) if local else name,
        "revision": REVISION if name != TOY and not local else None,
        "kind": TOY if name == TOY else "custom" if local else "gift-pretrain",
        "rows": len(dataset),
    }


def arrays(dataset) -> list[np.ndarray]:
    """Return float32 [variate, time] arrays; memory-mapped views may be read-only."""
    import pyarrow as pa

    out = []
    for chunk in dataset.with_format("arrow")[:].column("target").chunks:
        variates, points = chunk.offsets.to_numpy(), chunk.values.offsets.to_numpy()
        values = chunk.values.values
        if values.type != pa.float32():
            values = values.cast(pa.float32())
        values = values.to_numpy(zero_copy_only=False)
        lengths = np.diff(points)
        for row in range(len(chunk)):
            first, last = variates[row], variates[row + 1]
            if last == first or np.any(lengths[first:last] != lengths[first]):
                raise ValueError("Each target must be a nonempty [variate][time] list")
            out.append(values[points[first] : points[last]].reshape(last - first, -1))
    return out


def toy(split: str, seed: int = 7, series: int = 24, length: int = 512):
    from datasets import Dataset as HFDataset

    rng = np.random.default_rng(seed)
    time = np.arange(length)
    rows = {"item_id": [], "source": [], "freq": [], "target": []}
    for index in range(series):
        period = rng.choice([12, 24, 48])
        values = (
            rng.uniform(0.5, 3) * np.sin(2 * np.pi * time / period + rng.random())
            + rng.normal(0, 0.1, length)
            + 0.002 * time
        ).astype("float32")
        values[rng.random(length) < 0.02] = np.nan
        if ("validation" if index % 5 == 0 else "train") == split:
            rows["item_id"].append(f"{TOY}/{index}")
            rows["source"].append(TOY)
            rows["freq"].append("H")
            rows["target"].append([values.tolist()])
    if not rows["item_id"]:
        raise ValueError(f"The toy data has no {split} split")
    return HFDataset.from_dict(rows)


# Frequency bands for band-balanced sampling, keyed by the pandas alias without its multiple or
# anchor ("15T" -> "T", "W-SUN" -> "W").
BANDS = {
    "S": "seconds",
    "T": "minutes",
    "H": "hourly",
    "D": "daily",
    "W": "weekly",
    **dict.fromkeys(("M", "MS", "Q", "A"), "monthly"),
}


def band(freq: str) -> str:
    """Map a pandas frequency string such as 15T, 6H, W-SUN or Q-DEC to its band."""
    alias = freq.split("-")[0].lstrip("0123456789")
    if alias not in BANDS:
        raise ValueError(f"No frequency band for {freq!r}")
    return BANDS[alias]


FREQ_MINUTES = {"S": 1 / 60, "T": 1, "H": 60, "D": 1440, "W": 10080, "M": 43830, "MS": 43830,
                "Q": 131490, "A": 525960}


def freq_minutes(freq: str) -> float:
    """The spacing of a pandas frequency string such as 15T, 6H, W-SUN or Q-DEC, in minutes."""
    alias = freq.split("-")[0]
    unit = alias.lstrip("0123456789")
    return int(alias[: len(alias) - len(unit)] or 1) * FREQ_MINUTES[unit]


def coarse_mode(roll: float, stride: float, block: float):
    """coarse_points' mode for a roll under the sum of the three chances: False, True or PEAK."""
    return False if roll < stride else True if roll < stride + block else PEAK


def draw_coarsening(rng, augment: dict, points: int, full: int):
    """Draw the rate, the phase and the mode (see coarse_points), for a series of points.

    The rate falls to what leaves a full window of real points, so a series shorter than two
    windows is never resampled; a rate of 1 leaves the series as it is. peak_detect makes no draw
    of its own: it takes the top of the same roll, so a run without it draws the same.
    """
    stride, block, peak = (augment.get(name, 0) for name in COARSENINGS)
    roll = rng.random()
    if roll >= stride + block + peak:
        return 1, 0, False
    rate = min(int(rng.choice(COARSE_RATES)), points // full)
    if rate < 2:
        return 1, 0, False
    phase = int(rng.integers(min(rate, points - rate * full + 1)))
    return rate, phase, coarse_mode(roll, stride, block)


def draw_short_coarsening(rng, augment: dict, points: int, least: int):
    """As draw_coarsening, always coarse, for a series that leaves least points at rate 2 or more."""
    rate = min(int(rng.choice(COARSE_RATES)), points // least)
    stride, block, peak = (augment.get(name, 0) for name in COARSENINGS)
    roll = rng.random() * (stride + block + peak)
    phase = int(rng.integers(min(rate, points - rate * least + 1)))
    return rate, phase, coarse_mode(roll, stride, block) if stride + block + peak else True


def coarse_points(group, channels, first: int, stop: int, rate: int, phase: int, mode):
    """Points first to stop of the series at this rate: every rate-th real point (mode False),
    block means (True), or with PEAK each block of 2 * rate points' minimum and maximum.

    A block's mean skips its missing points, and a block with none observed is missing. A peak
    block gives its first minimum and last maximum in the order they were recorded, so every value
    is a real point at its own place in the sequence; with one observed point it gives that point
    and then a missing one, and with none, two missing. PEAK needs first and stop even, so blocks
    start at first and a pair never spans two crops' edge.
    """
    begin, end = phase + first * rate, phase + stop * rate
    if mode == PEAK:
        if first % 2 or stop % 2:
            raise ValueError("peak_detect's blocks need an even first and stop")
        blocks = group[channels, begin:end].reshape(len(channels), -1, 2 * rate)
        observed = np.isfinite(blocks)
        low = np.where(observed, blocks, np.inf).argmin(-1)
        high = 2 * rate - 1 - np.where(observed, blocks, -np.inf)[..., ::-1].argmax(-1)
        places = np.stack([np.minimum(low, high), np.maximum(low, high)], -1)
        kept = np.take_along_axis(blocks, places, -1).astype("float32")
        kept[..., 1] = np.where(low == high, np.nan, kept[..., 1])
        return kept.reshape(len(channels), -1)
    if not mode:
        return np.array(group[channels, begin:end:rate], copy=True)
    blocks = group[channels, begin:end].reshape(len(channels), -1, rate)
    observed = np.isfinite(blocks)
    seen = observed.sum(-1)
    total = np.where(observed, blocks, 0).sum(-1, dtype="float64")
    return np.where(seen > 0, total / np.maximum(seen, 1), np.nan).astype("float32")


def censor_view(values: np.ndarray, rng, cut: int) -> np.ndarray:
    """Clip each [V,T] channel, history and target, at order statistics of its first cut points.

    Every call clips the top and half of them the bottom too, so the view has the flat tops and
    floors of a saturated sensor. Each threshold is one observed value of that channel's history,
    with no interpolation, so no target point can set it. Missing points stay missing.
    """
    upper, lower, floored = rng.uniform(0.5, 1), rng.uniform(0, 0.5), rng.random() < 0.5
    result = values.copy()
    for channel, history in zip(result, result[:, : max(cut, 0)], strict=True):
        seen = np.sort(history[np.isfinite(history)])
        if seen.size:
            least = seen[int(lower * (seen.size - 1))] if floored else None
            kept = np.isfinite(channel)
            channel[kept] = np.clip(channel[kept], least, seen[int(upper * (seen.size - 1))])
    return result


def apply_augment(values: np.ndarray, rng, augment: dict, cut: int | None = None) -> np.ndarray:
    """Apply the per-window transforms to [V,T] values; every channel gets the same draws.

    cut is where the target starts, which censor needs. It runs after the flips, so its history
    is the one the model will see: a reversal swaps which points are the target.

    Missing points stay missing. causal_scale removes any offset and rescale exactly, up to float32
    rounding, so the offset is bounded in standard deviations to keep that rounding small, and
    each is skipped where it would overflow an observed point into a missing one or, for a rescale,
    push a channel down to the scale floor.
    """
    reverse, negate, shift, resize = rng.random(4)
    if reverse < augment.get("time_reversal", 0):
        values = values[:, ::-1].copy()
    if negate < augment.get("sign_flip", 0):
        values = -values
    if augment.get("censor") and rng.random() < augment["censor"]:
        if cut is None:
            raise ValueError("censor sets its thresholds from the history, so needs the cut")
        values = censor_view(values, rng, cut)
    shift = shift < augment.get("offset", 0)
    resize = resize < augment.get("rescale", 0)
    if not (shift or resize):
        return values
    observed = np.isfinite(values)
    clean = np.where(observed, values, 0).astype("float64")
    seen = np.maximum(observed.sum(-1, keepdims=True), 1)
    mean = clean.sum(-1, keepdims=True) / seen
    std = np.sqrt((np.where(observed, clean - mean, 0) ** 2).sum(-1, keepdims=True) / seen)
    result = clean
    if shift:
        result = result + rng.uniform(-OFFSET_RANGE, OFFSET_RANGE) * std
    if resize:
        factor = RESCALE_RANGE ** rng.uniform(-1, 1)
        if not (std[std > 0] * factor < RESCALE_MIN_STD).any():
            result = result * factor
    with np.errstate(over="ignore"):
        result = result.astype("float32")
    if not np.isfinite(result[observed]).all():
        return values
    return np.where(observed, result, values)


def block_mean(values: np.ndarray, factor: int) -> np.ndarray:
    """Average a [variate, time] array over consecutive blocks of factor real points.

    Blocks end at the last point, so a remainder is dropped from the start. A block under half
    observed is NaN.
    """
    variates, points = values.shape
    length = points // factor
    blocks = values[:, points - length * factor :].reshape(variates, length, factor)
    observed = np.isfinite(blocks)
    count = observed.sum(-1)
    total = np.where(observed, blocks, 0).sum(-1, dtype="float64")
    return np.where(2 * count >= factor, total / np.maximum(count, 1), np.nan).astype("float32")


def floor_shares(weights: np.ndarray, bands: list, floor: float) -> np.ndarray:
    """Raise each band's share of the weights to the floor, scaling the other bands down evenly.

    Scaling can sink another band below the floor, so that repeats. Sources keep their ratios
    within a band.
    """
    names = sorted(set(bands))
    if floor * len(names) > 1:
        raise ValueError(f"A rate floor of {floor} doesn't fit {len(names)} bands")
    member = {name: [i for i, b in enumerate(bands) if b == name] for name in names}
    totals = {name: weights[rows].sum() for name, rows in member.items()}
    raised, wanted = set(), {}
    while True:
        rest = sum(total for name, total in totals.items() if name not in raised)
        scale = (1 - floor * len(raised)) / rest
        wanted = {name: floor if name in raised else total * scale
                  for name, total in totals.items()}
        below = {name for name, share in wanted.items() if share < floor}
        if not below:
            break
        raised |= below
    result = weights.copy()
    for name, rows in member.items():
        result[rows] *= wanted[name] / totals[name]
    return result


def predictability(values: np.ndarray) -> float:
    """One minus the normalized spectral entropy of a [variate, time] tail: 0 is white noise, 1 a
    single sine. Channels are averaged, and a series with no spread scores 0."""
    tail = values[: SPECTRUM_CHANNELS, -SPECTRUM_POINTS:].astype("float64")
    observed = np.isfinite(tail)
    seen = np.maximum(observed.sum(-1, keepdims=True), 1)
    clean = np.where(observed, tail, 0)
    power = np.abs(np.fft.rfft(clean - clean.sum(-1, keepdims=True) / seen * observed)) ** 2
    power = power[:, 1:]
    bins = power.shape[1]
    total = power.sum(-1, keepdims=True)
    if bins < 2 or not total.any():
        return 0.0
    share = power / np.where(total > 0, total, 1)
    entropy = -(share * np.log(np.where(share > 0, share, 1))).sum(-1)
    return float(np.where(total[:, 0] > 0, 1 - entropy / np.log(bins), 0).mean())


class Windows(Dataset):
    def __init__(
        self,
        dataset,
        context: int,
        horizon: int,
        count: int,
        seed: int,
        power=0.0,
        band_weights: dict | None = None,
        exclude=(),
        far: int = 0,
        augment: dict | None = None,
        source_weights: dict | None = None,
        views: dict | None = None,
        drop_items=(),
        quota: int = 0,
        channel_limit: int = 0,
        min_observed: float = 0.0,
        target_points: int = 0,
        series_tilt: float = 0.0,
        aug_views: int = 1,
        aug_ramp: list | None = None,
        hold_out: float = 0.0,
        held_only: bool = False,
        polyphase_share: float = 0.0,
        polyphase_strides=(2, 4),
        polyphase_least: int = 0,
        polyphase_short_share: float | None = None,
        polyphase_long_share: float | None = None,
        rate_tags: bool = False,
        rate_floor: float = 0.0,
        coarse_min_points: int = 0,
        polyphase_joint_share: float = 0.0,
    ):
        """Draw a source with weight (its series count) ** power, then a series from it.

        Power 0 weights sources equally, 1 weights every series equally. With band_weights, draw
        a frequency band first, then a source within it by the same rule; bands with no source
        are dropped and the remaining weights rescaled. Sources in exclude are never drawn.

        With far, the history gains that many earlier points of the same series ahead of the
        context, padded with NaN where the series has none. Crops are drawn as without it, so a
        series shorter than context + far + horizon still gets every crop, not just its last one.

        With augment, a map from names in AUGMENTATIONS to probabilities, each window is transformed
        on its own from the window's seeded generator. Stride and block mean resample the series
        before the crop, so they read a crop rate times longer and the window stays full length;
        the other transforms apply to the crop, far points included. Training only.

        source_weights multiplies the named sources' weights. With views, a share of windows come
        from block means of series at the listed freqs: a factor is drawn evenly, then a source
        by the same weights, then a series. Means are computed here, once, for this dataset's own
        series, so a view holds only points of its parent; pass views for training data only.
        A view is resampled once already, so augment's stride and block mean skip it.

        These are for training data only. Series in drop_items are never drawn. With quota, a batch
        size, the sources of each batch of that size are placed along the sources' weights in
        frequency-band order, so every band and source gets its share of every batch to within one
        window, and draws of the source stop being independent. channel_limit caps a window's
        channels at a uniform draw from 1 to it. min_observed is the least share of observed points
        in a window's history and in its last target_points. series_tilt weights the series
        within a source by exp(tilt * z), where z is the series' standardized predictability.

        With aug_views m, each run of m windows from a multiple of m shares its series, channels,
        start and resampling, and draws its other transforms on its own. aug_ramp, [first, last],
        multiplies every augment chance by a share moving linearly over the windows. hold_out keeps
        this share of each source's series, rounded down, out of these windows, or with held_only
        keeps only them; the same seed gives the same split, and sources left empty go.

        A polyphase_share of windows comes from a polyphase subsequence of the series, every s-th
        real point at a random phase. Unlike augment's stride it also reaches short series, whose
        windows are padded. Such a window is resampled once, so augment's stride and block mean
        skip it, as they skip a view. polyphase_strides "period" draws each stride from a random
        stretch of that series alone, so its main cycle lands near 24 points; see channel_strides.
        polyphase_short_share and polyphase_long_share, set together, replace polyphase_share for
        series under and from two full windows, the length base coarsening needs.

        A polyphase_joint_share of windows, drawn first, comes from every phase of a stride at once:
        each phase is a channel of the one window, so all of a window's history comes before all of
        its target, and one id takes them to the variate block. The series' own channels are capped
        so the phases fit MAX_VARIATES. polyphase_strides picks the stride, as above.

        rate_floor is the least share of draws for each native rate among the sources left, from
        the freqs of those sources alone; the source weights are otherwise unchanged.

        coarse_min_points opens short_window: a coarse window of a series too short for two full
        windows, which augment never resamples, holding at least that many points before padding.
        Those windows are extra to the ones indexed here, so no draw of a window moves; see Extras.
        """
        self.context, self.horizon, self.count, self.seed = context, horizon, count, seed
        self.far, self.augment = far, augment or None
        self.dataset, self.values = dataset, arrays(dataset)
        self.quota, self.channel_limit = quota, channel_limit
        self.min_observed, self.target_points = min_observed, target_points
        self.aug_views, self.aug_ramp = aug_views, aug_ramp
        self.polyphase_share, self.polyphase_strides = polyphase_share, polyphase_strides
        self.polyphase_least = polyphase_least
        self.polyphase_short_share = polyphase_short_share
        self.polyphase_long_share = polyphase_long_share
        self.polyphase_joint_share = polyphase_joint_share
        self.rate_tags = rate_tags
        if (augment or {}).get("peak_detect") and (far + context + horizon) % 2:
            raise ValueError("peak_detect keeps pairs of points, so needs a window of even length")
        sources, excluded = {}, set(exclude)
        dropped = set(drop_items)
        items = list(dataset["item_id"]) if dropped else None  # a Column reads rows one by one
        for row, (source, values) in enumerate(zip(dataset["source"], self.values, strict=True)):
            if source not in excluded and values.shape[1] > horizon:
                if not dropped or items[row] not in dropped:
                    sources.setdefault(source, []).append(row)
        if hold_out:
            rng = np.random.default_rng([seed, HOLD_OUT_STREAM])
            for source, rows in list(sources.items()):
                held = set(rng.permutation(len(rows))[: int(hold_out * len(rows))].tolist())
                sources[source] = [r for i, r in enumerate(rows) if (i in held) == held_only]
            sources = {source: rows for source, rows in sources.items() if rows}
        self.sources = [np.array(rows) for rows in sources.values()]
        names = list(sources)
        if not self.sources or min(context, horizon, count) <= 0:
            raise ValueError("No series longer than the horizon, or an invalid window/count")
        weights = np.array([len(source) for source in self.sources], dtype=float) ** power
        if source_weights:
            if not set(source_weights) <= set(names):
                raise ValueError(f"No such source: {sorted(set(source_weights) - set(names))}")
            weights *= [source_weights.get(name, 1) for name in names]
            if not weights.sum():
                raise ValueError("source_weights leaves no source with positive weight")
        self.weights = weights / weights.sum() if power or source_weights else None
        if rate_floor:
            freq = dataset["freq"]
            rates = [freq_minutes(freq[int(rows[0])]) for rows in self.sources]
            self.weights = floor_shares(weights / weights.sum(), rates, rate_floor)
        self.bands = None
        if band_weights is not None:
            freq = dataset["freq"]
            self.bands = [band(freq[int(rows[0])]) for rows in self.sources]
            shares = np.array([band_weights.get(name, 0) for name in self.bands], dtype=float)
            for name in set(self.bands):
                member = [i for i, b in enumerate(self.bands) if b == name]
                shares[member] *= weights[member] / weights[member].sum()
            if not shares.sum():
                raise ValueError("No source falls in a band with positive weight")
            self.weights = shares / shares.sum()
        if quota:
            freq = dataset["freq"]
            bands = [band(freq[int(rows[0])]) for rows in self.sources]
            count = len(self.sources)
            self.quota_order = np.array(sorted(range(count), key=bands.__getitem__))
            shares = np.full(count, 1 / count) if self.weights is None else self.weights
            self.quota_edges = shares[self.quota_order].cumsum()
        self.series_weights = None
        if series_tilt:
            self.series_weights = []
            for rows in self.sources:
                scores = np.array([predictability(self.values[int(row)]) for row in rows])
                spread = scores.std()
                z = (scores - scores.mean()) / (spread if spread > 0 else 1)
                tilted = np.exp(series_tilt * z)
                self.series_weights.append(tilted / tilted.sum())
        self.view_share = 0
        if views is not None:
            self.build_views(views, weights)
        if coarse_min_points:
            self.build_short(coarse_min_points)

    def build_short(self, least: int):
        """Series for short_window, and each source's draw: its weight, among sources that have any."""
        full = self.far + self.context + self.horizon
        lengths = np.array([values.shape[1] for values in self.values])
        reach = (lengths // full < 2) & (lengths // least >= 2)
        shares = np.full(len(self.sources), 1 / len(self.sources))
        shares = shares if self.weights is None else self.weights
        kept = [(source[reach[source]], share) for source, share in
                zip(self.sources, shares, strict=True) if reach[source].any()]
        if not kept:
            raise ValueError(f"No series reaches a coarse window of {least} points")
        self.short_least = least
        self.short_sources = [rows for rows, _ in kept]
        share = np.array([share for _, share in kept])
        self.short_weights = share / share.sum()

    def build_views(self, views: dict, weights: np.ndarray):
        """Compute every view's block means into one buffer, and each factor's source draw."""
        wanted = set(views["freqs"])
        chosen = np.array([freq in wanted for freq in self.dataset["freq"]])
        lengths = np.array([values.shape[1] for values in self.values])
        variates = np.array([values.shape[0] for values in self.values])
        self.view_share, self.view_factors = views["share"], list(views["factors"])
        self.view_sources, self.view_weights = [], []
        self.view_offsets = np.full((len(self.values), len(self.view_factors)), -1, dtype="int64")
        used = 0
        for column, factor in enumerate(self.view_factors):
            member, shares = [], []
            for source, weight in zip(self.sources, weights, strict=True):
                rows = source[chosen[source] & (lengths[source] // factor > self.horizon)]
                if len(rows) and weight:
                    member.append(rows)
                    shares.append(weight)
                    sizes = variates[rows] * (lengths[rows] // factor)
                    self.view_offsets[rows, column] = used + np.cumsum(sizes) - sizes
                    used += int(sizes.sum())
            if not member:
                raise ValueError(f"No series for a view at factor {factor}")
            self.view_sources.append(member)
            self.view_weights.append(np.array(shares) / sum(shares))
        # One shared buffer: loader workers receive a handle to it rather than a copy.
        self.view_points = torch.empty(used, dtype=torch.float32).share_memory_()
        buffer = self.view_points.numpy()
        # In row order, every factor at once, so a cold disk reads each series once and in sequence.
        for row in np.flatnonzero((self.view_offsets >= 0).any(1)):
            for column, factor in enumerate(self.view_factors):
                first = self.view_offsets[row, column]
                if first >= 0:
                    means = block_mean(self.values[row], factor)
                    buffer[first : first + means.size] = means.ravel()

    def view(self, row: int, column: int) -> np.ndarray:
        """The [variate, time] block means of a series at the column'th factor."""
        variates, points = self.values[row].shape
        length = points // self.view_factors[column]
        first = int(self.view_offsets[row, column])
        return self.view_points[first : first + variates * length].numpy().reshape(variates, -1)

    def __getstate__(self):
        # Spawned loader workers get the dataset, which pickles as its Arrow file paths when it is
        # memory-mapped, and rebuild the views on it, so all processes share one page cache.
        return {**self.__dict__, "values": None}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.values = arrays(self.dataset)

    def band_shares(self) -> dict:
        """Expected share of windows per band, when sampling by band."""
        shares = {}
        for name, weight in zip(self.bands, self.weights, strict=True):
            shares[name] = shares.get(name, 0.0) + float(weight)
        return dict(sorted(shares.items()))

    def __len__(self):
        return self.count

    def quota_place(self, index: int) -> int:
        """The source of a window: evenly spaced draws along the weights, from one batch offset."""
        batch, slot = divmod(index, self.quota)
        offset = np.random.default_rng([self.seed, batch]).random()
        place = np.searchsorted(self.quota_edges, (slot + offset) / self.quota, side="right")
        return int(self.quota_order[min(place, len(self.quota_order) - 1)])

    def __getitem__(self, index):
        return self.window(index)

    def thinning_share(self, points: int) -> float:
        """The polyphase share of a series of this many points."""
        if self.polyphase_short_share is None:
            return self.polyphase_share
        full = self.far + self.context + self.horizon
        return self.polyphase_short_share if points < 2 * full else self.polyphase_long_share

    def thinning_stride(self, group: np.ndarray, rng) -> int:
        """A stride for one series, read from that series' own points and no other."""
        if self.polyphase_strides != "period":
            return int(rng.choice(self.polyphase_strides))
        from nanotsfm.model import PERIOD_SPAN, channel_strides

        start = rng.integers(max(group.shape[1] - PERIOD_SPAN, 0) + 1)
        span = torch.tensor(group[:SPECTRUM_CHANNELS, start : start + PERIOD_SPAN])
        strides, strength = channel_strides(span[None])
        return int(strides[0, strength[0].argmax()]) or int(rng.choice(PERIOD_FALLBACK))

    def short_window(self, index):
        """An extra coarse window of a short series, from its own stream of draws."""
        return self.window(index, short=True)

    def window(self, index, short=False):
        first_view = int(index) - int(index) % self.aug_views
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, first_view]))
        if short:
            first_view = int(index)
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, first_view, SHORT_STREAM]))
        flips = rng
        if self.aug_views > 1 and not short:
            entropy = [self.seed, int(index), self.aug_views]
            flips = np.random.default_rng(np.random.SeedSequence(entropy))
        augment = self.augment
        if self.aug_ramp:
            begin, end = self.aug_ramp
            share = begin + (end - begin) * first_view / max(self.count - 1, 1)
            augment = {name: chance * share for name, chance in augment.items()}
        for _ in range(100):
            column = None
            sources, weights = self.sources, self.weights
            if short:
                sources, weights = self.short_sources, self.short_weights
            elif self.view_share and rng.random() < self.view_share:
                column = int(rng.integers(len(self.view_factors)))
                sources, weights = self.view_sources[column], self.view_weights[column]
            if self.quota and column is None and not short:
                place = self.quota_place(first_view)
            elif weights is None:  # the baseline's uniform draw, kept for exact reproduction
                place = rng.integers(len(sources))
            else:
                place = rng.choice(len(sources), p=weights)
            source = sources[place]
            if self.series_weights and column is None and not short:
                row = int(rng.choice(source, p=self.series_weights[place]))
            else:
                row = int(source[rng.integers(len(source))])
            group = self.values[row] if column is None else self.view(row, column)
            thinned, spacing = False, 1 if column is None else self.view_factors[column]
            whole, joint, offset = group, 0, 0
            if (
                self.polyphase_joint_share
                and column is None
                and not short
                and rng.random() < self.polyphase_joint_share
            ):
                stride = self.thinning_stride(group, rng)
                offset = int(rng.integers(stride))
                count = (group.shape[1] - offset) // stride
                if stride <= MAX_VARIATES and count > max(self.horizon, self.polyphase_least - 1):
                    # Every phase keeps count points, so a crop ends on one step in all of them.
                    group, thinned, spacing = group[:, offset::stride][:, :count], True, stride
                    joint = stride
            share = self.thinning_share(group.shape[1])
            polyphase = share and column is None and not short
            if polyphase and not joint and rng.random() < share:
                stride = self.thinning_stride(group, rng)
                # Every channel keeps the same phase, so variates stay aligned in time.
                part = group[:, rng.integers(stride) :: stride]
                if part.shape[1] > max(self.horizon, self.polyphase_least - 1):
                    group, thinned, spacing = part, True, stride
            limit = min(group.shape[0], MAX_VARIATES)
            if self.channel_limit:
                limit = min(limit, int(rng.integers(1, self.channel_limit + 1)))
            if joint:
                limit = min(limit, MAX_VARIATES // joint)
            channels = np.sort(rng.choice(group.shape[0], limit, replace=False))
            rate, phase, mode = 1, 0, False
            if short:
                rate, phase, mode = draw_short_coarsening(rng, self.augment, group.shape[1],
                                                          self.short_least)
            elif self.augment and column is None and not thinned:
                full = self.far + self.context + self.horizon
                rate, phase, mode = draw_coarsening(rng, augment, group.shape[1], full)
            points = (group.shape[1] - phase) // rate
            if mode == PEAK:  # whole blocks of two points, from an even place
                points -= points % 2
            length = min(points, self.context + self.horizon)
            start = int(rng.integers(points - length + 1))
            first = max(0, start - self.far)
            if mode == PEAK:
                start -= start % 2
                first += first % 2
            if joint:
                crop = np.concatenate([
                    coarse_points(whole, channels, first, start + length, joint, offset + lead, False)
                    for lead in range(joint)
                ])
            else:
                crop = coarse_points(group, channels, first, start + length, rate, phase, mode)
            cut = length - self.horizon
            if self.augment:
                crop = apply_augment(crop, flips, augment, start - first + cut)
            # After a reversal the same split holds: the points are contiguous and in a new order.
            older, window = crop[:, : start - first], crop[:, start - first :]
            if self.min_observed:
                split = max(1, length - self.target_points)
                past, future = np.isfinite(window[:, :split]), np.isfinite(window[:, split:])
                if min(past.mean(), future.mean()) < self.min_observed:
                    continue
            if np.isfinite(window[:, :cut]).any() and np.isfinite(window[:, cut:]).any():
                history = np.full((crop.shape[0], self.far + self.context), np.nan, dtype="float32")
                history[:, -cut:] = window[:, :cut]
                history[:, -cut - older.shape[1] : -cut] = older
                sample = torch.from_numpy(history), torch.from_numpy(window[:, cut:])
                return (*sample, spacing * rate) if self.rate_tags else sample
        raise ValueError("Could not find an observed window after 100 attempts")


class Extras(Dataset):
    """Every batch of chosen, in order, followed by extra short-series windows of its own.

    chosen is a Subset of a Windows whose indices are consecutive, so a batch's first index marks
    the extra windows' draws, and a resumed run gets the same ones.
    """

    def __init__(self, chosen, batch: int, extra: int):
        self.chosen, self.batch, self.extra = chosen, batch, extra

    def __len__(self):
        return len(self.chosen) // self.batch * (self.batch + self.extra)

    def __getitem__(self, index):
        step, slot = divmod(int(index), self.batch + self.extra)
        if slot < self.batch:
            return self.chosen[step * self.batch + slot]
        windows = self.chosen.dataset
        return windows.short_window(self.chosen.indices[step * self.batch] + slot - self.batch)


def pack_windows(samples):
    """Pack aligned samples, padding only the variate axis; return history, target, IDs. Samples
    with a rate tag, from Windows' rate_tags, also give it per variate, and padding reads 1."""
    if not samples:
        raise ValueError("Cannot pack an empty batch")
    rows, row, used = [], [], 0
    context, horizon = samples[0][0].shape[-1], samples[0][1].shape[-1]
    for index, (history, target, *_) in enumerate(samples):
        variates = history.shape[0]
        if (
            history.shape != (variates, context)
            or target.shape != (variates, horizon)
            or not 0 < variates <= MAX_VARIATES
        ):
            raise ValueError("Samples must be aligned [V,C]/[V,H] with 1 <= V <= 32")
        if used + variates > MAX_VARIATES:
            rows.append(row)
            row, used = [], 0
        row.append((index, history, target))
        used += variates
    rows.append(row)
    width = max(sum(h.shape[0] for _, h, _ in row) for row in rows)
    history = torch.full((len(rows), width, context), float("nan"))
    target = torch.full((len(rows), width, horizon), float("nan"))
    ids = torch.full((len(rows), width), -1, dtype=torch.long)
    for batch, row in enumerate(rows):
        offset = 0
        for index, h, t in row:
            stop = offset + h.shape[0]
            history[batch, offset:stop] = h
            target[batch, offset:stop] = t
            ids[batch, offset:stop] = index
            offset = stop
    if len(samples[0]) == 2:
        return history, target, ids
    tags = torch.tensor([sample[2] for sample in samples])
    return history, target, ids, torch.where(ids >= 0, tags[ids.clamp_min(0)], 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("download", help=f"Download configs of {REPOSITORY} into the cache")
    fetch.add_argument("names", nargs="*", default=["GEP-M", "GEP-Val", "GEP-Test", "GEP-tasks"])
    args = parser.parse_args()
    from datasets import load_dataset

    for name in args.names:
        print(name, load_dataset(REPOSITORY, name, revision=REVISION))


if __name__ == "__main__":
    main()
