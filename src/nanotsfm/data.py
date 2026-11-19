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
# Training transforms of one real series; see apply_augment. stride and block_mean exclude each
# other, so a window is never resampled twice.
AUGMENTATIONS = ("sign_flip", "time_reversal", "stride", "block_mean", "offset", "rescale")
COARSENINGS = ("stride", "block_mean")
COARSE_RATES = (2, 3, 4)
OFFSET_RANGE = 3.0  # the largest offset, in the channel's standard deviations
RESCALE_RANGE = 2.0  # factors are log-uniform from 1 / RESCALE_RANGE to RESCALE_RANGE
# A rescale may not take a channel near model.MIN_SCALE, where the causal scale stops following it.
RESCALE_MIN_STD = 1e-3


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
    if Path(name).is_dir():
        return load_dataset(str(name), split=split)
    return load_dataset(REPOSITORY, name, split=split, revision=REVISION)


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


def draw_coarsening(rng, augment: dict, points: int, full: int):
    """Draw the rate, the phase and the mode (see coarse_points), for a series of points.

    The rate falls to what leaves a full window of real points, so a series shorter than two
    windows is never resampled; a rate of 1 leaves the series as it is.
    """
    stride, block = (augment.get(name, 0) for name in COARSENINGS)
    roll = rng.random()
    if roll >= stride + block:
        return 1, 0, False
    rate = min(int(rng.choice(COARSE_RATES)), points // full)
    if rate < 2:
        return 1, 0, False
    phase = int(rng.integers(min(rate, points - rate * full + 1)))
    return rate, phase, roll >= stride


def coarse_points(group, channels, first: int, stop: int, rate: int, phase: int, mean: bool):
    """Points first to stop of the series at this rate: every rate-th real point, or with mean the
    block means. A block's mean skips its missing points, and a block with none observed is
    missing."""
    begin, end = phase + first * rate, phase + stop * rate
    if not mean:
        return np.array(group[channels, begin:end:rate], copy=True)
    blocks = group[channels, begin:end].reshape(len(channels), -1, rate)
    observed = np.isfinite(blocks)
    seen = observed.sum(-1)
    total = np.where(observed, blocks, 0).sum(-1, dtype="float64")
    return np.where(seen > 0, total / np.maximum(seen, 1), np.nan).astype("float32")


def apply_augment(values: np.ndarray, rng, augment: dict) -> np.ndarray:
    """Apply the per-window transforms to [V,T] values; every channel gets the same draws.

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


class Windows(Dataset):
    def __init__(
        self,
        dataset,
        context: int,
        horizon: int,
        count: int,
        seed: int,
        power=0.0,
        augment: dict | None = None,
        channel_limit: int = 0,
        polyphase_share: float = 0.0,
        polyphase_strides=(2, 4),
    ):
        """Draw a source with weight (its series count) ** power, then a series from it.

        Power 0 weights sources equally, 1 weights every series equally.

        These are for training data only. With augment, a map from names in AUGMENTATIONS to
        probabilities, each window is transformed on its own from the window's seeded generator.
        Stride and block mean resample the series before the crop, so they read a crop rate times
        longer and the window stays full length; the other transforms apply to the crop.
        channel_limit caps a window's channels at a uniform draw from 1 to it.

        A polyphase_share of windows comes from a polyphase subsequence of the series, every s-th
        real point at a random phase, with s drawn from polyphase_strides. Unlike augment's stride
        it also reaches short series, whose windows are padded. Such a window is resampled once, so
        augment's stride and block mean skip it.
        """
        self.context, self.horizon, self.count, self.seed = context, horizon, count, seed
        self.augment, self.channel_limit = augment or None, channel_limit
        self.polyphase_share, self.polyphase_strides = polyphase_share, polyphase_strides
        self.dataset, self.values = dataset, arrays(dataset)
        sources = {}
        for row, (source, values) in enumerate(zip(dataset["source"], self.values, strict=True)):
            if values.shape[1] > horizon:
                sources.setdefault(source, []).append(row)
        self.sources = [np.array(rows) for rows in sources.values()]
        if not self.sources or min(context, horizon, count) <= 0:
            raise ValueError("No series longer than the horizon, or an invalid window/count")
        weights = np.array([len(source) for source in self.sources], dtype=float) ** power
        self.weights = weights / weights.sum() if power else None

    def __getstate__(self):
        # Spawned loader workers get the dataset, which pickles as its Arrow file paths when it is
        # memory-mapped, and rebuild the arrays on it, so all processes share one page cache.
        return {**self.__dict__, "values": None}

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.values = arrays(self.dataset)

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        for _ in range(100):
            if self.weights is None:  # the baseline's uniform draw, kept for exact reproduction
                source = self.sources[rng.integers(len(self.sources))]
            else:
                source = self.sources[rng.choice(len(self.sources), p=self.weights)]
            group = self.values[int(source[rng.integers(len(source))])]
            thinned = False
            if self.polyphase_share and rng.random() < self.polyphase_share:
                stride = int(rng.choice(self.polyphase_strides))
                # Every channel keeps the same phase, so variates stay aligned in time.
                part = group[:, rng.integers(stride) :: stride]
                if part.shape[1] > self.horizon:
                    group, thinned = part, True
            limit = min(group.shape[0], MAX_VARIATES)
            if self.channel_limit:
                limit = min(limit, int(rng.integers(1, self.channel_limit + 1)))
            channels = np.sort(rng.choice(group.shape[0], limit, replace=False))
            rate, phase, mean = 1, 0, False
            if self.augment and not thinned:
                full = self.context + self.horizon
                rate, phase, mean = draw_coarsening(rng, self.augment, group.shape[1], full)
            points = (group.shape[1] - phase) // rate
            length = min(points, self.context + self.horizon)
            start = int(rng.integers(points - length + 1))
            window = coarse_points(group, channels, start, start + length, rate, phase, mean)
            if self.augment:
                window = apply_augment(window, rng, self.augment)
            cut = length - self.horizon
            if np.isfinite(window[:, :cut]).any() and np.isfinite(window[:, cut:]).any():
                history = np.full((len(channels), self.context), np.nan, dtype="float32")
                history[:, -cut:] = window[:, :cut]
                return torch.from_numpy(history), torch.from_numpy(window[:, cut:])
        raise ValueError("Could not find an observed window after 100 attempts")


def pack_windows(samples):
    """Pack aligned samples, padding only the variate axis; return history, target, IDs."""
    if not samples:
        raise ValueError("Cannot pack an empty batch")
    rows, row, used = [], [], 0
    context, horizon = samples[0][0].shape[-1], samples[0][1].shape[-1]
    for index, (history, target) in enumerate(samples):
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
    return history, target, ids


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
