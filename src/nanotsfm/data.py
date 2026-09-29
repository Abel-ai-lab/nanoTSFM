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


class Windows(Dataset):
    def __init__(self, dataset, context: int, horizon: int, count: int, seed: int):
        self.context, self.horizon, self.count, self.seed = context, horizon, count, seed
        sources = {}
        for source, values in zip(dataset["source"], arrays(dataset), strict=True):
            if values.shape[1] > horizon:
                sources.setdefault(source, []).append(values)
        self.sources = list(sources.values())
        if not self.sources or min(context, horizon, count) <= 0:
            raise ValueError("No series longer than the horizon, or an invalid window/count")

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        for _ in range(100):
            source = self.sources[rng.integers(len(self.sources))]
            group = source[rng.integers(len(source))]
            channels = np.sort(
                rng.choice(
                    group.shape[0],
                    min(group.shape[0], MAX_VARIATES),
                    replace=False,
                )
            )
            length = min(group.shape[1], self.context + self.horizon)
            start = rng.integers(group.shape[1] - length + 1)
            window = np.array(group[channels, start : start + length], copy=True)
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
