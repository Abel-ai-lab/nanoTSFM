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


# Statistical sampling experiment configuration to avoid .yaml checks
# and changing train.py.
#
# Set STATISTICAL_SAMPLING = False for the original sampling behavior.
# Set STATISTICAL_SAMPLING = True for the statistical sampling experiment.
#
# Statistical training experiment requires source_power = 1.0.
STATISTICAL_SAMPLING = True
PROFILE_POWER = 0.5

# data.py is located at src/nanotsfm/data.py.
# parents[2] resolves to the repository root.
REPOSITORY_ROOT = Path(__file__).resolve().parents[2]

PROFILE_STATS_PATH = (
    REPOSITORY_ROOT
    / "statistical-approach"
    / "series_stats_GEP-L_train.parquet"
)

# All series with one or more missing/non-finite statistical features
# are assigned this separate profile.
MISSING_STATS_PROFILE = (-1, -1, -1, -1)


class Windows(Dataset):
    def __init__(
        self,
        dataset,
        context: int,
        horizon: int,
        count: int,
        seed: int,
        power=0.0,
        profile_power=1.0,
        profile_path=None,
    ):
        """Sample a source, then a statistical profile, then a series.

        Source probability is proportional to source_size ** power.

        Within each source, profile probability is proportional to
        profile_size ** profile_power.

        A series is then selected uniformly within the chosen profile.

        Statistical sampling is enabled only for the training configuration
        with source_power=1.0. Validation retains the original sampling.
        """

        self.context = context
        self.horizon = horizon
        self.count = count
        self.seed = seed

        # Enable statistical sampling only for the training configuration.
        #
        # Training passes source_power=1.0.
        # Validation omits power, so it uses the default power=0.0.
        use_statistical_sampling = (
            STATISTICAL_SAMPLING and np.isclose(float(power), 1.0)
        )

        if use_statistical_sampling:
            profile_power = PROFILE_POWER
            profile_path = PROFILE_STATS_PATH
        else:
            # Preserve original uniform-series sampling.
            profile_power = 1.0
            profile_path = None

        self.profile_power = float(profile_power)

        if (
            not np.isfinite(self.profile_power)
            or not 0 <= self.profile_power <= 1
        ):
            raise ValueError(
                "profile_power must be a finite number from 0 to 1"
            )

        # ---------------------------------------------------------
        # Load statistical profiles only for training.
        # ---------------------------------------------------------

        profiles_by_id = None

        if use_statistical_sampling:
            if profile_path is None or not Path(profile_path).is_file():
                raise FileNotFoundError(
                    f"Statistical profile file not found: {profile_path}"
                )

            import pandas as pd

            stats = pd.read_parquet(profile_path)

            features = [
                "relative_iqr",
                "skewness",
                "kurtosis",
                "autocorrelation",
            ]

            required = {"item_id", *features}
            missing_columns = required - set(stats.columns)

            if missing_columns:
                raise ValueError(
                    "Statistical profile file is missing columns: "
                    f"{sorted(missing_columns)}"
                )

            if stats["item_id"].duplicated().any():
                raise ValueError(
                    "Statistical profile file contains duplicate item_id values"
                )

            # Only use series eligible for training, if the column exists.
            if "eligible_for_training" in stats.columns:
                stats = stats[
                    stats["eligible_for_training"].fillna(False)
                ].copy()

            # Replace infinite values with NaN.
            stats[features] = stats[features].replace(
                [np.inf, -np.inf], np.nan
            )

            # Separate complete and incomplete statistical records.
            complete_mask = stats[features].notna().all(axis=1)

            complete_stats = stats.loc[complete_mask].copy()
            incomplete_stats = stats.loc[~complete_mask].copy()

            # -----------------------------------------------------
            # Construct global five-quantile profiles.
            #
            # Quantile ranks are calculated using complete records
            # only.
            # -----------------------------------------------------

            bin_columns = []

            for feature in features:
                bin_column = f"{feature}_bin"

                if not complete_stats.empty:
                    ranks = complete_stats[feature].rank(pct=True)

                    complete_stats[bin_column] = np.minimum(
                        (ranks * 5).astype(int),
                        4,
                    )

                bin_columns.append(bin_column)

            # Map complete series to their four-dimensional profiles.
            complete_profiles = {}

            if not complete_stats.empty:
                complete_stats["profile"] = list(
                    complete_stats[bin_columns].itertuples(
                        index=False,
                        name=None,
                    )
                )

                complete_profiles = dict(
                    zip(
                        complete_stats["item_id"],
                        complete_stats["profile"],
                        strict=True,
                    )
                )

            # Preserve incomplete series using a separate profile.
            incomplete_profiles = {
                item_id: MISSING_STATS_PROFILE
                for item_id in incomplete_stats["item_id"]
            }

            profiles_by_id = {
                **complete_profiles,
                **incomplete_profiles,
            }

            print(
                "Statistical profiles loaded:"
                f"\n  Complete series: {len(complete_profiles)}"
                f"\n  Incomplete series: {len(incomplete_profiles)}"
                f"\n  Total profiles: {len(set(profiles_by_id.values()))}"
                f"\n  Profile power: {self.profile_power}"
            )

        # ---------------------------------------------------------
        # Group series by source and statistical profile.
        # ---------------------------------------------------------

        source_names = []
        sources = []
        source_profiles = []

        source_indices = {}

        for item_id, source, values in zip(
            dataset["item_id"],
            dataset["source"],
            arrays(dataset),
            strict=True,
        ):
            # Preserve the original minimum-length filtering.
            if values.shape[1] <= horizon:
                continue

            # Create a source group if this is the first occurrence.
            if source not in source_indices:
                source_indices[source] = len(sources)

                source_names.append(source)
                sources.append([])
                source_profiles.append({})

            source_index = source_indices[source]

            # Add the series to its source.
            sources[source_index].append(values)

            # Add the series to its statistical profile group.
            if profiles_by_id is not None:
                if item_id not in profiles_by_id:
                    raise ValueError(
                        f"No eligible statistical profile found for {item_id}. "
                        "Check that the Parquet file matches the training dataset."
                    )

                profile = profiles_by_id[item_id]

                source_profiles[source_index].setdefault(
                    profile,
                    [],
                ).append(values)

        self.sources = sources

        if not self.sources or min(context, horizon, count) <= 0:
            raise ValueError("No series longer than the horizon, or an invalid window/count")

        # ---------------------------------------------------------
        # Source-level sampling.
        #
        # power=0: uniform source sampling (original behavior).
        # power=1: source probability proportional to source size.
        # ---------------------------------------------------------

        weights = np.array([len(source) for source in self.sources], dtype=float) ** power
        self.weights = weights / weights.sum() if power else None

        # ---------------------------------------------------------
        # Profile-level sampling.
        #
        # P(profile) is proportional to profile_size ** profile_power.
        #
        # profile_power=1.0: equivalent to uniform series sampling.
        # profile_power=0.5: moderate emphasis on smaller profiles.
        # profile_power=0.0: uniform profile sampling.
        # ---------------------------------------------------------

        self.source_profile_groups = None

        if profiles_by_id is not None:
            self.source_profile_groups = []

            for source_index, source in enumerate(self.sources):
                groups = list(
                    source_profiles[source_index].values()
                )

                profile_sizes = np.array(
                    [len(group) for group in groups],
                    dtype=float,
                )

                profile_weights = profile_sizes ** self.profile_power
                profile_weights /= profile_weights.sum()

                self.source_profile_groups.append(
                    (groups, profile_weights)
                )

        print(
            "Window sampler initialized:"
            f"\n  Sources: {len(self.sources)}"
            f"\n  Series: {sum(len(source) for source in self.sources)}"
            f"\n  Statistical sampling: {use_statistical_sampling}"
            f"\n  Source power: {power}"
        )

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, int(index)]))
        for _ in range(100):

            # Select a source.
            if self.weights is None:
                source_index = rng.integers(len(self.sources))
            else:
                source_index = rng.choice(
                    len(self.sources),
                    p=self.weights,
                )

            source = self.sources[source_index]

            # Select a series.
            if self.source_profile_groups is None:

                # Original uniform-series sampling.
                group = source[rng.integers(len(source))]

            else:

                # Select a statistical profile.
                profile_groups, profile_weights = (
                    self.source_profile_groups[source_index]
                )

                selected_profile = rng.choice(
                    len(profile_groups),
                    p=profile_weights,
                )

                profile = profile_groups[selected_profile]

                # Select a series uniformly within the profile.
                group = profile[rng.integers(len(profile))]

            # Select variates.
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
