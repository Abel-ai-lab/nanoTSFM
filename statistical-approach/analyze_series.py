
"""Calculate statistical features for time-series dataset items.

Example:
    uv run python statistical-approach/analyze_series.py --dataset GEP-M --split train --limit 100
    uv run python statistical-approach/analyze_series.py --dataset GEP-M --split train
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nanotsfm.data import arrays, load


def calculate_features(values: np.ndarray) -> dict:
    """Calculate statistical features for one variate of a time series."""
    values = np.asarray(values, dtype=np.float64)

    # Ignore NaN and infinite values for distribution statistics.
    finite = values[np.isfinite(values)]
    n = len(finite)

    if n == 0:
        return {
            "iqr": np.nan,
            "relative_iqr": np.nan,
            "skewness": np.nan,
            "kurtosis": np.nan,
            "autocorrelation": np.nan,
        }

    # Interquartile range: robust measure of absolute dispersion.
    q25, q75 = np.percentile(finite, [25, 75])
    iqr = q75 - q25

    # Relative IQR: dispersion relative to the absolute median.
    median = np.median(finite)

    relative_iqr = (
        iqr / abs(median)
        if abs(median) > 1e-10
        else np.nan
    )

    # Central moments.
    centered = finite - np.mean(finite)
    variance = np.mean(centered**2)

    skewness = np.nan
    kurtosis = np.nan

    if variance > 0 and n >= 3:
        m3 = np.mean(centered**3)
        g1 = m3 / variance**1.5

        # Bias-corrected sample skewness.
        skewness = np.sqrt(n * (n - 1)) / (n - 2) * g1

    if variance > 0 and n >= 4:
        m4 = np.mean(centered**4)
        g2 = m4 / variance**2 - 3

        # Bias-corrected excess kurtosis.
        kurtosis = (
            (n - 1) / ((n - 2) * (n - 3))
            * ((n + 1) * g2 + 6)
        )

    # Lag-1 autocorrelation.
    # Only use pairs adjacent in the original series where both
    # observations are finite. Do not bridge missing values.
    left = values[:-1]
    right = values[1:]

    valid_pairs = np.isfinite(left) & np.isfinite(right)
    left = left[valid_pairs]
    right = right[valid_pairs]

    autocorrelation = np.nan

    if len(left) >= 2:
        left_centered = left - np.mean(left)
        right_centered = right - np.mean(right)

        denominator = np.sqrt(
            np.sum(left_centered**2) * np.sum(right_centered**2)
        )

        if denominator > 0:
            autocorrelation = (
                np.sum(left_centered * right_centered) / denominator
            )

    return {
        "iqr": float(iqr),
        "relative_iqr": float(relative_iqr),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "autocorrelation": float(autocorrelation),
    }


def summarize_variates(values: np.ndarray) -> dict:
    """Calculate per-variate features and aggregate them into one item."""
    variate_features = [
        calculate_features(variate)
        for variate in values
    ]

    result = {}

    for feature in (
        "iqr",
        "relative_iqr",
        "skewness",
        "kurtosis",
        "autocorrelation",
    ):
        feature_values = np.array(
            [item[feature] for item in variate_features],
            dtype=np.float64,
        )

        valid = feature_values[np.isfinite(feature_values)]

        # Median represents the typical variate in this dataset item.
        result[feature] = (
            float(np.median(valid)) if len(valid) else np.nan
        )

        # Standard deviation measures variation between variates.
        result[f"{feature}_variate_std"] = (
            float(np.std(valid, ddof=1)) if len(valid) >= 2 else np.nan
        )

    return result


def get_optional_column(dataset, column_name, start, end):
    """Return a dataset column slice, or None if the column is absent."""
    if column_name not in dataset.column_names:
        return None

    return dataset[column_name][start:end]


def analyze_batch(dataset, start, end):
    """Analyze a subset of dataset rows."""
    batch = dataset.select(range(start, end))
    series_arrays = arrays(batch)

    item_ids = get_optional_column(batch, "item_id", 0, end - start)
    sources = get_optional_column(batch, "source", 0, end - start)
    frequencies = get_optional_column(batch, "freq", 0, end - start)

    rows = []

    for i, values in enumerate(series_arrays):
        # values has shape [variates, time].
        values = np.asarray(values, dtype=np.float32)

        num_variates, series_length = values.shape

        missing_proportion = float(
            1.0 - np.isfinite(values).sum() / values.size
        )

        row = {
            "dataset_index": start + i,
            "item_id": (
                item_ids[i] if item_ids is not None else str(start + i)
            ),
            "source": sources[i] if sources is not None else None,
            "freq": frequencies[i] if frequencies is not None else None,
            "num_variates": num_variates,
            "series_length": series_length,
            "missing_proportion": missing_proportion,

            # Mirrors the training Windows filter: train.py passes horizon=1.
            "eligible_for_training": series_length > 1,
        }

        row.update(summarize_variates(values))
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Calculate statistical features for time-series data."
    )

    parser.add_argument(
        "--dataset",
        default="GEP-M",
        help="Dataset name or local dataset path (default: GEP-M).",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split (default: train).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of dataset items to analyze.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Number of dataset items to process at a time.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output Parquet path.",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")

    dataset = load(args.dataset, args.split)

    total = len(dataset)
    if args.limit is not None:
        total = min(total, args.limit)

    output = args.output or Path(
        f"statistical-approach/data/series_stats_{args.dataset}_{args.split}.parquet"
    )
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Items to analyze: {total}")
    print(f"Output: {output}")

    writer = None

    try:
        for start in range(0, total, args.batch_size):
            end = min(start + args.batch_size, total)

            rows = analyze_batch(dataset, start, end)
            table = pa.Table.from_pylist(rows)

            if writer is None:
                writer = pq.ParquetWriter(output, table.schema)

            writer.write_table(table)

            print(f"Processed {end}/{total} items")

    finally:
        if writer is not None:
            writer.close()

    print(f"Finished. Results saved to {output}")


if __name__ == "__main__":
    main()
