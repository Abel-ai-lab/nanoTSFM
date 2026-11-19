
"""Precompute statistical features for time-series dataset items.

This script calculates per-item statistical features and saves them to
Parquet for use by the statistical sampling implementation.

Statistics are preprocessing artifacts. They are calculated before model
training and do not consume the model training time budget.

Examples:
    # Generate statistics for the complete GEP-M training split.
    uv run python statistical-approach/precompute_statistics.py \
        --dataset GEP-M --split train

    # Generate statistics for a limited test run.
    uv run python statistical-approach/precompute_statistics.py \
        --dataset GEP-M --split train --limit 100 \
        --output statistical-approach/data/test_stats.parquet

    # Overwrite an existing output file.
    uv run python statistical-approach/precompute_statistics.py \
        --dataset GEP-M --split train --overwrite
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from nanotsfm.data import arrays, load


FEATURES = (
    "iqr",
    "relative_iqr",
    "skewness",
    "kurtosis",
    "autocorrelation",
)


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
        skewness = (
            np.sqrt(n * (n - 1)) / (n - 2) * g1
        )

    if variance > 0 and n >= 4:
        m4 = np.mean(centered**4)
        g2 = m4 / variance**2 - 3

        # Bias-corrected excess kurtosis.
        kurtosis = (
            (n - 1) / ((n - 2) * (n - 3))
            * ((n + 1) * g2 + 6)
        )

    # Lag-1 autocorrelation.
    # Only use adjacent pairs where both observations are finite.
    # Do not bridge missing observations.
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
            np.sum(left_centered**2)
            * np.sum(right_centered**2)
        )

        if denominator > 0:
            autocorrelation = (
                np.sum(left_centered * right_centered)
                / denominator
            )

    return {
        "iqr": float(iqr),
        "relative_iqr": float(relative_iqr),
        "skewness": float(skewness),
        "kurtosis": float(kurtosis),
        "autocorrelation": float(autocorrelation),
    }


def summarize_variates(values: np.ndarray) -> dict:
    """Calculate per-variate features and aggregate them into one item.

    Each feature is calculated independently for every variate.
    The median across variates represents the typical variate.
    The standard deviation describes variation between variates.
    """
    variate_features = [
        calculate_features(variate)
        for variate in values
    ]

    result = {}

    for feature in FEATURES:
        feature_values = np.array(
            [item[feature] for item in variate_features],
            dtype=np.float64,
        )

        valid = feature_values[np.isfinite(feature_values)]

        result[feature] = (
            float(np.median(valid))
            if len(valid)
            else np.nan
        )

        result[f"{feature}_variate_std"] = (
            float(np.std(valid, ddof=1))
            if len(valid) >= 2
            else np.nan
        )

    return result


def get_optional_column(dataset, column_name):
    """Return a dataset column, or None if the column is absent."""
    if column_name not in dataset.column_names:
        return None

    return dataset[column_name]


def analyze_batch(dataset, start, end, horizon, seen_ids):
    """Analyze a subset of dataset rows.

    Args:
        dataset: Source dataset.
        start: Starting dataset index.
        end: Exclusive ending dataset index.
        horizon: Forecast horizon used for training eligibility.
        seen_ids: Set used to validate item ID uniqueness.

    Returns:
        List of statistical feature records.
    """
    batch = dataset.select(range(start, end))
    series_arrays = arrays(batch)

    item_ids = get_optional_column(batch, "item_id")
    sources = get_optional_column(batch, "source")
    frequencies = get_optional_column(batch, "freq")

    rows = []

    for i, values in enumerate(series_arrays):
        # The item ID is required to match statistics to training series.
        item_id = item_ids[i]

        if item_id is None:
            raise ValueError(
                f"Missing item_id at dataset index {start + i}."
            )

        if isinstance(item_id, (float, np.floating)):
            if not np.isfinite(item_id):
                raise ValueError(
                    f"Invalid item_id at dataset index {start + i}."
                )

        try:
            if item_id in seen_ids:
                raise ValueError(
                    f"Duplicate item_id found: {item_id!r}"
                )
            seen_ids.add(item_id)
        except TypeError as exc:
            raise ValueError(
                f"Unhashable item_id at dataset index {start + i}: "
                f"{item_id!r}"
            ) from exc

        # values has shape [variates, time].
        values = np.asarray(values, dtype=np.float32)

        if values.ndim != 2:
            raise ValueError(
                f"Expected a 2D array at dataset index {start + i}, "
                f"received shape {values.shape}."
            )

        num_variates, series_length = values.shape

        if num_variates == 0 or series_length == 0:
            raise ValueError(
                f"Empty time-series item at dataset index {start + i}."
            )

        missing_proportion = float(
            1.0 - np.isfinite(values).sum() / values.size
        )

        row = {
            "dataset_index": start + i,
            "item_id": item_id,
            "source": (
                sources[i] if sources is not None else None
            ),
            "freq": (
                frequencies[i] if frequencies is not None else None
            ),
            "num_variates": num_variates,
            "series_length": series_length,
            "missing_proportion": missing_proportion,

            # A series must be longer than the forecast horizon.
            "eligible_for_training": series_length > horizon,
        }

        row.update(summarize_variates(values))
        rows.append(row)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Precompute statistical features for time-series data."
        )
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
        "--horizon",
        type=int,
        default=1,
        help="Forecast horizon used to determine training eligibility.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output Parquet path.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting an existing output file.",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")

    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")

    if args.horizon <= 0:
        parser.error("--horizon must be positive.")

    # Load the requested dataset split.
    dataset = load(args.dataset, args.split)

    total = len(dataset)

    if args.limit is not None:
        total = min(total, args.limit)

    if total == 0:
        parser.error("The selected dataset contains no items.")

    # Generate a safe dataset label for the default output filename.
    dataset_label = (
        Path(args.dataset).name.replace("/", "_").replace("\\", "_")
    )

    if args.output is not None:
        output = args.output
    else:
        suffix = (
            f"_limit{total}"
            if args.limit is not None
            else ""
        )

        output = (
            Path("statistical-approach")
            / f"series_stats_{dataset_label}_{args.split}{suffix}.parquet"
        )

    output.parent.mkdir(parents=True, exist_ok=True)

    if output.exists() and not args.overwrite:
        parser.error(
            f"Output file already exists: {output}\n"
            "Use --overwrite to replace it."
        )

    # Write to a temporary file first so an interrupted run does not
    # leave a partially written file at the final output path.
    temporary_output = output.with_name(output.name + ".tmp")

    if temporary_output.exists():
        temporary_output.unlink()

    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Items to analyze: {total}")
    print(f"Batch size: {args.batch_size}")
    print(f"Forecast horizon: {args.horizon}")
    print(f"Output: {output}")

    # item_id is necessary for matching statistics to training series.
    if "item_id" not in dataset.column_names:
        parser.error(
            "Dataset does not contain the required item_id column."
        )

    writer = None
    seen_ids = set()
    processed = 0
    eligible_count = 0

    try:
        for start in range(0, total, args.batch_size):
            end = min(start + args.batch_size, total)

            rows = analyze_batch(
                dataset=dataset,
                start=start,
                end=end,
                horizon=args.horizon,
                seen_ids=seen_ids,
            )

            table = pa.Table.from_pylist(rows)

            if writer is None:
                writer = pq.ParquetWriter(
                    temporary_output,
                    table.schema,
                )

            writer.write_table(table)

            processed += len(rows)
            eligible_count += sum(
                row["eligible_for_training"]
                for row in rows
            )

            print(f"Processed {processed}/{total} items")

        if writer is None:
            raise RuntimeError(
                "No Parquet data was generated."
            )

        writer.close()
        writer = None

        # Only publish the output after the complete Parquet file
        # has been successfully written.
        temporary_output.replace(output)

    except Exception:
        if writer is not None:
            writer.close()

        if temporary_output.exists():
            temporary_output.unlink()

        raise

    print()
    print("Finished preprocessing.")
    print(f"Total items: {processed}")
    print(f"Eligible items: {eligible_count}")
    print(f"Ineligible items: {processed - eligible_count}")
    print(f"Unique item IDs: {len(seen_ids)}")
    print(f"Results saved to: {output}")


if __name__ == "__main__":
    main()
