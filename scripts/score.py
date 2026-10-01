"""Score GIFT-Eval forecasts: native metrics, and geometric means relative to Seasonal Naive.

python -m scripts.score --forecasts DIR --upstream external/gift-eval --tasks configs/gift-full.json
    --output gift.json [--workers N]

The forecasts come from `python -m nanotsfm.evaluation gift`. This file imports nothing from
nanotsfm, so participant code never runs in the process that scores it.
"""

import argparse
import csv
import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
METRICS = {"crps": "mean_weighted_sum_quantile_loss", "mase": "MASE[0.5]"}
UPSTREAM_REVISION = "9a014e9e8ea130ba39c100c60d5dcbab7db57ac9"
GIFT_DATA, GIFT_DATA_REVISION = "Salesforce/GiftEval", "30841734ac5cfddbd0c3bad6d09d2b6b32becbb0"
TASKS_SHA256 = "3d956b416e19ecb46a9c994942549352b3aedc1334498d8a57580875f24db32c"
TASK_COUNT = 97
REFERENCE = "results/seasonal_naive/all_results.csv"
REFERENCE_SHA256 = "d89f8247cf455a953cdfb961b1ddd8fe452bfd8e3131b641fcc54234b710d949"
PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def task_file(task: dict) -> str:
    """The file that holds one task's forecasts."""
    return f"{task['dataset'].replace('/', '__')}__{task['term']}.npz"


def leaderboard_name(dataset: str, term: str, properties: dict) -> str:
    parts = dataset.split("/")
    key = PRETTY_NAMES.get(parts[0].lower(), parts[0].lower())
    frequency = parts[1] if len(parts) > 1 else properties[key]["frequency"]
    return f"{key}/{frequency}/{term}"


def reference_metrics(path: Path) -> dict:
    with path.open(newline="") as stream:
        return {
            row["dataset"]: {
                key.removeprefix("eval_metrics/"): float(value)
                for key, value in row.items()
                if key.startswith("eval_metrics/")
            }
            for row in csv.DictReader(stream)
        }


def gift_data() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(GIFT_DATA, repo_type="dataset", revision=GIFT_DATA_REVISION))


def official_metrics():
    from gluonts.ev.metrics import (
        MAE,
        MAPE,
        MASE,
        MSE,
        MSIS,
        ND,
        NRMSE,
        RMSE,
        SMAPE,
        MeanWeightedSumQuantileLoss,
    )

    return [
        MSE(forecast_type="mean"),
        MSE(forecast_type=0.5),
        MAE(),
        MASE(),
        MAPE(),
        SMAPE(),
        MSIS(),
        RMSE(),
        NRMSE(),
        ND(),
        MeanWeightedSumQuantileLoss(quantile_levels=list(QUANTILES)),
    ]


def summarize(rows):
    """Geometric mean of task scores relative to Seasonal Naive; undefined ratios give None."""
    if not rows or len({(r["dataset"], r["term"]) for r in rows}) != len(rows):
        raise ValueError("Require nonempty, unique task rows")
    result = {}
    for label, metric in METRICS.items():
        values = np.array([r["metrics"][metric] for r in rows], dtype=float)
        references = np.array([r["seasonal_naive"][metric] for r in rows], dtype=float)
        valid = np.isfinite(values) & (values >= 0) & np.isfinite(references) & (references > 0)
        score = None
        if valid.all():
            if (values == 0).any():
                score = 0.0
            else:
                score = float(np.exp(np.mean(np.log(values) - np.log(references))))
        result[f"geometric_relative_{label}"] = score
    return result


def export_csv(rows, path: Path, properties: dict):
    metrics = list(rows[0]["metrics"])
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["dataset", "model", *["eval_metrics/" + m for m in metrics], "domain", "num_variates"]
        )
        for row in rows:
            name = leaderboard_name(row["dataset"], row["term"], properties)
            metadata = properties[name.split("/")[0]]
            writer.writerow(
                [
                    name,
                    "nanotsfm",
                    *[row["metrics"][m] for m in metrics],
                    metadata["domain"],
                    metadata["num_variates"],
                ]
            )
    temporary.replace(path)


def score_task(forecasts: Path, data_root: Path, upstream: Path, task: dict, reference: dict):
    """Score one task's saved forecasts against GIFT-Eval's test windows."""
    sys.path.insert(0, str(upstream / "src"))
    os.environ["GIFT_EVAL"] = str(data_root.resolve())
    from gift_eval.data import Dataset
    from gluonts.model import evaluate_forecasts
    from gluonts.model.forecast import QuantileForecast
    from gluonts.time_feature import get_seasonality

    started = time.monotonic()
    dataset = Dataset(task["dataset"], term=task["term"])
    horizon = dataset.prediction_length
    # Saved without pickles: every entry's [variates, horizon, quantiles] stacked, and the
    # variate count of each entry.
    with np.load(forecasts / task_file(task), allow_pickle=False) as saved:
        values, variates = saved["forecasts"], saved["variates"]
    if (
        values.dtype != np.float32
        or values.ndim != 3
        or values.shape[1:] != (horizon, len(QUANTILES))
        or variates.ndim != 1
        or int(variates.sum()) != len(values)
    ):
        raise ValueError(
            f"{task_file(task)} does not hold [variates, horizon, quantiles] forecasts"
        )
    used = 0

    def wrap():
        nonlocal used
        offset = 0
        for entry, channels in zip(dataset.test_data.input, variates.tolist(), strict=True):
            target = np.asarray(entry["target"])
            if channels != (1 if target.ndim == 1 else len(target)):
                raise ValueError(f"{task_file(task)}: forecast {used} has the wrong variates")
            prediction = values[offset : offset + channels]
            offset += channels
            used += 1
            array = prediction[0].T if target.ndim == 1 else prediction.transpose(2, 1, 0)
            if not np.isfinite(array).all():
                raise ValueError("Non-finite forecast")
            yield QuantileForecast(
                forecast_arrays=np.concatenate([array, array.mean(axis=0, keepdims=True)]),
                start_date=entry["start"] + target.shape[-1],
                forecast_keys=[str(q) for q in QUANTILES] + ["mean"],
                item_id=entry.get("item_id"),
            )

    result = (
        evaluate_forecasts(
            wrap(),
            test_data=dataset.test_data,
            metrics=official_metrics(),
            batch_size=32,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=get_seasonality(dataset.freq),
        )
        .iloc[0]
        .to_dict()
    )
    if used != len(variates):
        raise ValueError(f"{task_file(task)} holds {len(variates)} forecasts; the task has {used}")
    return {
        "dataset": task["dataset"],
        "term": task["term"],
        "metrics": {k: float(v) if np.isfinite(v) else None for k, v in result.items()},
        "seasonal_naive": reference,
        "elapsed_seconds": time.monotonic() - started,
        "prediction_length": horizon,
        "windows": dataset.windows,
    }


def score(
    forecasts: Path,
    upstream: Path,
    tasks_path: Path,
    output: Path,
    workers: int = 1,
    data_root: Path | None = None,
):
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    revision = subprocess.check_output(
        ["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != UPSTREAM_REVISION:
        raise ValueError(f"GIFT-Eval checkout must be pinned to {UPSTREAM_REVISION}")
    dirty = subprocess.check_output(
        ["git", "-C", str(upstream), "status", "--porcelain", "--untracked-files=no"], text=True
    ).strip()
    if dirty:
        raise ValueError("GIFT-Eval checkout has modified tracked files")
    reference = upstream / REFERENCE
    if sha256(reference) != REFERENCE_SHA256:
        raise ValueError(f"{reference} is not GIFT-Eval's Seasonal Naive table")
    if sha256(tasks_path) != TASKS_SHA256:
        raise ValueError(f"{tasks_path} is not the pinned {TASK_COUNT}-task GIFT-Eval suite")
    data_root = Path(data_root) if data_root else gift_data()
    if not data_root.is_dir():
        raise ValueError(f"GIFT-Eval data not found at {data_root}")
    if output.exists() or output.with_suffix(".csv").exists():
        raise ValueError(f"{output} already exists; pick a new output file")
    specification = json.loads(tasks_path.read_text())
    tasks = specification["tasks"]
    if len({(t["dataset"], t["term"]) for t in tasks}) != TASK_COUNT:
        raise ValueError(f"The task manifest must hold {TASK_COUNT} distinct tasks")
    made = json.loads((forecasts / "manifest.json").read_text())
    if made["data"]["kind"] == "toy":
        raise ValueError("A model trained on the toy data cannot receive a score")
    checkpoint = Path(made["checkpoint"])
    properties = json.loads((upstream / "notebooks/dataset_properties.json").read_text())
    references = reference_metrics(reference)
    for task in tasks:
        if not (data_root / task["dataset"]).resolve().is_relative_to(data_root.resolve()):
            raise ValueError("Task dataset path escapes the data root")
        task["reference"] = references[leaderboard_name(task["dataset"], task["term"], properties)]
    order = {(t["dataset"], t["term"]): i for i, t in enumerate(tasks)}
    started, rows = time.monotonic(), []

    def record(row):
        rows.append(row)
        print(f"{row['dataset']}/{row['term']}: {row['elapsed_seconds']:.2f}s", flush=True)

    jobs = [(forecasts, data_root, upstream, task, task["reference"]) for task in tasks]
    if workers == 1:
        for job in jobs:
            record(score_task(*job))
    else:
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            futures = [executor.submit(score_task, *job) for job in jobs]
            try:
                for future in as_completed(futures):
                    record(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    rows.sort(key=lambda r: order[(r["dataset"], r["term"])])
    export_csv(rows, output.with_suffix(".csv"), properties)
    summary = summarize(rows)
    write_json(
        output,
        {
            "evaluation": "gift-eval",
            "suite": specification["name"],
            "status": specification.get("status", "unapproved"),
            "upstream_revision": revision,
            "data": made["data"],
            "checkpoint_sha256": sha256(checkpoint) if checkpoint.exists() else None,
            "task_manifest_sha256": sha256(tasks_path),
            "expected_tasks": len(tasks),
            "workers": workers,
            "protocol": "gift-v-packed-v1",
            "reference": f"gift-eval-seasonal-naive:{sha256(reference)}",
            "note": "Native metrics and Seasonal-Naive-relative geometric means; no ranks.",
            "complete": True,
            "summary": summary,
            "rows": rows,
            "forecast_seconds": made["elapsed_seconds"],
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    print(
        f"GIFT-Eval: relative CRPS {summary['geometric_relative_crps']:.4f}, "
        f"relative MASE {summary['geometric_relative_mase']:.4f} over {len(rows)} tasks"
    )
    return output


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    for name in ("forecasts", "upstream", "tasks", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, help="GIFT-Eval data; default: the pinned cache")
    parser.add_argument("--workers", type=int, default=1)
    args = vars(parser.parse_args())
    args["tasks_path"] = args.pop("tasks")
    print(score(**args))


if __name__ == "__main__":
    main()
