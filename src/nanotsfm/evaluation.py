"""GIFT-Eval and GEP scoring: native metrics relative to Seasonal Naive."""

import argparse
import csv
import json
import multiprocessing
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import islice
from pathlib import Path

import numpy as np
import torch

from nanotsfm.data import (
    MAX_VARIATES,
    REVISION,
    arrays,
    load,
    pack_windows,
    sha256,
    write_json,
)
from nanotsfm.model import forecast, load_checkpoint

QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def seasonal_naive(history: np.ndarray, horizon: int, seasonality: int) -> np.ndarray:
    history = np.asarray(history, dtype=float)
    if history.ndim != 1 or not len(history) or horizon <= 0 or seasonality <= 0:
        raise ValueError("Invalid seasonal-naive input")
    # Forward-fill historical gaps; use zero before the first observation.
    last = np.where(np.isfinite(history), np.arange(len(history)), -1)
    np.maximum.accumulate(last, out=last)
    filled = np.where(last >= 0, history[np.maximum(last, 0)], 0.0)
    period = min(seasonality, len(filled))
    point = np.resize(filled[-period:], horizon)
    return np.repeat(point[:, None], len(QUANTILES), axis=1)


def metrics(history: np.ndarray, target: np.ndarray, forecast: np.ndarray, seasonality: int):
    history, target, forecast = map(np.asarray, (history, target, forecast))
    if forecast.shape != (len(target), len(QUANTILES)) or not np.isfinite(forecast).all():
        raise ValueError("Forecast must be finite [horizon, nine quantiles]")
    if np.any(np.diff(forecast, axis=-1) < 0):
        raise ValueError("Forecast quantiles cross")
    if seasonality <= 0:
        raise ValueError("seasonality must be positive")
    valid = np.isfinite(target)
    if not valid.any():
        raise ValueError("No observed evaluation targets")
    truth, pred = target[valid], forecast[valid]
    error = truth[:, None] - pred
    quantiles = np.asarray(QUANTILES)
    loss_sum = 2 * np.maximum(quantiles * error, (quantiles - 1) * error).sum(axis=0)
    denominator = float(np.abs(truth).sum())
    mae = float(np.abs(truth - pred[:, 4]).mean())
    lag = min(seasonality, max(1, len(history) - 1))
    differences = np.abs(history[lag:] - history[:-lag])
    differences = differences[np.isfinite(differences)]
    scale = float(differences.mean()) if len(differences) else 0.0
    return {
        "crps": float(loss_sum.mean() / denominator) if denominator else None,
        "MASE": mae / scale if scale > 0 else None,
        "MAE": mae,
        "observed_targets": int(valid.sum()),
        "absolute_target_sum": denominator,
        "quantile_loss_sum": loss_sum.tolist(),
    }


METRICS = {"crps": "mean_weighted_sum_quantile_loss", "mase": "MASE[0.5]"}


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


def forecast_many(model, histories: list, horizon: int, batch: int = 1024) -> list:
    context = model.config.context_length
    samples, owners = [], []
    for index, history in enumerate(histories):
        cropped = np.full((len(history), context), np.nan, dtype="float32")
        width = min(context, history.shape[-1])
        cropped[:, -width:] = history[:, -width:]
        for start in range(0, len(history), MAX_VARIATES):
            block = torch.from_numpy(cropped[start : start + MAX_VARIATES])
            samples.append((block, torch.empty(len(block), 0)))
            owners.append(index)
    parts = [[] for _ in histories]
    for start in range(0, len(samples), batch):
        chunk = samples[start : start + batch]
        packed, _, ids = pack_windows(chunk)
        prediction = forecast(model, packed, horizon, ids).cpu().numpy()
        ids = ids.numpy()
        for offset in range(len(chunk)):
            parts[owners[start + offset]].append(prediction[ids == offset])
    return [np.concatenate(part) for part in parts]


def aggregate(records) -> dict:
    """Sum CRPS across entries as GluonTS does; average MASE."""
    denominator = sum(r["absolute_target_sum"] for r in records)
    numerators = np.asarray([r["quantile_loss_sum"] for r in records]).sum(axis=0)
    mase = [r["MASE"] for r in records if r["MASE"] is not None]
    return {
        METRICS["crps"]: float(numerators.mean() / denominator) if denominator else None,
        METRICS["mase"]: float(np.mean(mase)) if mase else None,
    }


HELDOUT = {"validation": "GEP-Val", "test": "GEP-Test"}


def evaluate_heldout(checkpoint: Path, split: str, output: Path, device: str = "cpu"):
    if output.exists():
        raise FileExistsError("Choose a new metrics output path")
    model, saved = load_checkpoint(checkpoint, device)
    heldout = load(HELDOUT[split], split)
    by_source = {}
    for source, values in zip(heldout["source"], arrays(heldout), strict=True):
        by_source.setdefault(source, []).append(values)
    started = time.monotonic()
    rows = []
    for task in load("GEP-tasks", split).to_list():
        horizon, windows = task["prediction_length"], task["windows"]
        season = task["seasonality"]
        entries = []
        for values in by_source.get(task["source"], []):
            if values.shape[1] < task["min_length"]:
                continue
            for window in range(windows):
                cut = values.shape[1] - (windows - window) * horizon
                entries.append((values[:, :cut], values[:, cut : cut + horizon]))
        predictions = forecast_many(model, [history for history, _ in entries], horizon)
        scored, reference = [], []
        for (history, target), prediction in zip(entries, predictions, strict=True):
            for h, t, p in zip(history, target, prediction, strict=True):
                if np.isfinite(t).any():
                    scored.append(metrics(h, t, p, season))
                    reference.append(metrics(h, t, seasonal_naive(h, horizon, season), season))
        rows.append(
            {
                "dataset": task["source"],
                "term": task["term"],
                "prediction_length": horizon,
                "windows": windows,
                "forecasts": len(scored),
                "metrics": aggregate(scored),
                "seasonal_naive": aggregate(reference),
            }
        )
    result = {
        "evaluation": HELDOUT[split],
        "revision": REVISION,
        "data": saved.get("data"),
        "checkpoint_sha256": sha256(checkpoint),
        "summary": summarize(rows),
        "tasks": len(rows),
        "elapsed_seconds": time.monotonic() - started,
        "rows": rows,
    }
    write_json(output, result)
    summary = result["summary"]
    print(
        f"{HELDOUT[split]}: relative CRPS {summary['geometric_relative_crps']:.4f}, "
        f"relative MASE {summary['geometric_relative_mase']:.4f} over {len(rows)} tasks"
    )
    return output


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


def make_predictor(model, prediction_length, seasonality=1):
    from gluonts.model.forecast import QuantileForecast
    from gluonts.model.predictor import Predictor

    class NanoPredictor(Predictor):
        def __init__(self):
            super().__init__(prediction_length=prediction_length)

        def predict(self, dataset, **kwargs):
            iterator = iter(dataset)
            while entries := list(islice(iterator, 1024)):
                samples, layouts, baseline = [], [], []
                for entry in entries:
                    target = np.asarray(entry["target"], dtype="float32")
                    if target.ndim not in (1, 2) or target.shape[-1] == 0:
                        raise ValueError("Expected a nonempty univariate or multivariate target")
                    history = target[None] if target.ndim == 1 else target
                    layouts.append((target.ndim, len(history), target.shape[-1]))
                    if model is None:
                        baseline.extend(
                            [
                                seasonal_naive(channel, prediction_length, seasonality)
                                for channel in history
                            ]
                        )
                    else:
                        context = model.config.context_length
                        cropped = np.full((len(history), context), np.nan, dtype="float32")
                        width = min(context, history.shape[-1])
                        cropped[:, -width:] = history[:, -width:]
                        for start in range(0, len(history), MAX_VARIATES):
                            h = torch.from_numpy(cropped[start : start + MAX_VARIATES])
                            samples.append((h, torch.empty(len(h), 0)))
                if model is None:
                    predictions = np.stack(baseline)
                else:
                    blocks = []
                    for start in range(0, len(samples), 1024):
                        histories, _, ids = pack_windows(samples[start : start + 1024])
                        prediction = forecast(model, histories, prediction_length, ids).cpu()
                        blocks.append(prediction[ids >= 0].numpy())
                    predictions = np.concatenate(blocks)
                offset = 0
                for entry, (ndim, channels, length) in zip(entries, layouts, strict=True):
                    prediction = predictions[offset : offset + channels]
                    offset += channels
                    values = prediction[0].T if ndim == 1 else prediction.transpose(2, 1, 0)
                    if not np.isfinite(values).all():
                        raise ValueError("Non-finite forecast")
                    values = np.concatenate([values, values.mean(axis=0, keepdims=True)])
                    yield QuantileForecast(
                        forecast_arrays=values,
                        start_date=entry["start"] + length,
                        forecast_keys=[str(q) for q in QUANTILES] + ["mean"],
                        item_id=entry.get("item_id"),
                    )

    return NanoPredictor()


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


def evaluate_task(checkpoint, data_root, upstream, task, device, reference):
    sys.path.insert(0, str(upstream / "src"))
    os.environ["GIFT_EVAL"] = str(data_root.resolve())
    from gift_eval.data import Dataset
    from gluonts.model import evaluate_model
    from gluonts.time_feature import get_seasonality

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    started = time.monotonic()
    model, _ = load_checkpoint(checkpoint, device)
    dataset = Dataset(task["dataset"], term=task["term"])
    seasonality = get_seasonality(dataset.freq)

    def measure(predictor):
        result = (
            evaluate_model(
                predictor,
                test_data=dataset.test_data,
                metrics=official_metrics(),
                batch_size=32,
                axis=None,
                mask_invalid_label=True,
                allow_nan_forecast=False,
                seasonality=seasonality,
            )
            .iloc[0]
            .to_dict()
        )
        return {key: float(value) if np.isfinite(value) else None for key, value in result.items()}

    clean = measure(make_predictor(model, dataset.prediction_length))
    return {
        "dataset": task["dataset"],
        "term": task["term"],
        "metrics": clean,
        "seasonal_naive": reference,
        "elapsed_seconds": time.monotonic() - started,
        "prediction_length": dataset.prediction_length,
        "windows": dataset.windows,
    }


def run(
    checkpoint: Path,
    upstream: Path,
    tasks_path: Path,
    output: Path,
    device: str = "cpu",
    allow_toy: bool = False,
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
    partial = output.with_suffix(".partial.json")
    if not data_root.is_dir():
        raise ValueError(f"GIFT-Eval data not found at {data_root}")
    if output.exists() or output.with_suffix(".csv").exists():
        raise ValueError(f"{output} already exists; pick a new output file")
    if partial.exists():
        raise ValueError(f"{partial} is left from an interrupted run; delete it and rerun")
    specification = json.loads(tasks_path.read_text())
    if specification.get("upstream_revision", revision) != revision:
        raise ValueError("Task manifest uses a different upstream revision")
    tasks = specification["tasks"]
    if len({(t["dataset"], t["term"]) for t in tasks}) != TASK_COUNT:
        raise ValueError(f"The task manifest must hold {TASK_COUNT} distinct tasks")
    _, saved = load_checkpoint(checkpoint)
    data = saved["data"]
    if data["kind"] == "toy" and not allow_toy:
        raise ValueError("Toy-data checkpoints require --allow-toy and cannot receive a score")
    rows = []
    started = time.monotonic()
    provenance = {
        "evaluation": "gift-eval",
        "suite": specification["name"],
        "status": specification.get("status", "unapproved"),
        "upstream_revision": revision,
        "data": data,
        "checkpoint_sha256": sha256(checkpoint),
        "task_manifest_sha256": sha256(tasks_path),
        "expected_tasks": len(tasks),
        "workers": workers,
        "protocol": "gift-v-packed-v1",
        "reference": f"gift-eval-seasonal-naive:{sha256(reference)}",
        "note": "Native metrics and Seasonal-Naive-relative geometric means; no ranks.",
    }
    properties = json.loads((upstream / "notebooks/dataset_properties.json").read_text())
    references = reference_metrics(reference)
    for task in tasks:
        if not (data_root / task["dataset"]).resolve().is_relative_to(data_root.resolve()):
            raise ValueError("Task dataset path escapes the data root")
        task["reference"] = references[leaderboard_name(task["dataset"], task["term"], properties)]
    order = {(t["dataset"], t["term"]): i for i, t in enumerate(tasks)}

    def record(row):
        rows.append(row)
        rows.sort(key=lambda r: order[(r["dataset"], r["term"])])
        print(f"{row['dataset']}/{row['term']}: {row['elapsed_seconds']:.2f}s", flush=True)
        write_json(
            partial,
            {
                **provenance,
                "complete": False,
                "rows": rows,
                "elapsed_seconds": time.monotonic() - started,
            },
        )

    if workers == 1:
        for task in tasks:
            record(evaluate_task(checkpoint, data_root, upstream, task, device, task["reference"]))
    else:
        # Spawn avoids inheriting an initialized CUDA runtime through fork.
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            futures = [
                executor.submit(
                    evaluate_task, checkpoint, data_root, upstream, task, device, task["reference"]
                )
                for task in tasks
            ]
            try:
                for future in as_completed(futures):
                    record(future.result())
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    export_csv(rows, output.with_suffix(".csv"), properties)
    write_json(
        output,
        {
            **provenance,
            "complete": True,
            "summary": summarize(rows),
            "rows": rows,
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    partial.unlink(missing_ok=True)
    summary = summarize(rows)
    print(
        f"GIFT-Eval: relative CRPS {summary['geometric_relative_crps']:.4f}, "
        f"relative MASE {summary['geometric_relative_mase']:.4f} over {len(rows)} tasks"
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    heldout = sub.add_parser("gep", help="Score a checkpoint on GEP-Val or GEP-Test")
    for name in ("checkpoint", "output"):
        heldout.add_argument(f"--{name}", type=Path, required=True)
    heldout.add_argument("--split", choices=HELDOUT, required=True)
    heldout.add_argument("--device", default="cpu")
    gift = sub.add_parser("gift", help="Run the pinned GIFT-Eval suite")
    for name in ("checkpoint", "upstream", "tasks", "output"):
        gift.add_argument(f"--{name}", type=Path, required=True)
    gift.add_argument("--data-root", type=Path, help="GIFT-Eval data; default: the pinned cache")
    gift.add_argument("--device", default="cpu")
    gift.add_argument("--allow-toy", action="store_true")
    gift.add_argument("--workers", type=int, default=1)
    args = vars(parser.parse_args())
    command = args.pop("command")
    if command == "gep":
        print(evaluate_heldout(**args))
    else:
        args["tasks_path"] = args.pop("tasks")
        print(run(**args))


if __name__ == "__main__":
    main()
