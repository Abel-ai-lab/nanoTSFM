"""GEP scoring, and GIFT-Eval forecasts for scripts/score.py to score.

GEP-Val and GEP-Test are diagnostics, scored here. GIFT-Eval, the score, is scored in a separate
process that runs none of nanotsfm's code: this module only writes the forecasts.
"""

import argparse
import json
import multiprocessing
import os
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


GIFT_DATA, GIFT_DATA_REVISION = "Salesforce/GiftEval", "30841734ac5cfddbd0c3bad6d09d2b6b32becbb0"


def gift_data() -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(GIFT_DATA, repo_type="dataset", revision=GIFT_DATA_REVISION))


def task_file(task: dict) -> str:
    """The file that holds one task's forecasts, as scripts/score.py names it."""
    return f"{task['dataset'].replace('/', '__')}__{task['term']}.npz"


def forecast_entries(model, entries, horizon: int):
    """Forecast GIFT-Eval inputs in batches: each batch's stacked [variates, horizon, quantiles]
    forecasts, and the variate count of each of its entries."""
    context = model.config.context_length
    iterator = iter(entries)
    while batch := list(islice(iterator, 1024)):
        samples, variates = [], []
        for entry in batch:
            target = np.asarray(entry["target"], dtype="float32")
            if target.ndim not in (1, 2) or target.shape[-1] == 0:
                raise ValueError("Expected a nonempty univariate or multivariate target")
            history = target[None] if target.ndim == 1 else target
            variates.append(len(history))
            cropped = np.full((len(history), context), np.nan, dtype="float32")
            width = min(context, history.shape[-1])
            cropped[:, -width:] = history[:, -width:]
            for start in range(0, len(history), MAX_VARIATES):
                block = torch.from_numpy(cropped[start : start + MAX_VARIATES])
                samples.append((block, torch.empty(len(block), 0)))
        blocks = []
        for start in range(0, len(samples), 1024):
            histories, _, ids = pack_windows(samples[start : start + 1024])
            prediction = forecast(model, histories, horizon, ids).cpu()
            blocks.append(prediction[ids >= 0].numpy())
        yield np.concatenate(blocks), variates


def forecast_task(checkpoint, data_root, upstream, task, device, output):
    """Forecast one task's test windows from their inputs alone, and save the forecasts."""
    sys.path.insert(0, str(upstream / "src"))
    os.environ["GIFT_EVAL"] = str(data_root.resolve())
    from gift_eval.data import Dataset

    torch.set_num_threads(min(torch.get_num_threads(), 4))
    started = time.monotonic()
    model, _ = load_checkpoint(checkpoint, device)
    dataset = Dataset(task["dataset"], term=task["term"])
    parts = list(forecast_entries(model, dataset.test_data.input, dataset.prediction_length))
    np.savez(
        output / task_file(task),
        forecasts=np.concatenate([p for p, _ in parts]).astype(np.float32, copy=False),
        variates=np.array([n for _, counts in parts for n in counts], dtype=np.int64),
    )
    return f"{task['dataset']}/{task['term']}", time.monotonic() - started


def run(
    checkpoint: Path,
    upstream: Path,
    tasks_path: Path,
    output: Path,
    device: str = "cpu",
    workers: int = 1,
    data_root: Path | None = None,
):
    """Write forecasts for every GIFT-Eval task into `output`; scripts/score.py scores them."""
    if not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if output.exists():
        raise ValueError(f"{output} already exists; pick a new folder")
    partial = output.with_name(output.name + ".partial")
    if partial.exists():
        raise ValueError(f"{partial} is left from an interrupted run; delete it and rerun")
    _, saved = load_checkpoint(checkpoint)
    if saved["data"]["kind"] == "toy":
        raise ValueError("A model trained on the toy data cannot receive a score")
    data_root = Path(data_root) if data_root else gift_data()
    if not data_root.is_dir():
        raise ValueError(f"GIFT-Eval data not found at {data_root}")
    tasks = json.loads(tasks_path.read_text())["tasks"]
    partial.mkdir(parents=True)
    started = time.monotonic()
    jobs = [(checkpoint, data_root, upstream, task, device, partial) for task in tasks]
    if workers == 1:
        for job in jobs:
            name, seconds = forecast_task(*job)
            print(f"{name}: {seconds:.2f}s", flush=True)
    else:
        # Spawn avoids inheriting an initialized CUDA runtime through fork.
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=multiprocessing.get_context("spawn")
        ) as executor:
            futures = [executor.submit(forecast_task, *job) for job in jobs]
            try:
                for future in as_completed(futures):
                    name, seconds = future.result()
                    print(f"{name}: {seconds:.2f}s", flush=True)
            except BaseException:
                for future in futures:
                    future.cancel()
                raise
    write_json(
        partial / "manifest.json",
        {
            "checkpoint": str(checkpoint.resolve()),
            "data": saved["data"],
            "device": device,
            "tasks": len(tasks),
            "elapsed_seconds": time.monotonic() - started,
        },
    )
    partial.rename(output)
    print(f"Forecasts for {len(tasks)} tasks; score them with scripts/score.py")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    heldout = sub.add_parser("gep", help="Score a checkpoint on GEP-Val or GEP-Test")
    for name in ("checkpoint", "output"):
        heldout.add_argument(f"--{name}", type=Path, required=True)
    heldout.add_argument("--split", choices=HELDOUT, required=True)
    heldout.add_argument("--device", default="cpu")
    gift = sub.add_parser("gift", help="Forecast the pinned GIFT-Eval suite into a folder")
    for name in ("checkpoint", "upstream", "tasks", "output"):
        gift.add_argument(f"--{name}", type=Path, required=True)
    gift.add_argument("--data-root", type=Path, help="GIFT-Eval data; default: the pinned cache")
    gift.add_argument("--device", default="cpu")
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
