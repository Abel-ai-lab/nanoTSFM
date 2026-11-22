"""Train and validate nanoTSFM."""

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from nanotsfm.data import (
    AUGMENTATIONS,
    COARSENINGS,
    Windows,
    describe,
    load,
    pack_windows,
    write_json,
)
from nanotsfm.model import (
    QUANTILES,
    NanoTSFM,
    device_for,
    load_checkpoint,
    network_config,
    read_config,
)

MASK_RATE_MAX = 0.4
MASK_SPAN_MAX = 8
MIN_CONTEXT_PATCHES = 1
WARMUP_FRACTION = 0.05
FINAL_LR_FRACTION = 0.1
LOG_EVERY = 25
VALIDATION_WINDOWS = 1024
# Optional training fields and their defaults, which reproduce the runs made before they existed.
OPTIONAL = {
    "sign_flip": 0.0,  # record 4: chance a window is negated; see data.Windows
    "downsample": 0.0,  # record 4: chance a window is averaged over 2 to 4 steps
    "augment": None,  # e.g. {"sign_flip": 0.5}: chances per window; see data.AUGMENTATIONS
    "channel_limit": 0,  # each window keeps a uniform 1 to this many channels; 0 keeps them all
    "polyphase_share": 0.0,  # share of windows taken from every s-th point; see data.Windows
    "polyphase_strides": [2, 4],  # the strides s those windows draw from
}


def option(settings: dict, key: str):
    return settings.get(key, OPTIONAL[key])


def load_config(path: Path):
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or set(config) != {"model", "training"}:
        raise ValueError("Config must contain exactly model and training sections")
    model = read_config(config["model"])
    settings = config["training"]
    required = {
        "data",
        "steps",
        "batch_size",
        "learning_rate",
        "seed",
        "device",
        "max_seconds",
        "checkpoint_every",
        "source_power",
    }
    if not required <= set(settings) <= required | set(OPTIONAL):
        raise ValueError(f"Training fields must be {sorted(required)}, or {sorted(OPTIONAL)}")
    for key in ("steps", "batch_size", "checkpoint_every"):
        if not isinstance(settings[key], int) or settings[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "max_seconds"):
        if not isinstance(settings[key], (int, float)) or not 0 < settings[key] < float("inf"):
            raise ValueError(f"{key} must be finite and positive")
    if not isinstance(settings["seed"], int) or settings["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    power = settings["source_power"]
    if isinstance(power, bool) or not isinstance(power, (int, float)) or not 0 <= power <= 2:
        raise ValueError("source_power must be a number from 0 to 2")
    if not isinstance(settings["data"], str) or not settings["data"]:
        raise ValueError("data must name a dataset config or a local parquet folder")
    for key in ("sign_flip", "downsample"):
        value = option(settings, key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{key} must be a number from 0 to 1")
    augment = option(settings, "augment")
    if augment is not None and (
        not isinstance(augment, dict)
        or not set(augment) <= set(AUGMENTATIONS)
        or any(type(p) not in (int, float) or not 0 <= p <= 1 for p in augment.values())
    ):
        raise ValueError(f"augment must map some of {AUGMENTATIONS} to probabilities")
    if augment and sum(augment.get(name, 0) for name in COARSENINGS) > 1:
        raise ValueError(f"{COARSENINGS} exclude each other, so may sum to at most 1")
    limit = option(settings, "channel_limit")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("channel_limit must be a nonnegative integer")
    polyphase = option(settings, "polyphase_share")
    if type(polyphase) not in (int, float) or not 0 <= polyphase <= 1:
        raise ValueError("polyphase_share must be a number from 0 to 1")
    strides = option(settings, "polyphase_strides")
    if (
        not isinstance(strides, list)
        or not strides
        or any(type(stride) is not int or stride < 2 for stride in strides)
    ):
        raise ValueError("polyphase_strides must be integers of at least 2")
    return model, settings


def save_checkpoint(path, model, optimizer, settings, step, seconds, data):
    payload = {
        "format_version": 3,
        "model_config": asdict(model.config),
        "training": settings,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "data": data,
        "elapsed_seconds": seconds,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if next(model.parameters()).is_cuda else [],
    }
    temporary = path.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    observed = torch.isfinite(target)
    if prediction.shape != (*target.shape, len(QUANTILES)):
        raise ValueError("prediction and target shapes differ")
    if not observed.any():
        raise ValueError("Need at least one observed target")
    error = torch.where(observed, target, 0.0).unsqueeze(-1) - prediction
    quantiles = prediction.new_tensor(QUANTILES)
    loss = torch.maximum(quantiles * error, (quantiles - 1) * error)
    return (loss * observed.unsqueeze(-1)).sum() / (observed.sum() * len(QUANTILES))


@torch.no_grad()
def validate(model, batches, device) -> float | None:
    if not batches:
        return None
    model.eval()
    patch, total, count = model.config.patch_length, 0.0, 0
    for history, target, series_ids in batches:
        history, target = history.to(device), target.to(device)
        future = math.ceil(target.shape[-1] / patch)
        values = F.pad(history, (0, future * patch), value=float("nan"))
        quantiles, loc, scale = model.next_patch(values, series_ids.to(device))
        first = history.shape[-1] // patch - 1
        window = slice(first, first + future)
        target = F.pad(target, (0, future * patch - target.shape[-1]), value=float("nan"))
        target = target.unflatten(-1, (future, patch)).double()
        scaled = torch.asinh((target - loc[..., window, None]) / scale[..., window, None]).float()
        observed = int(torch.isfinite(scaled).sum())
        total += float(pinball_loss(quantiles[:, :, window], scaled)) * observed
        count += observed
    model.train()
    return total / count


def contiguous_patch_mask(shape, device) -> torch.Tensor:
    patches = shape[-1]
    span = torch.randint(1, MASK_SPAN_MAX + 1, (*shape[:-1], 1), device=device)
    rate = torch.rand(*shape[:-1], 1, device=device) * MASK_RATE_MAX
    starts = torch.rand(shape, device=device) < rate / span
    position = torch.arange(patches, device=device)
    latest = torch.where(starts, position, -MASK_SPAN_MAX).cummax(dim=-1).values
    return position - latest < span


def next_patch_loss(model, sequence: torch.Tensor, series_ids: torch.Tensor) -> torch.Tensor:
    patch = model.config.patch_length
    batch, variates, length = sequence.shape
    hidden = contiguous_patch_mask((batch, variates, length // patch), sequence.device)
    observed = torch.isfinite(sequence).unflatten(-1, (-1, patch)).any(-1)
    hidden &= observed.cumsum(-1) > MIN_CONTEXT_PATCHES
    seen = (observed & ~hidden).cummax(-1).values
    inputs = sequence.masked_fill(hidden.repeat_interleave(patch, -1), float("nan"))
    quantiles, loc, scale = model.next_patch(inputs, series_ids)
    target = sequence.unflatten(-1, (-1, patch))[:, :, 1:].double()
    target = torch.asinh((target - loc[..., :-1, None]) / scale[..., :-1, None]).float()
    seen = (observed & ~hidden).cummax(-1).values
    target = target.masked_fill(~seen[..., :-1, None], float("nan"))
    return pinball_loss(quantiles[:, :, :-1], target)


def learning_rate(step: int, steps: int, peak: float) -> float:
    warmup = max(1, int(WARMUP_FRACTION * steps))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * (
        FINAL_LR_FRACTION + (1 - FINAL_LR_FRACTION) * 0.5 * (1 + math.cos(math.pi * progress))
    )


def record_environment(output: Path):
    source = Path(__file__).parent
    (output / "code").mkdir()
    for path in source.glob("*.py"):
        shutil.copy2(path, output / "code" / path.name)

    def git(*args):
        result = subprocess.run(["git", *args], cwd=source, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None

    commit = git("rev-parse", "HEAD")
    inputs = [":/src", ":/configs", ":/scripts", ":/run.sh", ":/pyproject.toml", ":/uv.lock"]
    write_json(
        output / "environment.json",
        {
            "commit": commit,
            "uncommitted_changes": bool(git("status", "--porcelain", "--", *inputs))
            if commit
            else None,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "host": platform.node(),
        },
    )


def train(
    config_path: Path,
    output: Path,
    resume: Path | None = None,
    data: str | None = None,
    seed: int | None = None,
):
    model_config, settings = load_config(config_path)
    # config.json records the config as given, which the forecast reads in full; see network_config.
    recorded = model_config
    model_config = network_config(model_config)
    if data is not None:
        settings["data"] = str(data)
    if seed is not None:
        settings["seed"] = seed
    train_set = load(settings["data"], "train")
    identity = describe(settings["data"], train_set)
    device = device_for(settings["device"])
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    model = NanoTSFM(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"])
    first_step, previous_seconds = 0, 0.0
    checkpoint = None
    if resume:
        _, checkpoint = load_checkpoint(resume)
        # Through read_config, so checkpoints saved before a field existed compare at its default.
        saved_config = asdict(read_config(checkpoint["model_config"]))
        if checkpoint["data"] != identity or saved_config != asdict(model_config):
            raise ValueError("Resume requires identical data and model configuration")
        for field in ("steps", "batch_size", "learning_rate", "seed", "device"):
            if checkpoint["training"][field] != settings[field]:
                raise ValueError(f"Cannot change {field} during resume")
        for field in OPTIONAL:
            if option(checkpoint["training"], field) != option(settings, field):
                raise ValueError(f"Cannot change {field} during resume")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        first_step, previous_seconds = checkpoint["step"], checkpoint["elapsed_seconds"]
        torch.set_rng_state(checkpoint["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
    if first_step >= settings["steps"] or previous_seconds >= settings["max_seconds"]:
        raise ValueError("Checkpoint already reached the requested step or elapsed-time limit")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a fresh output directory, including for resume")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "config.json", {"model": asdict(recorded), "training": settings})
    record_environment(output)
    patch = model_config.patch_length
    future = math.ceil(model_config.prediction_length / patch) * patch
    batch_size = settings["batch_size"]
    # Windows splits off one point; training joins it back into the full sequence.
    windows = Windows(
        train_set,
        model_config.context_length + future - 1,
        1,
        settings["steps"] * batch_size,
        settings["seed"],
        settings["source_power"],
        option(settings, "sign_flip"),
        option(settings, "downsample"),
        option(settings, "augment"),
        option(settings, "channel_limit"),
        option(settings, "polyphase_share"),
        option(settings, "polyphase_strides"),
    )
    loader = DataLoader(
        Subset(windows, range(first_step * batch_size, len(windows))),
        batch_size=batch_size,
        collate_fn=pack_windows,
        num_workers=min(8, os.cpu_count() or 1) if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda",
        # A private generator keeps the global RNG, and so resumed runs, unchanged.
        generator=torch.Generator(),
    )
    # Fix validation sampling independently of the training seed.
    try:
        held_out = Windows(
            load(settings["data"], "validation"),
            model_config.context_length,
            model_config.prediction_length,
            VALIDATION_WINDOWS,
            0,
        )
        samples = [held_out[i] for i in range(len(held_out))]
        batches = [
            pack_windows(samples[i : i + batch_size]) for i in range(0, len(samples), batch_size)
        ]
    except ValueError:  # no validation split, or no series in it longer than the horizon
        batches = []
    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    autocast = torch.autocast("cuda", torch.bfloat16, enabled=bf16)
    # The clock runs from the first batch's arrival, so that worker startup and the first read
    # count as setup and data loading. Validation and saving are off it too.
    started, paused = None, 0.0
    step, last_loss, windows_seen, validation = first_step, None, 0, None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def elapsed():
        """Training seconds so far; validation and checkpoint saving are off the clock."""
        if started is None:
            return previous_seconds
        return previous_seconds + time.monotonic() - started - paused

    def save(seconds):
        path = output / "checkpoint.pt"
        save_checkpoint(path, model, optimizer, settings, step, seconds, identity)

    model.train()
    healthy = True
    # Set on the GPU when a gradient is not finite; read only when the loss is logged or saved.
    nonfinite = torch.zeros((), dtype=torch.bool, device=device)

    def check_finite():
        nonlocal healthy
        if not math.isfinite(float(loss.detach())) or bool(nonfinite):
            healthy = False
            raise ValueError(f"Non-finite loss or gradient by step {step}")

    interrupted = True  # until the training loop ends on its own
    try:
        with (output / "training.jsonl").open("w") as log:
            for history, target, series_ids in loader:
                if started is None:
                    started = time.monotonic()
                if elapsed() >= settings["max_seconds"]:
                    break
                # Joined after the transfer: a join on the CPU would leave pinned memory.
                history = history.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)
                sequence = torch.cat([history, target], dim=-1)
                series_ids = series_ids.to(device, non_blocking=True)
                for group in optimizer.param_groups:
                    group["lr"] = learning_rate(step, settings["steps"], settings["learning_rate"])
                optimizer.zero_grad(set_to_none=True)
                with autocast:
                    loss = next_patch_loss(model, sequence, series_ids)
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                nonfinite |= ~torch.isfinite(norm)
                optimizer.step()
                step += 1
                windows_seen += batch_size
                if step % LOG_EVERY == 0 or step == settings["steps"]:
                    check_finite()
                    last_loss = float(loss.detach())
                    record = {"step": step, "loss": last_loss, "seconds": round(elapsed(), 2)}
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                if step % settings["checkpoint_every"] == 0:
                    sync()  # finish queued training work before the clock pauses
                    check_finite()
                    seconds = elapsed()
                    pause = time.monotonic()
                    with autocast:
                        validation = validate(model, batches, device)
                    record = {"step": step, "validation_loss": validation}
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    save(seconds)
                    paused += time.monotonic() - pause
        if bool(nonfinite):
            raise ValueError(f"Non-finite gradient by step {step}")
        interrupted = False
    finally:
        sync()
        seconds = elapsed()
        # Keep completed steps on interruption, but never overwrite a good checkpoint with
        # weights from a non-finite update.
        healthy = healthy and not bool(nonfinite)
        if healthy:
            save(seconds)
        complete = step == settings["steps"] and not interrupted
        write_json(
            output / "run.json",
            {
                "status": ("completed" if complete else "partial") if healthy else "failed",
                "steps": step,
                "steps_this_run": step - first_step,
                "windows_this_run": windows_seen,
                "last_loss": last_loss,
                "validation_loss": validation,
                "elapsed_seconds": seconds,
                "steps_per_second": (step - first_step) / max(seconds - previous_seconds, 1e-9),
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None,
                "parameters": sum(p.numel() for p in model.parameters()),
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
                "data": identity,
            },
        )
    return output / "checkpoint.pt"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--data", help="override the config's data: a config name or a folder")
    parser.add_argument("--seed", type=int, help="override the config's seed")
    args = parser.parse_args()
    print(train(args.config, args.output, args.resume, args.data, args.seed))


if __name__ == "__main__":
    main()
