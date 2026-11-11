"""Train and validate nanoTSFM."""

import argparse
import copy
import json
import math
import os
import platform
import shutil
import subprocess
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset

from nanotsfm.data import (
    AUGMENTATIONS,
    BANDS,
    COARSENINGS,
    Extras,
    Windows,
    describe,
    load,
    pack_windows,
    write_json,
)
from nanotsfm.model import (
    MIN_SCALE,
    QUANTILES,
    WORD_SEGMENTS,
    Ensemble,
    ModelConfig,
    NanoTSFM,
    causal_scale,
    config_fields,
    detect_periods,
    device_for,
    load_checkpoint,
    patch_words,
)

MASK_RATE_MAX = 0.4
MASK_SPAN_MAX = 8
MIN_CONTEXT_PATCHES = 1
WARMUP_FRACTION = 0.05
FINAL_LR_FRACTION = 0.1
LOG_EVERY = 25
REPOSITORY = Path(__file__).resolve().parents[2]
VALIDATION_WINDOWS = 1024
# The last share of training keeps every letter's meaning, so its embedding can settle.
SHAPE_SETTLE_FRACTION = 0.2
# Keeps raw_pinball's quantiles within about 200 scales, where sinh is still a forecast.
RAW_MAX_QUANTILE = 6.0
FRAGMENT_TEMPERATURE = 0.1  # the common InfoNCE value
MASE_CLAMP = 0.1, 10.0  # the audit's bound on a window's lag-1 error in context spreads
REGRET_DECAY = 0.99  # the copy trails by about 100 steps, 2% of a 5,000-step run
REGRET_MAX_GAP = 1.0  # one wild window can't take the batch; gaps are usually hundredths
DEEP_CLUSTER_TOKENS = 256  # 65,536 pairs, cheap beside a step, and a batch's worth of tokens
RV_FLOOR = 1e-4  # a patch with a spread of 1% of the scale or less counts as flat
HEAD_REFIT_SHARE = 0.1  # of each source's series, kept out of training for head_refit
# Optional training fields and their defaults, which reproduce the runs made before they existed.
OPTIONAL = {
    "schedule": "cosine",  # or "wsd": warmup, constant peak, then a (1 - sqrt) cooldown
    "final_lr_fraction": FINAL_LR_FRACTION,
    "cooldown_fraction": 0.2,  # wsd's share of steps spent cooling down; also late_average's share
    "band_weights": None,  # e.g. {"minutes": 0.3, "hourly": 0.25, ...}; see data.BANDS
    "exclude_sources": None,  # Pretrain sources left out of every window; see lab/transfer
    "prefix_share": 0.0,  # share of windows forecast from only a few patches; see left_censor
    "prefix_patches": 4,  # the most patches such a window keeps
    "late_average": False,  # end on the uniform mean of the last cooldown_fraction of iterates
    "coarse_weight": 0.5,  # weight of the mean coarse head loss, with model.coarse_blocks
    "augment": None,  # e.g. {"sign_flip": 0.5}: chances per window; see data.AUGMENTATIONS
    "source_weights": None,  # e.g. {"azure_vm_traces_2017": 2}; multiplies a source's draw weight
    "views": None,  # e.g. {"share": 0.15, "freqs": ["5T"], "factors": [2, 12]}; see data.Windows
    "raw_pinball": False,  # pinball in raw units over the window's mean absolute value
    "fragment_weight": 0.0,  # weight of fragment_loss, a second model pass; 0 skips it
    "batch_quotas": False,  # every batch holds each band's and source's share; see data.Windows
    "channel_limit": 0,  # each window keeps a uniform 1 to this many channels; 0 keeps them all
    "min_observed": 0.0,  # least observed share of a window's history and of its target
    "series_tilt": 0.0,  # draw series within a source by predictability; see data.Windows
    "drop_items": None,  # a repository JSON file whose "drop" list holds item_ids left out
    "leave_out": None,  # sources this member leaves out too, for ensemble_vary
    "weight_decay": None,  # on kernels only, none on norms and biases; None is AdamW's default
    "log_loss": 0.0,  # eps: mean log(eps + a window's pinball) in place of the mean pinball
    "mase_weight": False,  # divide each window's loss by its lag-1 error; see mase_weights
    "regret_weight": 0.0,  # eta: weight windows by exp(eta * loss above a trailing copy's)
    "deep_cluster": 0.0,  # weight of deep_cluster_loss on tokens and next-patch words
    "fredf": 0.0,  # weight of an L1 between the spectra of each patch's median and target
    "end_blank_rate": 0.0,  # share of windows that hide everything from the forecast origin on
    "end_blank_jitter": 0,  # patches that origin may move either way; see end_blank
    "aug_views": 1,  # views of each window in a batch, which stays its size; see data.Windows
    "aug_ramp": None,  # [first, last] multipliers of the augment chances, linear over the steps
    "drop_window": None,  # [start, stop] shares of the steps in which model.drop_path acts
    "head_refit": 0,  # the last steps train only the head, on held-out series; see member_data
    "polyphase_share": 0.0,  # share of windows taken from every s-th point; see data.Windows
    "polyphase_strides": [2, 4],  # the strides s those windows draw from, or "period"
    "polyphase_least": 0,  # fewest points a thinned series needs; below two patches nothing is scored
    "polyphase_short_share": None,  # with the next, replaces polyphase_share under two windows
    "polyphase_long_share": None,  # and from two full windows up; see data.Windows
    "polyphase_joint_share": 0.0,  # windows of every phase as channels; needs model.polyphase_joint
    "rate_floor": 0.0,  # least share of draws for each native rate; see data.Windows
    "coarse_min_points": 0,  # the least points of an extra coarse window of a short series
    "coarse_extra": 0.0,  # such windows added to each batch, as a share of batch_size
    "adam_betas": [0.9, 0.999],  # AdamW's two decay rates; these are its defaults
    "length_buckets": None,  # rising token lengths of the forward groups; see bucketed_predict
    "frame_gain": 1.0,  # g: a chosen window's scale times g ** U(-1, 1); see frame_draws
    "frame_gain_share": 0.0,  # share of windows whose frame changes; 0 leaves every frame plain
    "frame_shift": 0.0,  # d: a chosen window's loc moves by d * U(-1, 1) scales
}
# A bucket drops a window's leading tokens, which these read by position or as a group.
BUCKETS_REFUSE = (
    "future_query",
    "far_tiers",
    "shape_codes",
    "block_heads",
    "seed_bias",
    "stat_pool",
    "gap_embedding",
    "phase_bins",
    "shifted_block",
    "markov_path",
    "residual_target",
)
# frame_draws' own stream, apart from the windows' and the global RNG, so runs stay paired.
FRAME_STREAM = 15485863
# The frame is changed between causal_scale and the tokens; these change or read it elsewhere.
FRAME_REFUSE = ("far_tiers", "residual_target", "refine_masked_stats", "scale_features",
                "scale_halflife", "future_query")
# Member k of an ensemble trains with seed + k * MEMBER_SEED_STRIDE, so the members of runs with
# nearby seeds never coincide; member 0 keeps the configured seed.
MEMBER_SEED_STRIDE = 1000
ENSEMBLE_FIELDS = {"ensemble_size", "ensemble_vary"}
MODEL_VARY = {"context_length", "patch_length", "width", "layers", "heads", "dropout"}
TRAINING_VARY = {
    "data",
    "steps",
    "batch_size",
    "learning_rate",
    "source_power",
    "leave_out",
    "raw_pinball",
    "mase_weight",
}


def option(settings: dict, key: str):
    return settings.get(key, OPTIONAL[key])


def check_training(settings):
    for key in ("steps", "batch_size", "checkpoint_every"):
        if not isinstance(settings[key], int) or settings[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    for key in ("learning_rate", "max_seconds"):
        if not isinstance(settings[key], (int, float)) or not 0 < settings[key] < float("inf"):
            raise ValueError(f"{key} must be finite and positive")
    if not isinstance(settings["seed"], int) or settings["seed"] < 0:
        raise ValueError("seed must be a nonnegative integer")
    betas = option(settings, "adam_betas")
    if (
        not isinstance(betas, (list, tuple))
        or len(betas) != 2
        or any(type(beta) is not float or not 0 <= beta < 1 for beta in betas)
    ):
        raise ValueError("adam_betas must be two floats from 0 to under 1")
    power = settings["source_power"]
    if isinstance(power, bool) or not isinstance(power, (int, float)) or not 0 <= power <= 1:
        raise ValueError("source_power must be a number from 0 to 1")
    if not isinstance(settings["data"], str) or not settings["data"]:
        raise ValueError("data must name a dataset config or a local parquet folder")
    if option(settings, "schedule") not in {"cosine", "wsd"}:
        raise ValueError("schedule must be cosine or wsd")
    final, cooldown = option(settings, "final_lr_fraction"), option(settings, "cooldown_fraction")
    if isinstance(final, bool) or not isinstance(final, (int, float)) or not 0 <= final <= 1:
        raise ValueError("final_lr_fraction must be a number from 0 to 1")
    if (
        isinstance(cooldown, bool)
        or not isinstance(cooldown, (int, float))
        or not (0 < cooldown <= 1 - WARMUP_FRACTION)
    ):
        raise ValueError(f"cooldown_fraction must be in (0, {1 - WARMUP_FRACTION}]")
    sources = option(settings, "source_weights")
    if sources is not None and (
        not isinstance(sources, dict)
        or not sources
        or any(type(w) not in (int, float) or not 0 <= w < float("inf") for w in sources.values())
    ):
        raise ValueError("source_weights must map source names to nonnegative multipliers")
    views = option(settings, "views")
    if views is not None and (
        not isinstance(views, dict)
        or set(views) != {"share", "freqs", "factors"}
        or type(views["share"]) not in (int, float)
        or not 0 < views["share"] <= 1
        or not isinstance(views["freqs"], list)
        or not views["freqs"]
        or not all(isinstance(freq, str) for freq in views["freqs"])
        or not isinstance(views["factors"], list)
        or not views["factors"]
        or not all(type(factor) is int and factor >= 2 for factor in views["factors"])
    ):
        raise ValueError("views must set a share in (0, 1], a list of freqs, and factors of 2 up")
    for key in ("raw_pinball", "batch_quotas", "mase_weight"):
        if not isinstance(option(settings, key), bool):
            raise ValueError(f"{key} must be true or false")
    for key in ("fragment_weight", "log_loss", "regret_weight", "deep_cluster", "fredf"):
        weight = option(settings, key)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
            raise ValueError(f"{key} must be a nonnegative number")
    decay = option(settings, "weight_decay")
    if decay is not None and (type(decay) not in (int, float) or decay < 0):
        raise ValueError("weight_decay must be a nonnegative number, or None")
    blank, jitter = option(settings, "end_blank_rate"), option(settings, "end_blank_jitter")
    if isinstance(blank, bool) or not isinstance(blank, (int, float)) or not 0 <= blank <= 1:
        raise ValueError("end_blank_rate must be a number from 0 to 1")
    if type(jitter) is not int or jitter < 0:
        raise ValueError("end_blank_jitter must be a nonnegative integer")
    limit = option(settings, "channel_limit")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0:
        raise ValueError("channel_limit must be a nonnegative integer")
    least = option(settings, "min_observed")
    if isinstance(least, bool) or not isinstance(least, (int, float)) or not 0 <= least <= 1:
        raise ValueError("min_observed must be a number from 0 to 1")
    tilt = option(settings, "series_tilt")
    if isinstance(tilt, bool) or not isinstance(tilt, (int, float)):
        raise ValueError("series_tilt must be a number")
    items = option(settings, "drop_items")
    if items is not None and not (isinstance(items, str) and (REPOSITORY / items).is_file()):
        raise ValueError("drop_items must be a file in the repository")
    left = option(settings, "leave_out")
    if left is not None and (
        not isinstance(left, list) or not all(isinstance(s, str) and s for s in left)
    ):
        raise ValueError("leave_out must be a list of source names")
    bands = option(settings, "band_weights")
    if bands is not None and (
        not isinstance(bands, dict)
        or not set(bands) <= set(BANDS.values())
        or any(type(w) not in (int, float) or w < 0 for w in bands.values())
        or not sum(bands.values())
    ):
        raise ValueError(f"band_weights must map some of {sorted(set(BANDS.values()))} to weights")
    exclude = option(settings, "exclude_sources")
    if exclude is not None and (
        not isinstance(exclude, list) or not all(isinstance(s, str) and s for s in exclude)
    ):
        raise ValueError("exclude_sources must be a list of source names")
    share, longest = option(settings, "prefix_share"), option(settings, "prefix_patches")
    if isinstance(share, bool) or not isinstance(share, (int, float)) or not 0 <= share < 1:
        raise ValueError("prefix_share must be a number in [0, 1)")
    if not isinstance(longest, int) or isinstance(longest, bool) or longest <= 0:
        raise ValueError("prefix_patches must be a positive integer")
    if not isinstance(option(settings, "late_average"), bool):
        raise ValueError("late_average must be true or false")
    weight = option(settings, "coarse_weight")
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or weight < 0:
        raise ValueError("coarse_weight must be a nonnegative number")
    augment = option(settings, "augment")
    if augment is not None and (
        not isinstance(augment, dict)
        or not set(augment) <= set(AUGMENTATIONS)
        or any(type(p) not in (int, float) or not 0 <= p <= 1 for p in augment.values())
    ):
        raise ValueError(f"augment must map some of {AUGMENTATIONS} to probabilities")
    if augment and sum(augment.get(name, 0) for name in COARSENINGS) > 1:
        raise ValueError(f"{COARSENINGS} exclude each other, so may sum to at most 1")
    copies = option(settings, "aug_views")
    if type(copies) is not int or copies <= 0 or settings["batch_size"] % copies:
        raise ValueError("aug_views must be a positive integer that divides batch_size")
    ramp = option(settings, "aug_ramp")
    if ramp is not None and (
        not isinstance(ramp, list)
        or len(ramp) != 2
        or any(type(share) not in (int, float) or not 0 <= share <= 1 for share in ramp)
    ):
        raise ValueError("aug_ramp must be [first, last], each from 0 to 1")
    if (copies > 1 or ramp is not None) and not augment:
        raise ValueError("aug_views and aug_ramp act on augment, so they need it")
    window = option(settings, "drop_window")
    if window is not None and (
        not isinstance(window, list)
        or len(window) != 2
        or any(type(share) not in (int, float) for share in window)
        or not 0 <= window[0] < window[1] <= 1
    ):
        raise ValueError("drop_window must be [start, stop], rising within 0 to 1")
    refit = option(settings, "head_refit")
    if type(refit) is not int or not 0 <= refit < settings["steps"]:
        raise ValueError("head_refit must be an integer from 0 to under steps")
    if refit and option(settings, "late_average"):
        raise ValueError("head_refit fits the head to the last weights, not to their average")
    polyphase = option(settings, "polyphase_share")
    if type(polyphase) not in (int, float) or not 0 <= polyphase <= 1:
        raise ValueError("polyphase_share must be a number from 0 to 1")
    strides = option(settings, "polyphase_strides")
    if strides != "period" and (
        not isinstance(strides, list)
        or not strides
        or any(type(stride) is not int or stride < 2 for stride in strides)
    ):
        raise ValueError("polyphase_strides must be \"period\" or integers of at least 2")
    thin = option(settings, "polyphase_least")
    if type(thin) is not int or thin < 0:
        raise ValueError("polyphase_least must be a nonnegative integer")
    reach = option(settings, "polyphase_short_share"), option(settings, "polyphase_long_share")
    if (reach[0] is None) != (reach[1] is None):
        raise ValueError("polyphase_short_share and polyphase_long_share work together")
    if reach[0] is not None and any(type(s) not in (int, float) or not 0 <= s <= 1 for s in reach):
        raise ValueError("polyphase_short_share and polyphase_long_share must be from 0 to 1")
    joint = option(settings, "polyphase_joint_share")
    if type(joint) not in (int, float) or not 0 <= joint <= 1:
        raise ValueError("polyphase_joint_share must be a number from 0 to 1")
    floor = option(settings, "rate_floor")
    if type(floor) not in (int, float) or not 0 <= floor < 1:
        raise ValueError("rate_floor must be a number in [0, 1)")
    if floor and bands is not None:
        raise ValueError("rate_floor sets shares by rate, and band_weights sets them by band")
    least, extra = option(settings, "coarse_min_points"), option(settings, "coarse_extra")
    if type(least) is not int or least < 0:
        raise ValueError("coarse_min_points must be a nonnegative integer")
    if type(extra) not in (int, float) or not 0 <= extra < 1:
        raise ValueError("coarse_extra must be a number in [0, 1)")
    if bool(least) != bool(extra):
        raise ValueError("coarse_min_points and coarse_extra work together, so set both or neither")
    if extra and not (augment and augment.get("stride", 0) + augment.get("block_mean", 0)):
        raise ValueError("coarse_extra draws as augment's stride and block mean do, so needs one")
    if extra and not round(settings["batch_size"] * extra):
        raise ValueError("coarse_extra rounds to no windows at this batch_size")
    edges = option(settings, "length_buckets")
    if edges is not None and (
        not isinstance(edges, list)
        or not edges
        or any(type(edge) is not int or edge <= 0 for edge in edges)
        or edges != sorted(set(edges))
    ):
        raise ValueError("length_buckets must be rising positive integers")


def check_buckets(config: ModelConfig, settings: dict):
    """length_buckets promises the loss of mask_padded with no buckets, which needs these."""
    edges = option(settings, "length_buckets")
    if edges is None:
        return
    future = math.ceil(config.prediction_length / config.patch_length) * config.patch_length
    tokens = (config.context_length + future) // config.patch_length
    if edges[-1] != tokens:
        raise ValueError(f"length_buckets must end at the window's {tokens} tokens")
    if not config.mask_padded:
        raise ValueError("length_buckets needs mask_padded, so a row's padding doesn't count")
    if option(settings, "channel_limit") != 1:
        raise ValueError("length_buckets moves single windows, so needs channel_limit 1")
    if option(settings, "polyphase_joint_share"):
        raise ValueError("length_buckets moves single windows, so excludes polyphase_joint_share")
    if option(settings, "regret_weight") or option(settings, "fragment_weight"):
        raise ValueError("regret_weight and fragment_weight run their own passes, unbucketed")
    used = [field for field in BUCKETS_REFUSE if getattr(config, field)]
    if used:
        raise ValueError(f"length_buckets can't drop tokens that {used} read")


def load_config(path: Path):
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or set(config) != {"model", "training"}:
        raise ValueError("Config must contain exactly model and training sections")
    model = ModelConfig(**config["model"])
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
    optional = set(OPTIONAL) | ENSEMBLE_FIELDS
    if not required <= set(settings) <= required | optional:
        raise ValueError(f"Training fields must be {sorted(required)}, or {sorted(optional)}")
    check_training(settings)
    if model.future_query and option(settings, "prefix_share"):
        raise ValueError("prefix_share hides patches of the next-patch loss, not of future_query")
    next_patch_only = (
        "raw_pinball",
        "fragment_weight",
        "log_loss",
        "mase_weight",
        "regret_weight",
        "deep_cluster",
        "fredf",
        "end_blank_rate",
    )
    if model.future_query and any(option(settings, key) for key in next_patch_only):
        raise ValueError(f"{next_patch_only} belong to the next-patch loss")
    if option(settings, "drop_window") and not model.drop_path:
        raise ValueError("drop_window times model.drop_path, so it needs it")
    gain, share, shift = (option(settings, k) for k in ("frame_gain", "frame_gain_share",
                                                         "frame_shift"))
    if not (isinstance(gain, (int, float)) and gain >= 1 and 0 <= share <= 1 and shift >= 0):
        raise ValueError("frame_gain must be at least 1, frame_gain_share 0 to 1, frame_shift >= 0")
    if share and (gain > 1 or shift):
        refused = [key for key in FRAME_REFUSE if getattr(model, key)]
        refused += [key for key in ("length_buckets", "fragment_weight") if option(settings, key)]
        if refused:
            raise ValueError(f"frame_gain doesn't reach the frame used by {refused}")
    if option(settings, "deep_cluster") and model.patch_length < WORD_SEGMENTS:
        raise ValueError(f"deep_cluster needs patch_length of at least {WORD_SEGMENTS}")
    size = settings.get("ensemble_size", 1)
    if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
        raise ValueError("ensemble_size must be a positive integer")
    vary = settings.get("ensemble_vary", [])
    if not isinstance(vary, list) or not all(isinstance(v, dict) for v in vary):
        raise ValueError("ensemble_vary must be a list of per-member overrides")
    for overrides in vary:
        if not set(overrides) <= MODEL_VARY | TRAINING_VARY:
            raise ValueError(f"ensemble_vary may set only {sorted(MODEL_VARY | TRAINING_VARY)}")
    member_plans(model, settings)  # validates every member's settings
    return model, settings


def member_plans(model_config: ModelConfig, settings: dict):
    """Each ensemble member's model config and training settings. Member k applies
    ensemble_vary[k % len(ensemble_vary)]; without ensemble fields, one member is the config."""
    vary = settings.get("ensemble_vary") or [{}]
    plans = []
    for k in range(settings.get("ensemble_size", 1)):
        overrides = vary[k % len(vary)]
        member = {key: value for key, value in settings.items() if key not in ENSEMBLE_FIELDS}
        member.update({key: v for key, v in overrides.items() if key in TRAINING_VARY})
        member["seed"] = settings["seed"] + MEMBER_SEED_STRIDE * k
        check_training(member)
        config = replace(
            model_config, **{key: v for key, v in overrides.items() if key in MODEL_VARY}
        )
        check_buckets(config, member)
        if option(member, "polyphase_joint_share") and not config.polyphase_joint:
            raise ValueError("polyphase_joint_share trains what model.polyphase_joint forecasts")
        plans.append((config, member))
    return plans


def save_checkpoint(path, model, optimizer, settings, step, seconds, data):
    payload = {
        "format_version": 3,
        "model_config": config_fields(model.config),
        **(
            {"ensemble": [config_fields(member.config) for member in model.members]}
            if isinstance(model, Ensemble)
            else {}
        ),
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


def pinball_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    levels=QUANTILES,
    weights: torch.Tensor | None = None,
    log_floor: float = 0.0,
    log_scales: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mean pinball. weights are per [B,V] window and act as if normalized to mean 1 over what they
    cover. log_floor averages log(log_floor + window pinball) over windows instead. log_scales are
    per level, as in Kendall's multi-task loss: each divides its level's loss by its exponential
    and pays itself back, so a level's own optimum doesn't move."""
    observed = torch.isfinite(target)
    if prediction.shape != (*target.shape, len(levels)):
        raise ValueError("prediction and target shapes differ")
    if not observed.any():
        raise ValueError("Need at least one observed target")
    error = torch.where(observed, target, 0.0).unsqueeze(-1) - prediction
    quantiles = prediction.new_tensor(levels)
    loss = torch.maximum(quantiles * error, (quantiles - 1) * error)
    if weights is None and not log_floor and log_scales is None:
        return (loss * observed.unsqueeze(-1)).sum() / (observed.sum() * len(levels))
    if log_scales is not None:
        loss = loss * (-log_scales).exp()
    dims = tuple(range(2, target.ndim))
    if log_floor:
        count = observed.sum(dims)
        pinball = (loss.mean(-1) * observed).sum(dims) / count.clamp_min(1)
        mass = count > 0 if weights is None else (count > 0) * weights
        result = ((log_floor + pinball).log() * mass).sum() / mass.sum()
    else:
        mass = observed if weights is None else observed * weights[(...,) + (None,) * len(dims)]
        result = (loss * mass.unsqueeze(-1)).sum() / (mass.sum() * len(levels))
    return result if log_scales is None else result + log_scales.mean()


def window_pinball(prediction, target, levels) -> torch.Tensor:
    """Each [B,V] window's mean pinball over its observed points, 0 with none."""
    observed = torch.isfinite(target)
    error = torch.where(observed, target, 0.0).unsqueeze(-1) - prediction
    quantiles = prediction.new_tensor(levels)
    loss = torch.maximum(quantiles * error, (quantiles - 1) * error).mean(-1)
    return (loss * observed).sum((2, 3)) / observed.sum((2, 3)).clamp_min(1)


def mase_weights(context: torch.Tensor) -> torch.Tensor:
    """One over each [B,V] window's mean lag-1 error, as a share of its context's spread.

    Training's pinball is already in units of the running scale, so the denominator must be free
    of units too, or the clamp would only sort series by size. It reads the context alone. A window
    with no neighboring points keeps weight 1."""
    steps = context.double().diff(dim=-1)
    usable = steps.isfinite()
    error = torch.where(usable, steps, 0).abs().sum(-1) / usable.sum(-1).clamp_min(1)
    _, spread = causal_scale(context, context.shape[-1])
    weight = 1 / (error / spread[..., 0]).clamp(*MASE_CLAMP)
    return torch.where(usable.any(-1), weight, 1.0).float()


@torch.no_grad()
def validate(model, batches, device) -> float | None:
    if not batches:
        return None
    model.eval()
    patch, total, count = model.config.patch_length, 0.0, 0
    for history, target, series_ids in batches:
        history, target = history.to(device), target.to(device)
        if model.config.future_query:
            loss = future_query_loss(model, history, target, series_ids.to(device))
            observed = int(torch.isfinite(target).sum())
            total += float(loss) * observed
            count += observed
            continue
        future = math.ceil(target.shape[-1] / patch)
        values = F.pad(history, (0, future * patch), value=float("nan"))
        quantiles, loc, scale = model.next_patch(values, series_ids.to(device))
        first = (history.shape[-1] - model.config.far_length) // patch - 1
        window = slice(first, first + future)
        target = F.pad(target, (0, future * patch - target.shape[-1]), value=float("nan"))
        target = target.unflatten(-1, (future, patch)).double()
        scaled = torch.asinh((target - loc[..., window, None]) / scale[..., window, None]).float()
        observed = int(torch.isfinite(scaled).sum())
        total += float(pinball_loss(quantiles[:, :, window], scaled, model.levels)) * observed
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


def hide_far(model, far: torch.Tensor) -> torch.Tensor:
    """Hide spans of far tokens, as next_patch_loss hides spans of recent patches."""
    hidden = [
        contiguous_patch_mask((*far.shape[:2], tokens), far.device).repeat_interleave(
            rate * model.config.patch_length, -1
        )
        for rate, tokens in reversed(model.config.far_tiers)
    ]
    return far.masked_fill(torch.cat(hidden, -1), float("nan"))


def left_censor(model, hidden, observed, series_ids, share: float, longest: int):
    """Turn a share of windows into forecasts from a fragment, as a short series would be.

    A chosen window keeps 1 to `longest` patches just before the forecast origin and hides the rest
    of its input. Hidden patches keep their targets, so the true future supervises it, and it has
    the geometry of a short history at forecast time. Variates of a window share one draw. A window
    with nothing observed in its fragment stays as it was. Returns the choice, [B,V], and hidden.
    """
    origin = (model.config.context_length - model.config.far_length) // model.config.patch_length
    windows, device = series_ids.clamp_min(0), series_ids.device
    count = int(windows.max()) + 1
    kept = torch.randint(1, longest + 1, (count,), device=device)[windows]
    position = torch.arange(hidden.shape[-1], device=device)
    fragment = (position < origin) & (position >= origin - kept[..., None])
    has = (observed & fragment).any(-1).float().flatten()
    has = torch.zeros(count, device=device).index_add_(0, windows.flatten(), has) > 0
    chosen = (torch.rand(count, device=device) < share)[windows] & has[windows]
    return chosen, torch.where(chosen[..., None], ~fragment, hidden)


def end_blank(model, hidden, observed, series_ids, rate: float, jitter: int):
    """Give a share of windows the mask a forecast has: everything from the origin on is hidden and
    nothing before it, with the origin moved up to `jitter` patches either way. Variates of a window
    share one draw, and hidden patches keep their targets. A variate with nothing left before the
    blank stays as it was."""
    origin = (model.config.context_length - model.config.far_length) // model.config.patch_length
    windows, device = series_ids.clamp_min(0), series_ids.device
    count = int(windows.max()) + 1
    moved = torch.randint(-jitter, jitter + 1, (count,), device=device)[windows]
    position = torch.arange(hidden.shape[-1], device=device)
    tail = position >= (origin + moved)[..., None]
    chosen = (torch.rand(count, device=device) < rate)[windows] & (observed & ~tail).any(-1)
    return torch.where(chosen[..., None], tail, hidden)


def rv_loss(model, features, target):
    """Huber loss of the training-only head on each next patch's log variance in scaled space."""
    observed = target.isfinite()
    count = observed.sum(-1)
    clean = torch.where(observed, target, 0.0)
    centered = (clean - clean.sum(-1, keepdim=True) / count[..., None].clamp_min(1)) * observed
    usable = count >= 2
    if not usable.any():
        return features.new_zeros(())
    variance = centered.square().sum(-1) / (count - 1).clamp_min(1)
    guess = model.rv_head(features[:, :, :-1]).float()[..., 0]
    return F.huber_loss(guess[usable], (variance[usable] + RV_FLOOR).log())


def next_next_loss(model, features, raw, loc, scale, seen):
    """Pinball of the training-only head on the patch after next, in each position's scaled space.
    raw is [B,V,P,L] from the second patch on, so its second row is what position 0 reads."""
    target = torch.asinh((raw[:, :, 1:] - loc[..., :-2, None]) / scale[..., :-2, None]).float()
    target = target.masked_fill(~seen[..., :-2, None], float("nan"))
    if not target.isfinite().any():
        return features.new_zeros(())
    shape = model.config.patch_length, len(model.levels)
    guess = model.next_head(features[:, :, :-2]).float().unflatten(-1, shape)
    return pinball_loss(guess.sort(dim=-1).values, target, model.levels)


def deep_cluster_loss(features, raw, seen):
    """Deep clustering's affinity loss on a sample of tokens: two tokens' cosine goes toward 1 if
    their next patches rise and fall alike, by patch_words, and toward 0 if not."""
    words = patch_words(raw)
    usable = (seen & (words >= 0)).nonzero()
    if len(usable) < 2:
        return features.new_zeros(())
    index = usable[torch.randperm(len(usable), device=usable.device)[:DEEP_CLUSTER_TOKENS]]
    index = index.unbind(-1)
    tokens = F.normalize(features[index].float(), dim=-1)
    chosen = words[index]
    return (tokens @ tokens.T - (chosen[None] == chosen[:, None]).float()).square().mean()


def fredf_loss(quantiles, target, levels):
    """L1 between the spectra of the median and the target on fully observed patches. The ortho
    norm keeps a bin about the size of a point's error, so the weight compares with pinball."""
    full = target.isfinite().all(-1)
    if not full.any():
        return quantiles.new_zeros(())
    median = quantiles[..., levels.index(0.5)][full].float()
    with torch.autocast(target.device.type, enabled=False):
        gap = torch.fft.rfft(median, norm="ortho") - torch.fft.rfft(target[full], norm="ortho")
    return (gap.real.abs() + gap.imag.abs()).mean()


def block_loss(model, layers, target):
    """Mean pinball of the training-only heads on the next patch, one on each time block."""
    shape = model.config.patch_length, len(model.levels)
    losses = []
    for head, tokens in zip(model.block_heads, layers, strict=True):
        guess = head(model.norm(tokens[:, :, :-1])).float().unflatten(-1, shape)
        losses.append(pinball_loss(guess.sort(dim=-1).values, target, model.levels))
    return sum(losses) / len(losses)


def coarse_loss(model, features, sequence, loc, scale, seen):
    """Mean pinball of the coarse heads, each on the mean of the patches after each position, in
    that position's scaled space. A block past the end of the sequence is not supervised."""
    patch = model.config.patch_length
    points = sequence.isfinite()
    totals = torch.where(points, sequence, 0).double().unflatten(-1, (-1, patch)).sum(-1)
    counts = points.unflatten(-1, (-1, patch)).sum(-1)
    losses = []
    for head, block in zip(model.coarse_heads, model.config.coarse_blocks, strict=True):
        # Position i is scored on patches i + 1 to i + block.
        total = totals.unfold(-1, block, 1).sum(-1)[..., 1:]
        count = counts.unfold(-1, block, 1).sum(-1)[..., 1:]
        positions = total.shape[-1]
        mean = torch.where(count > 0, total / count.clamp_min(1), torch.nan)
        target = torch.asinh((mean - loc[..., :positions]) / scale[..., :positions]).float()
        target = target.masked_fill(~seen[..., :positions], float("nan"))
        if target.isfinite().any():
            prediction = head(features[:, :, :positions]).float().sort(dim=-1).values
            losses.append(pinball_loss(prediction, target))
    return sum(losses) / max(len(losses), 1)


def raw_pinball_loss(quantiles, target, loc, scale, sequence, levels, **terms):
    """Pinball in raw units over each window's mean absolute value, as the score counts it.

    quantiles and target are [B,V,P,L,...] and [B,V,P,L]; loc and scale are those positions'."""
    points = sequence.isfinite()
    size = torch.where(points, sequence, 0).double().abs().sum(-1) / points.sum(-1).clamp_min(1)
    size = size.clamp_min(MIN_SCALE)[..., None, None]
    raw = quantiles.double().clamp(-RAW_MAX_QUANTILE, RAW_MAX_QUANTILE).sinh()
    raw = raw * scale[..., None, None] + loc[..., None, None]
    return pinball_loss(raw / size[..., None], target / size, levels, **terms).float()


def fragment_loss(model, sequence, observed, series_ids, features, longest: int, rates=None):
    """InfoNCE pulling each variate's token at the forecast origin, read from a few patches, toward
    the same token read from the whole window; other variates in the batch are the negatives."""
    patch = model.config.patch_length
    origin = (model.config.context_length - model.config.far_length) // patch
    nothing = torch.zeros_like(observed)
    chosen, hidden = left_censor(model, nothing, observed, series_ids, 1.0, longest)
    inputs = sequence.masked_fill(hidden.repeat_interleave(patch, -1), float("nan"))
    if model.config.far_length:
        inputs = F.pad(inputs, (model.config.far_length, 0), value=float("nan"))
    _, short, *_ = model.predict(inputs, series_ids, rates=rates)
    usable = chosen & (series_ids >= 0) & observed[..., origin - 1]
    if int(usable.sum()) < 2:
        return features.new_zeros(())
    anchors = F.normalize(short[:, :, origin - 1][usable], dim=-1)
    parents = F.normalize(features[:, :, origin - 1][usable], dim=-1)
    logits = (anchors @ parents.T).float() / FRAGMENT_TEMPERATURE
    return F.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


def future_query_loss(model, history, target, series_ids):
    quantiles, loc, scale = model.future_quantiles(history, series_ids, target.shape[-1])
    target = torch.asinh((target.double() - loc[..., None]) / scale[..., None]).float()
    return pinball_loss(quantiles, target)


def bucketed_predict(model, inputs, series_ids, rates, edges: list):
    """What predict returns, at full size, from fewer tokens. Each window of a [B,V,T] batch joins
    the shortest group that holds its real tokens and loses the leading tokens past that, so with
    mask_padded the windows, masks and loss stay those of one full pass, to float rounding.
    Positions dropped read 0, with loc 0 and scale 1, and are never scored: nothing is seen there.
    """
    patch = model.config.patch_length
    batch, variates, length = inputs.shape
    tokens = length // patch
    flat, ids = inputs.flatten(0, 1), series_ids.flatten()[:, None]
    tags = None if rates is None else rates.flatten()[:, None]
    real = flat.isfinite().unflatten(-1, (tokens, patch)).any(-1)
    needed = (tokens - real.float().argmax(-1)) * real.any(-1)  # an empty slot needs nothing
    groups = torch.searchsorted(needed.new_tensor(edges), needed)
    periods = None
    if model.config.phase_tokens or model.config.rope_periods:
        # Once, on the whole batch: an FFT plan is cached by shape, and a group's size keeps moving.
        periods = detect_periods(flat[None, :, : model.config.context_length])[0]
    order, parts = [], []
    for group, edge in enumerate(edges):
        chosen = (groups == group).nonzero()[:, 0]
        if not len(chosen):
            continue
        lead = tokens - edge
        tag = None if tags is None else tags[chosen]
        cycles = None if periods is None else periods[chosen, None]
        quantiles, features, loc, scale = model.predict(
            flat[chosen, None, lead * patch :], ids[chosen], rates=tag, lead=lead, periods=cycles
        )
        order.append(chosen)
        parts.append(
            (
                F.pad(quantiles, (0, 0, 0, 0, lead, 0)),
                F.pad(features, (0, 0, lead, 0)),
                F.pad(loc, (lead, 0)),
                F.pad(scale, (lead, 0), value=1.0),
            )
        )
    back = torch.cat(order).argsort()
    return [
        torch.cat(part)[back].squeeze(1).unflatten(0, (batch, variates)) for part in zip(*parts)
    ]


def frame_draws(settings: dict, series_ids: torch.Tensor, seed: int, step: int):
    """frame_gain's [B,V] gains and shifts, one draw per window shared by its variates, or None.

    A frame_gain_share of windows get a gain g ** U(-1, 1), log-uniform from 1 / g to g, and a
    shift of frame_shift * U(-1, 1); the rest keep gain 1 and shift 0. Draws come from their own
    stream of (seed, step), so neither the windows nor the global RNG move, and a resume repeats
    them. Three draws per window are always made, so arms with other settings stay paired.
    """
    gain, share, shift = (option(settings, k) for k in ("frame_gain", "frame_gain_share",
                                                         "frame_shift"))
    if not share or (gain <= 1 and not shift):
        return None
    windows = series_ids.clamp_min(0)
    count = int(windows.max()) + 1
    rng = np.random.default_rng(np.random.SeedSequence([seed, step, FRAME_STREAM]))
    chosen, power, offset = torch.from_numpy(rng.random((3, count))).to(series_ids.device)
    chosen = chosen < share
    gains = torch.where(chosen, float(gain) ** (2 * power - 1), 1.0)
    shifts = torch.where(chosen, shift * (2 * offset - 1), 0.0)
    return gains[windows], shifts[windows]


def next_patch_loss(
    model,
    sequence: torch.Tensor,
    series_ids: torch.Tensor,
    settings: dict | None = None,
    trailing=None,
    rates: torch.Tensor | None = None,
    frame=None,
) -> torch.Tensor:
    """The training loss. trailing is regret_weight's slowly updated copy of the model, and rates
    are rate_embedding's stride per [B,V] window. frame is frame_draws' gains and shifts: input
    and target are both read in each window's changed frame, and windows weigh max(gain, 1)."""
    settings = settings or {}
    if model.config.future_query:
        context = model.config.context_length
        available = min(sequence.shape[-1] - context, model.config.future_query_max_horizon)
        horizons = sorted({min(length, available) for length in (12, 128, 256, 512, 720, 900)})
        horizon = horizons[int(torch.randint(len(horizons), ()))]
        if not torch.isfinite(sequence[..., context : context + horizon]).any():
            horizon = available
        return future_query_loss(
            model, sequence[..., :context], sequence[..., context : context + horizon], series_ids
        )
    patch, far = model.config.patch_length, model.config.far_length
    older, sequence = sequence[..., :far], sequence[..., far:]
    batch, variates, length = sequence.shape
    hidden = contiguous_patch_mask((batch, variates, length // patch), sequence.device)
    observed = torch.isfinite(sequence).unflatten(-1, (-1, patch)).any(-1)
    hidden &= observed.cumsum(-1) > MIN_CONTEXT_PATCHES
    rate = option(settings, "end_blank_rate")
    if rate:
        jitter = option(settings, "end_blank_jitter")
        hidden = end_blank(model, hidden, observed, series_ids, rate, jitter)
    share = option(settings, "prefix_share")
    if share:
        longest = option(settings, "prefix_patches")
        chosen, hidden = left_censor(model, hidden, observed, series_ids, share, longest)
        older = older.masked_fill(chosen[..., None], float("nan"))
    seen = (observed & ~hidden).cummax(-1).values
    inputs = sequence.masked_fill(hidden.repeat_interleave(patch, -1), float("nan"))
    if far:
        inputs = torch.cat([hide_far(model, older), inputs], dim=-1)
    aux = bool(model.config.shape_codes or model.config.block_heads)
    edges = option(settings, "length_buckets")
    if edges:
        extras = None
        quantiles, features, loc, scale = bucketed_predict(model, inputs, series_ids, rates, edges)
    else:
        quantiles, features, loc, scale, *extras = model.predict(
            inputs, series_ids, aux, rates, frame=frame
        )
    shape_loss, layers = extras or (None, None)
    raw = sequence.unflatten(-1, (-1, patch))[:, :, 1:].double()
    target = torch.asinh((raw - loc[..., :-1, None]) / scale[..., :-1, None]).float()
    seen = (observed & ~hidden).cummax(-1).values
    target = target.masked_fill(~seen[..., :-1, None], float("nan"))
    weights = None
    if option(settings, "mase_weight"):
        weights = mase_weights(sequence[..., : model.config.context_length - far])
    if trailing is not None:
        with torch.no_grad():
            behind = trailing.predict(inputs, series_ids, rates=rates, frame=frame)[0][:, :, :-1]
            gap = window_pinball(quantiles[:, :, :-1].detach(), target, model.levels)
            gap = gap - window_pinball(behind, target, model.levels)
            eta = option(settings, "regret_weight")
            regret = (eta * gap.clamp(-REGRET_MAX_GAP, REGRET_MAX_GAP)).exp()
        weights = regret if weights is None else weights * regret
    if frame is not None:  # pinball shrinks as the gain grows, in these units
        gained = frame[0].clamp_min(1).float()
        weights = gained if weights is None else weights * gained
    terms = {
        "weights": weights,
        "log_floor": option(settings, "log_loss"),
        "log_scales": model.level_log_scales if model.config.learned_level_weights else None,
    }
    if option(settings, "raw_pinball"):
        loss = raw_pinball_loss(
            quantiles[:, :, :-1],
            raw.masked_fill(~seen[..., :-1, None], float("nan")),
            loc[..., :-1],
            scale[..., :-1],
            sequence,
            model.levels,
            **terms,
        )
    else:
        loss = pinball_loss(quantiles[:, :, :-1], target, model.levels, **terms)
    if model.config.coarse_blocks:
        coarse = coarse_loss(model, features, sequence, loc, scale, seen)
        loss = loss + option(settings, "coarse_weight") * coarse
    fragment = option(settings, "fragment_weight")
    if fragment:
        longest = option(settings, "prefix_patches")
        extra = fragment_loss(model, sequence, observed, series_ids, features, longest, rates)
        loss = loss + fragment * extra
    if model.config.rv_aux:
        loss = loss + model.config.rv_aux * rv_loss(model, features, target)
    if model.config.next_next:
        extra = next_next_loss(model, features, raw, loc, scale, seen)
        loss = loss + model.config.next_next * extra
    cluster = option(settings, "deep_cluster")
    if cluster:
        loss = loss + cluster * deep_cluster_loss(features[:, :, :-1], raw, seen[..., :-1])
    frequency = option(settings, "fredf")
    if frequency:
        loss = loss + frequency * fredf_loss(quantiles[:, :, :-1], target, model.levels)
    if model.config.block_heads:
        loss = loss + model.config.block_heads * block_loss(model, layers, target)
    return loss if shape_loss is None else loss + shape_loss


def make_optimizer(model, settings):
    """AdamW, with weight_decay on kernels only when it is set."""
    decay = option(settings, "weight_decay")
    rate, betas = settings["learning_rate"], tuple(option(settings, "adam_betas"))
    if decay is None:
        return torch.optim.AdamW(model.parameters(), lr=rate, betas=betas)
    kernels = [p for p in model.parameters() if p.dim() >= 2]
    others = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": kernels, "weight_decay": decay}, {"params": others, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=rate, betas=betas)


def cooldown_start(steps: int, cooldown: float) -> int:
    warmup = max(1, int(WARMUP_FRACTION * steps))
    return max(warmup, steps - max(1, int(cooldown * steps)))


def learning_rate(
    step: int,
    steps: int,
    peak: float,
    schedule: str = "cosine",
    final: float = FINAL_LR_FRACTION,
    cooldown: float = 0.2,
) -> float:
    warmup = max(1, int(WARMUP_FRACTION * steps))
    if step < warmup:
        return peak * (step + 1) / warmup
    if schedule == "wsd":
        start = cooldown_start(steps, cooldown)
        if step < start:
            return peak
        progress = (step - start) / max(1, steps - start)
        return peak * (final + (1 - final) * (1 - math.sqrt(progress)))
    progress = (step - warmup) / max(1, steps - warmup)
    return peak * (final + (1 - final) * 0.5 * (1 + math.cos(math.pi * progress)))


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
    if data is not None:  # overrides every member's data
        settings["data"] = str(data)
        for overrides in settings.get("ensemble_vary", []):
            overrides.pop("data", None)
    if seed is not None:
        settings["seed"] = seed
    plans = member_plans(model_config, settings)
    ensemble = len(plans) > 1
    loaded = {}  # one load per dataset and split, however many members read it

    def dataset(name, split):
        if (name, split) not in loaded:
            loaded[name, split] = load(name, split)
        return loaded[name, split]

    names = list(dict.fromkeys(member["data"] for _, member in plans))
    identity = describe(names[0], dataset(names[0], "train"))
    if len(names) > 1:
        identity["mixed_with"] = [describe(name, dataset(name, "train")) for name in names[1:]]
    device = device_for(settings["device"])
    torch.manual_seed(settings["seed"])
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    member_config, member = plans[0]
    model = NanoTSFM(member_config).to(device)
    optimizer = make_optimizer(model, member)
    first_step, previous_seconds = 0, 0.0
    checkpoint = None
    if resume:
        if ensemble:
            raise ValueError("Ensembles cannot resume; train them from scratch")
        if option(settings, "late_average"):
            raise ValueError("late_average keeps its mean in memory; train it from scratch")
        if option(settings, "regret_weight"):
            raise ValueError("regret_weight keeps its copy in memory; train it from scratch")
        if option(settings, "head_refit"):
            raise ValueError("head_refit changes data partway through; train it from scratch")
        _, checkpoint = load_checkpoint(resume)
        # Through ModelConfig, so checkpoints saved before a field existed compare at its default.
        saved_config = asdict(ModelConfig(**checkpoint["model_config"]))
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
    write_json(output / "config.json", {"model": config_fields(model_config), "training": settings})
    record_environment(output)

    def member_data(config, member, first_step):
        patch, far = config.patch_length, config.far_length
        future = math.ceil(config.prediction_length / patch) * patch
        # Windows splits off one point; training joins it back into the full sequence.
        context, horizon = config.context_length - far + future - 1, 1
        if config.future_query:
            context, horizon = config.context_length, config.future_query_max_horizon
        batch_size = member["batch_size"]
        exclude = (option(member, "exclude_sources") or []) + (option(member, "leave_out") or [])
        # A misspelled name would leave its source in training without a sign.
        if exclude and not set(exclude) <= set(dataset(member["data"], "train")["source"]):
            raise ValueError(f"exclude_sources must all be in {member['data']}: {exclude}")
        items = option(member, "drop_items")
        drop = json.loads((REPOSITORY / items).read_text())["drop"] if items else ()
        windows = Windows(
            dataset(member["data"], "train"),
            context,
            horizon,
            member["steps"] * batch_size,
            member["seed"],
            member["source_power"],
            option(member, "band_weights"),
            exclude,
            far,
            option(member, "augment"),
            option(member, "source_weights"),
            option(member, "views"),
            drop,
            batch_size if option(member, "batch_quotas") else 0,
            option(member, "channel_limit"),
            option(member, "min_observed"),
            future,
            option(member, "series_tilt"),
            option(member, "aug_views"),
            option(member, "aug_ramp"),
            HEAD_REFIT_SHARE if option(member, "head_refit") else 0.0,
            polyphase_share=option(member, "polyphase_share"),
            polyphase_strides=option(member, "polyphase_strides"),
            polyphase_least=option(member, "polyphase_least"),
            polyphase_short_share=option(member, "polyphase_short_share"),
            polyphase_long_share=option(member, "polyphase_long_share"),
            polyphase_joint_share=option(member, "polyphase_joint_share"),
            rate_floor=option(member, "rate_floor"),
            coarse_min_points=option(member, "coarse_min_points"),
            rate_tags=bool(config.rate_embedding),
        )
        extra = round(batch_size * option(member, "coarse_extra"))

        def loader_of(chosen, size=batch_size):
            return DataLoader(
                chosen,
                batch_size=size,
                collate_fn=pack_windows,
                num_workers=min(8, os.cpu_count() or 1) if device.type == "cuda" else 0,
                pin_memory=device.type == "cuda",
                # A private generator keeps the global RNG, and so resumed runs, unchanged.
                generator=torch.Generator(),
            )

        refit = option(member, "head_refit")
        stop = len(windows) - refit * batch_size
        chosen = Subset(windows, range(first_step * batch_size, stop))
        loader = loader_of(Extras(chosen, batch_size, extra), batch_size + extra) if extra \
            else loader_of(chosen)

        def refit_stream():
            """The main windows, then plain draws of the held-out series: every source equal, with
            no augment. Built only when reached, so its setup counts on the training clock."""
            yield from loader
            held = Windows(
                dataset(member["data"], "train"),
                context,
                horizon,
                refit * batch_size,
                member["seed"],
                exclude=exclude,
                far=far,
                drop_items=drop,
                hold_out=HEAD_REFIT_SHARE,
                held_only=True,
            )
            yield from loader_of(held)

        # Fix validation sampling independently of the training seed.
        try:
            held_out = Windows(
                dataset(member["data"], "validation"),
                config.context_length - far,
                config.prediction_length,
                VALIDATION_WINDOWS,
                0,
                exclude=exclude,
                far=far,
            )
            samples = [held_out[i] for i in range(len(held_out))]
            batches = [
                pack_windows(samples[i : i + batch_size])
                for i in range(0, len(samples), batch_size)
            ]
        except ValueError:  # no validation split, or no series in it longer than the horizon
            batches = []
        return windows, refit_stream() if refit else loader, batches

    bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported(including_emulation=False)
    autocast = torch.autocast("cuda", torch.bfloat16, enabled=bf16)
    # One training clock for every member: `spent` holds finished members' seconds, and the clock
    # runs only while a member trains: from its first batch's arrival, so that worker startup and
    # the first read count as setup and data loading. Validation and saving are off it too.
    clock = {"spent": previous_seconds, "started": None, "paused": 0.0}
    step, last_loss, windows_seen, validation = first_step, None, 0, None
    trained_steps = 0  # steps of every member this run, kept or not
    finished = []  # kept members: (model, steps, seconds, validation loss)
    stopped_early = False
    windows = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def elapsed():
        """Training seconds so far; validation and checkpoint saving are off the clock."""
        if clock["started"] is None:
            return clock["spent"]
        return clock["spent"] + time.monotonic() - clock["started"] - clock["paused"]

    # The ensemble forecasts through the longest member's context, and each member reads its part.
    longest = max(config.context_length for config, _ in plans)
    joined = replace(model_config, context_length=longest)

    def kept(models):
        return Ensemble(models, joined) if ensemble else models[0]

    def save(seconds, models=None):
        """Save the current model, or the given kept members with their total steps."""
        steps = sum(s for _, s, _, _ in finished) if models and finished else step
        models = models or [model]
        path = output / "checkpoint.pt"
        save_checkpoint(path, kept(models), optimizer, settings, steps, seconds, identity)

    healthy = True
    nonfinite = torch.zeros((), dtype=torch.bool, device=device)

    def check_finite():
        nonlocal healthy
        if not math.isfinite(float(loss.detach())) or bool(nonfinite):
            healthy = False
            raise ValueError(f"Non-finite loss or gradient by step {step}")

    interrupted = True  # until the training loop ends on its own
    try:
        with (output / "training.jsonl").open("w") as log:
            for index, (member_config, member) in enumerate(plans):
                if index:
                    # Stop adding members once the next one would likely run past the cap.
                    estimate = max(t * member["steps"] / s for _, s, t, _ in finished)
                    if clock["spent"] + estimate > settings["max_seconds"]:
                        stopped_early = True
                        break
                    torch.manual_seed(member["seed"])
                    model = NanoTSFM(member_config).to(device)
                    optimizer = make_optimizer(model, member)
                    step, first_step, last_loss, validation = 0, 0, None, None
                windows, loader, batches = member_data(member_config, member, first_step)
                schedule = [
                    option(member, k)
                    for k in ("schedule", "final_lr_fraction", "cooldown_fraction")
                ]
                tag = {"member": index} if ensemble else {}
                average, averaged = None, 0  # late_average: running mean, and iterates in it
                average_from = member["steps"]
                if option(member, "late_average"):
                    cooldown = option(member, "cooldown_fraction")
                    average_from = cooldown_start(member["steps"], cooldown)
                refit_start = member["steps"] - option(member, "head_refit")
                drop_window = option(member, "drop_window")
                # The vocabulary's codes are frozen with the rest during a refit.
                last_revival = min((1 - SHAPE_SETTLE_FRACTION) * member["steps"], refit_start)
                member_start = clock["spent"]
                model.train()
                trailing = None
                if option(member, "regret_weight"):
                    trailing = copy.deepcopy(model).eval().requires_grad_(False)
                # Set on the GPU when a gradient is not finite; read only when the loss is logged
                # or saved.
                nonfinite = torch.zeros((), dtype=torch.bool, device=device)
                for history, target, series_ids, *tags in loader:
                    if clock["started"] is None:
                        clock["started"], clock["paused"] = time.monotonic(), 0.0
                    if elapsed() >= settings["max_seconds"]:
                        break
                    # Joined after the transfer: a join on the CPU would leave pinned memory.
                    history = history.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)
                    sequence = torch.cat([history, target], dim=-1)
                    series_ids = series_ids.to(device, non_blocking=True)
                    rates = tags[0].to(device, non_blocking=True) if tags else None
                    if step == refit_start:
                        for name, parameter in model.named_parameters():
                            frozen = not name.startswith(("head.", "head_residual."))
                            parameter.requires_grad_(not frozen)
                    if drop_window or refit_start < member["steps"]:
                        start, stop = drop_window or (0, 1)
                        live = start <= step / member["steps"] < stop and step < refit_start
                        model.set_drop_live(live)
                    rate = learning_rate(step, member["steps"], member["learning_rate"], *schedule)
                    for group in optimizer.param_groups:
                        group["lr"] = rate
                    optimizer.zero_grad(set_to_none=True)
                    frame = frame_draws(member, series_ids, member["seed"], step)
                    with autocast:
                        loss = next_patch_loss(
                            model, sequence, series_ids, member, trailing, rates, frame
                        )
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    nonfinite |= ~torch.isfinite(norm)
                    optimizer.step()
                    if trailing is not None:
                        with torch.no_grad():
                            for old, new in zip(
                                trailing.parameters(), model.parameters(), strict=True
                            ):
                                old.lerp_(new, 1 - REGRET_DECAY)
                    if member_config.shape_codes and step < last_revival:
                        model.vocabulary.revive()
                    step += 1
                    trained_steps += 1
                    windows_seen += member["batch_size"] + round(
                        member["batch_size"] * option(member, "coarse_extra")
                    )
                    if step > average_from:
                        averaged += 1
                        with torch.no_grad():
                            if average is None:
                                average = [p.detach().clone() for p in model.parameters()]
                            for mean, parameter in zip(average, model.parameters(), strict=True):
                                mean.lerp_(parameter, 1 / averaged)
                    if step % LOG_EVERY == 0 or step == member["steps"]:
                        check_finite()
                        last_loss = float(loss.detach())
                        record = {**tag, "step": step, "loss": last_loss}
                        record["seconds"] = round(elapsed(), 2)
                        log.write(json.dumps(record) + "\n")
                        log.flush()
                    if step % member["checkpoint_every"] == 0:
                        sync()  # finish queued training work before the clock pauses
                        check_finite()
                        seconds = elapsed()
                        pause = time.monotonic()
                        with autocast:
                            validation = validate(model, batches, device)
                        record = {**tag, "step": step, "validation_loss": validation}
                        log.write(json.dumps(record) + "\n")
                        log.flush()
                        if not finished:  # later ensemble members save once, when they finish
                            save(seconds)
                        clock["paused"] += time.monotonic() - pause
                if bool(nonfinite):
                    raise ValueError(f"Non-finite gradient by step {step}")
                if average is not None:
                    with torch.no_grad():
                        for mean, parameter in zip(average, model.parameters(), strict=True):
                            parameter.copy_(mean)
                sync()
                clock["spent"], clock["started"] = elapsed(), None
                if average is not None:  # what is kept is the mean, so report the mean
                    with autocast:
                        validation = validate(model, batches, device)
                    record = {**tag, "step": step, "averaged": averaged}
                    log.write(json.dumps({**record, "validation_loss": validation}) + "\n")
                    log.flush()
                # The first member is kept even if the cap cut it short, as a single model is;
                # later members count only if they finished.
                if not index or step == member["steps"]:
                    finished.append((model, step, clock["spent"] - member_start, validation))
                    if ensemble:
                        save(clock["spent"], [m for m, *_ in finished])
                if clock["spent"] >= settings["max_seconds"]:
                    stopped_early = index + 1 < len(plans)
                    break
        interrupted = False
    finally:
        sync()
        seconds = elapsed()
        # Keep completed steps on interruption, but never overwrite a good checkpoint with
        # weights from a non-finite update.
        healthy = healthy and not bool(nonfinite)
        models = [m for m, *_ in finished] or [model]
        if healthy:
            save(seconds, models)
        final = kept(models)
        if ensemble:
            steps_kept = sum(s for _, s, _, _ in finished) or step
            complete = len(finished) == len(plans) and all(
                s == plan[1]["steps"] for (_, s, _, _), plan in zip(finished, plans, strict=True)
            )
            losses = [v for *_, v in finished if v is not None]
            validation = sum(losses) / len(losses) if losses else validation
        else:
            steps_kept, complete = step, step == settings["steps"]
        status = "completed" if complete and not interrupted else "partial"
        write_json(
            output / "run.json",
            {
                "status": status if healthy else "failed",
                "steps": steps_kept,
                "steps_this_run": trained_steps if ensemble else step - first_step,
                "windows_this_run": windows_seen,
                "last_loss": last_loss,
                "validation_loss": validation,
                "elapsed_seconds": seconds,
                "steps_per_second": (trained_steps if ensemble else step - first_step)
                / max(seconds - previous_seconds, 1e-9),
                "device": str(device),
                "device_name": torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None,
                "parameters": sum(p.numel() for p in final.parameters()),
                "peak_cuda_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
                "data": identity,
                **({"band_shares": windows.band_shares()} if windows and windows.bands else {}),
                **(
                    {
                        "ensemble_size": len(plans),
                        "ensemble_members": len(finished),
                        "ensemble_stopped_early": stopped_early,
                        "member_steps": [s for _, s, _, _ in finished],
                        "member_seconds": [round(t, 2) for _, _, t, _ in finished],
                        "member_validation_loss": [v for *_, v in finished],
                    }
                    if ensemble
                    else {}
                ),
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
