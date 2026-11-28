"""Use the fixed, one-worker GIFT predictor with locked history on an archived checkpoint."""

import argparse
from functools import partial
from pathlib import Path
from unittest.mock import patch

import torch

from nanotsfm import evaluation
from nanotsfm.data import sha256, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for field in ("checkpoint", "upstream", "tasks", "data-root", "output", "hardware-receipt"):
        p.add_argument(f"--{field}", required=True, type=Path)
    p.add_argument("--checkpoint-sha256", required=True)
    p.add_argument("--device", choices=("cuda",), default="cuda")
    a = p.parse_args()
    if sha256(a.checkpoint) != a.checkpoint_sha256:
        raise ValueError("Unexpected checkpoint bytes")
    device = torch.device(a.device)
    props = torch.cuda.get_device_properties(device)
    if "A100" not in props.name or props.total_memory < 79 * 2**30:
        raise ValueError("The forecast must use an A100 80 GB")
    torch.cuda.reset_peak_memory_stats(device)
    loader = partial(evaluation.load_checkpoint, locked_history=True)
    with patch.object(evaluation, "load_checkpoint", loader):
        evaluation.run(a.checkpoint, a.upstream, a.tasks, a.output, a.device, 1, a.data_root)
    torch.cuda.synchronize(device)
    write_json(a.hardware_receipt, {"checkpoint_sha256": a.checkpoint_sha256,
                                    "device_name": props.name, "device_total_bytes": props.total_memory,
                                    "peak_cuda_bytes": torch.cuda.max_memory_allocated(device)})


if __name__ == "__main__":
    main()
