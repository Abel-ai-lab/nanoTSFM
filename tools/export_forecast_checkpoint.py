"""Export the exact forecast weights from a resumable nanoTSFM checkpoint."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch


FORECAST_FIELDS = ("format_version", "model_config", "model")
PROVENANCE_FIELDS = ("step", "elapsed_seconds", "data")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export(checkpoint, expected_sha256, output):
    if checkpoint.resolve() == output.resolve() or output.exists():
        raise FileExistsError(output)
    if digest(checkpoint) != expected_sha256:
        raise ValueError("The source checkpoint does not match the supplied SHA-256")
    original = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    if original.get("format_version") != 3 or "optimizer" not in original:
        raise ValueError("Expected a resumable version 3 training checkpoint")
    retained = {key: original[key] for key in (*FORECAST_FIELDS, *PROVENANCE_FIELDS) if key in original}
    tensors = retained["model"]
    if not tensors or any(not isinstance(tensor, torch.Tensor) for tensor in tensors.values()):
        raise ValueError("Unexpected model state")
    with tempfile.TemporaryDirectory(prefix="forecast-export-", dir=output.parent) as directory:
        staged = Path(directory) / output.name
        torch.save(retained, staged)
        restored = torch.load(staged, weights_only=True, map_location="cpu", mmap=True)
        if restored.keys() != retained.keys() or restored["model"].keys() != tensors.keys():
            raise ValueError("Exported checkpoint structure changed")
        for key, tensor in tensors.items():
            actual = restored["model"][key]
            if not torch.equal(tensor, actual) or tensor.dtype != actual.dtype or tensor.stride() != actual.stride():
                raise ValueError(f"Export changed model tensor {key}")
        for key in retained.keys() - {"model"}:
            if restored[key] != retained[key]:
                raise ValueError(f"Export changed metadata {key}")
        os.link(staged, output)
    return {
        "source_sha256": expected_sha256,
        "forecast_sha256": digest(output),
        "source_bytes": checkpoint.stat().st_size,
        "forecast_bytes": output.stat().st_size,
        "model_tensors": len(tensors),
        "model_values": sum(tensor.numel() for tensor in tensors.values()),
        "exact_tensor_values": True,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    args = parser.parse_args()
    print(json.dumps(export(args.checkpoint, args.checkpoint_sha256, args.output), indent=2))
