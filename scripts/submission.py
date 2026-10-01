"""Package, check and verify submissions.

python -m scripts.submission package records/NAME RUN_DIR...
python -m scripts.submission check records/NAME [--runs RUN_DIR...]
python -m scripts.submission verify records/NAME --output DIR [--runs RUN...]
python -m scripts.submission guard CHANGED_FILE...
"""

import argparse
import hashlib
import json
import math
import os
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import yaml

TIME_CAP_SECONDS = 3600
TIME_MARGIN_SECONDS = 10
MIN_RUNS = 3  # runs a submission reports, and runs the maintainers retrain
SEED_SD = 0.007  # GIFT-Eval spread between seeds of the baseline
Z = 2.33  # one-sided p < 0.01
ROOT = Path(__file__).resolve().parents[1]
RECORDS = "records"
FROZEN = "src/nanotsfm/evaluation.py"
FIXED = (FROZEN, "configs/gift-full.json", "scripts/submission.py", ".github/")
# What decides training; the fixed evaluation files do not.
TRAINING = (
    "src",
    "configs",
    "pyproject.toml",
    "uv.lock",
    f":!{FROZEN}",
    ":!configs/gift-full.json",
)
UPSTREAM = "https://github.com/Abel-ai-lab/nanoTSFM"
FILES = ("README.md", "result.json")
VERIFIED = "verified.json"  # written by verify, then committed by the maintainers
TEAM_FIELDS = ("team", "description", "members", "ai_disclosure")
MAX_MEMBERS = 2
MAX_FILE_BYTES = 2**20
MAX_LOG_POINTS = 1000


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True)


def front_matter(readme: Path) -> dict:
    text = readme.read_text()
    if not text.startswith("---\n"):
        raise ValueError(f"{readme} must start with the front matter from records/template")
    return yaml.safe_load(text.split("---\n", 2)[1]) or {}


def check_team(team: dict, readme: Path):
    for field in TEAM_FIELDS:
        if not team.get(field) or "YOUR_" in json.dumps(team[field]):
            raise ValueError(f"Fill in {field} in {readme}")
    members = team["members"]
    if (
        not isinstance(members, list)
        or not 1 <= len(members) <= MAX_MEMBERS
        or not all(isinstance(m, dict) and m.get("name") and m.get("github") for m in members)
        or len({m["github"].lower() for m in members}) != len(members)
    ):
        raise ValueError(f"List 1 to {MAX_MEMBERS} members, each with a name and a GitHub handle")


def check_result(result: dict):
    for key in ("commit", "config", "data", "gift_eval", "runs"):
        if not result.get(key):
            raise ValueError(f"result.json lacks {key}; package it with ./run.sh submit")
    if result.get("uncommitted_changes") is not False:
        raise ValueError("The runs had uncommitted changes; commit, push and train again")
    if result["config"]["training"]["max_seconds"] > TIME_CAP_SECONDS:
        raise ValueError(f"max_seconds exceeds the {TIME_CAP_SECONDS}-second training cap")
    if result["data"]["kind"] == "toy":
        raise ValueError("A model trained on the toy data is not eligible")
    runs = result["runs"]
    if len(runs) < MIN_RUNS or len({r["seed"] for r in runs}) != len(runs):
        raise ValueError(f"Report at least {MIN_RUNS} runs with different seeds")
    for run in runs:
        if run["training_seconds"] > TIME_CAP_SECONDS + TIME_MARGIN_SECONDS:
            raise ValueError(f"{run['run']} exceeds the {TIME_CAP_SECONDS}-second training cap")
        if not run.get("gift_eval"):
            raise ValueError(f"{run['run']} has no GIFT-Eval score")


def check_folder(folder: Path):
    names = {p.name for p in folder.iterdir() if not p.name.startswith(".")}
    missing, extra = sorted(set(FILES) - names), sorted(names - {*FILES, VERIFIED})
    if missing or extra:
        raise ValueError(
            f"{folder} must hold {', '.join(FILES)} and nothing else; missing {missing}, "
            f"extra {extra}"
        )


def check_training(commit: str):
    """Require the work tree to train exactly what the runs' commit trained."""
    exists = ["git", "-C", str(ROOT), "cat-file", "-e", f"{commit}^{{commit}}"]
    if subprocess.run(exists, stderr=subprocess.DEVNULL).returncode:
        raise ValueError(f"The runs' commit {commit} is not here; push it with your branch")
    diff = ["git", "-C", str(ROOT), "diff", "--quiet", commit, "--", *TRAINING]
    if subprocess.run(diff).returncode:
        raise ValueError(f"The training code differs from the runs' commit {commit[:7]}")


def summarize(scores: list[dict]) -> dict:
    crps = [s["geometric_relative_crps"] for s in scores]
    mase = [s["geometric_relative_mase"] for s in scores]
    return {
        "crps": statistics.mean(crps),
        "crps_sd": statistics.stdev(crps),
        "mase": statistics.mean(mase),
        "mase_sd": statistics.stdev(mase),
        "runs": len(scores),
    }


def compare(new: dict, old: dict) -> dict:
    """A record needs a mean ahead by Z seed-noise standard errors (one-sided p < 0.01)."""
    margin = Z * SEED_SD * math.sqrt(1 / new["runs"] + 1 / old["runs"])
    gap = old["crps"] - new["crps"]
    return {"gap": round(gap, 4), "margin": round(margin, 4), "beats": gap >= margin}


def records(tree: str | None = None) -> list[tuple[str, dict]]:
    """Merged records as (folder, GIFT-Eval summary), oldest first, from the work tree or a tree.

    A record's summary is that of the maintainers' retrains when its folder holds them.
    """
    if tree is None:
        names = sorted(p.name for p in (ROOT / RECORDS).iterdir() if p.is_dir())

        def read(name, file):
            path = ROOT / RECORDS / name / file
            return path.read_text() if path.exists() else None
    else:
        names = sorted(git("ls-tree", "--name-only", f"{tree}:{RECORDS}").split())
        files = set(git("ls-tree", "-r", "--name-only", tree, RECORDS).split())

        def read(name, file):
            path = f"{RECORDS}/{name}/{file}"
            return git("show", f"{tree}:{path}") if path in files else None

    return [
        (name, json.loads(read(name, VERIFIED) or read(name, "result.json"))["gift_eval"])
        for name in names
        if name != "template"
    ]


def current_record(folder: Path, tree: str | None = None) -> tuple[str, dict] | None:
    earlier = [(f, s) for f, s in records(tree) if f != folder.name]
    return min(earlier, key=lambda item: item[1]["crps"]) if earlier else None


def package(folder: Path, run_dirs: list[Path]) -> Path:
    if not (folder / "README.md").exists():
        raise FileNotFoundError(f"Copy records/template to {folder} and fill in README.md")
    if len(run_dirs) < MIN_RUNS:
        raise ValueError(f"Package at least {MIN_RUNS} runs with different seeds")
    runs, configs, environments = [], [], []
    for run_dir in run_dirs:
        if not (run_dir / "gift.json").exists():
            raise FileNotFoundError(f"Score {run_dir.name} on GIFT-Eval: ./run.sh eval")
        environment = json.loads((run_dir / "environment.json").read_text())
        if environment["uncommitted_changes"] is not False:
            raise ValueError(f"{run_dir.name} had uncommitted changes; commit and train again")
        config = json.loads((run_dir / "config.json").read_text())
        run = json.loads((run_dir / "run.json").read_text())
        lines = (run_dir / "training.jsonl").read_text().splitlines()
        entries = [json.loads(line) for line in lines if line.strip()]
        losses = [e for e in entries if "loss" in e]
        stride = max(1, -(-len(losses) // MAX_LOG_POINTS))
        kept = {id(e) for e in losses[::stride] + losses[-1:]}

        def summary(name, run_dir=run_dir):
            path = run_dir / name
            return json.loads(path.read_text())["summary"] if path.exists() else None

        runs.append(
            {
                "run": run_dir.name,
                "seed": config["training"]["seed"],
                "checkpoint_sha256": sha256(run_dir / "checkpoint.pt"),
                "device": run["device_name"],
                "parameters": run["parameters"],
                "data": run["data"],
                "steps": run["steps"],
                "training_seconds": run["elapsed_seconds"],
                "gift_eval": summary("gift.json"),
                "gep_val": summary("gep-val.json"),
                "gep_test": summary("gep-test.json"),
                "log": [e for e in entries if "loss" not in e or id(e) in kept],
            }
        )
        configs.append(config)
        environments.append(environment)
    if len({e["commit"] for e in environments}) != 1:
        raise ValueError("Train every run at the same commit")
    unseeded = [json.dumps({**c, "training": {**c["training"], "seed": None}}) for c in configs]
    if len(set(unseeded)) != 1 or len({json.dumps(r["data"]) for r in runs}) != 1:
        raise ValueError("Runs may differ only in their seed")
    shared = {k: runs[0][k] for k in ("parameters", "data")}
    for run in runs:
        del run["parameters"], run["data"]
    result = {
        "commit": environments[0]["commit"],
        "uncommitted_changes": False,
        "torch": environments[0]["torch"],
        "cuda": environments[0]["cuda"],
        **shared,
        "config": configs[0],
        "gift_eval": summarize([r["gift_eval"] for r in runs]),
        "runs": runs,
    }
    (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return folder


def interface_check(checkpoint: Path, result: dict, run: dict):
    from dataclasses import asdict

    import torch

    from nanotsfm.model import forecast, load_checkpoint

    if sha256(checkpoint) != run["checkpoint_sha256"]:
        raise ValueError(f"{checkpoint} is not the checkpoint of {run['run']}")
    model, saved = load_checkpoint(checkpoint)
    expected = {**result["config"]["training"], "seed": run["seed"]}
    if (
        asdict(model.config) != result["config"]["model"]
        or saved["training"] != expected
        or saved["data"] != result["data"]
        or saved["step"] != run["steps"]
        or abs(saved["elapsed_seconds"] - run["training_seconds"]) > 1e-6
    ):
        raise ValueError(f"The checkpoint of {run['run']} does not match result.json")
    torch.manual_seed(123)
    history = torch.randn(2, 3, model.config.context_length)
    history[..., :3] = float("nan")
    ids = torch.tensor([[0, 0, 1], [0, 1, 1]])
    horizon = model.config.prediction_length + 3
    prediction = forecast(model, history, horizon, ids)
    if (
        prediction.shape != (2, 3, horizon, 9)
        or not torch.isfinite(prediction).all()
        or (prediction.diff(dim=-1) < 0).any()
    ):
        raise ValueError("The model fails the forecast interface check")


def check(folder: Path, run_dirs=()) -> dict:
    check_folder(folder)
    team = front_matter(folder / "README.md")
    check_team(team, folder / "README.md")
    result = json.loads((folder / "result.json").read_text())
    check_result(result)
    by_name = {r["run"]: r for r in result["runs"]}
    for run_dir in run_dirs:
        if run_dir.name not in by_name:
            raise ValueError(f"{run_dir.name} is not in result.json")
        interface_check(run_dir / "checkpoint.pt", result, by_name[run_dir.name])
    summary = {
        "valid": True,
        "team": team["team"],
        "gift_eval": {k: round(v, 4) for k, v in result["gift_eval"].items()},
    }
    other = current_record(folder)
    if other:
        name, best = other
        summary["record"] = {name: round(best["crps"], 4)}
        summary.update(compare(result["gift_eval"], best))
    return summary


def gift_eval_checkout() -> Path:
    from nanotsfm.evaluation import UPSTREAM_REVISION

    upstream = ROOT / "external/gift-eval"
    if not upstream.exists():
        url = "https://github.com/SalesforceAIResearch/gift-eval.git"
        subprocess.run(["git", "clone", "-q", url, str(upstream)], check=True)
        subprocess.run(
            ["git", "-C", str(upstream), "checkout", "-q", UPSTREAM_REVISION], check=True
        )
    return upstream


def python(module: str, *args) -> None:
    """Run a module in its own process, so that participant code never loads into this one."""
    subprocess.run([sys.executable, "-m", module, *map(str, args)], cwd=ROOT, check=True)


def verify(folder: Path, output: Path, device="auto", official=UPSTREAM, only=()) -> dict:
    """Retrain the final configuration with new random seeds, then judge it on those runs.

    The seeds are drawn once and kept in `output`, and runs already verified there are skipped,
    so the work can be split across jobs.
    """
    git("fetch", "--quiet", official, "main")
    check(folder)
    result = json.loads((folder / "result.json").read_text())
    check_training(result["commit"])
    if result["data"]["kind"] == "custom":
        print("Custom data: build it with the command in the report before verifying.")
    output.mkdir(parents=True, exist_ok=True)
    try:  # new seeds, so that reported runs picked from many seeds cannot set a record
        with (output / "seeds.json").open("x") as stream:
            stream.write(json.dumps(random.SystemRandom().sample(range(2**31), MIN_RUNS)) + "\n")
    except FileExistsError:
        pass
    seeds = json.loads((output / "seeds.json").read_text())
    names = [f"run-{k}" for k in range(1, len(seeds) + 1)]
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    for name, seed in zip(names, seeds, strict=True):
        target = output / name
        if (only and name not in only) or (target / "verified.json").exists():
            continue
        config = json.loads(json.dumps(result["config"]))
        config["training"].update(device=device, seed=seed)
        target.mkdir(parents=True, exist_ok=True)
        (target / "config.yaml").write_text(yaml.safe_dump(config))
        started = time.monotonic()
        python("nanotsfm.train", "--config", target / "config.yaml", "--output", target / "run")
        wall = time.monotonic() - started
        python(
            "nanotsfm.evaluation",
            "gift",
            *("--checkpoint", target / "run" / "checkpoint.pt", "--output", target / "gift.json"),
            *("--upstream", gift_eval_checkout(), "--tasks", ROOT / "configs/gift-full.json"),
            *("--device", device, "--workers", min(16, cores or 1)),
        )
        run = json.loads((target / "run" / "run.json").read_text())
        if run["elapsed_seconds"] > TIME_CAP_SECONDS + TIME_MARGIN_SECONDS:
            raise ValueError(f"{name} trained for {run['elapsed_seconds']:.0f} s, over the cap")
        verified = {
            "seed": seed,
            "training_seconds": run["elapsed_seconds"],
            "wall_seconds": wall,  # the whole training process, setup included; for reference
            "device": run["device_name"],
            "gift_eval": json.loads((target / "gift.json").read_text())["summary"],
        }
        (target / "verified.json").write_text(json.dumps(verified) + "\n")
        score = verified["gift_eval"]["geometric_relative_crps"]
        print(f"{name}, seed {seed}: verified {score:.4f}")
    done = [output / name / "verified.json" for name in names]
    if not all(path.exists() for path in done):
        return {"verified": False, "remaining": [p.parent.name for p in done if not p.exists()]}
    runs = [json.loads(path.read_text()) for path in done]
    verified = {"gift_eval": summarize([r["gift_eval"] for r in runs]), "runs": runs}
    (folder / VERIFIED).write_text(json.dumps(verified, indent=2) + "\n")
    # Reported runs ahead of the retrains by more than run noise suggest picked seeds.
    reproduced = not compare(result["gift_eval"], verified["gift_eval"])["beats"]
    verdict = {
        "verified": True,
        "gift_eval": verified["gift_eval"],
        "reported": {"crps": round(result["gift_eval"]["crps"], 4), "reproduced": reproduced},
        "written": str(folder / VERIFIED),
    }
    other = current_record(folder, tree="FETCH_HEAD")
    if not other:
        return verdict
    name, best = other
    verdict.update(record=name, **compare(verified["gift_eval"], best))
    return verdict


def guard(changed: list[str]) -> str:
    for name in changed:
        path = ROOT / name
        if path.is_file() and path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"{name} is over 1 MB; keep runs, checkpoints and data out of git")
        if name.startswith(FIXED):
            raise ValueError(f"{name} is fixed; change it through an issue")
    folders = {Path(n).parts[1] for n in changed if n.startswith(f"{RECORDS}/")}
    folders.discard("template")
    if not folders:
        return "No record folder; a maintainer reviews this pull request."
    if len(folders) > 1:
        raise ValueError(f"One record per pull request; this one touches {sorted(folders)}")
    (name,) = folders
    folder = ROOT / RECORDS / name
    check_folder(folder)
    check_team(front_matter(folder / "README.md"), folder / "README.md")
    result = json.loads((folder / "result.json").read_text())
    check_result(result)
    check_training(result["commit"])  # after the merge, main trains what the runs trained
    other = current_record(folder)
    if not other:
        return f"{RECORDS}/{name}/ is well formed; a maintainer verifies it by retraining."
    verdict = compare(result["gift_eval"], other[1])
    claim = "claims a record" if verdict["beats"] else "does not beat the record"
    return (
        f"{RECORDS}/{name}/ is well formed and {claim} (gap {verdict['gap']}, needed "
        f"{verdict['margin']}); a maintainer verifies it by retraining."
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("package", help="Write result.json from three or more runs")
    pack.add_argument("folder", type=Path, help="Record folder, such as records/2026-10-01_name")
    pack.add_argument("runs", type=Path, nargs="+", help="Run folders, such as runs/final-s7")
    inspect = sub.add_parser("check", help="Check a record folder")
    inspect.add_argument("folder", type=Path)
    inspect.add_argument("--runs", type=Path, nargs="*", default=(), help="Run folders to check")
    retrain = sub.add_parser("verify", help="Retrain a submission with new seeds (maintainers)")
    retrain.add_argument("folder", type=Path)
    retrain.add_argument("--output", type=Path, required=True, help="Folder for the retrains")
    retrain.add_argument("--device", default="auto")
    retrain.add_argument("--official", default=UPSTREAM, help="The official repository")
    retrain.add_argument(
        "--runs", nargs="*", default=(), help="Retrain only these of run-1, run-2 and run-3"
    )
    screen = sub.add_parser("guard", help="Check a pull request's changed files")
    screen.add_argument("changed", nargs="*")
    args = parser.parse_args()
    try:
        if args.command == "package":
            print(package(args.folder, args.runs))
        elif args.command == "check":
            print(json.dumps(check(args.folder, args.runs), indent=2))
        elif args.command == "verify":
            verdict = verify(args.folder, args.output, args.device, args.official, args.runs)
            print(json.dumps(verdict, indent=2))
        else:
            print(guard(args.changed))
    except (ValueError, FileNotFoundError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
