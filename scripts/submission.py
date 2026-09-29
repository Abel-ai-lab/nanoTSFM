"""Package and verify submissions.

python -m scripts.submission package runs/RUN submissions/TEAM [--base COMMIT]
python -m scripts.submission check submissions/TEAM [--checkpoint PATH]
python -m scripts.submission verify submissions/TEAM --output DIR
python -m scripts.submission guard CHANGED_FILE...
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

TIME_CAP_SECONDS = 3600
TIME_MARGIN_SECONDS = 10
TOLERANCE = 0.01
ROOT = Path(__file__).resolve().parents[1]
FROZEN = "src/nanotsfm/evaluation.py"
UPSTREAM = "https://github.com/Abel-ai-lab/nanoTSFM"
FILES = ("README.md", "result.json", "changes.diff")
DIFF_PATHS = ("src", "configs", "scripts", "run.sh", "pyproject.toml")
TEAM_FIELDS = ("team", "members", "repository", "ai_disclosure")
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
        raise ValueError(f"{readme} must start with the front matter from submissions/template")
    return yaml.safe_load(text.split("---\n", 2)[1]) or {}


def check_team(team: dict, readme: Path):
    for field in TEAM_FIELDS:
        if not team.get(field) or "YOUR_" in json.dumps(team[field]):
            raise ValueError(f"Fill in {field} in {readme}")
    members = team["members"]
    if (
        not isinstance(members, list)
        or not 1 <= len(members) <= MAX_MEMBERS
        or not all(isinstance(m, dict) and m.get("name") and m.get("email") for m in members)
        or len({m["email"].lower() for m in members}) != len(members)
    ):
        raise ValueError(f"List 1 to {MAX_MEMBERS} members, each with a name and a unique email")


def check_result(result: dict):
    if result.get("uncommitted_changes") is not False:
        raise ValueError("The run had uncommitted changes; commit, push and train again")
    settings = result["config"]["training"]
    if settings["max_seconds"] > TIME_CAP_SECONDS:
        raise ValueError(f"max_seconds exceeds the {TIME_CAP_SECONDS}-second training cap")
    if result["training_seconds"] > TIME_CAP_SECONDS + TIME_MARGIN_SECONDS:
        raise ValueError(f"The run exceeds the {TIME_CAP_SECONDS}-second training cap")
    if result["data"]["kind"] == "toy":
        raise ValueError("A model trained on the toy data is not eligible")
    if not result.get("gift_eval"):
        raise ValueError("result.json has no GIFT-Eval score")


def check_folder(folder: Path):
    names = {p.name for p in folder.iterdir() if not p.name.startswith(".")}
    missing, extra = sorted(set(FILES) - names), sorted(names - set(FILES))
    if missing or extra:
        raise ValueError(
            f"{folder} must hold exactly {', '.join(FILES)}; missing {missing}, extra {extra}"
        )


def frozen_sha256(base: str) -> str:
    try:
        return hashlib.sha256(
            subprocess.check_output(["git", "-C", str(ROOT), "show", f"{base}:{FROZEN}"])
        ).hexdigest()
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Base commit {base} is not here; fetch {UPSTREAM} main") from error


def package(run_dir: Path, folder: Path, base: str | None = None) -> Path:
    if not (folder / "README.md").exists():
        raise FileNotFoundError(f"Copy submissions/template to {folder} and fill in README.md")
    if not (run_dir / "gift.json").exists():
        raise FileNotFoundError(f"Score the run on GIFT-Eval first: ./run.sh eval {run_dir.name}")
    run = json.loads((run_dir / "run.json").read_text())
    environment = json.loads((run_dir / "environment.json").read_text())
    if environment["uncommitted_changes"] is not False:
        raise ValueError("The run had uncommitted changes; commit, push and train again")
    commit = environment["commit"]
    if base is None:
        try:
            git("fetch", "--quiet", UPSTREAM, "main")
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Cannot fetch {UPSTREAM}; pass --base COMMIT") from error
        base = git("merge-base", commit, "FETCH_HEAD").strip()
    base = git("rev-parse", base).strip()
    (folder / "changes.diff").write_text(git("diff", base, commit, "--", *DIFF_PATHS))

    def summary(name):
        path = run_dir / name
        return json.loads(path.read_text())["summary"] if path.exists() else None

    lines = (run_dir / "training.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    losses = [r for r in records if "loss" in r]
    stride = max(1, -(-len(losses) // MAX_LOG_POINTS))
    kept = {id(r) for r in losses[::stride] + losses[-1:]}
    log = [r for r in records if "loss" not in r or id(r) in kept]
    result = {
        "run": run_dir.name,
        "commit": commit,
        "base": base,
        "uncommitted_changes": environment["uncommitted_changes"],
        "evaluation_sha256": sha256(run_dir / "code" / "evaluation.py"),
        "checkpoint_sha256": sha256(run_dir / "checkpoint.pt"),
        "torch": environment["torch"],
        "cuda": environment["cuda"],
        "device": run["device_name"],
        "parameters": run["parameters"],
        "steps": run["steps"],
        "training_seconds": run["elapsed_seconds"],
        "data": run["data"],
        "config": json.loads((run_dir / "config.json").read_text()),
        "gift_eval": summary("gift.json"),
        "gep_val": summary("gep-val.json"),
        "gep_test": summary("gep-test.json"),
        "log": log,
    }
    (folder / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return folder


def check(folder: Path, checkpoint=None):
    check_folder(folder)
    team = front_matter(folder / "README.md")
    check_team(team, folder / "README.md")
    result = json.loads((folder / "result.json").read_text())
    check_result(result)
    if result["evaluation_sha256"] != frozen_sha256(result["base"]):
        raise ValueError("The run changed evaluation.py, which must stay as supplied")
    if checkpoint:
        from dataclasses import asdict

        import torch

        from nanotsfm.model import forecast, load_checkpoint

        if sha256(checkpoint) != result["checkpoint_sha256"]:
            raise ValueError("The checkpoint is not the one result.json records")
        model, saved = load_checkpoint(checkpoint)
        if (
            asdict(model.config) != result["config"]["model"]
            or saved["training"] != result["config"]["training"]
            or saved["data"] != result["data"]
            or saved["step"] != result["steps"]
            or abs(saved["elapsed_seconds"] - result["training_seconds"]) > 1e-6
        ):
            raise ValueError("The checkpoint does not match result.json")
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
    return {
        "valid": True,
        "team": team["team"],
        "members": [m["name"] for m in team["members"]],
        "run": result["run"],
        "training_seconds": round(result["training_seconds"]),
        "gift_eval": result["gift_eval"],
    }


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


def verify(folder: Path, output: Path, device: str = "auto", official: str = UPSTREAM) -> dict:
    import datetime

    from nanotsfm.evaluation import run as evaluate_gift
    from nanotsfm.train import train

    git("fetch", "--quiet", official, "main")
    check(folder)
    result = json.loads((folder / "result.json").read_text())
    if subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", result["base"], "FETCH_HEAD"]
    ).returncode:
        raise ValueError(f"Base {result['base']} is not a commit of {official} main")
    head = git("rev-parse", "HEAD").strip()
    if head != result["commit"]:
        raise ValueError(f"Check out the submission's commit {result['commit']}, not {head}")
    if result["data"]["kind"] == "custom":
        print("Custom data: build it with the command in the team's README before verifying.")
    output.mkdir(parents=True, exist_ok=False)
    config = result["config"]
    config["training"]["device"] = device
    (output / "config.yaml").write_text(yaml.safe_dump(config))
    checkpoint = train(output / "config.yaml", output / "run")
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    workers = min(16, cores or 1)
    scored = evaluate_gift(
        checkpoint,
        gift_eval_checkout(),
        ROOT / "configs/gift-full.json",
        output / "gift.json",
        device=device,
        workers=workers,
    )
    verified = json.loads(scored.read_text())["summary"]
    run = json.loads((output / "run" / "run.json").read_text())
    reported = result["gift_eval"]["geometric_relative_crps"]
    row = {
        "team": front_matter(folder / "README.md")["team"],
        "gift_crps": round(verified["geometric_relative_crps"], 4),
        "gift_mase": round(verified["geometric_relative_mase"], 4),
        "parameters": run["parameters"],
        "training_seconds": round(run["elapsed_seconds"]),
        "commit": result["commit"],
        "verified": datetime.date.today().isoformat(),
    }
    (output / "verified.json").write_text(json.dumps(row, indent=2) + "\n")
    if run["elapsed_seconds"] > TIME_CAP_SECONDS + TIME_MARGIN_SECONDS:
        raise ValueError(f"The retrain took {run['elapsed_seconds']:.0f} s, over the cap")
    if abs(verified["geometric_relative_crps"] - reported) > TOLERANCE:
        raise ValueError(
            f"Retrained GIFT-Eval {row['gift_crps']} differs from the reported {reported:.4f}"
        )
    return row


def guard(changed: list[str]) -> str:
    for name in changed:
        path = ROOT / name
        if path.is_file() and path.stat().st_size > MAX_FILE_BYTES:
            raise ValueError(f"{name} is over 1 MB; keep runs, checkpoints and data out of git")
    folders = {Path(n).parts[1] for n in changed if n.startswith("submissions/")}
    folders.discard("template")
    if not folders:
        return "Not a submission; a maintainer reviews it."
    if len(folders) > 1:
        raise ValueError(f"One submission per pull request; this one touches {sorted(folders)}")
    (team,) = folders
    outside = [n for n in changed if not n.startswith(f"submissions/{team}/")]
    if outside:
        raise ValueError(f"A submission pull request changes only submissions/{team}/: {outside}")
    folder = ROOT / "submissions" / team
    check_folder(folder)
    check_team(front_matter(folder / "README.md"), folder / "README.md")
    result = json.loads((folder / "result.json").read_text())
    check_result(result)
    for key in ("commit", "base", "checkpoint_sha256", "parameters"):
        if not result.get(key):
            raise ValueError(f"result.json lacks {key}; package it with ./run.sh submit")
    return f"Submission submissions/{team}/ is well formed; a maintainer verifies it by retraining."


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    pack = sub.add_parser("package", help="Write result.json and changes.diff for one run")
    pack.add_argument("run", type=Path, help="Run folder, such as runs/final")
    pack.add_argument("folder", type=Path, help="Submission folder, such as submissions/my-team")
    pack.add_argument("--base", help="The nanoTSFM commit you started from; default: merge-base")
    inspect = sub.add_parser("check", help="Check a submission folder")
    inspect.add_argument("folder", type=Path)
    inspect.add_argument("--checkpoint", type=Path, help="The run's checkpoint.pt")
    retrain = sub.add_parser("verify", help="Retrain and score a submission (maintainers)")
    retrain.add_argument("folder", type=Path)
    retrain.add_argument("--output", type=Path, required=True, help="A new folder for the retrain")
    retrain.add_argument("--device", default="auto")
    retrain.add_argument("--official", default=UPSTREAM, help="The official repository to check")
    screen = sub.add_parser("guard", help="Check a pull request's changed files")
    screen.add_argument("changed", nargs="*")
    args = parser.parse_args()
    try:
        if args.command == "package":
            print(package(args.run, args.folder, args.base))
        elif args.command == "check":
            print(json.dumps(check(args.folder, args.checkpoint), indent=2))
        elif args.command == "verify":
            row = verify(args.folder, args.output, args.device, args.official)
            print("Verified; add this row to docs/leaderboard.csv:")
            print(",".join(str(value) for value in row.values()))
        else:
            print(guard(args.changed))
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        sys.exit(f"error: {error}")


if __name__ == "__main__":
    main()
