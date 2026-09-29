"""Draw the record history from records/*/, or print the README's record table with --table."""

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
# GIFT-Eval leaderboard at gift-eval commit 9a014e9: zero-shot models without test leakage.
MILESTONES = [
    ("TimesFM-3, best on the leaderboard", 0.456),
    ("Toto-2.0-4m, the same design trained far longer", 0.524),
    ("TinyCast, 146K parameters", 0.545),
    ("Moirai-small", 0.650),
]


def load() -> list[tuple[str, dict, dict]]:
    out = []
    for folder in sorted((ROOT / "records").iterdir()):
        if folder.is_dir() and folder.name != "template":
            team = yaml.safe_load((folder / "README.md").read_text().split("---\n", 2)[1])
            out.append((folder.name, team, json.loads((folder / "result.json").read_text())))
    return sorted(out, key=lambda entry: -entry[2]["gift_eval"]["crps"])  # each record improves


def table(entries) -> str:
    rows = ["| # | GIFT-Eval CRPS | Description | Date | Record | Contributors |"]
    rows.append("| ---: | --- | --- | --- | --- | --- |")
    for number, (folder, team, result) in enumerate(entries, 1):
        score = result["gift_eval"]
        handles = [m["github"] for m in team["members"]]
        people = ", ".join(f"[@{h}](https://github.com/{h})" for h in handles)
        crps = f"{score['crps']:.4f} ± {score['crps_sd']:.4f}"
        date = folder.split("_", 1)[0]
        link = f"[{folder}](records/{folder}/)"
        rows.append(f"| {number} | {crps} | {team['description']} | {date} | {link} | {people} |")
    return "\n".join(rows)


def figure(entries, output: Path):
    import matplotlib.pyplot as plt

    numbers = list(range(1, len(entries) + 1))
    means = [r["gift_eval"]["crps"] for _, _, r in entries]
    spreads = [r["gift_eval"]["crps_sd"] for _, _, r in entries]
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    right = len(entries) + 0.6
    for label, score in MILESTONES:
        ax.axhline(score, color="gray", lw=1, ls=":", zorder=1)
        ax.text(right, score - 0.003, label, ha="right", va="bottom", color="gray", fontsize=8.5)
    ax.step(numbers + [right], means + means[-1:], where="post", color="C1", lw=2, zorder=2)
    ax.errorbar(numbers, means, yerr=spreads, fmt="o", color="C1", ms=7, capsize=3, zorder=3)
    for number, (_, team, _), mean in zip(numbers, entries, means, strict=True):
        ax.annotate(
            f"{team['team']}  {mean:.3f}",
            (number, mean),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9,
            color="C1",
            fontweight="bold",
        )
    ax.set_xlim(0.5, right)
    ax.set_ylim(max(means) + 0.03, 0.43)
    ax.set_xticks(numbers)
    ax.set_xlabel("Record")
    ax.set_ylabel("GIFT-Eval relative CRPS (better ↑)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white")
    return output


def main():
    entries = load()
    if sys.argv[1:] == ["--table"]:
        print(table(entries))
    else:
        print(figure(entries, Path(sys.argv[1] if len(sys.argv) > 1 else "records.png")))


if __name__ == "__main__":
    main()
