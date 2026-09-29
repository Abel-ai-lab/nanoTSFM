import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# GIFT-Eval leaderboard at gift-eval commit 9a014e9: zero-shot models without test leakage.
# Parameters from each model's Hugging Face safetensors metadata; TiRex from its paper.
MODELS = [
    ("TimesFM-3", 330.7e6, 0.456, (0, 11, "center")),
    ("TimesFM-2.5", 231.3e6, 0.490, (0, -13, "center")),
    ("Chronos-2", 119.5e6, 0.485, (0, 11, "center")),
    ("TiRex", 35e6, 0.488, (0, 11, "center")),
    ("Toto-2.0-22m", 21.9e6, 0.496, (0, -13, "center")),
    ("Moirai-small", 13.8e6, 0.650, (9, 0, "left")),
    ("FlowState", 9.1e6, 0.502, (0, 11, "center")),
    ("Toto-2.0-4m", 4.1e6, 0.524, (-9, 0, "right")),
    ("TinyCast", 146.5e3, 0.545, (0, 11, "center")),
]
REFERENCES = [("Seasonal Naive", 1.000), ("AutoARIMA", 0.912)]
MODEL, SUBMISSION, REFERENCE = "C0", "C1", "gray"


def submissions(path: Path) -> list[tuple[str, int, float]]:
    with path.open(newline="") as stream:
        return [
            (r["team"], int(r["parameters"]), float(r["gift_crps"])) for r in csv.DictReader(stream)
        ]


def frontier(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    best, kept = float("inf"), []
    for params, score in sorted(points):
        if score < best:
            best = score
            kept.append((params, score))
    return kept


def main():
    output = Path(sys.argv[1] if len(sys.argv) > 1 else "docs/leaderboard.png")
    fig, ax = plt.subplots(figsize=(8, 4.6), dpi=200)
    entries = submissions(Path("docs/leaderboard.csv"))
    for label, y in REFERENCES:
        ax.axhline(y, color=REFERENCE, lw=1, ls=":", zorder=1)
        ax.text(4.2e8, y - 0.008, label, ha="right", va="bottom", color=REFERENCE, fontsize=9)
    edge = frontier([(p, s) for _, p, s, _ in MODELS] + [(p, s) for _, p, s in entries])
    ax.step(*zip(*edge, strict=True), where="post", color="gray", ls="--", lw=1.2, zorder=2)
    ax.annotate(
        "Pareto frontier",
        edge[0],
        xytext=(60, -8),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8.5,
        color="gray",
        style="italic",
    )
    for name, params, score, (dx, dy, ha) in MODELS:
        ax.scatter(params, score, s=46, color=MODEL, zorder=3)
        ax.annotate(
            name,
            (params, score),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va="center",
            fontsize=9,
        )
    for index, (team, params, score) in enumerate(entries):
        ax.scatter(params, score, s=110, color=SUBMISSION, edgecolor="white", lw=1.5, zorder=4)
        ax.annotate(
            f"{team}  {score:.3f}",
            (params, score),
            xytext=(10, 12 + 14 * (index % 3)),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=9.5,
            color=SUBMISSION,
            fontweight="bold",
        )
    ax.set_xscale("log")
    ax.set_xlim(8e4, 4.5e8)
    ax.set_ylim(1.04, 0.40)
    ax.set_xticks([1e5, 1e6, 1e7, 1e8], ["100K", "1M", "10M", "100M"])
    ax.set_xlabel("Parameters")
    ax.set_ylabel("GIFT-Eval relative CRPS (better ↑)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, facecolor="white")
    print(output)


if __name__ == "__main__":
    main()
