"""Build the record page from records/*/, or print the README's record table.

python scripts/records.py DIR      # DIR/index.html, DIR/records.svg and DIR/scatter.svg
python scripts/records.py --table
"""

import html
import json
import math
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/Abel-ai-lab/nanoTSFM"
# GIFT-Eval leaderboard at gift-eval commit 9a014e9: zero-shot models without test leakage.
# Parameter counts are the safetensors totals on each model's Hugging Face page.
MILESTONES = {
    "TimesFM-3": {"mase": 0.667, "crps": 0.456, "parameters": 330_710_976},
    "Toto-2.0-4m": {"mase": 0.757, "crps": 0.524, "parameters": 4_144_448},
    "TinyCast": {"mase": 0.774, "crps": 0.545, "parameters": 146_505},
    "Moirai-small": {"mase": 0.946, "crps": 0.650, "parameters": 13_827_528},
}
# (text, link, Simple Icons logo and its color)
LINKS = [
    ("Code", REPO, "github/F4F5F1"),
    ("Data", "https://huggingface.co/datasets/abel-lab/nanoTSFM-pretrain", "huggingface/FFD21E"),
    ("Rules", f"{REPO}/blob/main/docs/rules.md", None),
    ("Submit a result", f"{REPO}/blob/main/docs/submission.md", None),
]
ABOUT = """nanoTSFM is an open benchmark for training small time-series foundation models on a fixed
budget. A run trains a 3.3M-parameter simplified Toto 2.0 for at most one hour on one A100 80GB,
using data from GIFT-Eval Pretrain, and is then scored zero-shot on GIFT-Eval."""
SETUP = [
    (
        "Model",
        "3.3M-parameter simplified Toto 2.0; everything but the forecast interface may change",
    ),
    ("Data", "GIFT-Eval Pretrain, as the GEP-S, GEP-M and GEP-L slices on Hugging Face"),
    ("Budget", "At most 3,600 seconds of training per run on one A100 80GB"),
    (
        "Score",
        "GIFT-Eval CRPS relative to Seasonal Naive, 97 tasks, geometric mean; lower is better",
    ),
]
RULE = """A record is the mean of three or more runs at one commit, retrained by the maintainers. It
must improve on the previous record by more than seed noise: 0.013 with three runs each."""
FOOTNOTE = """Gray points and dashed lines mark published models on the GIFT-Eval leaderboard. In
the two panels by metric, each small gray dot is one verified run and the orange line is the record,
the mean of its runs. Records are decided on CRPS."""
FONTS = (
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500"
    "&family=IBM+Plex+Sans:ital,wght@0,400;0,600;1,600&display=swap"
)
TOKENS = """--paper: #FFFFFF; --ink: #14202B; --text: #3E4A56; --muted: #6B7682; --rule: #E1E4E0;
  --accent: #D9622B; --milestone: #8C99A6; --run: #A9B1BA;"""
DARK = """--paper: #10181F; --ink: #F4F5F1; --text: #C9D3DC; --muted: #8C99A6; --rule: #2A3845;
  --accent: #F08A55; --milestone: #6B7682; --run: #56626E; color-scheme: dark;"""
STYLE = f"""
:root {{ {TOKENS} }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ {DARK} }} }}
:root[data-theme="dark"] {{ {DARK} }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--paper); color: var(--text);
  font: 16px/1.6 "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif; }}
a {{ color: var(--ink); text-underline-offset: 3px; }}
a:focus-visible, summary:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 3px; }}
.band {{ background: #14202B; color: #C9D3DC; }}
.band .inner {{ max-width: 880px; margin: 0 auto; padding-inline: 20px; padding-block: 56px 44px; }}
.band h1 {{ margin: 0; color: #F4F5F1; font-size: 44px; font-weight: 600; line-height: 1.1; }}
.band p {{ margin: 12px 0 24px; font-size: 18px; max-width: 60ch; }}
.buttons {{ display: flex; flex-wrap: wrap; gap: 10px; }}
.buttons a {{ display: inline-flex; align-items: center; gap: 8px; color: #F4F5F1;
  border: 1px solid #3A4A5A; border-radius: 6px; padding: 7px 14px; text-decoration: none;
  font-size: 14px; }}
.buttons a:hover {{ border-color: #C9D3DC; }}
main {{ max-width: 880px; margin: 0 auto; padding-inline: 20px; padding-block: 36px 64px; }}
h2 {{ color: var(--ink); font-size: 22px; font-weight: 600; margin: 44px 0 12px; }}
.figure {{ width: min(1040px, calc(100vw - 40px)); position: relative; left: 50%;
  transform: translateX(-50%); margin: 8px 0 0; }}
.figure .scroll svg {{ min-width: 860px; }}
.figure + .figure {{ margin-top: 20px; }}
figcaption {{ color: var(--muted); font-size: 14px; margin-top: 6px; }}
svg {{ display: block; width: 100%; height: auto; }}
svg .title {{ fill: var(--ink); font: 600 17px "IBM Plex Sans", Arial, sans-serif; }}
svg .sub {{ fill: var(--muted); font: 13px "IBM Plex Sans", Arial, sans-serif; }}
svg .tick {{ fill: var(--muted); font: 12px "IBM Plex Mono", Menlo, monospace; }}
svg .grid {{ stroke: var(--rule); }}
svg .ref {{ stroke: var(--milestone); stroke-dasharray: 5 5; }}
svg .rlabel {{ fill: var(--milestone); font: 12px "IBM Plex Sans", Arial, sans-serif; }}
svg .run {{ fill: var(--run); }}
svg .model {{ fill: var(--milestone); }}
svg .key {{ fill: var(--paper); stroke: var(--milestone); }}
svg .best {{ fill: none; stroke: var(--accent); stroke-width: 2.5; }}
svg .rec {{ fill: var(--accent); stroke: var(--paper); stroke-width: 1.5; }}
svg .change {{ fill: var(--accent); font: 12.5px "IBM Plex Sans", Arial, sans-serif; }}
svg .seed {{ fill: none; stroke-width: 1.2; }}
svg [data-tip] {{ transform-box: fill-box; transform-origin: center; }}
svg [data-tip].on {{ transform: scale(1.6); }}
.tip {{ position: fixed; z-index: 10; pointer-events: none; max-width: 280px; white-space: pre-line;
  background: var(--ink); color: var(--paper); border-radius: 6px; padding: 8px 12px;
  font-size: 13px; line-height: 1.5; box-shadow: 0 4px 14px rgb(0 0 0 / 0.25); }}
.tip b {{ display: block; }}
dl {{ display: grid; grid-template-columns: 90px 1fr; gap: 6px 16px; margin: 16px 0 0; }}
dt {{ color: var(--muted); }} dd {{ margin: 0; }}
.scroll {{ overflow-x: auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
th {{ text-align: left; font-weight: 600; color: var(--muted); font-size: 13px; }}
th, td {{ padding: 8px 12px 8px 0; border-bottom: 1px solid var(--rule); vertical-align: top; }}
td.n {{ font-family: "IBM Plex Mono", Menlo, monospace; font-variant-numeric: tabular-nums;
  white-space: nowrap; }}
details {{ border-top: 1px solid var(--rule); padding: 10px 0; }}
details:last-of-type {{ border-bottom: 1px solid var(--rule); }}
summary {{ cursor: pointer; color: var(--ink); }}
.dim {{ color: var(--muted); }}
.body {{ padding-top: 10px; display: grid; gap: 14px; }}
.body p {{ margin: 0; }}
.changes {{ margin: 0; padding-left: 18px; font: 14px "IBM Plex Mono", Menlo, monospace; }}
@media (max-width: 600px) {{ .band h1 {{ font-size: 36px; }} dl {{ grid-template-columns: 1fr; }} }}
"""
# The README shows the figures as images, outside the page's styles.
STANDALONE = """<style>
.title { fill: #14202B; font: 600 17px Arial, sans-serif; }
.sub { fill: #6B7682; font: 13px Arial, sans-serif; }
.tick { fill: #6B7682; font: 12px Menlo, Consolas, monospace; }
.grid { stroke: #E1E4E0; } .ref { stroke: #8C99A6; stroke-dasharray: 5 5; }
.rlabel { fill: #8C99A6; font: 12px Arial, sans-serif; } .run { fill: #A9B1BA; }
.model { fill: #8C99A6; } .key { fill: #FFFFFF; stroke: #8C99A6; }
.best { fill: none; stroke: #D9622B; stroke-width: 2.5; }
.rec { fill: #D9622B; stroke: #FFFFFF; stroke-width: 1.5; }
.change { fill: #D9622B; font: 12.5px Arial, sans-serif; }
</style><rect width="100%" height="100%" fill="#FFFFFF"/>"""
SEEDS = ["var(--accent)", "var(--ink)", "var(--milestone)", "var(--muted)"]
# Show the tooltip of the point nearest the pointer, as plotting libraries do.
SCRIPT = """
const tip = document.querySelector(".tip");
const points = [...document.querySelectorAll(".figure [data-tip]")];
let shown = null;
function show(point) {
  if (point === shown) return;
  shown?.classList.remove("on");
  shown = point;
  tip.hidden = !point;
  if (!point) return;
  point.classList.add("on");
  const [head, ...rest] = point.dataset.tip.split("\\n");
  const title = document.createElement("b");
  title.textContent = head;
  tip.replaceChildren(title, rest.join("\\n"));
  const box = point.getBoundingClientRect();
  let left = box.right + 10;
  if (left + tip.offsetWidth > innerWidth - 8) left = box.left - 10 - tip.offsetWidth;
  const top = box.top + box.height / 2 - tip.offsetHeight / 2;
  tip.style.left = Math.max(8, left) + "px";
  tip.style.top = Math.max(8, Math.min(innerHeight - tip.offsetHeight - 8, top)) + "px";
}
function nearest(event) {
  let best = null, reach = 10;
  for (const point of points) {
    const box = point.getBoundingClientRect();
    const x = box.left + box.width / 2 - event.clientX;
    const away = Math.hypot(x, box.top + box.height / 2 - event.clientY) - box.width / 2;
    if (away < reach) [best, reach] = [point, away];
  }
  show(best);
}
for (const svg of document.querySelectorAll(".figure svg")) {
  svg.addEventListener("pointermove", nearest);
  svg.addEventListener("pointerdown", nearest);
  svg.addEventListener("pointerleave", (event) => event.pointerType === "touch" || show(null));
}
addEventListener("pointerdown", (event) => event.target.closest(".figure svg") || show(null));
addEventListener("scroll", () => show(null), true);
"""


def load() -> list[tuple[str, dict, dict]]:
    """(folder, front matter, result) for each record, oldest first."""
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
        crps = f"${score['crps']:.4f} \\pm {score['crps_sd']:.4f}$"
        date = folder.split("_", 1)[0]
        link = f"[{folder}](records/{folder}/)"
        rows.append(f"| {number} | {crps} | {team['description']} | {date} | {link} | {people} |")
    return "\n".join(rows)


def esc(text) -> str:
    return html.escape(str(text))


def size(parameters: int) -> str:
    """A parameter count in short form: 147K, 3.3M, 331M."""
    for unit, scale in (("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if parameters >= scale:
            value = parameters / scale
            return f"{value:.0f}{unit}" if value >= 10 else f"{value:.1f}".removesuffix(".0") + unit
    return str(parameters)


def radius(parameters: int) -> float:
    """Point radius in the overall plot, growing with the logarithm of the parameter count."""
    return max(3.0, 3 * math.log10(parameters) - 12)


def people(team: dict) -> str:
    return ", ".join(
        f'<a href="https://github.com/{esc(m["github"])}">@{esc(m["github"])}</a>'
        for m in team["members"]
    )


def svg(width, height, label, parts, standalone) -> str:
    head = f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{label}"'
    if standalone:
        head += f' xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        return head + STANDALONE + "".join(parts) + "</svg>"
    return head + ">" + "".join(parts) + "</svg>"


def tips(standalone):
    """Text the page shows when a point is hovered; the README's image has no tooltips."""

    def tip(*lines):
        return "" if standalone else f' data-tip="{esc(chr(10).join(lines))}"'

    def run(number, run):
        score = run["gift_eval"]
        return tip(
            f"Record {number}, seed {run['seed']}",
            f"MASE {score['geometric_relative_mase']:.4f}",
            f"CRPS {score['geometric_relative_crps']:.4f}",
            f"{run['training_seconds']:.0f} s on {run['device']}",
        )

    def record(number, folder, team, result):
        score = result["gift_eval"]
        return tip(
            f"Record {number} · {folder.split('_', 1)[0]}",
            team["description"],
            f"MASE {score['mase']:.4f} ± {score['mase_sd']:.4f}",
            f"CRPS {score['crps']:.4f} ± {score['crps_sd']:.4f}",
            f"{size(result['parameters'])} parameters · {score['runs']} runs",
            ", ".join(f"@{m['github']}" for m in team["members"]),
        )

    def model(name, scores):
        lines = (f"MASE {scores['mase']:.3f}", f"CRPS {scores['crps']:.3f}")
        return tip(name, *lines, f"{size(scores['parameters'])} parameters")

    return run, record, model


def ticks(lo, hi):
    tick = round(lo * 20) / 20
    while tick <= hi + 1e-9:
        if tick >= lo:
            yield tick
        tick = round(tick + 0.05, 2)


def progress_chart(entries, standalone=False) -> str:
    """MASE and CRPS by record, side by side: each run a gray dot, the record a step line."""
    width, height, top, bottom = 1040, 400, 86, 44
    count = len(entries)
    run_tip, record_tip, _ = tips(standalone)
    parts = []
    for panel, metric in enumerate(("mase", "crps")):
        left = panel * width // 2 + 64
        right = (panel + 1) * width // 2 - 146
        name = metric.upper()
        key = f"geometric_relative_{metric}"
        runs = [run["gift_eval"][key] for _, _, r in entries for run in r["runs"]]
        means = [r["gift_eval"][metric] for _, _, r in entries]
        marks = [(model, scores[metric]) for model, scores in MILESTONES.items()]
        lo = min(runs + [value for _, value in marks]) - 0.01
        hi = max(runs) + 0.01
        xs = [left + (right - left) * (i + 0.5) / count for i in range(count)]
        span = (height - top - bottom) / (hi - lo)
        ys = {v: top + span * (hi - v) for v in runs + means + [v for _, v in marks]}
        parts.append(
            f'<text class="title" x="{left}" y="26">GIFT-Eval relative {name} '
            "(lower is better)</text>"
        )
        for tick in ticks(lo, hi):
            y = top + span * (hi - tick)
            parts.append(
                f'<line class="grid" x1="{left}" x2="{right}" y1="{y:.1f}" y2="{y:.1f}"/>'
                f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">'
                f"{tick:.2f}</text>"
            )
        for model, value in marks:
            parts.append(
                f'<line class="ref" x1="{left}" x2="{right}" y1="{ys[value]:.1f}" '
                f'y2="{ys[value]:.1f}"/><text class="rlabel" x="{right + 8}" '
                f'y="{ys[value] + 4:.1f}">{esc(model)} {value:.3f}</text>'
            )
        for i, (_, _, result) in enumerate(entries):
            for j, run in enumerate(result["runs"]):
                dx = (j - (len(result["runs"]) - 1) / 2) * 7
                parts.append(
                    f'<circle class="run" cx="{xs[i] + dx:.1f}" '
                    f'cy="{ys[run["gift_eval"][key]]:.1f}" r="3.5"{run_tip(i + 1, run)}/>'
                )
        path = f"M{xs[0]:.1f},{ys[means[0]]:.1f}"
        for i in range(1, count):
            path += f" H{xs[i]:.1f} V{ys[means[i]]:.1f}"
        parts.append(f'<path class="best" d="{path} H{right}"/>')
        for i, (folder, team, result) in enumerate(entries):
            label = team["description"] if i else "baseline"
            y = ys[means[i]]
            parts.append(
                f'<circle class="rec" cx="{xs[i]:.1f}" cy="{y:.1f}" r="5"'
                f"{record_tip(i + 1, folder, team, result)}/>"
                f'<text class="change" transform="translate({xs[i] + 8:.1f},{y - 10:.1f}) '
                f'rotate(-28)">{esc(label)}</text>'
                f'<text class="tick" x="{xs[i]:.1f}" y="{height - bottom + 20}" '
                f'text-anchor="middle">{i + 1}</text>'
            )
        parts.append(
            f'<text class="sub" x="{(left + right) / 2:.1f}" y="{height - 6}" '
            'text-anchor="middle">Record</text>'
        )
    return svg(width, height, "MASE and CRPS by record", parts, standalone)


def scatter_chart(entries, standalone=False) -> str:
    """CRPS against MASE: records and published models as points sized by parameter count."""
    width, height, left, right, top, bottom = 1040, 460, 64, 40, 56, 56
    _, record_tip, model_tip = tips(standalone)
    points = [(r["gift_eval"]["mase"], r["gift_eval"]["crps"]) for _, _, r in entries]
    points += [(scores["mase"], scores["crps"]) for scores in MILESTONES.values()]
    lows = [min(p[axis] for p in points) - 0.02 for axis in (0, 1)]
    highs = [max(p[axis] for p in points) + 0.02 for axis in (0, 1)]

    def x(mase):
        return left + (width - left - right) * (mase - lows[0]) / (highs[0] - lows[0])

    def y(crps):
        return top + (height - top - bottom) * (highs[1] - crps) / (highs[1] - lows[1])

    parts = [
        f'<text class="title" x="{left}" y="26">GIFT-Eval relative CRPS against MASE '
        "(lower left is better)</text>"
    ]
    for tick in ticks(lows[1], highs[1]):
        parts.append(
            f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(tick):.1f}" '
            f'y2="{y(tick):.1f}"/><text class="tick" x="{left - 8}" y="{y(tick) + 4:.1f}" '
            f'text-anchor="end">{tick:.2f}</text>'
        )
    for tick in ticks(lows[0], highs[0]):
        parts.append(
            f'<line class="grid" x1="{x(tick):.1f}" x2="{x(tick):.1f}" y1="{top}" '
            f'y2="{height - bottom}"/><text class="tick" x="{x(tick):.1f}" '
            f'y="{height - bottom + 20}" text-anchor="middle">{tick:.2f}</text>'
        )
    for model, scores in MILESTONES.items():
        cx, cy, r = x(scores["mase"]), y(scores["crps"]), radius(scores["parameters"])
        parts.append(
            f'<circle class="model" cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}"'
            f"{model_tip(model, scores)}/>"
            f'<text class="rlabel" x="{cx + r + 5:.1f}" y="{cy + 4:.1f}">{esc(model)} · '
            f"{size(scores['parameters'])}</text>"
        )
    means = [(x(r["gift_eval"]["mase"]), y(r["gift_eval"]["crps"])) for _, _, r in entries]
    path = " L".join(f"{cx:.1f},{cy:.1f}" for cx, cy in means)
    parts.append(f'<path class="best" d="M{path}"/>')
    for i, (folder, team, result) in enumerate(entries):
        label = team["description"] if i else "baseline"
        (cx, cy), r = means[i], radius(result["parameters"])
        parts.append(
            f'<circle class="rec" cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}"'
            f"{record_tip(i + 1, folder, team, result)}/>"
            f'<text class="change" x="{cx - r - 5:.1f}" y="{cy - 8:.1f}" text-anchor="end">'
            f"{i + 1}. {esc(label)} · {size(result['parameters'])}</text>"
        )
    # The size legend sits in the corner no model reaches: high MASE with low CRPS.
    edge, base = width - right - 28, height - bottom - 24
    for k, parameters in enumerate((1_000_000_000, 10_000_000, 100_000)):
        r = radius(parameters)
        cx = edge - 30 - 66 * k
        parts.append(
            f'<circle class="key" cx="{cx:.1f}" cy="{base - r:.1f}" r="{r:.1f}"/>'
            f'<text class="rlabel" x="{cx + r + 4:.1f}" y="{base - r + 4:.1f}">'
            f"{size(parameters)}</text>"
        )
    parts.append(
        f'<text class="rlabel" x="{edge}" y="{base - 40}" text-anchor="end">'
        "Point size: parameters (log scale)</text>"
    )
    middle = top + (height - top - bottom) / 2
    parts.append(
        f'<text class="sub" x="{left + (width - left - right) / 2:.1f}" y="{height - 8}" '
        'text-anchor="middle">Relative MASE</text>'
        f'<text class="sub" transform="translate(16,{middle:.1f}) rotate(-90)" '
        'text-anchor="middle">Relative CRPS</text>'
    )
    return svg(width, height, "CRPS against MASE", parts, standalone)


def loss_chart(result) -> str:
    """Training loss of every run against step."""
    width, height, left, bottom, top, right = 960, 260, 56, 30, 14, 16
    runs = result["runs"]
    losses = [e["loss"] for r in runs for e in r["log"] if "loss" in e]
    steps = max(e["step"] for r in runs for e in r["log"])
    ordered = sorted(losses)
    lo, hi = ordered[0], ordered[int(0.98 * (len(ordered) - 1))]

    def x(step):
        return left + (width - left - right) * step / steps

    def y(v):
        return top + (height - top - bottom) * (hi - min(v, hi)) / (hi - lo)

    parts = [
        f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(v):.1f}" y2="{y(v):.1f}"/>'
        f'<text class="tick" x="{left - 8}" y="{y(v) + 4:.1f}" text-anchor="end">{v:.2f}</text>'
        for v in (lo, (lo + hi) / 2, hi)
    ]
    for i, run in enumerate(runs):
        color = SEEDS[i % len(SEEDS)]
        points = " ".join(
            f"{x(e['step']):.1f},{y(e['loss']):.1f}" for e in run["log"] if "loss" in e
        )
        parts.append(
            f'<polyline class="seed" style="stroke:{color}" points="{points}">'
            f"<title>seed {run['seed']}</title></polyline>"
            f'<text class="tick" style="fill:{color}" x="{width - right}" '
            f'y="{top + 14 + 16 * i}" text-anchor="end">seed {run["seed"]}</text>'
        )
    parts.append(
        f'<text class="tick" x="{left}" y="{height - 8}">step 0</text>'
        f'<text class="tick" x="{width - right}" y="{height - 8}" text-anchor="end">'
        f"step {steps:,}</text>"
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Training loss">'
        + "".join(parts)
        + "</svg>"
    )


def flatten(config: dict) -> dict:
    return {
        f"{section}.{key}": value
        for section, values in config.items()
        for key, value in values.items()
        if key not in ("seed", "device")
    }


def changes(result, previous) -> str:
    if previous is None:
        return "<p>The starting point: <code>configs/baseline.yaml</code>.</p>"
    new, old = flatten(result["config"]), flatten(previous["config"])
    rows = [
        f"<li>{esc(key)}: {esc(old.get(key, '—'))} → {esc(new.get(key, '—'))}</li>"
        for key in sorted(new.keys() | old.keys())
        if new.get(key) != old.get(key)
    ]
    if result["data"] != previous["data"]:
        rows.append(
            f"<li>data: {esc(previous['data']['name'])} → {esc(result['data']['name'])}</li>"
        )
    body = "".join(rows) or "<li>No configuration change; the change is in the code.</li>"
    return f'<ul class="changes">{body}</ul>'


def fmt(summary, key="geometric_relative_crps") -> str:
    return f"{summary[key]:.4f}" if summary else "—"


def records_table(entries) -> str:
    rows = "".join(
        f'<tr><td class="n">{i}</td><td class="n">{esc(f.split("_", 1)[0])}</td>'
        f"<td>{esc(t['description'])}</td>"
        f'<td class="n">{r["gift_eval"]["crps"]:.4f} ± {r["gift_eval"]["crps_sd"]:.4f}</td>'
        f'<td class="n">{r["gift_eval"]["mase"]:.4f}</td>'
        f'<td class="n">{r["gift_eval"]["runs"]}</td><td>{people(t)}</td></tr>'
        for i, (f, t, r) in enumerate(entries, 1)
    )
    return (
        '<div class="scroll"><table><tr><th>#</th><th>Date</th><th>Change</th>'
        "<th>GIFT-Eval CRPS</th><th>MASE</th><th>Runs</th><th>Contributors</th></tr>"
        f"{rows}</table></div>"
    )


def details(entries) -> str:
    out = []
    for i, (folder, team, result) in reversed(list(enumerate(entries, 1))):
        previous = entries[i - 2][2] if i > 1 else None
        runs = "".join(
            f'<tr><td class="n">{run["seed"]}</td><td class="n">{fmt(run["gift_eval"])}</td>'
            f'<td class="n">{fmt(run["gift_eval"], "geometric_relative_mase")}</td>'
            f'<td class="n">{fmt(run["gep_val"])}</td><td class="n">{fmt(run["gep_test"])}</td>'
            f'<td class="n">{run["training_seconds"]:.0f} s</td><td>{esc(run["device"])}</td></tr>'
            for run in result["runs"]
        )
        commit = result["commit"]
        out.append(
            f"<details><summary>Record {i}: {esc(team['description'])} "
            f'<span class="dim">({result["gift_eval"]["crps"]:.4f})</span></summary>'
            '<div class="body"><div class="scroll"><table><tr><th>Seed</th>'
            "<th>GIFT-Eval CRPS</th><th>MASE</th><th>GEP-Val</th><th>GEP-Test</th>"
            f"<th>Training</th><th>GPU</th></tr>{runs}</table></div>"
            f"<p><b>Change from the previous record</b></p>{changes(result, previous)}"
            f'<p><b>Training loss</b></p><div class="scroll">{loss_chart(result)}</div>'
            f'<p class="dim"><a href="{REPO}/tree/main/records/{esc(folder)}">Report</a> · '
            f'<a href="{REPO}/commit/{esc(commit)}">Commit {esc(commit[:7])}</a> · '
            f"{people(team)}</p></div></details>"
        )
    return "".join(out)


def page(entries) -> str:
    buttons = "".join(
        f'<a href="{url}">'
        + (
            f'<img src="https://cdn.simpleicons.org/{logo}" alt="" width="16" height="16">'
            * bool(logo)
        )
        + f"{esc(text)}</a>"
        for text, url, logo in LINKS
    )
    setup = "".join(f"<dt>{name}</dt><dd>{esc(value)}</dd>" for name, value in SETUP)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>nanoTSFM</title>
<meta name="description" content="nanoTSFM: training a time-series foundation model on one A100,
targeting GIFT-Eval.">
<link rel="stylesheet" href="{FONTS}">
<style>{STYLE}</style>
</head>
<body>
<header class="band"><div class="inner">
<h1><i>nano</i>TSFM</h1>
<p>Training a time-series foundation model on one A100, targeting GIFT-Eval.</p>
<nav class="buttons">{buttons}</nav>
</div></header>
<main>
<figure class="figure"><div class="scroll">{scatter_chart(entries)}</div></figure>
<figure class="figure"><div class="scroll">{progress_chart(entries)}</div>
<figcaption>{FOOTNOTE}</figcaption></figure>
<h2>The task</h2><p>{ABOUT}</p><dl>{setup}</dl>
<h2>Records</h2><p>{RULE}</p>{records_table(entries)}
<h2>Record details</h2>{details(entries)}
</main>
<div class="tip" role="tooltip" hidden></div>
<script>{SCRIPT}</script>
</body>
</html>
"""


def main():
    entries = load()
    if sys.argv[1:] == ["--table"]:
        print(table(entries))
        return
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "site")
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page(entries))
    (out / "records.svg").write_text(progress_chart(entries, standalone=True))
    (out / "scatter.svg").write_text(scatter_chart(entries, standalone=True))
    print(out / "index.html")


if __name__ == "__main__":
    main()
