"""Build the record page from records/*/, or print the README's record table.

python scripts/records.py DIR      # the page, its two figures and the badge, in DIR
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
# GIFT-Eval leaderboard at gift-eval commit 9a014e9: models without test leakage. Parameter counts
# are the safetensors totals on each model's Hugging Face page. The leaderboard ranks entries by
# their average rank over tasks; its first, STRIDE w/ Synapse, is an agentic system with no
# published size, which the figures draw as a diamond.
MILESTONES = {
    "STRIDE w/ Synapse": {"mase": 0.625, "crps": 0.423, "parameters": None, "short": "STRIDE"},
    "TimesFM-3": {"mase": 0.667, "crps": 0.456, "parameters": 330_710_976},
    "Toto-2.0-4m": {"mase": 0.757, "crps": 0.524, "parameters": 4_144_448},
    "TinyCast": {"mase": 0.774, "crps": 0.545, "parameters": 146_505},
    "Moirai-small": {"mase": 0.946, "crps": 0.650, "parameters": 13_827_528},
}
# More models from the same leaderboard, from well-known labs, drawn in the overall plot only: they
# fill its gaps, so the climb from the baseline goes in even steps.
STEPS = {
    "Chronos-2": {"mase": 0.698, "crps": 0.485, "parameters": 119_477_664},
    "Moirai-2.0-small": {"mase": 0.728, "crps": 0.516, "parameters": 11_387_208},
    "YingLong-50m": {"mase": 0.822, "crps": 0.567, "parameters": 36_264_718},
    "YingLong-6m": {"mase": 0.880, "crps": 0.609, "parameters": 7_319_566},
}
# (text, link, Simple Icons logo and its color)
LINKS = [
    ("Code", REPO, "github/F4F5F1"),
    ("Data", "https://huggingface.co/datasets/abel-lab/nanoTSFM-pretrain", "huggingface/FFD21E"),
    ("Rules", f"{REPO}/blob/main/docs/rules.md", None),
    ("Submit a result", f"{REPO}/blob/main/docs/submission.md", None),
]
# The task as a short specification; KaTeX renders the TeX between dollar signs.
DOCS = f"{REPO}/blob/main/docs"
ABOUT = f"""nanoTSFM is an open benchmark for training small time-series foundation models on a
fixed budget. A run trains on GIFT-Eval Pretrain and is then scored on GIFT-Eval datasets it has
never seen; the best verified score holds the record. Each try takes minutes and ends in one
verified number, so nanoTSFM is also a small environment for
<a href="{DOCS}/directions.md#agents-and-recursive-self-improvement">recursive self-improvement</a>,
where an AI agent runs the loop."""
SPEC = [
    ("Model", "3.3M parameters in the baseline; free to change"),
    ("Data", f'<a href="{DOCS}/data.md">GIFT-Eval Pretrain</a>'),
    (
        "Budget",
        rf'$\leq 3600\,\text{{s}}$ of <a href="{DOCS}/rules.md#budget">training</a> '
        r"on $1 \times$ A100 80GB",
    ),
    ("Score", f'<a href="{DOCS}/rules.md#score">CRPS</a> on GIFT-Eval, zero-shot; lower is better'),
    (
        "Submission",
        r"$\geq 3$ repeated runs at one commit; a record needs $\geq 0.013$ below the last "
        f'(<a href="{DOCS}/submission.md#the-record-rule">rule</a>)',
    ),
]
# The whole process, left to right: (title, two detail lines, whether participants may change it).
PROCESS = [
    ("Pretrain corpus", "GIFT-Eval Pretrain", "nothing else", False),
    ("Data pipeline", "selection, mixing", "preprocessing", True),
    ("Model + training", "any architecture", "any optimizer", True),
    ("Forecast", "fixed interface", "nine quantiles", False),
    ("Evaluation", "GIFT-Eval, 97 tasks", "zero-shot CRPS", False),
    ("Submission", "3+ repeated runs", "retrained by us", False),
]
# The training clock covers one box of the process: (box, label, detail).
CLOCK = (2, "1 hour on 1 A100", "training steps only")
MATH = """<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.18.1/dist/katex.min.css"
  integrity="sha384-1vdNCNel6Tx/NQa8IR1mGOGKsbGreCkOPfbtPPnUURJ5Tu2PRVfQ/7KLZC+Pi1p1"
  crossorigin="anonymous">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.18.1/dist/katex.min.js"
  integrity="sha384-ycJ6GAwiS15LoUPipwJOrWTvkUHl/YqELValBwI5I4awP1EeEQJYarj+w85ntcz7"
  crossorigin="anonymous"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.18.1/dist/contrib/auto-render.min.js"
  integrity="sha384-bjyGPfbij8/NDKJhSGZNP/khQVgtHUE5exjm4Ydllo42FwIgYsdLO2lXGmRBf5Mz"
  crossorigin="anonymous" onload="renderMathInElement(document.querySelector('.spec'),
  {delimiters: [{left: '$', right: '$', display: false}], throwOnError: false})"></script>"""
RULE = "Each record is the mean of its runs, retrained by the maintainers before it counts."
FOOTNOTE = """Gray points and dashed lines mark published models on the GIFT-Eval leaderboard; the
diamond is STRIDE w/ Synapse, ranked first on the leaderboard, an agentic system. In the two
panels by metric, each small gray dot is one verified run and the orange line is the record, the
mean of its runs. Records are decided on CRPS."""
FONTS = (
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500"
    "&family=IBM+Plex+Sans:ital,wght@0,400;0,600;1,600&display=swap"
)
TOKENS = """--paper: #FFFFFF; --ink: #14202B; --text: #3E4A56; --muted: #6B7682; --rule: #E1E4E0;
  --accent: #D9622B; --milestone: #8C99A6; --run: #A9B1BA; --mine: #FBE6DA;"""
DARK = """--paper: #10181F; --ink: #F4F5F1; --text: #C9D3DC; --muted: #8C99A6; --rule: #2A3845;
  --accent: #F08A55; --milestone: #6B7682; --run: #56626E; --mine: #3A2419; color-scheme: dark;"""
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
.band .latest {{ margin: 20px 0 0; font-size: 14px; max-width: none; }}
.band .latest a, .band .latest b {{ color: #F4F5F1; }}
.buttons {{ display: flex; flex-wrap: wrap; gap: 10px; }}
.buttons a {{ display: inline-flex; align-items: center; gap: 8px; color: #F4F5F1;
  border: 1px solid #3A4A5A; border-radius: 6px; padding: 7px 14px; text-decoration: none;
  font-size: 14px; }}
.buttons a:hover {{ border-color: #C9D3DC; }}
main {{ max-width: 880px; margin: 0 auto; padding-inline: 20px; padding-block: 36px 64px; }}
h2 {{ color: var(--ink); font-size: 22px; font-weight: 600; margin: 44px 0 12px; }}
.figure {{ width: min(1040px, calc(100vw - 40px)); position: relative; left: 50%;
  transform: translateX(-50%); margin: 8px 0 0; }}
.figure + .figure {{ margin-top: 20px; }}
.narrow {{ display: none; }}
.narrow svg {{ max-width: 480px; margin: 0 auto; }}
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
svg .star {{ fill: var(--accent); }}
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
.tip.linked {{ pointer-events: auto; }}
.tip a {{ color: var(--paper); text-underline-offset: 3px; }}
dl {{ display: grid; grid-template-columns: 90px 1fr; gap: 6px 16px; margin: 16px 0 0;
  align-items: baseline; }}
dt {{ color: var(--muted); }} dd {{ margin: 0; }}
.spec .katex {{ font-size: 1.05em; }}
.flow {{ margin: 20px 0 12px; }}
.flow .box {{ fill: var(--paper); stroke: var(--milestone); }}
.flow .box.mine {{ fill: var(--mine); stroke: var(--accent); }}
.flow .name {{ fill: var(--ink); font: 600 13px "IBM Plex Sans", Arial, sans-serif; }}
.flow .line {{ fill: var(--muted); font: 12px "IBM Plex Sans", Arial, sans-serif; }}
.flow .arrow {{ stroke: var(--milestone); fill: none; }}
.flow .head {{ fill: var(--milestone); }}
.flow .brace {{ fill: none; stroke: var(--ink); stroke-width: 1.5; }}
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
.body .scroll svg {{ min-width: 560px; }}
.runs td:last-child {{ white-space: nowrap; }}
/* Below 900px the figures switch to their narrow layouts. */
@media (max-width: 900px) {{ .wide {{ display: none; }} .narrow {{ display: block; }} }}
/* On phones each record becomes a short block in place of a table row. */
@media (max-width: 600px) {{
  .band h1 {{ font-size: 36px; }} dl {{ grid-template-columns: 1fr; }}
  .records, .records tbody {{ display: block; }}
  .records tr {{ display: flex; flex-wrap: wrap; gap: 2px 14px; padding: 10px 0;
    border-bottom: 1px solid var(--rule); }}
  .records tr:first-child {{ display: none; }}
  .records td {{ border: 0; padding: 0; }}
  .records td:nth-child(2) {{ order: 1; }}
  .records td:nth-child(3) {{ flex: 1 0 80%; color: var(--ink); }}
  .records td[data-label]::before {{ content: attr(data-label) " "; color: var(--muted);
    font-family: "IBM Plex Sans", Arial, sans-serif; }}
  .runs th, .runs td {{ padding-right: 8px; font-size: 13px; }}
  .runs th:last-child, .runs td:last-child {{ display: none; }}
}}
"""
# The README shows the figures as images, outside the page's styles.
STANDALONE = """<style>
.title { fill: #14202B; font: 600 17px Arial, sans-serif; }
.sub { fill: #6B7682; font: 13px Arial, sans-serif; }
.tick { fill: #6B7682; font: 12px Menlo, Consolas, monospace; }
.grid { stroke: #E1E4E0; } .ref { stroke: #8C99A6; stroke-dasharray: 5 5; }
.rlabel { fill: #8C99A6; font: 12px Arial, sans-serif; } .run { fill: #A9B1BA; }
.model { fill: #8C99A6; } .key { fill: #FFFFFF; stroke: #8C99A6; } .star { fill: #D9622B; }
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
  // A record's tooltip links to its report, so the tooltip itself takes the pointer.
  tip.classList.toggle("linked", Boolean(point.dataset.link));
  if (point.dataset.link) {
    const report = Object.assign(document.createElement("a"), { href: point.dataset.link });
    report.textContent = "Read the report";
    tip.append("\\n", report);
  }
  const box = point.getBoundingClientRect(), wide = tip.offsetWidth, tall = tip.offsetHeight;
  let left = box.right + 10, top = box.top + box.height / 2 - tall / 2;
  if (left + wide > innerWidth - 8) left = box.left - 10 - wide;
  if (left < 8) {
    left = Math.min(Math.max(8, box.left + box.width / 2 - wide / 2), innerWidth - wide - 8);
    top = box.bottom + 10 + tall > innerHeight - 8 ? box.top - 10 - tall : box.bottom + 10;
  }
  tip.style.left = left + "px";
  tip.style.top = Math.max(8, Math.min(innerHeight - tall - 8, top)) + "px";
}
let closing = null;
function leave() {
  // Give the pointer time to reach a linked tooltip before it closes.
  clearTimeout(closing);
  if (shown?.dataset.link) closing = setTimeout(() => show(null), 400);
  else show(null);
}
function nearest(event) {
  let best = null, reach = event.pointerType === "touch" ? 24 : 10;
  for (const point of points) {
    const box = point.getBoundingClientRect();
    if (!box.width) continue;
    const x = box.left + box.width / 2 - event.clientX;
    const away = Math.hypot(x, box.top + box.height / 2 - event.clientY) - box.width / 2;
    if (away < reach) [best, reach] = [point, away];
  }
  if (best) {
    clearTimeout(closing);
    show(best);
  } else if (shown) leave();
}
for (const svg of document.querySelectorAll(".figure svg")) {
  svg.addEventListener("pointermove", nearest);
  svg.addEventListener("pointerdown", nearest);
  svg.addEventListener("pointerleave", (event) => event.pointerType === "touch" || leave());
}
tip.addEventListener("pointerenter", () => clearTimeout(closing));
tip.addEventListener("pointerleave", leave);
addEventListener("pointerdown", (event) => event.target.closest(".figure svg, .tip") || show(null));
addEventListener("scroll", () => show(null), true);
"""


def load() -> list[tuple[str, dict, dict]]:
    """(folder, front matter, result) for each record, oldest first.

    When the maintainers' retrains are in the folder, they replace the reported score.
    """
    out = []
    for folder in sorted((ROOT / "records").iterdir()):
        if folder.is_dir() and folder.name != "template":
            team = yaml.safe_load((folder / "README.md").read_text().split("---\n", 2)[1])
            result = json.loads((folder / "result.json").read_text())
            if (folder / "verified.json").exists():
                verified = json.loads((folder / "verified.json").read_text())
                result["gift_eval"], result["retrains"] = verified["gift_eval"], verified["runs"]
            out.append((folder.name, team, result))
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


def latest(entries) -> str:
    """The latest record in one line, with its contributors."""
    folder, team, result = entries[-1]
    return (
        f"Latest record: <b>{result['gift_eval']['crps']:.3f}</b> relative CRPS, "
        f"{esc(team['description'])}, by {people(team)} on {esc(folder.split('_', 1)[0])}"
    )


def badge(entries) -> str:
    """The README's record badge, as a shields.io endpoint."""
    _, team, result = entries[-1]
    who = ", ".join(f"@{m['github']}" for m in team["members"])
    message = f"{result['gift_eval']['crps']:.3f} by {who}"
    return json.dumps(
        {"schemaVersion": 1, "label": "world record", "message": message, "color": "D9622B"}
    )


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
    """Text the page shows when a record or model is hovered; the README's image has none."""

    def tip(*lines):
        return "" if standalone else f' data-tip="{esc(chr(10).join(lines))}"'

    def record(number, folder, team, result):
        score = result["gift_eval"]
        report = "" if standalone else f' data-link="{REPO}/tree/main/records/{esc(folder)}"'
        return report + tip(
            f"Record {number} · {folder.split('_', 1)[0]}",
            team["description"],
            f"MASE {score['mase']:.4f} ± {score['mase_sd']:.4f}",
            f"CRPS {score['crps']:.4f} ± {score['crps_sd']:.4f}",
            f"{size(result['parameters'])} parameters · {score['runs']} runs",
            ", ".join(f"@{m['github']}" for m in team["members"]),
        )

    def model(name, scores):
        lines = (f"MASE {scores['mase']:.3f}", f"CRPS {scores['crps']:.3f}")
        if scores["parameters"] is None:
            return tip(name, *lines, "Ranked first on the leaderboard; an agentic system")
        return tip(name, *lines, f"{size(scores['parameters'])} parameters")

    return record, model


def process(narrow=False) -> str:
    """The whole process as boxes and arrows; the parts participants may change are orange.

    A brace marks the box the training clock covers: beside it on phones, under it otherwise.
    """
    count, (clock, label, detail) = len(PROCESS), CLOCK
    if narrow:  # one box per row, top to bottom
        width, wide, tall, gap = 400, 262, 46, 18
        spots = [(0, k * (tall + gap)) for k in range(count)]
        legend = spots[-1][1] + tall + 26
    else:  # one row, left to right
        width, wide, tall, gap = 880, 128, 76, 22
        spots = [(k * (wide + gap), 0) for k in range(count)]
        legend = tall + 82
    x, y = spots[clock]
    if narrow:  # a brace opening to the right, then the label beside it
        edge, mid, end = x + wide + 8, y + tall / 2, y + tall
        brace = (
            f"M{edge},{y} Q{edge + 5},{y} {edge + 5},{y + 5} V{mid - 5} "
            f"Q{edge + 5},{mid} {edge + 10},{mid} Q{edge + 5},{mid} {edge + 5},{mid + 5} "
            f"V{end - 5} Q{edge + 5},{end} {edge},{end}"
        )
        text = f'x="{edge + 18}" y="{mid - 3}"', f'x="{edge + 18}" y="{mid + 14}"'
    else:  # a brace opening downward, then the label under it
        edge, mid, end = y + tall + 8, x + wide / 2, x + wide
        brace = (
            f"M{x},{edge} Q{x},{edge + 5} {x + 5},{edge + 5} H{mid - 5} "
            f"Q{mid},{edge + 5} {mid},{edge + 10} Q{mid},{edge + 5} {mid + 5},{edge + 5} "
            f"H{end - 5} Q{end},{edge + 5} {end},{edge}"
        )
        anchor = f'x="{mid}" text-anchor="middle"'
        text = f'{anchor} y="{edge + 28}"', f'{anchor} y="{edge + 45}"'
    parts = [
        f'<path class="brace" d="{brace}"/><text class="name" {text[0]}>{label}</text>'
        f'<text class="line" {text[1]}>{detail}</text>'
    ]
    for k, ((x, y), (name, first, second, mine)) in enumerate(zip(spots, PROCESS, strict=True)):
        parts.append(
            f'<rect class="box{" mine" * mine}" x="{x + 0.5}" y="{y + 0.5}" width="{wide - 1}" '
            f'height="{tall - 1}" rx="8"/>'
        )
        if narrow:
            parts.append(
                f'<text class="name" x="{x + 14}" y="{y + 20}">{name}</text>'
                f'<text class="line" x="{x + 14}" y="{y + 37}">{first} · {second}</text>'
            )
        else:
            parts.append(
                f'<text class="name" x="{x + 12}" y="{y + 24}">{name}</text>'
                f'<text class="line" x="{x + 12}" y="{y + 45}">{first}</text>'
                f'<text class="line" x="{x + 12}" y="{y + 62}">{second}</text>'
            )
        if k:  # an arrow from the previous box
            if narrow:
                mid, start, end = x + wide / 2, y - gap + 3, y - 4
                parts.append(
                    f'<path class="arrow" d="M{mid},{start} V{end}"/>'
                    f'<path class="head" d="M{mid - 4},{end - 5} L{mid},{end} '
                    f'L{mid + 4},{end - 5}Z"/>'
                )
            else:
                mid, start, end = y + tall / 2, x - gap + 3, x - 4
                parts.append(
                    f'<path class="arrow" d="M{start},{mid} H{end}"/>'
                    f'<path class="head" d="M{end - 5},{mid - 4} L{end},{mid} '
                    f'L{end - 5},{mid + 4}Z"/>'
                )
    for k, (label, mine) in enumerate((("Yours to change", True), ("Fixed", False))):
        x = k * 150
        parts.append(
            f'<rect class="box{" mine" * mine}" x="{x + 0.5}" y="{legend - 11.5}" width="16" '
            f'height="14" rx="3"/><text class="line" x="{x + 24}" y="{legend}">{label}</text>'
        )
    height = legend + 8
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="The process from data to '
        f"record; orange parts are yours to change, and the one-hour clock covers training steps "
        f'only">{"".join(parts)}</svg>'
    )


def ticks(lo, hi):
    tick = round(lo * 20) / 20
    while tick <= hi + 1e-9:
        if tick >= lo:
            yield tick
        tick = round(tick + 0.05, 2)


def progress_chart(entries, standalone=False, narrow=False) -> str:
    """MASE and CRPS by record: each run a gray dot, the record a step line.

    The two panels sit side by side, or one above the other in the narrow layout for phones.
    """
    # One panel's width and height, then its top, bottom, left and right margins.
    span, block, top, bottom, lead, trail = 520, 400, 86, 44, 64, 146
    if narrow:
        span, block, top, bottom, lead, trail = 400, 350, 50, 44, 46, 92
    count = len(entries)
    record_tip, _ = tips(standalone)
    parts = []
    for panel, metric in enumerate(("mase", "crps")):
        across, down = (0, panel * block) if narrow else (panel * span, 0)
        left, right = across + lead, across + span - trail
        name = metric.upper()
        key = f"geometric_relative_{metric}"
        runs = [run["gift_eval"][key] for _, _, r in entries for run in dots(r)]
        means = [r["gift_eval"][metric] for _, _, r in entries]
        marks = [
            (scores.get("short", model), scores[metric]) for model, scores in MILESTONES.items()
        ]
        lo = min(runs + [value for _, value in marks]) - 0.01
        hi = max(runs) + 0.01
        xs = [left + (right - left) * (i + 0.5) / count for i in range(count)]
        scale = (block - top - bottom) / (hi - lo)
        # The axis is reversed so that progress climbs: lower scores sit higher.
        ys = {v: down + top + scale * (v - lo) for v in runs + means + [v for _, v in marks]}
        parts.append(
            f'<text class="title" x="{4 if narrow else left}" y="{down + 26}">'
            f"{name} by record (up is better)</text>"
        )
        for tick in ticks(lo, hi):
            y = down + top + scale * (tick - lo)
            parts.append(
                f'<line class="grid" x1="{left}" x2="{right}" y1="{y:.1f}" y2="{y:.1f}"/>'
                f'<text class="tick" x="{left - 8}" y="{y + 4:.1f}" text-anchor="end">'
                f"{tick:.2f}</text>"
            )
        for model, value in marks:
            label = esc(model) if narrow else f"{esc(model)} {value:.3f}"
            parts.append(
                f'<line class="ref" x1="{left}" x2="{right}" y1="{ys[value]:.1f}" '
                f'y2="{ys[value]:.1f}"/><text class="rlabel" x="{right + 8}" '
                f'y="{ys[value] + 4:.1f}">{label}</text>'
            )
        for i, (_, _, result) in enumerate(entries):
            for j, run in enumerate(dots(result)):
                dx = (j - (len(dots(result)) - 1) / 2) * 7
                parts.append(
                    f'<circle class="run" cx="{xs[i] + dx:.1f}" '
                    f'cy="{ys[run["gift_eval"][key]]:.1f}" r="3.5"/>'
                )
        path = f"M{xs[0]:.1f},{ys[means[0]]:.1f}"
        for i in range(1, count):
            path += f" H{xs[i]:.1f} V{ys[means[i]]:.1f}"
        parts.append(f'<path class="best" d="{path} H{right}"/>')
        # Labels keep clear of the points and of the step line, inside the plot.
        taken = [(xs[i] - 6, ys[m] - 6, xs[i] + 6, ys[m] + 6) for i, m in enumerate(means)]
        corners = [(xs[0], ys[means[0]])]
        for i in range(1, count):
            corners += [(xs[i], ys[means[i - 1]]), (xs[i], ys[means[i]])]
        corners.append((right, ys[means[-1]]))
        for (ax, ay), (bx, by) in zip(corners, corners[1:], strict=False):
            steps = max(1, int(math.hypot(bx - ax, by - ay) / 6))
            for k in range(steps + 1):
                px, py = ax + (bx - ax) * k / steps, ay + (by - ay) * k / steps
                taken.append((px - 3, py - 3, px + 3, py + 3))
        frame = (left, down + top - 30, right, down + block - bottom)
        for i, (folder, team, result) in enumerate(entries):
            label = team["description"] if i else "baseline"
            y = ys[means[i]]
            parts.append(
                f'<circle class="rec" cx="{xs[i]:.1f}" cy="{y:.1f}" r="5"'
                f"{record_tip(i + 1, folder, team, result)}/>"
                f'<text class="tick" x="{xs[i]:.1f}" y="{down + block - bottom + 20}" '
                f'text-anchor="middle">{i + 1}</text>'
            )
            if not narrow:  # the narrow layout has no room; the record's number stands for it
                where = place(xs[i], y, 5, label, taken, frame)
                parts.append(f'<text class="change" {where}>{esc(label)}</text>')
        parts.append(
            f'<text class="sub" x="{(left + right) / 2:.1f}" y="{down + block - 6}" '
            'text-anchor="middle">Record</text>'
        )
    width, height = (span, 2 * block) if narrow else (2 * span, block)
    return svg(width, height, "MASE and CRPS by record", parts, standalone)


def place(cx, cy, r, text, taken, frame) -> str:
    """Attributes for a label beside a point: the first side where it fits and covers nothing."""
    wide, tall = 6.8 * len(text), 12

    def free(side):
        x, y, _ = side
        inside = frame[0] <= x and x + wide <= frame[2] and frame[1] <= y and y + tall <= frame[3]
        return inside and not any(
            x < b[2] and b[0] < x + wide and y < b[3] and b[1] < y + tall for b in taken
        )

    near, up, down = 0.7 * r + 3, cy - 0.7 * r - 2 - tall, cy + 0.7 * r + 2
    sides = [  # right of the point, then left, below and above, then the four corners
        (cx + r + 5, cy - tall / 2, "start"),
        (cx - r - 5 - wide, cy - tall / 2, "end"),
        (cx - wide / 2, cy + r + 4, "middle"),
        (cx - wide / 2, cy - r - 4 - tall, "middle"),
        (cx + near, up, "start"),
        (cx + near, down, "start"),
        (cx - near - wide, up, "end"),
        (cx - near - wide, down, "end"),
    ]
    x, y, anchor = next(filter(free, sides), sides[0])
    taken.append((x, y, x + wide, y + tall))
    x += {"start": 0, "middle": wide / 2, "end": wide}[anchor]
    return f'x="{x:.1f}" y="{y + tall - 2:.1f}" text-anchor="{anchor}"'


def scatter_chart(entries, standalone=False, narrow=False) -> str:
    """CRPS against MASE, best at the upper right: records and published models as points.

    Both axes are reversed, so progress climbs to the upper right as in the charts by record. A
    point's size shows its parameter count. The narrow layout for phones labels records by number.
    """
    width, height, left, right, top, bottom = 1040, 460, 64, 40, 56, 56
    if narrow:
        width, height, left, right, top, bottom = 400, 414, 46, 14, 62, 50
    record_tip, model_tip = tips(standalone)
    points = [(r["gift_eval"]["mase"], r["gift_eval"]["crps"]) for _, _, r in entries]
    published = MILESTONES | STEPS
    points += [(scores["mase"], scores["crps"]) for scores in published.values()]
    # Room on the better side, so the star stands apart from even the best model.
    spans = [max(p[axis] for p in points) - min(p[axis] for p in points) for axis in (0, 1)]
    lows = [min(p[axis] for p in points) - 0.25 * spans[axis] for axis in (0, 1)]
    highs = [max(p[axis] for p in points) + 0.02 for axis in (0, 1)]

    def x(mase):
        return left + (width - left - right) * (highs[0] - mase) / (highs[0] - lows[0])

    def y(crps):
        return top + (height - top - bottom) * (crps - lows[1]) / (highs[1] - lows[1])

    parts = [f'<text class="title" x="{left}" y="26">CRPS vs MASE</text>']
    if narrow:
        parts = [
            '<text class="title" x="4" y="24">CRPS vs MASE</text>'
            f'<text class="sub" x="4" y="{top - 10}">Relative CRPS</text>'
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
    # The size legend sits in the corner no model reaches: high MASE with low CRPS, the upper left.
    start, line = left + (8 if narrow else 20), top + 30
    taken = [(start, line - 40, start + (180 if narrow else 230), line + 18)]
    parts.append(
        f'<text class="rlabel" x="{start}" y="{line - 26}">Point size: parameters'
        + " (log scale)" * (not narrow)
        + "</text>"
    )
    for parameters in (100_000, 10_000_000, 1_000_000_000):
        r, label = radius(parameters), size(parameters)
        parts.append(
            f'<circle class="key" cx="{start + r:.1f}" cy="{line}" r="{r:.1f}"/>'
            f'<text class="rlabel" x="{start + 2 * r + 5:.1f}" y="{line + 4}">{label}</text>'
        )
        start += 2 * r + 5 + 7 * len(label) + 18
    models = [
        (x(scores["mase"]), y(scores["crps"]), radius(scores["parameters"] or 1e7), name, scores)
        for name, scores in published.items()
    ]
    records = [
        (x(r["gift_eval"]["mase"]), y(r["gift_eval"]["crps"]), radius(r["parameters"]))
        for _, _, r in entries
    ]
    # Labels keep clear of every point and of the line between records.
    taken += [(cx - r, cy - r, cx + r, cy + r) for cx, cy, r, *_ in models + records]
    for (ax, ay, _), (bx, by, _) in zip(records, records[1:], strict=False):
        steps = max(1, int(math.hypot(bx - ax, by - ay) / 6))
        for k in range(steps + 1):
            px, py = ax + (bx - ax) * k / steps, ay + (by - ay) * k / steps
            taken.append((px - 3, py - 3, px + 3, py + 3))
    frame = (left, top - 20, width, height - bottom)
    # A star marks the best corner, the lowest MASE and the lowest CRPS, drawn over the points.
    sx, sy = width - right, top
    star = " ".join(
        f"{sx + (10 if k % 2 == 0 else 4.2) * math.cos(math.radians(-90 + 36 * k)):.1f},"
        f"{sy + (10 if k % 2 == 0 else 4.2) * math.sin(math.radians(-90 + 36 * k)):.1f}"
        for k in range(10)
    )
    corner = f'<polygon class="star" points="{star}"><title>Best corner</title></polygon>'
    taken.append((sx - 12, sy - 12, sx + 12, sy + 12))
    path = " L".join(f"{cx:.1f},{cy:.1f}" for cx, cy, _ in records)
    parts.append(f'<path class="best" d="M{path}"/>')
    labels = []
    for i, (folder, team, result) in enumerate(entries):
        cx, cy, r = records[i]
        label = (
            f"{i + 1}. {team['description'] if i else 'baseline'} · {size(result['parameters'])}"
        )
        if narrow:
            label = str(i + 1)
        parts.append(
            f'<circle class="rec" cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}"'
            f"{record_tip(i + 1, folder, team, result)}/>"
        )
        labels.append(
            f'<text class="change" {place(cx, cy, r, label, taken, frame)}>{esc(label)}</text>'
        )
    for cx, cy, r, name, scores in models:
        short = scores.get("short", name)
        if scores["parameters"] is None:  # an agentic system: a diamond, with no size
            label = short if narrow else f"{name} · agentic"
            parts.append(
                f'<path class="model" d="M{cx:.1f},{cy - r:.1f} L{cx + r:.1f},{cy:.1f} '
                f'L{cx:.1f},{cy + r:.1f} L{cx - r:.1f},{cy:.1f}Z"{model_tip(name, scores)}/>'
            )
        else:
            label = short if narrow else f"{name} · {size(scores['parameters'])}"
            parts.append(
                f'<circle class="model" cx="{cx:.1f}" cy="{cy:.1f}" r="{r:.1f}"'
                f"{model_tip(name, scores)}/>"
            )
        labels.append(
            f'<text class="rlabel" {place(cx, cy, r, label, taken, frame)}>{esc(label)}</text>'
        )
    middle = top + (height - top - bottom) / 2
    parts.append(
        f'<text class="sub" x="{left + (width - left - right) / 2:.1f}" y="{height - 8}" '
        'text-anchor="middle">Relative MASE</text>'
        + f'<text class="sub" transform="translate(16,{middle:.1f}) rotate(-90)" '
        'text-anchor="middle">Relative CRPS</text>' * (not narrow)
    )
    return svg(width, height, "CRPS against MASE", parts + labels + [corner], standalone)


def dots(result) -> list[dict]:
    """The runs a record's mean comes from: the maintainers' retrains, or the reported runs."""
    return result.get("retrains", result["runs"])


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
        f'<td class="n" data-label="CRPS">{r["gift_eval"]["crps"]:.4f} ± '
        f"{r['gift_eval']['crps_sd']:.4f}</td>"
        f'<td class="n" data-label="MASE">{r["gift_eval"]["mase"]:.4f}</td>'
        f'<td class="n" data-label="Runs">{r["gift_eval"]["runs"]}</td><td>{people(t)}</td></tr>'
        for i, (f, t, r) in enumerate(entries, 1)
    )
    return (
        '<div class="scroll"><table class="records"><tr><th>#</th><th>Date</th><th>Change</th>'
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
        retrains = ""
        if "retrains" in result:
            scores = [f"{fmt(run['gift_eval'])}" for run in result["retrains"]]
            retrains = (
                "<p>The record is the mean of the maintainers' retrains, with new seeds: CRPS "
                f"{', '.join(scores[:-1])} and {scores[-1]}. The team's runs:</p>"
            )
        out.append(
            f"<details><summary>Record {i}: {esc(team['description'])} "
            f'<span class="dim">({result["gift_eval"]["crps"]:.4f})</span></summary>'
            f'<div class="body">{retrains}<div class="scroll"><table class="runs"><tr><th>Seed</th>'
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
    spec = "".join(f"<dt>{name}</dt><dd>{value}</dd>" for name, value in SPEC)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>nanoTSFM</title>
<link rel="icon" href="https://abel.ai/favicon.ico" sizes="32x32">
<link rel="apple-touch-icon" href="https://abel.ai/apple-touch-icon.png">
<meta name="description" content="nanoTSFM: hill-climbing GIFT-Eval with one A100 and one hour.">
<link rel="stylesheet" href="{FONTS}">
{MATH}
<style>{STYLE}</style>
</head>
<body>
<header class="band"><div class="inner">
<h1><i>nano</i>TSFM</h1>
<p>Hill-climbing GIFT-Eval with one A100 and one hour.</p>
<nav class="buttons">{buttons}</nav>
<p class="latest">{latest(entries)}</p>
</div></header>
<main>
<figure class="figure wide">{scatter_chart(entries)}</figure>
<figure class="figure wide">{progress_chart(entries)}<figcaption>{FOOTNOTE}</figcaption></figure>
<figure class="figure narrow">{scatter_chart(entries, narrow=True)}</figure>
<figure class="figure narrow">{progress_chart(entries, narrow=True)}
<figcaption>{FOOTNOTE}</figcaption></figure>
<h2>The task</h2><p>{ABOUT}</p>
<figure class="flow wide">{process()}</figure>
<figure class="flow narrow">{process(narrow=True)}</figure>
<dl class="spec">{spec}</dl>
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
    (out / "badge.json").write_text(badge(entries))
    print(out / "index.html")


if __name__ == "__main__":
    main()
