"""Build the record page from records/*/, or print the README's record table.

python scripts/site.py DIR      # DIR/index.html and DIR/records.svg
python scripts/site.py --table
"""

import html
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = "https://github.com/Abel-ai-lab/nanoTSFM"
SITE = "https://abel-ai-lab.github.io/nanoTSFM/"
# GIFT-Eval leaderboard at gift-eval commit 9a014e9: zero-shot models without test leakage.
MILESTONES = [
    ("TimesFM-3", 0.456),
    ("Toto-2.0-4m", 0.524),
    ("TinyCast", 0.545),
    ("Moirai-small", 0.650),
]
FONTS = (
    "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600"
    "&family=IBM+Plex+Sans:ital,wght@0,400;0,600;0,700;1,400&display=swap"
)
STYLE = """
:root {
  --paper: #F4F5F1; --surface: #FFFFFF; --ink: #14202B; --text: #3E4A56; --muted: #6B7682;
  --rule: #D5DAD3; --accent: #D9622B; --good: #1F6B47; --milestone: #8C99A6;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper: #10181F; --surface: #17222C; --ink: #F4F5F1; --text: #C9D3DC; --muted: #8C99A6;
    --rule: #2A3845; --accent: #F08A55; --good: #7FD1A6; --milestone: #6B7682;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --paper: #10181F; --surface: #17222C; --ink: #F4F5F1; --text: #C9D3DC; --muted: #8C99A6;
  --rule: #2A3845; --accent: #F08A55; --good: #7FD1A6; --milestone: #6B7682;
  color-scheme: dark;
}
* { box-sizing: border-box; }
body {
  margin: 0; background: var(--paper); color: var(--text);
  font: 16px/1.55 "IBM Plex Sans", "Helvetica Neue", Arial, sans-serif;
}
main { max-width: 1040px; margin: 0 auto; padding-inline: 20px; padding-block: 28px 64px;
  display: flex; flex-direction: column; gap: 56px; }
a { color: var(--accent); text-underline-offset: 3px; }
a:focus-visible, summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; }
h1, h2, h3 { color: var(--ink); margin: 0; text-wrap: balance; }
h2 { font-size: 22px; font-weight: 600; }
.mono, table, .num { font-family: "IBM Plex Mono", Menlo, Consolas, monospace;
  font-variant-numeric: tabular-nums; }
nav { display: flex; flex-wrap: wrap; gap: 8px 24px; align-items: baseline;
  justify-content: space-between; }
/* The name as a series: "nano" observed on a solid line, "TSFM" forecast on a dashed one. */
.wm { display: inline-flex; align-items: baseline; font-size: 22px; line-height: 1.2;
  text-decoration: none; }
.wm i, .wm b { padding-bottom: 4px; background: left bottom / 100% 2px no-repeat; }
.wm i { font: italic 400 1em "IBM Plex Sans", Arial, sans-serif; color: var(--muted);
  padding-right: 2px; background-image: linear-gradient(var(--muted), var(--muted)); }
.wm b { font: 600 1em "IBM Plex Mono", Menlo, monospace; color: var(--ink); letter-spacing: -0.03em;
  background-image: repeating-linear-gradient(90deg, var(--accent) 0 6px, transparent 6px 10px); }
.wm.small { font-size: 1em; }
nav .links { display: flex; flex-wrap: wrap; gap: 8px 20px; font-size: 15px; }
.eyebrow { font: 500 13px "IBM Plex Mono", Menlo, monospace; letter-spacing: 0.12em;
  text-transform: uppercase; color: var(--muted); margin: 0; }
.hero { display: grid; grid-template-columns: minmax(0, 1.1fr) minmax(0, 1fr); gap: 32px 48px;
  align-items: end; }
.score { font: 500 clamp(64px, 12vw, 112px)/1 "IBM Plex Mono", Menlo, monospace;
  color: var(--ink); letter-spacing: -0.02em; margin: 8px 0 4px; }
.spread { font: 500 18px "IBM Plex Mono", Menlo, monospace; color: var(--muted); margin: 0 0 12px; }
.holder { font-size: 18px; color: var(--ink); margin: 0; }
.lede { margin: 0; max-width: 62ch; }
.facts { display: flex; flex-wrap: wrap; gap: 12px 32px; margin: 16px 0 0; padding: 0;
  list-style: none; }
.facts b { display: block; font: 500 20px "IBM Plex Mono", Menlo, monospace; color: var(--ink); }
.facts span { font-size: 13px; color: var(--muted); }
section { display: flex; flex-direction: column; gap: 16px; }
.chart { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px;
  padding: 12px; overflow-x: auto; }
.chart svg { display: block; width: 100%; min-width: 560px; height: auto; }
.chart .axis, .chart .tick { fill: var(--muted); font: 12px "IBM Plex Mono", Menlo, monospace; }
.chart .grid { stroke: var(--rule); stroke-width: 1; }
.chart .milestone { stroke: var(--milestone); stroke-dasharray: 2 5; stroke-width: 1.5; }
.chart .mlabel { fill: var(--milestone); font: 12px "IBM Plex Sans", Arial, sans-serif; }
.chart .line { fill: none; stroke: var(--accent); stroke-width: 3; stroke-linejoin: round; }
.chart .dot { fill: var(--accent); stroke: var(--surface); stroke-width: 2; }
.chart .bar { stroke: var(--accent); stroke-width: 1.5; }
.chart .plabel { fill: var(--ink); font: 600 13px "IBM Plex Sans", Arial, sans-serif; }
.chart .seed { fill: none; stroke-width: 1.2; }
.scroll { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 14px; }
th { text-align: left; font: 600 12px "IBM Plex Sans", Arial, sans-serif; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.08em; }
th, td { padding: 10px 12px; border-bottom: 1px solid var(--rule); vertical-align: top; }
td.text { font-family: "IBM Plex Sans", Arial, sans-serif; color: var(--ink); }
tr.best td { color: var(--good); }
tr.best td.text { color: var(--ink); }
details { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; }
summary { cursor: pointer; padding: 16px 20px; display: flex; flex-wrap: wrap; gap: 4px 16px;
  align-items: baseline; list-style: none; }
summary::-webkit-details-marker { display: none; }
summary::before { content: "+"; font: 500 16px "IBM Plex Mono", monospace; color: var(--muted);
  width: 12px; }
details[open] summary::before { content: "\\2212"; }
summary .title { font-weight: 600; color: var(--ink); font-size: 17px; }
summary .num { color: var(--accent); }
summary .meta { color: var(--muted); font-size: 14px; }
.body { padding: 0 20px 20px; display: grid; gap: 20px; }
.body h3 { font-size: 13px; text-transform: uppercase; letter-spacing: 0.08em;
  color: var(--muted); font-weight: 600; }
.changes { margin: 0; padding: 0; list-style: none; font: 14px "IBM Plex Mono", Menlo, monospace; }
.changes li { padding: 2px 0; }
.links { display: flex; flex-wrap: wrap; gap: 8px 20px; font-size: 14px; }
footer { color: var(--muted); font-size: 13px; border-top: 1px solid var(--rule);
  padding-top: 16px; }
@media (max-width: 720px) { .hero { grid-template-columns: 1fr; } }
@media (prefers-reduced-motion: no-preference) { details[open] .body { animation: open .2s; } }
@keyframes open { from { opacity: 0.4; } to { opacity: 1; } }
"""
SEEDS = ["var(--accent)", "var(--good)", "var(--milestone)", "var(--ink)"]
# The README shows the chart as an image, outside the page's styles.
CHART_STYLE = """<style>
.grid { stroke: #D5DAD3; } .milestone { stroke: #8C99A6; stroke-dasharray: 2 5; stroke-width: 1.5; }
.tick, .axis { fill: #6B7682; font: 12px Menlo, Consolas, monospace; }
.mlabel { fill: #8C99A6; font: 12px Arial, sans-serif; }
.line { fill: none; stroke: #D9622B; stroke-width: 3; } .bar { stroke: #D9622B; stroke-width: 1.5; }
.dot { fill: #D9622B; stroke: #FFFFFF; stroke-width: 2; }
.plabel { fill: #14202B; font: 600 13px Arial, sans-serif; }
</style><rect width="100%" height="100%" fill="#FFFFFF"/>"""


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
        crps = f"{score['crps']:.4f} ± {score['crps_sd']:.4f}"
        date = folder.split("_", 1)[0]
        link = f"[{folder}](records/{folder}/)"
        rows.append(f"| {number} | {crps} | {team['description']} | {date} | {link} | {people} |")
    return "\n".join(rows)


def esc(text) -> str:
    return html.escape(str(text))


def people(team: dict) -> str:
    return ", ".join(
        f'<a href="https://github.com/{esc(m["github"])}">@{esc(m["github"])}</a>'
        for m in team["members"]
    )


def record_chart(entries, standalone=False) -> str:
    """Record history, lower CRPS drawn higher, with published models as dotted lines."""
    width, height, left, right, top, bottom = 960, 380, 64, 250, 24, 44
    means = [r["gift_eval"]["crps"] for _, _, r in entries]
    lo = min([m for _, m in MILESTONES] + means) - 0.01
    hi = max(means) + 0.02
    count = len(entries)

    def x(i):
        return left + (width - left - right) * (i + 0.5) / count

    def y(v):
        return top + (height - top - bottom) * (hi - v) / (hi - lo)

    parts = []
    tick = round(lo * 20) / 20
    while tick <= hi:
        if tick >= lo:
            parts.append(
                f'<line class="grid" x1="{left}" x2="{width - right}" y1="{y(tick):.1f}" '
                f'y2="{y(tick):.1f}"/><text class="tick" x="{left - 10}" y="{y(tick) + 4:.1f}" '
                f'text-anchor="end">{tick:.2f}</text>'
            )
        tick = round(tick + 0.05, 2)
    for label, value in MILESTONES:
        parts.append(
            f'<line class="milestone" x1="{left}" x2="{width - right}" y1="{y(value):.1f}" '
            f'y2="{y(value):.1f}"/><text class="mlabel" x="{width - right + 10}" '
            f'y="{y(value) + 4:.1f}">{esc(label)} {value:.3f}</text>'
        )
    points = [(x(i), y(m)) for i, m in enumerate(means)]
    path = f"M{points[0][0]:.1f},{points[0][1]:.1f}"
    for px, py in points[1:]:
        path += f" H{px:.1f} V{py:.1f}"
    path += f" H{width - right:.1f}"
    parts.append(f'<path class="line" d="{path}"/>')
    for i, (folder, team, result) in enumerate(entries):
        score = result["gift_eval"]
        px, py = points[i]
        spread = score["crps_sd"] * (height - top - bottom) / (hi - lo)
        parts.append(
            f"<g><title>#{i + 1} {esc(team['team'])}: {score['crps']:.4f} ± "
            f"{score['crps_sd']:.4f} over {score['runs']} runs</title>"
            f'<line class="bar" x1="{px:.1f}" x2="{px:.1f}" y1="{py - spread:.1f}" '
            f'y2="{py + spread:.1f}"/><circle class="dot" cx="{px:.1f}" cy="{py:.1f}" r="7"/>'
            f'<text class="plabel" x="{px + 12:.1f}" y="{py - 12:.1f}">#{i + 1} '
            f"{esc(team['team'])} {score['crps']:.3f}</text></g>"
        )
        parts.append(
            f'<text class="tick" x="{px:.1f}" y="{height - bottom + 22}" '
            f'text-anchor="middle">{esc(folder.split("_", 1)[0])}</text>'
        )
    parts.append(
        f'<text class="axis" x="14" y="{top + (height - top - bottom) / 2:.1f}" '
        f'transform="rotate(-90 14 {top + (height - top - bottom) / 2:.1f})" '
        f'text-anchor="middle">GIFT-Eval relative CRPS, better ↑</text>'
    )
    head = f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Record history"'
    if standalone:
        head += f' xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        return head + CHART_STYLE + "".join(parts) + "</svg>"
    return head + ">" + "".join(parts) + "</svg>"


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
        )
        parts.append(
            f'<text class="tick" style="fill:{color}" x="{width - right}" '
            f'y="{top + 14 + 16 * i}" text-anchor="end">seed {run["seed"]}</text>'
        )
    parts.append(
        f'<text class="tick" x="{left}" y="{height - 8}">step 0</text>'
        f'<text class="tick" x="{width - right}" y="{height - 8}" text-anchor="end">'
        f"step {steps:,}</text>"
    )
    return f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Training loss">' + (
        "".join(parts) + "</svg>"
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


def detail(number, folder, team, result, previous, is_current) -> str:
    score = result["gift_eval"]
    rows = "".join(
        f"<tr><td>{run['seed']}</td><td>{fmt(run['gift_eval'])}</td>"
        f"<td>{fmt(run['gift_eval'], 'geometric_relative_mase')}</td>"
        f"<td>{fmt(run['gep_val'])}</td><td>{fmt(run['gep_test'])}</td>"
        f"<td>{run['training_seconds']:.0f} s</td><td>{run['steps']:,}</td>"
        f'<td class="text">{esc(run["device"] or "CPU")}</td></tr>'
        for run in result["runs"]
    )
    commit = result["commit"]
    return f"""<details{" open" if is_current else ""}>
<summary><span class="title">#{number} {esc(team["description"])}</span>
<span class="num">{score["crps"]:.4f} ± {score["crps_sd"]:.4f}</span>
<span class="meta">{esc(folder.split("_", 1)[0])} · {people(team)}</span></summary>
<div class="body">
<div class="scroll"><table>
<tr><th>Seed</th><th>GIFT-Eval CRPS</th><th>MASE</th><th>GEP-Val</th><th>GEP-Test</th>
<th>Training</th><th>Steps</th><th>GPU</th></tr>{rows}</table></div>
<div><h3>Change from the previous record</h3>{changes(result, previous)}</div>
<div><h3>Training loss</h3><div class="chart">{loss_chart(result)}</div></div>
<p class="links"><a href="{REPO}/tree/main/records/{esc(folder)}">Report</a>
<a href="{REPO}/blob/main/records/{esc(folder)}/result.json">result.json</a>
<a href="{REPO}/commit/{esc(commit)}">Commit <span class="mono">{esc(commit[:7])}</span></a></p>
</div></details>"""


def page(entries, built: str) -> str:
    number = len(entries)
    folder, team, result = entries[-1]
    score = result["gift_eval"]
    first = entries[0][2]["gift_eval"]["crps"]
    gain = 100 * (first - score["crps"]) / first
    table = "".join(
        f'<tr class="{"best" if i == number else ""}"><td>{i}</td>'
        f"<td>{r['gift_eval']['crps']:.4f} ± {r['gift_eval']['crps_sd']:.4f}</td>"
        f"<td>{r['gift_eval']['mase']:.4f}</td><td>{r['gift_eval']['runs']}</td>"
        f'<td class="text">{esc(t["description"])}</td><td>{esc(f.split("_", 1)[0])}</td>'
        f'<td class="text">{people(t)}</td></tr>'
        for i, (f, t, r) in enumerate(entries, 1)
    )
    details = "".join(
        detail(i, f, t, r, entries[i - 2][2] if i > 1 else None, i == number)
        for i, (f, t, r) in reversed(list(enumerate(entries, 1)))
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>nanoTSFM World Record</title>
<meta name="description" content="The nanoTSFM world record: GIFT-Eval relative CRPS after at
most one hour of training on one A100.">
<link rel="stylesheet" href="{FONTS}">
<style>{STYLE}</style>
</head>
<body>
<main>
<nav><a class="wm" href="{REPO}" aria-label="nanoTSFM on GitHub"><i>nano</i><b>TSFM</b></a>
<span class="links"><a href="{REPO}/blob/main/docs/rules.md">Rules</a>
<a href="{REPO}/blob/main/docs/submission.md">Submit a result</a>
<a href="{REPO}">GitHub</a></span></nav>
<header class="hero">
<div><p class="eyebrow">World record #{number} · {esc(folder.split("_", 1)[0])}</p>
<p class="score">{score["crps"]:.4f}</p>
<p class="spread">± {score["crps_sd"]:.4f} over {score["runs"]} runs</p>
<p class="holder">{esc(team["description"])} · {people(team)}</p></div>
<div><p class="lede">Train a time-series foundation model for at most one hour on one A100,
then forecast 97 GIFT-Eval tasks it has never seen. The score is CRPS relative to Seasonal
Naive, averaged geometrically: lower is better, and 1 matches Seasonal Naive. Each record is
the mean of three or more verified runs.</p>
<ul class="facts"><li><b>{number}</b><span>records</span></li>
<li><b>−{gain:.1f}%</b><span>since the baseline</span></li>
<li><b>{MILESTONES[0][1]:.3f}</b><span>best published model</span></li></ul></div>
</header>
<section><h2>Record history</h2><div class="chart">{record_chart(entries)}</div></section>
<section><h2>Records</h2><div class="scroll"><table>
<tr><th>#</th><th>GIFT-Eval CRPS</th><th>MASE</th><th>Runs</th><th>Change</th><th>Date</th>
<th>Contributors</th></tr>{table}</table></div></section>
<section><h2>Each record</h2>{details}</section>
<footer><span class="wm small"><i>nano</i><b>TSFM</b></span> · built from
<a href="{REPO}/tree/main/records">records/</a> at
<span class="mono">{esc(built[:7])}</span>. A new record must beat the last by more than seed
noise; <a href="{REPO}/blob/main/docs/submission.md">here is how to submit</a>.</footer>
</main>
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
    built = (
        os.environ.get("GITHUB_SHA")
        or subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    )
    (out / "index.html").write_text(page(entries, built))
    (out / "records.svg").write_text(record_chart(entries, standalone=True))
    print(out / "index.html")


if __name__ == "__main__":
    main()
