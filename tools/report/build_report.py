# /// script
# requires-python = ">=3.10"
# dependencies = ["markdown>=3.5"]
# ///
"""Build docs/split_report.html: notebook outputs + expert interpretation + recommendation.

Inputs (tools/report/_work/):
  report_data.json        from export_notebook.py (charts, tables, printed output, notebook markdown)
  interp_<key>.json       per-section interpretation (A, B, C, D, decision)
  synthesis.json          executive summary, key findings, final recommendation

The HTML is self-contained: D3 v7 is inlined once, every chart is redrawn from the notebook's own
payload and drawing function, so it is interactive offline.

Usage: uv run tools/report/build_report.py
"""
import datetime as dt
import html
import json
import re
from pathlib import Path

import markdown

REPORT = Path(__file__).resolve().parent
WORK = REPORT / "_work"
D3 = REPORT.parents[1] / "notebooks" / "vendor" / "d3.v7.min.js"
OUT = REPORT.parents[1] / "docs" / "split_report.html"

SECTION_ORDER = ["A", "B", "C", "D", "decision"]
MD = markdown.Markdown(extensions=["tables", "fenced_code", "sane_lists"])


def md(text):
    MD.reset()
    return MD.convert(text or "")


def esc(t):
    return html.escape(t or "")


def load_json(name, default=None):
    p = WORK / name
    return json.loads(p.read_text()) if p.exists() else default


CHECK_ANCHORS = {}


def evidence_links(ids, item_ids):
    out = []
    for e in ids or []:
        e = str(e).strip()
        target = e if e in item_ids else CHECK_ANCHORS.get(e)
        out.append(f'<a class="ev" href="#{esc(target)}" title="go to the output that records {esc(e)}">{esc(e)}</a>'
                   if target else f'<span class="ev">{esc(e)}</span>')
    return " ".join(out)


def chart_html(it):
    cid = "c_" + re.sub(r"[^a-zA-Z0-9_]", "_", it["id"])
    w = it["payload"].get("opts", {}).get("width", 720)
    h = it["payload"].get("opts", {}).get("height", 320)
    payload = json.dumps(it["payload"]).replace("</", "<\\/")  # a "</script>" inside data would end the tag
    return f"""<div class="chart-wrap"><div class="chart" id="{cid}" style="width:{w}px;min-height:{h}px"></div></div>
<script>(function(){{ const payload = {payload}; const draw = {it['func']};
  const el = document.getElementById("{cid}");
  try {{ draw(el, payload.data, d3, payload.opts); }} catch (e) {{ el.innerHTML = '<pre class="err">chart error: ' + e + '</pre>'; }}
}})();</script>"""


def interp_block(entry):
    if not entry:
        return ""
    cav = entry.get("caveats") or ""
    return (f'<div class="interp"><div class="interp-label">Interpretation</div>{md(entry.get("interpretation", ""))}'
            + (f'<div class="caveat"><b>Caveats.</b> {md(cav)}</div>' if cav.strip() else "") + "</div>")


def section_html(sec, interp, item_ids):
    key, title = sec["key"], sec["title"]
    items_i = (interp or {}).get("items", {})
    parts = [f'<section id="sec-{key}"><h2>{esc(title)}</h2>']
    if interp:
        parts.append(f'<div class="headline">{md(interp.get("headline", ""))}</div>')
        parts.append(f'<div class="overview">{md(interp.get("overview", ""))}</div>')
        if interp.get("implications"):
            parts.append('<div class="implications"><h4>Implications for the split</h4><ul>'
                         + "".join(f"<li>{md(x)}</li>" for x in interp["implications"]) + "</ul></div>")
    parts.append('<h3 class="detail-h">Detailed results</h3>')
    first_md = True
    for it in sec["items"]:
        iid = it["id"]
        if it["type"] == "markdown":
            text = it["text"]
            if first_md:   # section heading cell: its title is already the <h2>
                text = "\n".join(text.splitlines()[1:])
                first_md = False
            if text.strip():
                parts.append(f'<div class="nbmd" id="{esc(iid)}"><div class="nbmd-label">What this tests (from the notebook)</div>{md(text)}</div>')
            continue
        parts.append(f'<div class="result" id="{esc(iid)}"><div class="rid">{esc(iid)}</div>')
        if it["type"] == "text":
            parts.append(f'<pre class="out">{esc(it["text"])}</pre>')
        elif it["type"] == "table":
            parts.append(f'<div class="table-wrap">{it["html"]}</div>')
        elif it["type"] == "chart":
            parts.append(chart_html(it))
        parts.append(interp_block(items_i.get(iid)) + "</div>")
    if interp and interp.get("open_questions"):
        parts.append('<div class="openq"><h4>Open questions</h4><ul>'
                     + "".join(f"<li>{md(x)}</li>" for x in interp["open_questions"]) + "</ul></div>")
    parts.append("</section>")
    return "\n".join(parts)


RULE_RE = re.compile(r"^\[(R\d+[a-z]?(?:-T\d)?s?)\]", re.M)  # decision-rule lines, e.g. "[R3-T1] ..."
CHECK_RE = re.compile(r"\[(?:PASS|FAIL|INFO)[^\]]*\]\s+([A-D]\d+[a-z]?)\s*:")


def check_anchor_map(data):
    """Check id (e.g. 'D9') -> id of the text output where record_check printed it."""
    m = {}
    for s in data["sections"]:
        for it in s["items"]:
            if it["type"] == "text":
                for cid in CHECK_RE.findall(it["text"]) + RULE_RE.findall(it["text"]):
                    m.setdefault(cid, it["id"])
    return m


def fold_diagram():
    """Grid of Congresses x folds for the recommended rolling-origin split (mirrors the split table)."""
    cols = list(range(108, 120))
    rows = [(f"Fold {c}", c, "dev") for c in range(112, 118)] + [("Locked test", 118, "locked"), ("Prospective", 119, "prosp")]
    style = {"hist": ("#e4e3df", "#52514e", "history"), "train": ("#2a78d6", "#fff", "train"),
             "val": ("#eda100", "#0b0b0b", "validate"), "dev": ("#1baf7a", "#fff", "test"),
             "locked": ("#4a3aa7", "#fff", "locked test"), "prosp": ("#e87ba4", "#0b0b0b", "prospective")}
    head = "<tr><th></th>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    body = ""
    for label, c, kind in rows:
        cells = ""
        for k in cols:
            if k == 108 and c >= 112:
                role = "hist"
            elif kind == "prosp":
                role = "train" if k <= 118 else "prosp"
            elif k <= c - 2:
                role = "train"
            elif k == c - 1:
                role = "val"
            elif k == c:
                role = kind
            else:
                role = None
            if role:
                bg, fg, txt = style[role]
                cells += f'<td style="background:{bg};color:{fg}" title="{label}: {k} = {txt}">{txt}</td>'
            else:
                cells += "<td></td>"
        body += f"<tr><th>{label}</th>{cells}</tr>"
    legend = " ".join(f'<span class="lg" style="background:{bg};color:{fg}">{txt}</span>'
                      for bg, fg, txt in style.values())
    return (f'<div class="folds"><table>{head}{body}</table><div class="fold-legend">{legend}</div>'
            '<p class="fold-note">Each row is one honest "next Congress" forecast. Tune on the validate Congress, then refit '
            'on train + validate before scoring the test Congress. 108 is message-passing history only (GNN burn-in). '
            'Prospective: refit on 108-118, score 119 bills whose label windows have closed.</p></div>')



def simple_split_diagram():
    """One-row version of the fold grid for the simple split (train 108-115 / val 116 / test 117)."""
    cols = list(range(108, 120))
    role = {**{c: ("#2a78d6", "#fff", "train") for c in range(108, 116)}, 116: ("#eda100", "#0b0b0b", "validate"),
            117: ("#1baf7a", "#fff", "test"), 118: ("#4a3aa7", "#fff", "held back"), 119: ("#e4e3df", "#52514e", "excluded")}
    head = "<tr><th></th>" + "".join(f"<th>{c}</th>" for c in cols) + "</tr>"
    cells = "".join(f'<td style="background:{role[c][0]};color:{role[c][1]}" title="{c} = {role[c][2]}">{role[c][2]}</td>' for c in cols)
    return f'<div class="folds"><table>{head}<tr><th>Simple split</th>{cells}</tr></table></div>'


def conclusion_html(item_ids):
    """Final conclusion (tools/report/_work/conclusion.md): [ID] references become evidence links."""
    path = WORK / "conclusion.md"
    if not path.exists():
        return ""
    text = path.read_text()
    text = re.sub(r"\[([A-D]\d+[a-z]?|R\d+[a-z]?(?:-T\d)?s?)\](?!\()",
                  lambda m: evidence_links([m.group(1)], item_ids), text)
    rendered = md(text)
    rendered = rendered.replace("<p>{{SIMPLE_SPLIT_DIAGRAM}}</p>", simple_split_diagram())
    rendered = rendered.replace("<p>{{ROLLING_DIAGRAM}}</p>", fold_diagram())
    rendered = rendered.replace("<h2>", '<h2 id="conclusion">', 1)
    return f'<section class="conclusion">{rendered}</section>'


def recommendation_html(syn, item_ids, short=False):
    rec = syn.get("recommendation", {})
    rows = "".join(f"<tr><td>{esc(r.get('role'))}</td><td>{esc(r.get('congresses'))}</td><td>{md(r.get('notes', ''))}</td></tr>"
                   for r in rec.get("split_table", []))
    table = f'<table class="split"><thead><tr><th>Role</th><th>Congresses</th><th>Notes</th></tr></thead><tbody>{rows}</tbody></table>'
    per_task = "".join(f'<div class="task"><h4>{esc(t.get("task"))}</h4>{md(t.get("recommendation", ""))}'
                       + (f'<p class="reporting"><b>Reporting:</b> {md(t.get("reporting", ""))}</p>' if t.get("reporting") else "")
                       + "</div>" for t in syn.get("per_task", []))
    if short:
        compact = "".join(f"<tr><td>{esc(r.get('role'))}</td><td>{esc(r.get('congresses'))}</td></tr>"
                          for r in rec.get("split_table", []))
        return (f'<div class="bottomline"><div class="bl-label">Bottom line</div><div class="bl-head">{md(rec.get("headline", ""))}</div>'
                f'{fold_diagram()}<table class="split"><thead><tr><th>Role</th><th>Congresses</th></tr></thead><tbody>{compact}</tbody></table>'
                f'<p class="bl-more"><a href="#recommendation">Full recommendation, protocol and justification &darr;</a></p></div>')
    just = "".join(f'<li>{md(j.get("reason", ""))}<div class="evs">{evidence_links(j.get("evidence"), item_ids)}</div></li>'
                   for j in rec.get("justification", []))
    alts = "".join(f'<tr><td>{md(a.get("option", ""))}</td><td>{md(a.get("reason", ""))}<div class="evs">{evidence_links(a.get("evidence"), item_ids)}</div></td></tr>'
                   for a in rec.get("alternatives_rejected", []))
    risks = "".join(f'<tr><td>{md(r.get("risk", ""))}</td><td>{md(r.get("mitigation", ""))}</td></tr>' for r in rec.get("risks_and_mitigations", []))
    proto = "".join(f"<li>{md(p)}</li>" for p in rec.get("protocol", []))
    change = "".join(f"<li>{md(c)}</li>" for c in rec.get("what_would_change_this", []))
    return f"""<section id="recommendation"><h2>Recommended train / validation / test split</h2>
<div class="rec-head">{md(rec.get('headline', ''))}</div>
{md(rec.get('plain_explanation', ''))}
<h3>The split</h3>{fold_diagram()}{table}
<h3>Per task</h3><div class="tasks">{per_task}</div>
<h3>Evaluation protocol</h3><ol class="protocol">{proto}</ol>
<h3>Why this split</h3><ul class="just">{just}</ul>
<h3>Alternatives considered and rejected</h3><table class="alts"><thead><tr><th>Option</th><th>Why not</th></tr></thead><tbody>{alts}</tbody></table>
<h3>Risks and mitigations</h3><table class="risks"><thead><tr><th>Risk</th><th>Mitigation</th></tr></thead><tbody>{risks}</tbody></table>
<h3>What would change this recommendation</h3><ul>{change}</ul></section>"""


CSS = """
:root { --ink:#0b0b0b; --ink2:#52514e; --muted:#7a7975; --line:#e4e3df; --bg:#fcfcfb; --accent:#2a78d6; --soft:#f3f6fb; --warn:#fff7e6; }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--ink); font:15px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }
main { max-width: 1120px; margin: 0 auto; padding: 32px 28px 80px; }
h1 { font-size: 30px; line-height:1.2; margin: 0 0 6px; }
h2 { font-size: 23px; margin: 56px 0 12px; padding-top: 12px; border-top: 2px solid var(--line); }
h3 { font-size: 18px; margin: 28px 0 10px; }
h4 { font-size: 15px; margin: 14px 0 6px; }
a { color: var(--accent); }
.meta { color: var(--ink2); font-size: 13px; margin-bottom: 24px; }
.toc { background:#fff; border:1px solid var(--line); border-radius:8px; padding:12px 20px; margin: 20px 0; }
.toc ol { margin: 6px 0; padding-left: 20px; } .toc li { margin: 2px 0; }
.bottomline { background: var(--soft); border-left: 4px solid var(--accent); border-radius: 6px; padding: 16px 22px; margin: 22px 0; }
.bl-label, .interp-label, .nbmd-label { font-size: 11px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--ink2); }
.bl-head { font-size: 18px; font-weight: 600; margin: 6px 0 12px; }
.rec-head { font-size: 19px; font-weight: 600; background: var(--soft); border-left: 4px solid var(--accent); padding: 12px 18px; border-radius: 6px; margin: 10px 0 16px; }
.headline { font-size: 17px; font-weight: 600; margin: 4px 0 12px; }
.overview p { margin: 0 0 10px; }
.implications, .openq { background:#fff; border:1px solid var(--line); border-radius:8px; padding: 6px 20px 8px; margin: 14px 0; }
.detail-h { color: var(--ink2); }
.nbmd { color: var(--ink2); font-size: 14px; border-left: 3px solid var(--line); padding: 2px 0 2px 14px; margin: 26px 0 10px; }
.nbmd p { margin: 6px 0; }
.result { background:#fff; border:1px solid var(--line); border-radius:8px; padding: 12px 16px 6px; margin: 12px 0 18px; position: relative; }
.rid { position:absolute; top:6px; right:10px; font: 11px ui-monospace, Menlo, monospace; color: var(--muted); }
pre.out { font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, monospace; background:#f7f7f5; border:1px solid var(--line); border-radius:6px; padding:10px 12px; overflow:auto; max-height: 460px; white-space: pre; }
.table-wrap { overflow:auto; max-height: 460px; }
.table-wrap table, table.split, table.alts, table.risks { border-collapse: collapse; font-size: 13px; width: 100%; }
.table-wrap th, .table-wrap td, table.split th, table.split td, table.alts th, table.alts td, table.risks th, table.risks td { border-bottom: 1px solid var(--line); padding: 5px 8px; text-align: left; vertical-align: top; }
table.split th, table.alts th, table.risks th { background: #f7f7f5; }
.chart-wrap { overflow-x:auto; padding: 4px 0; }
.interp { background: var(--soft); border-radius: 6px; padding: 10px 14px; margin: 10px 0 8px; font-size: 14.5px; }
.interp p { margin: 4px 0 8px; }
.caveat { background: var(--warn); border-radius: 4px; padding: 6px 10px; margin-top: 6px; font-size: 13.5px; }
.caveat p { display:inline; margin:0; }
.findings li { margin: 8px 0; }
.ev { display:inline-block; font: 11px ui-monospace, Menlo, monospace; background:#eef2f8; color: var(--ink2); border-radius: 3px; padding: 0 5px; margin: 2px 3px 0 0; text-decoration:none; }
.evs { margin-top: 2px; }
.tasks { display:grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.task { background:#fff; border:1px solid var(--line); border-radius:8px; padding: 4px 16px 10px; }
.protocol li, .just li { margin: 6px 0; }
.glossary dt { font-weight: 600; margin-top: 8px; } .glossary dd { margin: 2px 0 6px 0; color: var(--ink2); }
pre.err { color: #c00; }
.folds { margin: 12px 0 14px; overflow-x: auto; }
.folds table { border-collapse: separate; border-spacing: 2px; font-size: 11.5px; }
.folds th { font-weight: 600; color: var(--ink2); padding: 2px 6px; text-align: right; white-space: nowrap; }
.folds thead th, .folds tr:first-child th { text-align: center; }
.folds td { min-width: 58px; height: 24px; text-align: center; border-radius: 3px; background: #f7f7f5; }
.fold-legend { margin: 6px 0; } .lg { display:inline-block; font-size: 11px; border-radius: 3px; padding: 1px 7px; margin-right: 4px; }
.conclusion { background:#fff; border:2px solid var(--accent); border-radius:10px; padding: 4px 26px 18px; margin-top: 56px; }
.conclusion h2 { border-top: none; margin-top: 18px; }
.conclusion blockquote { border-left: 4px solid var(--accent); margin: 10px 0; padding: 6px 16px; background: var(--soft); }
.conclusion table:not(.folds table) { border-collapse: collapse; font-size: 14px; margin: 8px 0 14px; }
.conclusion th, .conclusion td { border-bottom: 1px solid var(--line); padding: 5px 10px; text-align: left; }
.fold-note { font-size: 13px; color: var(--ink2); margin: 4px 0 10px; }
@media (max-width: 800px) { .tasks { grid-template-columns: 1fr; } }
"""


def main():
    data = load_json("report_data.json")
    syn = load_json("synthesis.json", {})
    interps = {k: load_json(f"interp_{k}.json") for k in SECTION_ORDER}
    sections = {s["key"]: s for s in data["sections"]}
    item_ids = {it["id"] for s in data["sections"] for it in s["items"]}
    CHECK_ANCHORS.update(check_anchor_map(data))
    missing = [k for k, v in interps.items() if v is None]

    title = syn.get("title") or "Choosing a train / validation / test split for the Congress graph tasks"
    toc = "".join(f'<li><a href="#sec-{k}">{esc(sections[k]["title"])}</a></li>' for k in SECTION_ORDER if k in sections)
    findings = "".join(f'<li>{md(f.get("finding", ""))}<div class="evs">{evidence_links(f.get("evidence"), item_ids)}</div></li>'
                       for f in syn.get("key_findings", []))
    gloss = "".join(f"<dt>{esc(g.get('term'))}</dt><dd>{md(g.get('definition', ''))}</dd>" for g in syn.get("glossary", []))
    body = [
        f"<h1>{esc(title)}</h1>",
        f'<div class="meta">CS224W project &middot; Congress.gov / GovInfo data, Congresses 108&ndash;118 &middot; '
        f'generated {dt.date.today().isoformat()} from <code>notebooks/01_split_analysis.ipynb</code> outputs</div>',
        recommendation_html(syn, item_ids, short=True) if syn else "",
        '<h2 id="summary">Executive summary</h2>' + md(syn.get("executive_summary", "")) if syn else "",
        ('<h3>Key findings</h3><ol class="findings">' + findings + "</ol>") if findings else "",
        f'<nav class="toc"><b>Contents</b><ol><li><a href="#summary">Executive summary</a></li>{toc}'
        '<li><a href="#recommendation">Recommended split</a></li><li><a href="#method">Method and glossary</a></li><li><a href="#conclusion"><b>Conclusion: the split we will use</b></a></li></ol></nav>',
    ]
    if "intro" in sections:
        intro_md = next((i["text"] for i in sections["intro"]["items"] if i["type"] == "markdown"), "")
        body.append('<h2 id="sec-intro">The question and the tasks</h2>' + md("\n".join(intro_md.splitlines()[1:])))
    for k in SECTION_ORDER:
        if k in sections:
            body.append(section_html(sections[k], interps.get(k), item_ids))
    if syn:
        body.append(recommendation_html(syn, item_ids))
    body.append('<section id="method"><h2>Method and glossary</h2>' + md(syn.get("methods_note", ""))
                + (f'<h3>Glossary</h3><dl class="glossary">{gloss}</dl>' if gloss else "") + "</section>")
    body.append(conclusion_html(item_ids))

    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>{esc(title)}</title>
<style>{CSS}</style><script>{D3.read_text()}</script></head>
<body><main>{''.join(body)}</main></body></html>"""
    OUT.write_text(page)
    n_charts = sum(1 for s in data["sections"] for i in s["items"] if i["type"] == "chart")
    covered = sum(len((interps.get(k) or {}).get("items", {})) for k in SECTION_ORDER)
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB): {n_charts} charts, {covered} interpreted items"
          + (f"; MISSING interpretations: {missing}" if missing else "") + ("" if syn else "; no synthesis.json yet"))


if __name__ == "__main__":
    main()
