"""Extract every output of the executed notebook into structured data for the HTML report.

Reads notebooks/01_split_analysis.ipynb (as saved after running it in JupyterLab) and writes:
  tools/report/_work/report_data.json   ordered sections -> items (markdown, text, table, chart)
  tools/report/_work/digest_<sec>.md    readable per-section digest (for writing interpretations)

Each item has a stable id like "B-c12-o1" (section, notebook cell index, output index).
Usage: python tools/report/export_notebook.py
"""
import html
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NOTEBOOK = REPO / "notebooks" / "01_split_analysis.ipynb"
WORK = Path(__file__).resolve().parent / "_work"

SECTION_RE = re.compile(r"^##\s+(?:([A-D])\.|(Decision))", re.M)
CHART_RE = re.compile(r"const payload = (\{.*?\});\s*const draw = (function.*?);\s*function go\(\)", re.S)


def src(cell):
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def text_of(o):
    t = o.get("text") or o.get("data", {}).get("text/plain", "")
    return "".join(t) if isinstance(t, list) else t


def html_table_to_text(h, max_rows=40):
    """Rough text rendering of a pandas HTML table for the digest."""
    rows = re.findall(r"<tr.*?>(.*?)</tr>", h, re.S)
    out = []
    for r in rows[: max_rows + 1]:
        cells = [html.unescape(re.sub(r"<.*?>", "", c)).strip() for c in re.findall(r"<t[hd].*?>(.*?)</t[hd]>", r, re.S)]
        out.append(" | ".join(cells))
    if len(rows) > max_rows + 1:
        out.append(f"... ({len(rows) - max_rows - 1} more rows)")
    return "\n".join(out)


def chart_digest(payload, max_rows=60):
    o = payload.get("opts", {})
    data = payload.get("data")
    lines = [f"title: {o.get('title', '')}", f"subtitle: {o.get('subtitle', '')}"]
    keys = {k: v for k, v in o.items() if k in ("x", "y", "v", "series", "yfmt", "fmt", "ylabel", "xlabel")}
    if keys:
        lines.append(f"encoding: {keys}")
    if isinstance(data, list) and data and isinstance(data[0], dict):
        cols = list(data[0].keys())
        lines.append("data (" + str(len(data)) + " rows): " + " | ".join(cols))
        for r in data[:max_rows]:
            lines.append("  " + " | ".join(str(r.get(c)) for c in cols))
        if len(data) > max_rows:
            lines.append(f"  ... ({len(data) - max_rows} more rows)")
    else:
        lines.append("data: " + json.dumps(data)[:3000])
    return "\n".join(lines)


def main():
    nb = json.loads(NOTEBOOK.read_text())
    WORK.mkdir(parents=True, exist_ok=True)
    sections, cur = [], {"key": "intro", "title": "Introduction and setup", "items": []}
    n_charts = 0
    for ci, cell in enumerate(nb["cells"]):
        s = src(cell)
        if cell["cell_type"] == "markdown":
            m = SECTION_RE.search(s)
            if m and s.lstrip().startswith("##"):
                sections.append(cur)
                key = m.group(1) or "decision"
                cur = {"key": key, "title": s.strip().splitlines()[0].lstrip("# ").strip(), "items": []}
            cur["items"].append({"id": f"{cur['key']}-c{ci}-md", "type": "markdown", "text": s})
            continue
        texts = []
        for oi, o in enumerate(cell.get("outputs", [])):
            if o["output_type"] == "stream":
                texts.append(text_of(o))
                continue
            data = o.get("data", {})
            h = data.get("text/html")
            h = "".join(h) if isinstance(h, list) else h
            if h and "const payload" in h:
                m = CHART_RE.search(h)
                if not m:
                    continue
                payload = json.loads(m.group(1))
                n_charts += 1
                cur["items"].append({"id": f"{cur['key']}-c{ci}-o{oi}", "type": "chart", "payload": payload,
                                     "func": m.group(2), "title": payload["opts"].get("title", "")})
            elif h:
                cur["items"].append({"id": f"{cur['key']}-c{ci}-o{oi}", "type": "table", "html": h})
            elif o["output_type"] == "error":
                texts.append("ERROR: " + o.get("ename", "") + ": " + o.get("evalue", ""))
            else:
                t = text_of(o)
                if t:
                    texts.append(t)
        joined = "".join(texts).strip()
        if joined:
            # Charts printed placeholders nowhere in the kernel run; text goes before the cell's charts.
            item = {"id": f"{cur['key']}-c{ci}-txt", "type": "text", "text": joined, "code_comment": s.splitlines()[0] if s else ""}
            idx = next((i for i, it in enumerate(cur["items"]) if it["id"].startswith(f"{cur['key']}-c{ci}-o")), None)
            cur["items"].insert(idx if idx is not None else len(cur["items"]), item)
    sections.append(cur)
    (WORK / "report_data.json").write_text(json.dumps({"sections": sections}, indent=1))

    for sec in sections:
        out = [f"# Digest: {sec['title']} (section key {sec['key']})", ""]
        for it in sec["items"]:
            out.append(f"\n---\n## [{it['id']}] {it['type']}")
            if it["type"] == "markdown":
                out.append(it["text"])
            elif it["type"] == "text":
                out.append("```\n" + it["text"] + "\n```")
            elif it["type"] == "table":
                out.append("```\n" + html_table_to_text(it["html"]) + "\n```")
            elif it["type"] == "chart":
                out.append("```\n" + chart_digest(it["payload"]) + "\n```")
        (WORK / f"digest_{sec['key']}.md").write_text("\n".join(out))
    print(f"sections: {[ (s['key'], len(s['items'])) for s in sections ]}; charts: {n_charts}")
    for sec in sections:
        print(f"  {sec['key']}: " + ", ".join(f"{t}={sum(1 for i in sec['items'] if i['type'] == t)}"
                                              for t in ("markdown", "text", "table", "chart")))


if __name__ == "__main__":
    main()
