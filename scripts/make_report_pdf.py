"""REPORT.md -> REPORT.pdf (markdown-it + headless Chrome; no network)."""

import base64
import re
import subprocess
import tempfile
from pathlib import Path

from markdown_it import MarkdownIt

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
root = Path(__file__).resolve().parent.parent
md = (root / "REPORT.md").read_text()
html = MarkdownIt("commonmark").enable("table").render(md)


def inline(m):
    data = base64.b64encode((root / m.group(1)).read_bytes()).decode()
    return f'src="data:image/png;base64,{data}"'


html = re.sub(r'src="(reports/figures/[^"]+)"', inline, html)
css = """body{font-family:Helvetica,Arial,sans-serif;font-size:10.5pt;line-height:1.4;margin:0}
h1{font-size:20pt}h2{font-size:15pt;margin-top:18px}h3{font-size:12pt}
table{border-collapse:collapse;font-size:8.5pt;margin:8px 0;width:100%}th,td{border:1px solid #bbb;padding:3px 5px;text-align:left}
th{background:#eee}img{max-width:80%;display:block;margin:8px auto}code{font-size:9pt}"""
with tempfile.TemporaryDirectory() as t:
    f = Path(t) / "r.html"
    f.write_text(f"<!doctype html><meta charset=utf-8><style>{css}</style>{html}")
    subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-pdf-header-footer",
            f"--print-to-pdf={root / 'REPORT.pdf'}",
            f.as_uri(),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
print("wrote REPORT.pdf")
