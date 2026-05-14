"""Export every notebook and script to MD, HTML, and PDF in code/exports/.

Notebooks (code/notebooks/*.ipynb) -> notebook_<name>.{md,html,pdf}
Scripts   (code/scripts/*.py)      -> script_<name>.{md,html,pdf}

Run:  python code/scripts/export_all.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE_DIR = HERE.parent
NB_DIR = CODE_DIR / "notebooks"
PY_DIR = CODE_DIR / "scripts"
OUT_DIR = CODE_DIR / "exports"
OUT_DIR.mkdir(parents=True, exist_ok=True)

THIS_FILE = Path(__file__).name


def run(cmd):
    print(">>", " ".join(str(c) for c in cmd))
    return subprocess.run(cmd)


def html_to_pdf(html_path: Path, pdf_path: Path) -> bool:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(f"   ! playwright not installed; skipping {pdf_path.name}")
        return False
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(html_path.absolute().as_uri(), wait_until="networkidle")
            page.pdf(
                path=str(pdf_path),
                format="A4",
                print_background=True,
                margin={"top": "10mm", "bottom": "10mm",
                        "left": "10mm", "right": "10mm"},
            )
            browser.close()
        return True
    except Exception as e:
        print(f"   ! playwright PDF failed for {pdf_path.name}: {e}")
        return False


def export_notebook(nb: Path):
    out_base = f"notebook_{nb.stem}"
    run([sys.executable, "-m", "nbconvert", "--to", "markdown",
         "--output", out_base, "--output-dir", str(OUT_DIR), str(nb)])
    run([sys.executable, "-m", "nbconvert", "--to", "html", "--embed-images",
         "--output", out_base, "--output-dir", str(OUT_DIR), str(nb)])
    html_to_pdf(OUT_DIR / f"{out_base}.html", OUT_DIR / f"{out_base}.pdf")


def export_script(py: Path):
    out_base = f"script_{py.stem}"
    source = py.read_text(encoding="utf-8")

    md_path = OUT_DIR / f"{out_base}.md"
    md_path.write_text(
        f"# `{py.name}`\n\n```python\n{source}\n```\n", encoding="utf-8"
    )

    html_path = OUT_DIR / f"{out_base}.html"
    try:
        from pygments import highlight
        from pygments.lexers import PythonLexer
        from pygments.formatters import HtmlFormatter
        formatter = HtmlFormatter(full=True, linenos=True,
                                  style="default", title=py.name)
        html_path.write_text(
            highlight(source, PythonLexer(), formatter), encoding="utf-8"
        )
    except ImportError:
        html_path.write_text(
            f"<!doctype html><html><body><h1>{py.name}</h1>"
            f"<pre>{source}</pre></body></html>",
            encoding="utf-8",
        )

    html_to_pdf(html_path, OUT_DIR / f"{out_base}.pdf")


def main():
    notebooks = sorted(
        p for p in NB_DIR.glob("*.ipynb")
        if ".ipynb_checkpoints" not in str(p)
    )
    scripts = sorted(
        p for p in PY_DIR.glob("*.py")
        if p.name not in {"__init__.py", THIS_FILE}
    )

    for nb in notebooks:
        print(f"\n== {nb.name}")
        export_notebook(nb)

    for py in scripts:
        print(f"\n== {py.name}")
        export_script(py)

    print(f"\nDone. {len(notebooks)} notebooks + {len(scripts)} scripts -> {OUT_DIR}")


if __name__ == "__main__":
    main()
