#!/usr/bin/env python
"""T32 — Render ``docs/report.md`` to ``docs/report.pdf``.

Uses the pure-Python ``markdown-pdf`` package so the build has zero
system dependencies (no pandoc, no LaTeX, no wkhtmltopdf). The output
PDF is committed alongside the markdown so a reader who clones the
repo already has the static artefact — this script only needs to run
when the markdown changes.

Usage
-----
::

    python scripts/build_report_pdf.py
    # or, via the Makefile:
    make report-pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
INPUT = REPO_ROOT / "docs/report.md"
OUTPUT = REPO_ROOT / "docs/report.pdf"


def main() -> int:
    try:
        from markdown_pdf import MarkdownPdf, Section  # noqa: PLC0415
    except ImportError as e:
        raise SystemExit(
            "markdown-pdf not installed. Run `pip install markdown-pdf` "
            "(it is not in the core dependency list because the PDF is "
            "already committed to the repo)."
        ) from e

    if not INPUT.exists():
        raise SystemExit(f"missing {INPUT}")

    text = INPUT.read_text(encoding="utf-8")
    # Drop the YAML frontmatter — markdown-pdf does not parse it and
    # leaves the fence markers visible in the output. We rely on the
    # first ``# <title>`` line for the PDF title instead.
    if text.startswith("---\n"):
        _, _, body = text.partition("---\n")[2].partition("---\n")
        text = body.lstrip()
    else:
        body = text

    pdf = MarkdownPdf(toc_level=2)
    pdf.meta["title"] = "GreekKV v0.1.0 technical report"
    pdf.meta["author"] = "Giorgos Mponobas"
    pdf.add_section(Section(text))
    pdf.save(str(OUTPUT))

    size_kb = OUTPUT.stat().st_size / 1024
    print(f"Wrote {OUTPUT.relative_to(REPO_ROOT)}  ({size_kb:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
