"""Write deep-research Markdown reports to PDF via Qt."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtGui import QTextDocument


def write_research_report_pdf(
    report_markdown: str,
    dest_path: Path,
    *,
    document_stylesheet: str = "",
) -> Path:
    """Render Markdown to PDF using QTextDocument.printToPdf."""
    return write_markdown_pdf(
        report_markdown,
        dest_path,
        document_stylesheet=document_stylesheet,
    )


def write_markdown_pdf(
    markdown: str,
    dest_path: Path,
    *,
    document_stylesheet: str = "",
) -> Path:
    """Render Markdown to PDF using QTextDocument.printToPdf."""
    dest = Path(dest_path)
    if dest.suffix.lower() != ".pdf":
        dest = dest.with_suffix(".pdf")
    dest.parent.mkdir(parents=True, exist_ok=True)

    doc = QTextDocument()
    if document_stylesheet:
        doc.setDefaultStyleSheet(document_stylesheet)
    doc.setMarkdown((markdown or "").strip())
    doc.printToPdf(str(dest))
    return dest
