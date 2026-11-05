"""Folder scanning and file reading for the supported document types."""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Iterable, Iterator

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".rst", ".pdf", ".docx", ".html", ".htm"}
IGNORED_DIRS = {"node_modules", "__pycache__", "venv", ".venv"}  # hidden dirs (.git etc.) are skipped too


def scan_folders(folders: Iterable[str | Path],
                 extensions: set[str] = SUPPORTED_EXTENSIONS) -> Iterator[Path]:
    """Recursively yield every supported file under the given folders (deduplicated)."""
    seen: set[Path] = set()
    for folder in folders:
        root = Path(folder).expanduser().resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Not a folder: {root}")
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = sorted(d for d in dirnames
                                 if d not in IGNORED_DIRS and not d.startswith("."))
            for name in sorted(filenames):
                path = Path(dirpath) / name
                if path.suffix.lower() in extensions and path not in seen:
                    seen.add(path)
                    yield path


def read_file(path: str | Path) -> str:
    """Extract plain text from a supported file."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _read_pdf(path)
    if ext == ".docx":
        return _read_docx(path)
    if ext in {".html", ".htm"}:
        return _read_html(path)
    return _read_text(path)


def _read_text(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return raw.decode(enc)
        except UnicodeError:
            continue
    return raw.decode("latin-1", errors="replace")


def _read_pdf(path: Path) -> str:
    from pypdf import PdfReader  # lazy import: only needed for PDFs

    reader = PdfReader(str(path))
    return "\n\n".join((page.extract_text() or "") for page in reader.pages)


def _read_docx(path: Path) -> str:
    import docx  # python-docx

    document = docx.Document(str(path))
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            parts.append(" | ".join(cell.text.strip() for cell in row.cells))
    return "\n\n".join(parts)


def _read_html(path: Path) -> str:
    html = _read_text(path)
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
        return soup.get_text("\n")
    except ImportError:  # crude fallback
        return re.sub(r"<[^>]+>", " ", html)
