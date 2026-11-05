"""FAISS-backed knowledge base with incremental sync and semantic search."""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .chunker import chunk_text
from .embedder import MODEL_NAME, SentenceTransformerEmbedder
from .file_reader import read_file, scan_folders

DEFAULT_INDEX_DIR = ".knowledge_index"


@dataclass
class SearchResult:
    score: float        # cosine similarity, higher = more relevant
    path: str
    chunk_index: int
    text: str


class KnowledgeBase:
    """Stores chunk embeddings in FAISS (IndexIDMap over IndexFlatIP) plus JSON metadata."""

    def __init__(self, index_dir: str | Path = DEFAULT_INDEX_DIR,
                 embedder=None, chunk_words: int = 150, overlap_words: int = 30):
        self.index_dir = Path(index_dir)
        self.embedder = embedder or SentenceTransformerEmbedder()
        self.chunk_words = chunk_words
        self.overlap_words = overlap_words
        self.index = None                       # faiss index, created on first add
        self.chunks: dict[str, dict] = {}       # chunk id -> {path, chunk_index, text}
        self.manifest: dict[str, dict] = {}     # file path -> {mtime, size, hash, ids}
        self.next_id = 0
        self._load()

    # ------------------------------------------------------------------ persistence
    @property
    def _index_file(self) -> Path:
        return self.index_dir / "index.faiss"

    @property
    def _meta_file(self) -> Path:
        return self.index_dir / "meta.json"

    def _load(self) -> None:
        if not (self._index_file.exists() and self._meta_file.exists()):
            return
        import faiss

        meta = json.loads(self._meta_file.read_text(encoding="utf-8"))
        saved_model = meta.get("model")
        current_model = getattr(self.embedder, "model_name", saved_model)
        if saved_model != current_model:
            raise RuntimeError(f"Index was built with {saved_model!r} but {current_model!r} is "
                               "configured. Delete the index folder and re-index.")
        self.index = faiss.read_index(str(self._index_file))
        self.chunks = meta["chunks"]
        self.manifest = meta["manifest"]
        self.next_id = meta["next_id"]

    def save(self) -> None:
        import faiss

        self.index_dir.mkdir(parents=True, exist_ok=True)
        if self.index is not None:
            faiss.write_index(self.index, str(self._index_file))
        meta = {"model": getattr(self.embedder, "model_name", MODEL_NAME),
                "next_id": self.next_id, "chunks": self.chunks, "manifest": self.manifest}
        self._meta_file.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------------ indexing
    def sync(self, folders: Iterable[str | Path]) -> dict:
        """Bring the index in line with the folders: add new, re-embed changed, drop deleted."""
        roots = [Path(f).expanduser().resolve() for f in folders]
        current = {str(p): p for p in scan_folders(roots)}
        stats = {"added": 0, "updated": 0, "removed": 0, "unchanged": 0,
                 "skipped": 0, "chunks": 0}

        for path in list(self.manifest):
            if path not in current and any(_is_under(Path(path), r) for r in roots):
                self._remove_file(path)
                stats["removed"] += 1

        for path_str, path in current.items():
            st = path.stat()
            entry = self.manifest.get(path_str)
            if entry and entry["mtime"] == st.st_mtime and entry["size"] == st.st_size:
                stats["unchanged"] += 1
                continue
            try:
                text = read_file(path)
            except Exception as exc:  # unreadable / corrupt file: skip, don't crash
                print(f"  ! skipped {path}: {exc}", file=sys.stderr)
                stats["skipped"] += 1
                continue

            digest = hashlib.sha1(text.encode("utf-8", "ignore")).hexdigest()
            if entry and entry["hash"] == digest:      # touched but content identical
                entry.update(mtime=st.st_mtime, size=st.st_size)
                stats["unchanged"] += 1
                continue

            if entry:
                self._remove_file(path_str)
                stats["updated"] += 1
            else:
                stats["added"] += 1
            pieces = chunk_text(text, self.chunk_words, self.overlap_words)
            self._add_chunks(path_str, pieces, st, digest)
            stats["chunks"] += len(pieces)

        self.save()
        return stats

    def _add_chunks(self, path: str, pieces: list[str], st, digest: str) -> None:
        ids: list[int] = []
        if pieces:
            import faiss

            vectors = self.embedder.encode(pieces)
            if self.index is None:
                self.index = faiss.IndexIDMap(faiss.IndexFlatIP(vectors.shape[1]))
            ids = list(range(self.next_id, self.next_id + len(pieces)))
            self.next_id += len(pieces)
            self.index.add_with_ids(vectors, np.array(ids, dtype="int64"))
            for i, (cid, piece) in enumerate(zip(ids, pieces)):
                self.chunks[str(cid)] = {"path": path, "chunk_index": i, "text": piece}
        self.manifest[path] = {"mtime": st.st_mtime, "size": st.st_size, "hash": digest, "ids": ids}

    def _remove_file(self, path: str) -> None:
        ids = self.manifest.pop(path, {}).get("ids", [])
        if ids and self.index is not None:
            self.index.remove_ids(np.array(ids, dtype="int64"))
        for cid in ids:
            self.chunks.pop(str(cid), None)

    # ------------------------------------------------------------------ searching
    def search(self, query: str, k: int = 5, min_score: float = 0.0,
               unique_files: bool = True) -> list[SearchResult]:
        """Return the k most semantically similar chunks (best chunk per file by default)."""
        if not query.strip() or self.index is None or self.index.ntotal == 0:
            return []
        q = self.embedder.encode([query])
        fetch = min(self.index.ntotal, k * 5 if unique_files else k)
        scores, ids = self.index.search(q, fetch)

        results: list[SearchResult] = []
        seen: set[str] = set()
        for score, cid in zip(scores[0], ids[0]):
            if cid == -1 or score < min_score:
                continue
            chunk = self.chunks[str(int(cid))]
            if unique_files and chunk["path"] in seen:
                continue
            seen.add(chunk["path"])
            results.append(SearchResult(float(score), chunk["path"], chunk["chunk_index"], chunk["text"]))
            if len(results) == k:
                break
        return results

    def stats(self) -> dict:
        return {"files": len(self.manifest), "chunks": len(self.chunks),
                "vectors": 0 if self.index is None else self.index.ntotal,
                "model": getattr(self.embedder, "model_name", MODEL_NAME),
                "index_dir": str(self.index_dir)}


def _is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
