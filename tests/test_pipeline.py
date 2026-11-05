"""Pipeline tests using a lightweight fake embedder (no model download needed)."""
import hashlib
import re
import shutil
from pathlib import Path

import numpy as np
import pytest

from knowledge_search import KnowledgeBase
from knowledge_search.chunker import chunk_text

SAMPLE = Path(__file__).parent.parent / "sample_data"


class FakeEmbedder:
    """Hashed bag-of-words with crude stemming: deterministic stand-in for MiniLM."""
    model_name = "fake"

    def encode(self, texts):
        out = np.zeros((len(texts), 256), dtype="float32")
        for row, text in enumerate(texts):
            for w in re.findall(r"[a-z]+", text.lower()):
                out[row, int(hashlib.md5(w[:5].encode()).hexdigest(), 16) % 256] += 1
        out /= np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)
        return out


@pytest.fixture
def kb(tmp_path):
    return KnowledgeBase(tmp_path / "idx", embedder=FakeEmbedder())


def test_chunker_overlap_and_limits():
    text = " ".join(f"w{i}" for i in range(400))
    chunks = chunk_text(text, chunk_words=150, overlap_words=30)
    assert len(chunks) >= 3 and all(len(c.split()) <= 150 for c in chunks)
    assert chunks[0].split()[-30:] == chunks[1].split()[:30]
    assert chunk_text("   ") == []


def test_index_and_search(kb):
    stats = kb.sync([SAMPLE])
    assert stats["added"] == 6 and stats["skipped"] == 0
    top = kb.search("bedtime routine for insomnia", k=3)
    assert top[0].path.endswith("sleep_routine.md")
    assert kb.search("decorators wrap functions", k=1)[0].path.endswith("learning_python.md")
    assert kb.search("   ") == []


def test_incremental_update_remove_and_persistence(kb, tmp_path):
    data = tmp_path / "notes"
    shutil.copytree(SAMPLE, data)
    kb.sync([data])
    assert kb.sync([data])["unchanged"] == 6          # nothing re-embedded

    (data / "journal" / "running_plan.txt").write_text("Totally new text about volcano geology.")
    (data / "articles" / "compound_interest.txt").unlink()
    s = kb.sync([data])
    assert s["updated"] == 1 and s["removed"] == 1

    reloaded = KnowledgeBase(tmp_path / "idx", embedder=FakeEmbedder())   # from disk
    assert reloaded.stats()["files"] == 5
    assert reloaded.search("volcano geology", k=1)[0].path.endswith("running_plan.txt")
    assert all("compound_interest" not in r.path for r in reloaded.search("investing money", k=5))
