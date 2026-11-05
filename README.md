# Personal Knowledge Search

Search your notes, articles and documents **by meaning**, not exact keywords.
Searching *"how do I sleep better"* finds a note titled *"Insomnia and my bedtime routine"*
even though they share no keywords.

| Component        | Choice                                   |
|------------------|------------------------------------------|
| Embedding model  | `all-MiniLM-L6-v2` (384-dim, runs on CPU) |
| Vector database  | FAISS (`IndexIDMap` over `IndexFlatIP`)   |
| File formats     | `.txt .md .rst .pdf .docx .html`          |

## Quick start

```bash
python -m venv venv && source venv/bin/activate     # Windows: venv\Scripts\activate
pip install -r requirements.txt

python main.py index ./sample_data                  # first run downloads the model (~90 MB)
python main.py search "how do I sleep better"
python main.py search "putting money away for retirement" -k 3
python main.py chat                                 # interactive mode
python main.py stats
```

Point `index` at your real folders: `python main.py index ~/Documents/notes ~/articles`.
Re-run it any time: only new, changed, or deleted files are processed.

## How it works

```
folders ──► scan_folders ──► read_file ──► chunk_text ──► MiniLM embeddings ──► FAISS index
                                                                                    ▲
query ───────────────────────────────────────► MiniLM embedding ──► top-k cosine ───┘
```

1. **Folder scanning** (`file_reader.py`): recursive walk; skips hidden folders, `venv`, `node_modules`.
2. **File reading**: one extractor per format; unreadable files are skipped with a warning.
3. **Chunking** (`chunker.py`): paragraph-aware ~150-word chunks with 30-word overlap, because
   MiniLM only reads the first 256 tokens of its input.
4. **Embedding** (`embedder.py`): vectors are L2-normalised so inner product equals cosine similarity.
5. **Index** (`indexer.py`): FAISS holds vectors; `.knowledge_index/meta.json` holds chunk text and a
   manifest (mtime, size, content hash) for incremental sync. `IndexIDMap` lets us delete a changed
   file's old vectors.
6. **Search**: embed the query, take top-k nearest chunks, return the best chunk per file.

## Project layout

```
main.py                    CLI (index / search / chat / stats)
knowledge_search/
  file_reader.py           scanning + text extraction
  chunker.py               overlapping chunks
  embedder.py              all-MiniLM-L6-v2 wrapper
  indexer.py               FAISS index, incremental sync, search
tests/test_pipeline.py     tests (use a fake embedder, no download needed)
sample_data/               demo notes
```

## Use from Python

```python
from knowledge_search import KnowledgeBase
kb = KnowledgeBase(".knowledge_index")
kb.sync(["./sample_data"])
for r in kb.search("fixing slow database queries", k=3):
    print(f"{r.score:.2f}  {r.path}\n   {r.text[:100]}")
```

## Tuning and extensions

- **Scores**: cosine similarity, roughly 0.3+ is usually a real match; use `--min-score 0.3` to cut noise.
- **Scale**: `IndexFlatIP` is exact and fine up to ~100k chunks. Beyond that, switch to `IndexHNSWFlat`
  or `IndexIVFFlat` in `_add_chunks`.
- **Ideas**: Streamlit/web UI, watch mode, hybrid keyword + semantic search, a larger model such as
  `all-mpnet-base-v2` (delete the index folder after changing models), RAG answers with an LLM.

## Testing

```bash
pytest -q
```
