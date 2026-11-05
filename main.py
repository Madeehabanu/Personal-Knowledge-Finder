"""Command-line interface for Personal Knowledge Search.

  python main.py index ./my_notes ./articles     # build / update the index
  python main.py search "how do I sleep better"  # one-off query
  python main.py chat                            # interactive search loop
  python main.py stats                           # index summary
"""
from __future__ import annotations

import argparse
import sys
import textwrap

from knowledge_search.indexer import DEFAULT_INDEX_DIR, KnowledgeBase


def snippet(text: str, width: int = 280) -> str:
    text = " ".join(text.split())
    return textwrap.fill(text[:width] + ("…" if len(text) > width else ""),
                         width=96, initial_indent="     ", subsequent_indent="     ")


def show(results) -> None:
    if not results:
        print("No matches found.")
        return
    for rank, r in enumerate(results, 1):
        print(f"\n{rank}. [{r.score:.3f}] {r.path}  (chunk {r.chunk_index})")
        print(snippet(r.text))
    print()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Search your notes by meaning, not keywords.")
    parser.add_argument("--index-dir", default=DEFAULT_INDEX_DIR, help="where the index is stored")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="scan folders and build/update the index")
    p_index.add_argument("folders", nargs="+")

    p_search = sub.add_parser("search", help="run a single query")
    p_search.add_argument("query", nargs="+")
    p_search.add_argument("-k", type=int, default=5, help="number of results")
    p_search.add_argument("--min-score", type=float, default=0.0)
    p_search.add_argument("--all-chunks", action="store_true", help="allow several hits per file")

    p_chat = sub.add_parser("chat", help="interactive search prompt")
    p_chat.add_argument("-k", type=int, default=5)

    sub.add_parser("stats", help="show index statistics")
    args = parser.parse_args(argv)

    try:
        kb = KnowledgeBase(args.index_dir)
        if args.command == "index":
            print("Scanning and embedding (first run downloads the model, ~90 MB)…")
            s = kb.sync(args.folders)
            print(f"Done: {s['added']} added, {s['updated']} updated, {s['removed']} removed, "
                  f"{s['unchanged']} unchanged, {s['skipped']} skipped  ({s['chunks']} new chunks)")
        elif args.command == "search":
            show(kb.search(" ".join(args.query), k=args.k, min_score=args.min_score,
                           unique_files=not args.all_chunks))
        elif args.command == "chat":
            print(f"{kb.stats()['files']} files indexed. Type a question, or 'q' to quit.")
            while True:
                try:
                    query = input("\n🔎 > ").strip()
                except (EOFError, KeyboardInterrupt):
                    break
                if query.lower() in {"q", "quit", "exit"}:
                    break
                show(kb.search(query, k=args.k))
        elif args.command == "stats":
            for key, value in kb.stats().items():
                print(f"{key:>10}: {value}")
    except (NotADirectoryError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
