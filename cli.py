"""
cli.py — DocQuery batch-scan CLI

    python cli.py scan "What is the effective date?" ./contracts/
    python cli.py scan "What is the invoice number?" ./invoice.pdf

Asks the same question of a single file, or of every supported file in a
folder, one at a time. Each file gets its own fresh VectorStore, so a
question about one document is never answered using another document's
chunks -- this is a batch field-extraction tool, not the conversational,
multi-document assistant app.py/main.py give you. Reuses the same
services.ingest / services.qa_engine pipeline as both of those, so
answers here are identical in quality (retrieval, reranking, confidence,
caching) to what you'd get asking one document at a time in the app.

Prints "filename: answer" pairs to stdout; per-file errors (unreadable
file, no usable text, etc.) are caught and printed inline rather than
aborting the whole scan.
"""

from dotenv import load_dotenv
load_dotenv()

import argparse
import os
import sys

from services import ingest
from services.vector_store import VectorStore
from services.qa_engine import answer_question
from services.loaders import is_supported, SUPPORTED_EXTENSIONS


def _iter_files(path: str):
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for name in sorted(os.listdir(path)):
            full = os.path.join(path, name)
            if os.path.isfile(full) and is_supported(name):
                yield full
    else:
        raise SystemExit(f"'{path}' is not a file or directory.")


def scan(question: str, path: str, use_cache: bool = True) -> int:
    """Returns the number of files answered without error (for exit-code use)."""
    files = list(_iter_files(path))
    if not files:
        supported = ", ".join(f".{e}" for e in SUPPORTED_EXTENSIONS)
        raise SystemExit(f"No supported files found under '{path}'. Supported: {supported}")

    ok_count = 0
    for filepath in files:
        filename = os.path.basename(filepath)
        print(f"\n=== {filename} ===")

        try:
            with open(filepath, "rb") as f:
                file_bytes = f.read()

            # Fresh store + registry per file -- deliberate isolation, see
            # module docstring.
            vector_store = VectorStore()
            doc_registry = {}

            ingest.process_upload(file_bytes, filename, vector_store, doc_registry, use_cache=use_cache)
            result = answer_question(question, vector_store)

            print(result["answer"])
            print(f"[confidence: {result['confidence']}]")
            ok_count += 1

        except Exception as e:
            print(f"[error] {e}")

    return ok_count


def main():
    parser = argparse.ArgumentParser(
        prog="docquery-cli",
        description="Ask the same question of one file, or every supported file in a folder.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_parser = subparsers.add_parser("scan", help="Ask a question of a file or folder of files.")
    scan_parser.add_argument("question", help="The question to ask of each document.")
    scan_parser.add_argument("path", help="A file, or a folder of files.")
    scan_parser.add_argument("--no-cache", action="store_true", help="Skip the on-disk document cache.")

    args = parser.parse_args()

    if args.command == "scan":
        answered = scan(args.question, args.path, use_cache=not args.no_cache)
        print(f"\n{answered} file(s) answered.")
        sys.exit(0 if answered > 0 else 1)


if __name__ == "__main__":
    main()
