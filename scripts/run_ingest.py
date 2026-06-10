import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rag_chatbot.pipeline.ingest import run_index, run_load


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["load", "index"], required=True)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--documents-output", default="data/processed/documents.json")
    parser.add_argument("--chunks-output", default="data/processed/chunks.json")
    parser.add_argument("--index-dir", default="artifacts/faiss_index")
    args = parser.parse_args()

    if args.stage == "load":
        documents = run_load(args.raw_dir, args.documents_output)
        print(f"Loaded {len(documents)} documents.")
        return

    chunks = run_index(args.raw_dir, args.index_dir, args.chunks_output)
    print(f"Indexed {len(chunks)} chunks into {args.index_dir}.")


if __name__ == "__main__":
    main()

