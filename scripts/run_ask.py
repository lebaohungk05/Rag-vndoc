import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from rag_chatbot.pipeline.ask import run_ask


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", required=True)
    parser.add_argument("--index-dir", default="artifacts/faiss_index")
    args = parser.parse_args()

    result = run_ask(question=args.question, index_dir=args.index_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

