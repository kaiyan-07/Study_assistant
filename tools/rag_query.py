import os
import sys
import json
import argparse

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))

if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from services.assistant_service import ask_question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", required=True, type=str)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--model", type=str, default="llama3")
    parser.add_argument("--json", action="store_true")

    args = parser.parse_args()

    try:
        result = ask_question(
            query=args.query,
            top_k=args.top_k,
            model_name=args.model
        )

        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print("RAG_RESULT_BEGIN")
            print(f"QUERY: {result['query']}")
            print(f"TOP_K: {result['top_k']}")
            print(f"MODEL: {result['model_name']}")
            print("ANSWER:")
            print(result["answer"])
            print("RAG_RESULT_END")

    except Exception as e:
        error_result = {
            "error": str(e)
        }

        if args.json:
            print(json.dumps(error_result, ensure_ascii=False, indent=2))
        else:
            print(f"ERROR: {str(e)}")

        sys.exit(1)


if __name__ == "__main__":
    main()