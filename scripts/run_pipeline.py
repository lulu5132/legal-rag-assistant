from __future__ import annotations

import argparse

from src.config import load_config
from src.pipeline import answer_query


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Legal RAG query pipeline")
    parser.add_argument(
        "--config",
        default="config/settings.example.yaml",
        help="Path to YAML config file",
    )
    parser.add_argument("--query", required=True, help="Question to ask")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild FAISS index from documents",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    print("\n=== Runtime LLM ===")
    print(f"provider: {config.model.provider}")
    print(f"model: {config.model.llm_model}")
    print(f"api_base: {config.model.api_base}")

    answer, sources = answer_query(config, args.query, rebuild=args.rebuild)

    print("\n=== Structured Answer ===")
    print(answer)

    if sources:
        print("\n=== Top Sources (preview) ===")
        for i, source in enumerate(sources, start=1):
            print(f"{i}. {source}")


if __name__ == "__main__":
    main()
