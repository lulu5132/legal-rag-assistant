from __future__ import annotations

import os

from dotenv import load_dotenv
from llama_index.core.base.embeddings.base import BaseEmbedding

from src.config import AppConfig


def build_embedding(config: AppConfig) -> BaseEmbedding:
    embedding_type = config.model.embedding_type.lower().strip()

    if embedding_type == "local":
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding

        return HuggingFaceEmbedding(model_name=config.model.embedding_model)

    if embedding_type == "openai":
        from llama_index.embeddings.openai import OpenAIEmbedding

        load_dotenv()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when embedding_type=openai")
        return OpenAIEmbedding(model=config.model.embedding_model, api_key=api_key)

    raise ValueError(f"Unsupported embedding_type: {embedding_type}")
