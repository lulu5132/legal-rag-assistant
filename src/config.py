from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class ProjectConfig(BaseModel):
    name: str = "legal-rag-assistant"


class PathsConfig(BaseModel):
    docs_dir: str = "data/processed"
    index_dir: str = "indexes/faiss"


class RetrievalConfig(BaseModel):
    similarity_top_k: int = 5
    keyword_top_k: int = 8
    semantic_top_k: int = 8
    final_top_k: int = 4


class ModelConfig(BaseModel):
    provider: str = "ollama"
    llm_model: str = "deepseek-r1:8b"
    temperature: float = 0.1
    embedding_type: str = "local"
    embedding_model: str = "models/bge-small-zh-v1.5"
    embedding_dim: int = 384
    api_base: str = "http://localhost:11434/v1"


class OutputConfig(BaseModel):
    use_json_schema: bool = True


class AppConfig(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(config_path: str) -> AppConfig:
    with Path(config_path).open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig.model_validate(raw)
