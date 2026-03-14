from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Tuple

import faiss
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.vector_stores.faiss import FaissVectorStore
from pydantic import BaseModel, ValidationError

from src.config import AppConfig
from src.embeddings import build_embedding
from src.models import complete_text


def _ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def _index_exists(index_dir: str) -> bool:
    return os.path.isdir(index_dir) and len(os.listdir(index_dir)) > 0


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", text.lower())


def _cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a * a for a in vec_a) ** 0.5
    norm_b = sum(b * b for b in vec_b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


class CitationItem(BaseModel):
    quote: str
    source_hint: str


class LegalAnswerSchema(BaseModel):
    answer: str
    key_points: list[str]
    risk_level: str
    citations: list[CitationItem]


def load_documents(config: AppConfig):
    docs_dir = config.paths.docs_dir
    if not os.path.isdir(docs_dir):
        raise FileNotFoundError(f"Docs directory not found: {docs_dir}")

    reader = SimpleDirectoryReader(input_dir=docs_dir, recursive=True)
    documents = reader.load_data()
    if not documents:
        raise ValueError(f"No documents found under: {docs_dir}")
    return documents


def build_or_load_index(config: AppConfig, rebuild: bool = False) -> VectorStoreIndex:
    _ensure_dir(config.paths.index_dir)
    embed_model = build_embedding(config)

    if rebuild or not _index_exists(config.paths.index_dir):
        documents = load_documents(config)
        # Infer actual embedding dimension to avoid config mismatch.
        inferred_dim = len(embed_model.get_query_embedding("法律合规"))
        faiss_index = faiss.IndexFlatL2(inferred_dim)
        vector_store = FaissVectorStore(faiss_index=faiss_index)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        index = VectorStoreIndex.from_documents(
            documents,
            storage_context=storage_context,
            embed_model=embed_model,
        )
        index.storage_context.persist(persist_dir=config.paths.index_dir)
        return index

    vector_store = FaissVectorStore.from_persist_dir(config.paths.index_dir)
    storage_context = StorageContext.from_defaults(
        vector_store=vector_store,
        persist_dir=config.paths.index_dir,
    )
    return load_index_from_storage(storage_context=storage_context, embed_model=embed_model)


def _keyword_retrieve(config: AppConfig, query: str, documents: list[Any]) -> list[NodeWithScore]:
    query_tokens = set(_tokenize(query))
    scored: list[NodeWithScore] = []

    for doc in documents:
        text = getattr(doc, "text", "") or ""
        if not text.strip():
            continue
        tokens = _tokenize(text)
        if not tokens:
            continue
        overlap = sum(1 for t in tokens if t in query_tokens)
        score = overlap / max(len(set(query_tokens)), 1)
        if score <= 0:
            continue
        preview = text[:1200]
        node = TextNode(text=preview)
        scored.append(NodeWithScore(node=node, score=score))

    scored.sort(key=lambda n: n.score or 0.0, reverse=True)
    return scored[: config.retrieval.keyword_top_k]


def _hybrid_retrieve(config: AppConfig, index: VectorStoreIndex, query: str) -> list[NodeWithScore]:
    semantic_retriever = index.as_retriever(similarity_top_k=config.retrieval.semantic_top_k)
    semantic_nodes = semantic_retriever.retrieve(query)

    keyword_nodes = _keyword_retrieve(config, query, load_documents(config))

    fused: dict[str, NodeWithScore] = {}
    for rank, n in enumerate(semantic_nodes, start=1):
        key = getattr(n.node, "node_id", "") or n.node.text[:80]
        score = 1.0 / (60 + rank)
        if key not in fused:
            fused[key] = NodeWithScore(node=n.node, score=score)
        else:
            fused[key].score = (fused[key].score or 0.0) + score

    for rank, n in enumerate(keyword_nodes, start=1):
        key = getattr(n.node, "node_id", "") or n.node.text[:80]
        score = 1.0 / (60 + rank)
        if key not in fused:
            fused[key] = NodeWithScore(node=n.node, score=score)
        else:
            fused[key].score = (fused[key].score or 0.0) + score

    return sorted(fused.values(), key=lambda x: x.score or 0.0, reverse=True)


def _rerank_by_embedding(config: AppConfig, query: str, candidates: list[NodeWithScore]) -> list[NodeWithScore]:
    embed_model = build_embedding(config)
    query_vec = embed_model.get_query_embedding(query)

    rescored: list[NodeWithScore] = []
    for cand in candidates:
        text = cand.node.text[:1500]
        text_vec = embed_model.get_text_embedding(text)
        sim = _cosine_similarity(query_vec, text_vec)
        rescored.append(NodeWithScore(node=cand.node, score=sim))

    rescored.sort(key=lambda x: x.score or 0.0, reverse=True)
    return rescored[: config.retrieval.final_top_k]


def _extract_json_block(text: str) -> str:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("Model response does not contain JSON object")
    return match.group(0)


def _generate_structured_answer(config: AppConfig, query: str, contexts: list[str]) -> dict[str, Any]:
    schema_hint = {
        "answer": "string",
        "key_points": ["string", "..."],
        "risk_level": "low|medium|high",
        "citations": [
            {"quote": "string", "source_hint": "string"},
        ],
    }
    context_block = "\n\n".join(
        [f"[Context {i + 1}]\n{c}" for i, c in enumerate(contexts)]
    )
    prompt = (
        "你是法律合规分析助手。请严格根据提供的上下文回答，"
        "如果上下文不足请在 answer 中明确说明。\n"
        "必须只返回一个 JSON 对象，不要输出任何额外文本。\n"
        f"JSON Schema 示例: {json.dumps(schema_hint, ensure_ascii=False)}\n\n"
        f"用户问题: {query}\n\n"
        f"检索上下文:\n{context_block}"
    )

    raw_text = complete_text(config, prompt)
    json_text = _extract_json_block(raw_text)
    payload = json.loads(json_text)
    parsed = LegalAnswerSchema.model_validate(payload)
    return parsed.model_dump()


def answer_query(config: AppConfig, query: str, rebuild: bool = False) -> Tuple[str, list[str]]:
    index = build_or_load_index(config, rebuild=rebuild)
    candidates = _hybrid_retrieve(config, index, query)
    reranked = _rerank_by_embedding(config, query, candidates)

    sources = []
    seen = set()
    for n in reranked:
        preview = n.node.text[:220].replace("\n", " ")
        if preview in seen:
            continue
        seen.add(preview)
        sources.append(preview)
    contexts = [n.node.text for n in reranked]

    if not config.output.use_json_schema:
        prompt = f"问题: {query}\n\n参考资料:\n" + "\n\n".join(contexts)
        answer = complete_text(config, prompt)
        return answer, sources

    try:
        structured = _generate_structured_answer(config, query, contexts)
    except (ValueError, json.JSONDecodeError, ValidationError, Exception) as exc:
        fallback = {
            "answer": f"结构化输出失败: {exc}",
            "key_points": [],
            "risk_level": "medium",
            "citations": [],
        }
        return json.dumps(fallback, ensure_ascii=False, indent=2), sources

    return json.dumps(structured, ensure_ascii=False, indent=2), sources
