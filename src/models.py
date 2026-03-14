from __future__ import annotations

import os

from dotenv import load_dotenv
from openai import OpenAI

from src.config import AppConfig


def _resolve_provider_config(config: AppConfig) -> tuple[str, str]:
    load_dotenv()
    provider = config.model.provider.lower().strip()

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is missing in environment.")
        api_base = config.model.api_base or "https://api.openai.com/v1"
    elif provider == "deepseek":
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY is missing in environment.")
        api_base = config.model.api_base or "https://api.deepseek.com/v1"
    elif provider == "ollama":
        # Ollama local OpenAI-compatible endpoint usually does not require a real key.
        api_key = os.getenv("OLLAMA_API_KEY", "ollama")
        api_base = config.model.api_base or "http://localhost:11434/v1"
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return api_key, api_base


def complete_text(config: AppConfig, prompt: str) -> str:
    api_key, api_base = _resolve_provider_config(config)
    provider = config.model.provider.lower().strip()
    timeout = None if provider == "ollama" else config.model.request_timeout_sec

    client = OpenAI(
        api_key=api_key,
        base_url=api_base,
        timeout=timeout,
        max_retries=config.model.max_retries,
    )

    resp = client.chat.completions.create(
        model=config.model.llm_model,
        temperature=config.model.temperature,
        messages=[{"role": "user", "content": prompt}],
        timeout=timeout,
    )
    content = resp.choices[0].message.content
    return content or ""


def build_llm(config: AppConfig) -> OpenAI:
    """Compatibility wrapper retained for older call sites."""
    api_key, api_base = _resolve_provider_config(config)
    return OpenAI(
        api_key=api_key,
        base_url=api_base,
        timeout=config.model.request_timeout_sec,
        max_retries=config.model.max_retries,
    )
