"""
航空航天 RAG 系统配置：从 .env 加载密钥与运行参数。
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent

_ENV_FILE = _PROJECT_ROOT / ".env"
load_dotenv(_ENV_FILE)

DEFAULT_DATA_PATH = "./data/nasa"
DEFAULT_INDEX_PATH = "./vector_index"


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.getenv(key)
    if raw is None:
        return default
    return raw.lower() in ("1", "true", "yes", "on")


def _env_int(key: str, default: int) -> int:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(key: str, default: float) -> float:
    raw = os.getenv(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _default_faiss_use_gpu() -> bool:
    """有 CUDA + faiss-gpu 时默认开启 FAISS GPU。"""
    try:
        import torch

        if not torch.cuda.is_available():
            return False
        import faiss

        return faiss.get_num_gpus() > 0
    except Exception:
        return False


@dataclass
class RAGConfig:
    """航空航天 RAG 配置（密钥与 API 地址来自 .env）"""

    data_path: str = DEFAULT_DATA_PATH
    index_save_path: str = DEFAULT_INDEX_PATH
    source_mode: str = "aerospace"

    llm_provider: str = "openai"

    openai_api_key: str = ""
    openai_base_url: str = "https://api.openai.com/v1"
    llm_model: str = "gpt-4o-mini"

    local_llm_base_url: str = "http://localhost:8000/v1"
    local_llm_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
    local_llm_api_key: str = "not-needed"

    deepl_api_key: str = ""
    deepl_api_url: str = "https://api-free.deepl.com/v2/translate"
    deepl_api_mode: str = "deepl"

    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    top_k: int = 3
    translate_query_for_bm25: bool = True
    translate_query_for_vector: bool = True
    temperature: float = 0.1
    max_tokens: int = 2048

    embedding_device: str = "auto"
    faiss_use_gpu: bool = False
    faiss_gpu_id: int = 0
    embedding_batch_size: int = 64
    embedding_encode_batch_size: int = 0
    llm_max_workers: int = 4

    def __post_init__(self):
        if not self.openai_api_key and not self.local_llm_base_url:
            raise ValueError(
                "请在项目根目录 .env 中至少配置 OPENAI_API_KEY 或 LOCAL_LLM_BASE_URL。"
            )
        provider = (self.llm_provider or "openai").lower()
        if provider == "openai" and not self.openai_api_key:
            raise ValueError(
                "当前 LLM_PROVIDER=openai，但未配置 OPENAI_API_KEY。"
            )
        if provider == "local" and not self.local_llm_base_url:
            raise ValueError(
                "当前 LLM_PROVIDER=local，但未配置 LOCAL_LLM_BASE_URL。"
            )

    @property
    def active_llm_model(self) -> str:
        if (self.llm_provider or "openai").lower() == "local":
            return self.local_llm_model
        return self.llm_model

    @property
    def active_llm_base_url(self) -> str:
        if (self.llm_provider or "openai").lower() == "local":
            return self.local_llm_base_url
        return self.openai_base_url

    @property
    def active_llm_api_key(self) -> str:
        if (self.llm_provider or "openai").lower() == "local":
            return self.local_llm_api_key or "not-needed"
        return self.openai_api_key

    def resolve_data_path(self) -> Path:
        p = Path(self.data_path)
        if p.is_absolute():
            return p
        return (_PROJECT_ROOT / p).resolve()

    @classmethod
    def from_env(cls, **overrides) -> "RAGConfig":
        params = {
            "data_path": _env("RAG_DATA_PATH", DEFAULT_DATA_PATH),
            "index_save_path": _env("RAG_INDEX_PATH", DEFAULT_INDEX_PATH),
            "source_mode": _env("SOURCE_MODE", "aerospace"),
            "llm_provider": _env("LLM_PROVIDER", "openai").lower(),
            "openai_api_key": _env("OPENAI_API_KEY"),
            "openai_base_url": _env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            "llm_model": _env("OPENAI_MODEL", "gpt-4o-mini"),
            "local_llm_base_url": _env("LOCAL_LLM_BASE_URL", "http://localhost:8000/v1"),
            "local_llm_model": _env(
                "LOCAL_LLM_MODEL", "meta-llama/Llama-3.3-70B-Instruct-Turbo"
            ),
            "local_llm_api_key": _env("LOCAL_LLM_API_KEY", "not-needed"),
            "deepl_api_key": _env("DEEPL_API_KEY"),
            "deepl_api_url": _env(
                "DEEPL_API_URL", "https://api-free.deepl.com/v2/translate"
            ),
            "deepl_api_mode": _env("DEEPL_API_MODE", "deepl"),
            "embedding_model": _env("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5"),
            "top_k": _env_int("RAG_TOP_K", 3),
            "translate_query_for_bm25": _env_bool("TRANSLATE_QUERY_FOR_BM25", True),
            "translate_query_for_vector": _env_bool("TRANSLATE_QUERY_FOR_VECTOR", True),
            "temperature": _env_float("LLM_TEMPERATURE", 0.1),
            "max_tokens": _env_int("LLM_MAX_TOKENS", 2048),
            "embedding_device": _env("EMBEDDING_DEVICE", "auto"),
            "faiss_use_gpu": _env_bool("FAISS_USE_GPU", _default_faiss_use_gpu()),
            "faiss_gpu_id": _env_int("FAISS_GPU_ID", 0),
            "embedding_batch_size": _env_int("EMBEDDING_BATCH_SIZE", 64),
            "embedding_encode_batch_size": _env_int("EMBEDDING_ENCODE_BATCH_SIZE", 0),
            "llm_max_workers": _env_int("LLM_MAX_WORKERS", 4),
        }
        params.update(overrides)
        return cls(**params)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "data_path": self.data_path,
            "index_save_path": self.index_save_path,
            "source_mode": self.source_mode,
            "llm_provider": self.llm_provider,
            "openai_base_url": self.openai_base_url,
            "llm_model": self.llm_model,
            "local_llm_base_url": self.local_llm_base_url,
            "local_llm_model": self.local_llm_model,
            "active_llm_model": self.active_llm_model,
            "embedding_model": self.embedding_model,
            "top_k": self.top_k,
            "translate_query_for_bm25": self.translate_query_for_bm25,
            "translate_query_for_vector": self.translate_query_for_vector,
            "deepl_api_url": self.deepl_api_url,
            "deepl_api_mode": self.deepl_api_mode,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "embedding_device": self.embedding_device,
            "faiss_use_gpu": self.faiss_use_gpu,
            "embedding_batch_size": self.embedding_batch_size,
            "embedding_encode_batch_size": self.embedding_encode_batch_size,
            "llm_max_workers": self.llm_max_workers,
        }


def get_active_config() -> RAGConfig:
    return RAGConfig.from_env(
        data_path=DEFAULT_DATA_PATH,
        index_save_path=DEFAULT_INDEX_PATH,
        source_mode="aerospace",
        translate_query_for_bm25=True,
        translate_query_for_vector=True,
    )


def build_thesis_config() -> RAGConfig:
    """评测脚本兼容别名。"""
    return get_active_config()
