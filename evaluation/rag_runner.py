"""对接 RAG 流程，批量生成 RAGAS 所需的 predictions。"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import RAGConfig, get_active_config, build_thesis_config  # noqa: E402
from main import AerospaceRAGSystem  # noqa: E402
from evaluation.utils import normalize_eval_item  # noqa: E402

logger = logging.getLogger(__name__)


class EvaluationRunner:
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        *,
        context_mode: str = "parent",
        context_max_length: int = 8000,
        quiet: bool = True,
    ):
        self.config = config or get_active_config()
        self.context_mode = context_mode
        self.context_max_length = context_max_length
        self.quiet = quiet
        self.system: Optional[AerospaceRAGSystem] = None

    def setup(self) -> None:
        self.system = AerospaceRAGSystem(config=self.config)
        self.system.initialize_system()
        self.system.build_knowledge_base()

    def run_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        if self.system is None:
            raise RuntimeError("请先调用 setup()")

        item = normalize_eval_item(item)
        question = item["question"]
        search_query = self.system.generation_module.query_rewrite(question)

        vector_q, bm25_q = self.system._retrieval_queries(search_query)
        chunks = self.system.retrieval_module.hybrid_search(
            search_query,
            top_k=self.config.top_k,
            vector_query=vector_q,
            bm25_query=bm25_q,
        )
        parents = self.system.data_module.get_parent_documents(chunks)
        answer_docs = (
            chunks if self.context_mode == "chunk" else (parents or chunks)
        )

        gen = self.system.generation_module
        ctx_len = self.context_max_length
        generation_prompt = gen.build_basic_answer_prompt(
            question, answer_docs, max_length=ctx_len
        )
        response = (
            gen.generate_basic_answer(question, answer_docs, max_length=ctx_len)
            if answer_docs
            else ""
        )

        return {
            "id": item.get("id"),
            "question": question,
            "search_query": search_query,
            "generation_prompt": generation_prompt,
            "response": response,
            "retrieved_contexts": [c.page_content for c in chunks],
            "generation_contexts": [p.page_content for p in parents],
            "ground_truth": item["ground_truth"],
            "ground_truth_contexts": item["ground_truth_contexts"],
            "relevant_chunk_ids": item.get("relevant_chunk_ids", []),
            "type": item.get("type"),
        }

    def run_dataset(
        self, items: List[Dict[str, Any]], *, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        subset = items[:limit] if limit else items
        results: List[Dict[str, Any]] = []
        for i, raw in enumerate(subset, 1):
            item = normalize_eval_item(raw)
            if not self.quiet:
                print(f"[{i}/{len(subset)}] {item.get('id')}: {item['question'][:60]}...")
            try:
                results.append(self.run_item(item))
            except Exception as e:
                logger.exception("评测失败: %s", item.get("id"))
                results.append(
                    {
                        "id": item.get("id"),
                        "question": item.get("question"),
                        "error": str(e),
                        "ground_truth": item.get("ground_truth"),
                        "ground_truth_contexts": item.get("ground_truth_contexts", []),
                    }
                )
        return results


def build_run_metadata(config: RAGConfig, **extra: Any) -> Dict[str, Any]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "config": config.to_dict(),
        **extra,
    }


def config_from_name(name: str) -> RAGConfig:
    if name.lower() in ("thesis", "nasa", "aerospace", "default"):
        return build_thesis_config()
    raise ValueError(f"未知配置: {name}")
