"""对接 RAG 流程，批量生成 RAGAS 所需的 predictions。"""

from __future__ import annotations

import logging
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import RAGConfig, get_active_config  # noqa: E402
from main import AerospaceRAGSystem  # noqa: E402
from evaluation.utils import normalize_eval_item  # noqa: E402

logger = logging.getLogger(__name__)


class EvaluationRunner:
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        *,
        eval_prompt: bool = True,
        eval_temperature: float = 0.0,
        quiet: bool = True,
    ):
        base = config or get_active_config()
        self.config = replace(base, temperature=eval_temperature)
        self.eval_prompt = eval_prompt
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
        route_type, rewritten_query = self.system.route_and_rewrite(
            question, rewrite=not self.eval_prompt
        )
        chunks, trace = self.system.retrieve_documents(
            question,
            rewritten_query=rewritten_query,
            route_type=route_type,
        )
        response, generation_prompt, context_docs = self.system.generate_answer(
            question,
            chunks,
            route_type,
            eval_mode=self.eval_prompt,
        )

        return {
            "id": item.get("id"),
            "question": question,
            "route_type": route_type,
            "search_query": trace["search_query"],
            "vector_query": trace["vector_query"],
            "bm25_query": trace["bm25_query"],
            "generation_prompt": generation_prompt,
            "response": response,
            "retrieved_contexts": [c.page_content for c in chunks],
            "retrieved_chunk_ids": [
                c.metadata.get("chunk_id") for c in chunks if c.metadata.get("chunk_id")
            ],
            "generation_contexts": [c.page_content for c in chunks],
            "ground_truth": item["ground_truth"],
            "ground_truth_contexts": item["ground_truth_contexts"],
            "relevant_chunk_ids": item.get("relevant_chunk_ids", []),
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
