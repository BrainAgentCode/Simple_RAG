#!/usr/bin/env python3
"""对 predictions.jsonl 运行 RAGAS。"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

warnings.filterwarnings("ignore")

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from evaluation.utils import load_jsonl


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def _as_list(value: Any) -> List[str]:
    if not value:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _resolve_device(requested: str) -> str:
    key = (requested or "auto").strip().lower()
    if key in ("cpu", "cuda", "mps"):
        return key
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _load_rows(path: Path, max_samples: int | None) -> List[Dict[str, Any]]:
    rows = [r for r in load_jsonl(path) if not r.get("error")]
    if not rows:
        raise SystemExit("predictions 为空")
    if max_samples is not None:
        if max_samples <= 0:
            raise SystemExit("--max-samples 须 > 0")
        rows = rows[:max_samples]
    return rows


def _build_llm():
    from openai import OpenAI
    from ragas.llms import llm_factory

    api_key = _env("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("请在 .env 配置 OPENAI_API_KEY")

    model = _env("RAGAS_LLM_MODEL") or _env("OPENAI_MODEL", "gpt-4o-mini")
    client = OpenAI(
        api_key=api_key,
        base_url=_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        timeout=float(_env("RAGAS_LLM_TIMEOUT", "3600")),
    )
    max_tokens = int(_env("RAGAS_LLM_MAX_TOKENS", "4096"))
    return llm_factory(model, client=client, temperature=0, max_tokens=max_tokens), model


def _build_embeddings():
    from ragas.embeddings import LangchainEmbeddingsWrapper

    if _env("RAGAS_EMBEDDING_BACKEND", "local").lower() == "openai":
        from langchain_openai import OpenAIEmbeddings

        api_key = _env("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("RAGAS_EMBEDDING_BACKEND=openai 需要 OPENAI_API_KEY")
        model = _env("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")
        emb = OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            base_url=_env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
        return LangchainEmbeddingsWrapper(emb), f"openai/{model}"

    from config import get_active_config
    from langchain_huggingface import HuggingFaceEmbeddings

    cfg = get_active_config()
    model = _env("RAGAS_EMBEDDING_MODEL") or cfg.embedding_model
    device = _resolve_device(_env("RAGAS_EMBEDDING_DEVICE") or cfg.embedding_device)
    emb = HuggingFaceEmbeddings(
        model_name=model,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": True},
    )
    return LangchainEmbeddingsWrapper(emb), f"{model}@{device}"


def _build_metrics(*, with_recall: bool, full: bool) -> List[Any]:
    from ragas.metrics._answer_relevance import AnswerRelevancy
    from ragas.metrics._context_precision import ContextPrecision
    from ragas.metrics._faithfulness import Faithfulness

    metrics: List[Any] = [
        ContextPrecision(),
        Faithfulness(),
        AnswerRelevancy(strictness=1),
    ]
    if with_recall:
        from ragas.metrics._context_recall import ContextRecall

        metrics.insert(1, ContextRecall())
    if full:
        from ragas.metrics._answer_correctness import AnswerCorrectness

        metrics.append(AnswerCorrectness())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS 评测")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--full",
        action="store_true",
        help="包含 answer_correctness（更慢）",
    )
    args = parser.parse_args()

    rows = _load_rows(args.predictions, args.max_samples)
    has_ref = any(r.get("ground_truth_contexts") for r in rows)
    llm, llm_name = _build_llm()
    embeddings, emb_label = _build_embeddings()
    metrics = _build_metrics(with_recall=has_ref, full=args.full)

    from datasets import Dataset
    from ragas import evaluate
    from ragas.run_config import RunConfig

    dataset = Dataset.from_dict(
        {
            "question": [r.get("question", "") for r in rows],
            "contexts": [_as_list(r.get("retrieved_contexts")) for r in rows],
            "answer": [r.get("response", "") for r in rows],
            "ground_truth": [r.get("ground_truth", "") for r in rows],
            "reference_contexts": [
                _as_list(r.get("ground_truth_contexts")) for r in rows
            ],
        }
    )

    workers = int(_env("RAGAS_MAX_WORKERS", "2"))
    job_timeout = int(_env("RAGAS_TIMEOUT", "3600"))
    print(f"RAGAS: {len(rows)} samples, {len(metrics)} metrics, workers={workers}, timeout={job_timeout}s")
    print(f"LLM: {llm_name}")
    print(f"Embedding: {emb_label}")

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        run_config=RunConfig(
            max_workers=workers,
            timeout=job_timeout,
        ),
    )

    out_dir = args.output_dir or args.predictions.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    df = result.to_pandas()
    df.to_csv(out_dir / "ragas_scores.csv", index=False)
    numeric = df.select_dtypes(include="number").columns
    summary = {c: float(df[c].mean()) for c in numeric}
    failed = int(df[numeric].isna().any(axis=1).sum()) if len(numeric) else 0

    with open(out_dir / "ragas_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_samples": len(rows),
                "metrics": [m.name for m in metrics],
                "failed_jobs": failed,
                "mean_scores": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== mean scores ===")
    for name, value in summary.items():
        print(f"  {name}: {value:.4f}")
    if failed:
        print(f"  ({failed} rows with NaN)")
    print(f"saved: {out_dir / 'ragas_scores.csv'}")


if __name__ == "__main__":
    main()
