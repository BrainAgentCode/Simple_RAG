#!/usr/bin/env python3
"""对 predictions.jsonl 运行 RAGAS。"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 减少 HF / transformers / 依赖库的噪音
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=r".*google\.generativeai.*")
warnings.filterwarnings("ignore", message=r".*Unrecognized keys in `rope_parameters`.*")
warnings.filterwarnings("ignore", message=r".*unauthenticated requests to the HF Hub.*")

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")

from evaluation.utils import load_jsonl


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


def build_dataset(rows: List[Dict[str, Any]], context_field: str):
    from datasets import Dataset

    questions, contexts, answers, ground_truths, reference_contexts = [], [], [], [], []
    for row in rows:
        if row.get("error"):
            continue
        ctx = row.get(context_field) or row.get("retrieved_contexts") or []
        if isinstance(ctx, str):
            ctx = [ctx]
        ref = row.get("ground_truth_contexts") or []
        if isinstance(ref, str):
            ref = [ref] if ref else []

        questions.append(row.get("question", ""))
        contexts.append(list(ctx))
        answers.append(row.get("response", ""))
        ground_truths.append(row.get("ground_truth", ""))
        reference_contexts.append(list(ref))

    return Dataset.from_dict(
        {
            "question": questions,
            "contexts": contexts,
            "answer": answers,
            "ground_truth": ground_truths,
            "reference_contexts": reference_contexts,
        }
    )


def _load_metrics(
    llm: Any, embeddings: Any
) -> Tuple[List[Any], List[Any], List[Any]]:
    """
    加载与 evaluate() 兼容的 Metric 实例。

    注意：ragas.metrics.collections 里的类属于新架构，不能用于 evaluate()；
    必须用 ragas.metrics._* 模块里的 dataclass Metric。
    """
    strictness = int(_env("RAGAS_ANSWER_RELEVANCY_STRICTNESS", "1"))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        from ragas.metrics._answer_correctness import AnswerCorrectness
        from ragas.metrics._answer_relevance import AnswerRelevancy
        from ragas.metrics._context_precision import ContextPrecision
        from ragas.metrics._context_recall import ContextRecall
        from ragas.metrics._faithfulness import Faithfulness
        from ragas.metrics.base import Metric

    ctx_prec = ContextPrecision()
    ctx_rec = ContextRecall()
    faith = Faithfulness()
    ans_rel = AnswerRelevancy(strictness=strictness)
    ans_corr = AnswerCorrectness()

    for m in (ctx_prec, ctx_rec, faith, ans_rel, ans_corr):
        if not isinstance(m, Metric):
            raise TypeError(f"{type(m).__name__} 不是 evaluate() 支持的 Metric 类型")

    retrieval = [ctx_prec, ctx_rec]
    generation = [faith, ans_rel, ans_corr]
    all_metrics = [ctx_prec, ctx_rec, faith, ans_rel, ans_corr]
    return retrieval, generation, all_metrics


def build_llm_and_embeddings():
    from config import get_active_config
    from rag_modules.runtime_accel import resolve_embedding_device

    cfg = get_active_config()
    api_key = _env("OPENAI_API_KEY")
    base_url = _env("OPENAI_BASE_URL", "https://api.openai.com/v1")
    llm_model = _env("RAGAS_LLM_MODEL") or _env("OPENAI_MODEL", "gpt-4o-mini")

    llm = None
    embeddings = None

    try:
        from openai import OpenAI
        from ragas.llms import llm_factory

        client = OpenAI(api_key=api_key, base_url=base_url)
        llm = llm_factory(llm_model, client=client, temperature=0)
    except Exception:
        pass

    if llm is None:
        from langchain_openai import ChatOpenAI

        chat = ChatOpenAI(
            model=llm_model, api_key=api_key, base_url=base_url, temperature=0
        )
        try:
            from ragas.llms import LangchainLLMWrapper

            llm = LangchainLLMWrapper(chat)
        except ImportError:
            llm = chat

    if _env("RAGAS_EMBEDDING_BACKEND", "local").lower() == "openai":
        emb_model = _env("RAGAS_EMBEDDING_MODEL", "text-embedding-3-small")
        try:
            from openai import OpenAI
            from ragas.embeddings import OpenAIEmbeddings

            client = OpenAI(api_key=api_key, base_url=base_url)
            embeddings = OpenAIEmbeddings(model=emb_model, client=client)
        except Exception:
            from langchain_openai import OpenAIEmbeddings

            lc_emb = OpenAIEmbeddings(
                model=emb_model, api_key=api_key, base_url=base_url
            )
            try:
                from ragas.embeddings import LangchainEmbeddingsWrapper

                embeddings = LangchainEmbeddingsWrapper(lc_emb)
            except ImportError:
                embeddings = lc_emb
    else:
        model = _env("RAGAS_EMBEDDING_MODEL") or cfg.embedding_model
        device = _env("RAGAS_EMBEDDING_DEVICE") or cfg.embedding_device
        try:
            from ragas.embeddings import HuggingFaceEmbeddings

            embeddings = HuggingFaceEmbeddings(
                model=model,
                device=resolve_embedding_device(device),
            )
        except Exception:
            from langchain_huggingface import HuggingFaceEmbeddings

            lc_emb = HuggingFaceEmbeddings(
                model_name=model,
                model_kwargs={"device": resolve_embedding_device(device)},
                encode_kwargs={"normalize_embeddings": True},
            )
            try:
                from ragas.embeddings import LangchainEmbeddingsWrapper

                embeddings = LangchainEmbeddingsWrapper(lc_emb)
            except ImportError:
                embeddings = lc_emb

    return llm, embeddings


def _run_config(workers: int) -> Any:
    from ragas.run_config import RunConfig

    timeout = int(_env("RAGAS_TIMEOUT", "180"))
    return RunConfig(max_workers=max(1, workers), timeout=timeout)


def _run_evaluate(
    dataset,
    metrics,
    llm: Any,
    embeddings: Any,
    *,
    run_config: Any,
):
    from ragas import evaluate

    n = len(dataset)
    m = len(metrics)
    print(
        f"  任务数 ≈ {n} 条 × {m} 指标 = {n * m} 个评分任务"
        f"（每条 context 可能触发多次 LLM 调用，非卡死）"
    )
    print(f"  并行 workers={run_config.max_workers}, timeout={run_config.timeout}s")

    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "metrics": metrics,
        "llm": llm,
        "embeddings": embeddings,
        "run_config": run_config,
    }
    try:
        return evaluate(**kwargs)
    except TypeError:
        kwargs.pop("run_config", None)
        return evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
        )


def _mean_scores(result) -> Dict[str, float]:
    df = result.to_pandas()
    return {c: float(df[c].mean()) for c in df.select_dtypes(include="number").columns}


def _merge_score_frames(frames: List["pd.DataFrame"]) -> "pd.DataFrame":
    import pandas as pd

    if not frames:
        return pd.DataFrame()
    merged = frames[0].copy()
    for df in frames[1:]:
        for col in df.select_dtypes(include="number").columns:
            if col not in merged.columns:
                merged[col] = df[col].values
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="RAGAS 评测")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument(
        "--context-field",
        default="auto",
        choices=["auto", "retrieved_contexts", "generation_contexts"],
        help="auto=检索指标用 retrieved，faithfulness 用 generation（推荐）",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="RAGAS 并行 worker 数（默认 RAGAS_MAX_WORKERS 或 LLM_MAX_WORKERS 或 8）",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="只评测前 N 条 predictions（快速 smoke test）",
    )
    args = parser.parse_args()

    from config import get_active_config

    cfg = get_active_config()
    workers = args.workers
    if workers is None:
        workers = int(_env("RAGAS_MAX_WORKERS") or str(cfg.llm_max_workers or 8))
    run_config = _run_config(workers)

    if not _env("OPENAI_API_KEY"):
        raise SystemExit("请在 .env 配置 OPENAI_API_KEY（RAGAS judge 需要）")

    rows = load_jsonl(args.predictions)
    if not rows:
        raise SystemExit("predictions 为空")

    total_rows = len(rows)
    if args.max_samples is not None:
        if args.max_samples <= 0:
            raise SystemExit("--max-samples 须 > 0")
        rows = rows[: args.max_samples]
        print(f"仅评测前 {len(rows)}/{total_rows} 条 (--max-samples={args.max_samples})")

    out_dir = args.output_dir or args.predictions.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    has_ref = any(r.get("ground_truth_contexts") for r in rows if not r.get("error"))
    llm, embeddings = build_llm_and_embeddings()

    try:
        retrieval_metrics, generation_metrics, all_metrics = _load_metrics(
            llm, embeddings
        )
    except ImportError as e:
        raise SystemExit("请安装: pip install ragas datasets") from e

    summary: Dict[str, float] = {}
    details = []

    if args.context_field == "auto":
        ds_retrieval = build_dataset(rows, "retrieved_contexts")
        r_metrics = [retrieval_metrics[0]]
        if has_ref:
            r_metrics.append(retrieval_metrics[1])
        else:
            print("无 ground_truth_contexts，跳过 context_recall")

        print(f"RAGAS 检索指标 {len(ds_retrieval)} 条 (context=retrieved_contexts)")
        r1 = _run_evaluate(
            ds_retrieval, r_metrics, llm, embeddings, run_config=run_config
        )
        summary.update(_mean_scores(r1))
        details.append(r1.to_pandas())

        ds_gen = build_dataset(rows, "generation_contexts")
        print(f"RAGAS 生成指标 {len(ds_gen)} 条 (context=generation_contexts)")
        r2 = _run_evaluate(
            ds_gen, generation_metrics, llm, embeddings, run_config=run_config
        )
        summary.update(_mean_scores(r2))
        details.append(r2.to_pandas())
        context_note = "auto (retrieval=retrieved_contexts, faithfulness=generation_contexts)"
    else:
        ds = build_dataset(rows, args.context_field)
        if has_ref:
            metrics = all_metrics
        else:
            metrics = [retrieval_metrics[0], *generation_metrics]
        print(f"RAGAS 评测 {len(ds)} 条，context={args.context_field}")
        result = _run_evaluate(
            ds, metrics, llm, embeddings, run_config=run_config
        )
        summary = _mean_scores(result)
        details.append(result.to_pandas())
        context_note = args.context_field

    scores_df = _merge_score_frames(details)
    scores_df.to_csv(out_dir / "ragas_scores.csv", index=False)
    with open(out_dir / "ragas_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "num_samples": len([r for r in rows if not r.get("error")]),
                "num_predictions_total": total_rows,
                "max_samples": args.max_samples,
                "context_field": context_note,
                "mean_scores": summary,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== RAGAS ===")
    for name, value in summary.items():
        print(f"  {name}: {value:.4f}")
    print(f"\n明细: {out_dir / 'ragas_scores.csv'}")


if __name__ == "__main__":
    main()
