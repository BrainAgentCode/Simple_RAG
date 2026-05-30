#!/usr/bin/env python3
"""运行 RAG 并导出 predictions.jsonl（供 RAGAS 使用）。"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from evaluation.rag_runner import EvaluationRunner, build_run_metadata, config_from_name
from evaluation.utils import load_eval_dataset, save_jsonl, write_review

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 评测：生成 predictions.jsonl")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent / "datasets" / "eval_all.json",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", choices=["aerospace", "thesis"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--context-mode",
        choices=["parent", "chunk"],
        default="parent",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    from config import get_active_config

    config = config_from_name(args.config) if args.config else get_active_config()
    items = load_eval_dataset(args.dataset)
    if not items:
        raise SystemExit(f"评测集为空: {args.dataset}")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.output_dir or (_PROJECT_ROOT / "evaluation" / "results" / f"run_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = EvaluationRunner(
        config=config, context_mode=args.context_mode, quiet=not args.verbose
    )
    print("初始化 RAG...")
    runner.setup()

    predictions = runner.run_dataset(items, limit=args.limit)
    pred_path = out_dir / "predictions.jsonl"
    save_jsonl(predictions, pred_path)
    review_path = out_dir / "review.md"
    write_review(predictions, review_path)

    meta = build_run_metadata(
        config,
        dataset=str(args.dataset),
        limit=args.limit,
        context_mode=args.context_mode,
        num_samples=len(predictions),
    )
    with open(out_dir / "run_metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"已保存 {len(predictions)} 条 -> {pred_path}")
    print(f"人工查看 -> {review_path}")
    print(f"下一步: python evaluation/run_ragas.py --predictions {pred_path}")


if __name__ == "__main__":
    main()
