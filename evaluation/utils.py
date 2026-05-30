"""RAGAS 评测工具。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


def load_eval_dataset(path: Path | str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"评测集应为 JSON 数组: {path}")
    return data


def save_jsonl(records: List[Dict[str, Any]], path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path | str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_eval_item(item: Dict[str, Any]) -> Dict[str, Any]:
    gt = item.get("ground_truth") or item.get("expected_answer") or ""
    ctx = item.get("ground_truth_contexts") or []
    if isinstance(ctx, str):
        ctx = [ctx] if ctx else []
    return {
        **item,
        "ground_truth": gt,
        "ground_truth_contexts": list(ctx),
        "relevant_chunk_ids": list(item.get("relevant_chunk_ids") or []),
    }


def _truncate(text: str, limit: int = 1200) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n\n... [truncated, total {len(text)} chars]"


def _block(title: str, body: str) -> List[str]:
    """用定界符包裹正文，避免 prompt 里的 ``` 破坏 Markdown 渲染。"""
    return [
        f"### {title}",
        "",
        "-----",
        body or "(empty)",
        "-----",
        "",
    ]


def write_review(predictions: List[Dict[str, Any]], path: Path | str) -> None:
    """生成 review.md：question / response / generation prompt / 检索 context。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = [
        "# RAG 评测明细",
        "",
        "说明：Generation prompt 是实际送入 LLM 的完整文本；"
        "Retrieved contexts 是检索到的 chunk 原文（RAGAS 用这个评 retrieval）。",
        "",
    ]
    for i, row in enumerate(predictions, 1):
        lines.append(f"## {i}. {row.get('id', 'unknown')}")
        lines.append("")
        lines.append(f"**Question:** {row.get('question', '')}")
        lines.append("")
        if row.get("search_query") and row["search_query"] != row.get("question"):
            lines.append(f"**Search query:** {row['search_query']}")
            lines.append("")
        if row.get("relevant_chunk_ids"):
            lines.append(f"**Annotated chunk ids:** {row['relevant_chunk_ids']}")
            lines.append("")
        if row.get("ground_truth"):
            lines.extend(_block("Ground truth", row["ground_truth"]))
        if row.get("error"):
            lines.extend(_block("Error", row["error"]))
            lines.append("---")
            lines.append("")
            continue

        lines.extend(_block("Response", str(row.get("response") or "")))
        lines.extend(_block("Generation prompt", str(row.get("generation_prompt") or "")))

        retrieved = row.get("retrieved_contexts") or []
        if retrieved:
            parts = []
            for j, ctx in enumerate(retrieved, 1):
                parts.append(f"[Retrieved #{j}]\n{_truncate(ctx, 800)}")
            lines.extend(_block("Retrieved contexts", "\n\n".join(parts)))

        gt_ctx = row.get("ground_truth_contexts") or []
        if gt_ctx:
            parts = []
            for j, ctx in enumerate(gt_ctx, 1):
                parts.append(f"[Annotated #{j}]\n{_truncate(ctx, 800)}")
            lines.extend(_block("Annotated contexts (for RAGAS recall)", "\n\n".join(parts)))

        lines.append("---")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
