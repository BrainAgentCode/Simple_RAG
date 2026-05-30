#!/usr/bin/env python3
"""
生成 RAGAS 评测集。

两种来源（二选一）：
  llm    — 将整篇文档（带 chunk_id 标注）发给 LLM，生成 question / expected_answer / relevant_chunk_ids
  thesis — 导入 Master_Thesis_RAG 人工标注测试集，并从本地 chunks 填充 ground_truth_contexts

示例:
  # LLM 从语料随机抽 3 篇文档，每篇生成 2 题
  python evaluation/generate_dataset.py llm \\
    --chunks-dir data/nasa/chunks --max-docs 3 --per-doc 2 \\
    -o evaluation/datasets/eval_all.json

  # 导入 thesis 测试集
  python evaluation/generate_dataset.py thesis \\
    --chunks-dir data/nasa/chunks --skip-missing \\
    -o evaluation/datasets/eval_all.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv(_PROJECT_ROOT / ".env")


def _default_chunks_dir() -> Path:
    from config import get_active_config

    return get_active_config().resolve_data_path() / "chunks"


def _default_thesis_dir() -> Path:
    return Path(__file__).resolve().parent / "datasets" / "thesis"


def load_chunks(chunks_dir: Path) -> List[Dict[str, Any]]:
    root = Path(chunks_dir).resolve()
    if not root.exists():
        raise FileNotFoundError(f"分块目录不存在: {root}")

    chunks: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for path in sorted(root.rglob("*.json")):
        if path.name.endswith(".partial.json"):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue
        for item in data if isinstance(data, list) else [data]:
            if not isinstance(item, dict):
                continue
            cid = str(item.get("chunk_id") or "").strip()
            if not cid or cid in seen or not item.get("content"):
                continue
            seen.add(cid)
            chunks.append(item)
    return chunks


def doc_key(chunk: Dict[str, Any]) -> str:
    meta = chunk.get("metadata") or {}
    url = str(meta.get("url") or "").strip()
    if url:
        return url
    cid = str(chunk.get("chunk_id") or "")
    if ".pdf_" in cid:
        return cid.split(".pdf_")[0]
    file_name = str(meta.get("file_name") or "")
    if file_name.endswith(".pdf"):
        return Path(file_name).stem
    return cid.split("_")[0] if "_" in cid else cid


def chunk_body(chunk: Dict[str, Any]) -> str:
    title = str(chunk.get("title") or "").strip()
    content = str(chunk.get("content") or "").strip()
    return f"{title}\n{content}".strip() if title else content


def group_by_document(chunks: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        groups[doc_key(chunk)].append(chunk)
    for key in groups:
        groups[key].sort(key=lambda c: str(c.get("chunk_id") or ""))
    return dict(groups)


def format_document(chunks: List[Dict[str, Any]], *, max_chars: int) -> str:
    parts: List[str] = []
    total = 0
    for chunk in chunks:
        cid = str(chunk.get("chunk_id") or "")
        body = chunk_body(chunk)
        block = f"[chunk_id: {cid}]\n{body}"
        if total and total + len(block) > max_chars:
            parts.append("[... 后续 chunk 已截断 ...]")
            break
        parts.append(block)
        total += len(block)
    return "\n\n".join(parts)


def build_llm():
    import os

    from langchain_openai import ChatOpenAI

    from config import get_active_config

    cfg = get_active_config()
    kwargs: Dict[str, Any] = {
        "model": os.getenv("EVAL_GEN_MODEL") or cfg.llm_model,
        "api_key": cfg.openai_api_key or os.getenv("OPENAI_API_KEY"),
        "base_url": cfg.openai_base_url,
        "temperature": 0,
    }
    # OpenAI 兼容 API 可强制 JSON，减少解析失败
    try:
        return ChatOpenAI(**kwargs, model_kwargs={"response_format": {"type": "json_object"}})
    except TypeError:
        return ChatOpenAI(**kwargs)


def _extract_json_blob(text: str) -> str:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        text = fence.group(1).strip()

    for opener, closer in (("[", "]"), ("{", "}")):
        start = text.find(opener)
        if start < 0:
            continue
        depth = 0
        for i in range(start, len(text)):
            ch = text[i]
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    return text[start : i + 1]
    return text


def _normalize_json_text(text: str) -> str:
    text = _extract_json_blob(text)
    text = re.sub(r",\s*([}\]])", r"\1", text)  # 去掉 trailing comma
    return text.strip()


def _parse_json_array(text: str) -> List[Dict[str, Any]]:
    last_err: Optional[Exception] = None
    for raw in {text, _normalize_json_text(text)}:
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            last_err = e
            continue
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("items", "questions", "data", "results"):
                val = data.get(key)
                if isinstance(val, list):
                    return val
            if all(k in data for k in ("question", "expected_answer")):
                return [data]
        last_err = ValueError(f"unexpected JSON shape: {type(data)}")

    preview = text.strip().replace("\n", " ")[:160]
    raise ValueError(f"无法解析 LLM 返回的 JSON: {preview!r}") from last_err


def _word_overlap(reference: str, text: str) -> float:
    ref = set(re.findall(r"\w+", reference.lower()))
    if not ref:
        return 0.0
    body = set(re.findall(r"\w+", text.lower()))
    return len(ref & body) / len(ref)


def _validate_qa(
    question: str,
    answer: str,
    chunk_ids: List[str],
    chunks: List[Dict[str, Any]],
    *,
    min_overlap: float,
) -> bool:
    if len(question) < 20 or len(answer) < 30:
        return False
    if not chunk_ids or len(chunk_ids) > 2:
        return False
    bodies = [chunk_body(c) for c in chunks if str(c["chunk_id"]) in chunk_ids]
    if not bodies:
        return False
    combined = "\n".join(bodies)
    return _word_overlap(answer, combined) >= min_overlap


def generate_for_document(
    llm,
    doc_id: str,
    chunks: List[Dict[str, Any]],
    *,
    per_doc: int,
    max_chars: int,
    max_retries: int = 2,
) -> List[Dict[str, Any]]:
    valid_ids = {str(c["chunk_id"]) for c in chunks}
    doc_text = format_document(chunks, max_chars=max_chars)
    base_prompt = f"""You are building a high-quality RAG evaluation dataset.

Document sections are labeled with [chunk_id: ...]. Create exactly {per_doc} question-answer pairs.

Strict rules:
1. Each question must target ONE specific fact from ONE primary chunk.
2. Pick exactly 1 primary chunk_id per question. You may add at most 1 secondary chunk_id only if strictly needed.
3. expected_answer must be 1-3 sentences, directly supported by the listed chunk(s). Do not add outside knowledge.
4. Prefer concrete questions (what / which / how / why) about names, numbers, methods, or findings.
5. Avoid vague questions like "Summarize this document" or "What is discussed here".
6. Use English for both question and expected_answer.

Return ONLY valid JSON (no markdown, no comments) in this exact shape:
{{"items": [
  {{
    "question": "...",
    "expected_answer": "...",
    "relevant_chunk_ids": ["primary_chunk_id"]
  }}
]}}

Document ({doc_id}):
{doc_text}
"""
    raw_items: List[Dict[str, Any]] = []
    for attempt in range(1, max_retries + 1):
        prompt = base_prompt
        if attempt > 1:
            prompt += "\n\nIMPORTANT: Your previous response was not valid JSON. Return ONLY the JSON object."
        try:
            resp = llm.invoke(prompt)
            raw_items = _parse_json_array(str(resp.content))
            break
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  JSON 解析失败 (attempt {attempt}/{max_retries}): {e}")
            if attempt >= max_retries:
                print(f"  跳过文档 {doc_id}")
                return []

    items: List[Dict[str, Any]] = []
    for i, row in enumerate(raw_items, 1):
        q = str(row.get("question") or "").strip()
        a = str(row.get("expected_answer") or "").strip()
        ids = [str(x) for x in (row.get("relevant_chunk_ids") or []) if str(x) in valid_ids][:2]
        if not _validate_qa(q, a, ids, chunks, min_overlap=0.2):
            print(f"  跳过无效条目: overlap/格式不符 (q={q[:50]}...)")
            continue
        ctx = [chunk_body(c) for c in chunks if str(c["chunk_id"]) in ids]
        items.append(
            {
                "id": f"llm_{doc_id}_{i}",
                "question": q,
                "ground_truth": a,
                "ground_truth_contexts": ctx,
                "relevant_chunk_ids": ids,
                "metadata": {"source": "llm", "doc_id": doc_id},
            }
        )
        if len(items) >= per_doc:
            break
    return items


def generate_llm_dataset(
    chunks_dir: Path,
    *,
    max_docs: int,
    per_doc: int,
    seed: int,
    stems: List[str],
    max_chars: int,
) -> List[Dict[str, Any]]:
    chunks = load_chunks(chunks_dir)
    groups = group_by_document(chunks)
    if not groups:
        raise ValueError(f"未加载到任何文档: {chunks_dir}")

    if stems:
        keys = [k for k in stems if k in groups]
        missing = [s for s in stems if s not in groups]
        if missing:
            print(f"警告: 未找到文档 {missing}")
    else:
        keys = sorted(groups.keys())
        rng = random.Random(seed)
        rng.shuffle(keys)
        keys = keys[:max_docs]

    if not keys:
        raise ValueError("没有可生成评测题的文档")

    llm = build_llm()
    all_items: List[Dict[str, Any]] = []
    skipped = 0
    for doc_id in keys:
        doc_chunks = groups[doc_id]
        print(f"生成 {doc_id} ({len(doc_chunks)} chunks)...")
        items = generate_for_document(
            llm, doc_id, doc_chunks, per_doc=per_doc, max_chars=max_chars
        )
        if not items:
            skipped += 1
        print(f"  -> {len(items)} 题")
        all_items.extend(items)
    if skipped:
        print(f"共跳过 {skipped} 篇文档（JSON 解析失败或无有效题目）")
    if not all_items:
        raise ValueError("LLM 未生成任何有效题目")
    return all_items


def _index_chunks(chunks: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_id: Dict[str, Dict[str, Any]] = {}
    by_url: Dict[str, Dict[str, Any]] = {}
    for chunk in chunks:
        cid = str(chunk.get("chunk_id") or "")
        if cid:
            by_id[cid] = chunk
        url = str((chunk.get("metadata") or {}).get("url") or "")
        if url:
            by_url[url] = chunk
    return {**by_id, **by_url}


def import_thesis_dataset(
    chunks_dir: Path,
    thesis_dir: Path,
    *,
    skip_missing: bool,
) -> List[Dict[str, Any]]:
    chunks = load_chunks(chunks_dir)
    index = _index_chunks(chunks)
    if not index:
        raise ValueError(f"未加载到任何 chunk: {chunks_dir}")

    type_map = [
        (thesis_dir / "TEST_DATABASE_SIMPLE.json", "factual"),
        (thesis_dir / "TEST_DATABASE_OPEN_MINDED.json", "analytic"),
    ]
    items: List[Dict[str, Any]] = []
    for path, qtype in type_map:
        if not path.exists():
            raise FileNotFoundError(f"缺少 thesis 文件: {path}")
        with open(path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        for row in rows:
            thesis_ids = list(row.get("relevant_chunk_ids") or [])
            mapped: List[str] = []
            contexts: List[str] = []
            missing: List[str] = []
            for tid in thesis_ids:
                chunk = index.get(tid)
                if chunk:
                    mapped.append(tid if tid.startswith("http") else str(chunk["chunk_id"]))
                    body = chunk_body(chunk)
                    if body not in contexts:
                        contexts.append(body)
                else:
                    missing.append(tid)
            if missing:
                msg = f"id={row.get('id')}: 语料中缺少 {missing}"
                if skip_missing:
                    print(f"跳过 {msg}")
                    continue
                raise ValueError(msg)
            items.append(
                {
                    "id": f"{qtype}_{row.get('id')}",
                    "question": row.get("question", ""),
                    "type": qtype,
                    "ground_truth": row.get("expected_answer", ""),
                    "ground_truth_contexts": contexts,
                    "relevant_chunk_ids": mapped,
                    "metadata": {"source": "master_thesis_rag"},
                }
            )
    if not items:
        raise ValueError("没有任何 thesis 条目映射成功")
    return items


def write_dataset(items: List[Dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成 RAGAS 评测集")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_llm = sub.add_parser("llm", help="LLM 根据整篇文档生成 Q&A")
    p_llm.add_argument("--chunks-dir", type=Path, default=None)
    p_llm.add_argument("--max-docs", type=int, default=5, help="随机抽取文档数")
    p_llm.add_argument("--per-doc", type=int, default=1, help="每篇文档生成题数（建议 1-2）")
    p_llm.add_argument("--seed", type=int, default=42)
    p_llm.add_argument("--stem", action="append", default=[], help="指定文档（PDF 编号或 lesson URL）")
    p_llm.add_argument("--max-chars", type=int, default=100_000, help="每篇文档送入 LLM 的最大字符数")
    p_llm.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "datasets" / "eval_all.json",
    )

    p_thesis = sub.add_parser("thesis", help="导入 Master_Thesis_RAG 人工标注测试集")
    p_thesis.add_argument("--chunks-dir", type=Path, default=None)
    p_thesis.add_argument("--thesis-dir", type=Path, default=_default_thesis_dir())
    p_thesis.add_argument("--skip-missing", action="store_true")
    p_thesis.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "datasets" / "eval_all.json",
    )

    args = parser.parse_args()
    chunks_dir = args.chunks_dir or _default_chunks_dir()

    if args.mode == "llm":
        items = generate_llm_dataset(
            chunks_dir,
            max_docs=args.max_docs,
            per_doc=args.per_doc,
            seed=args.seed,
            stems=list(args.stem or []),
            max_chars=args.max_chars,
        )
    else:
        items = import_thesis_dataset(
            chunks_dir, args.thesis_dir, skip_missing=args.skip_missing
        )

    write_dataset(items, args.output)
    print(f"已写入 {len(items)} 条 -> {args.output}")


if __name__ == "__main__":
    main()
