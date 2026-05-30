"""
从航空航天文档中构建父文档内容。

优先级：
1. Abstract → Conclusions → Introduction
2. LLM 概要（结构化均无时）
3. 检索命中片段的前、后相邻章节
4. 全文档章节拼接
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

SectionLike = Union[Dict[str, Any], Any]

SUMMARY_KINDS = ("abstract", "conclusions", "introduction")

SUMMARY_TITLE_ALIASES: Dict[str, set] = {
    "abstract": {"abstract", "summary", "executive summary", "摘要"},
    "introduction": {"introduction", "intro", "background", "引言", "背景"},
    "conclusions": {
        "conclusions",
        "conclusion",
        "summary and conclusions",
        "concluding remarks",
        "结论",
    },
}

INLINE_SECTION_PATTERNS = (
    ("abstract", r"(?is)\*\*(?:abstract|summary)\*\*:?\s*(.+?)(?=\n\s*\*\*[^*]+\*\*:?|\Z)"),
    (
        "conclusions",
        r"(?is)\*\*(?:conclusions?)\*\*:?\s*(.+?)(?=\n\s*\*\*(?:figure|references?)\b[^*]*\*\*:?|\n\s*\*\*[^*]+\*\*:?|\Z)",
    ),
    (
        "introduction",
        r"(?is)\*\*introduction\*\*:?\s*(.+?)(?=\n\s*\*\*[^*]+\*\*:?|\Z)",
    ),
)

MIN_SUMMARY_WORDS = 8
MAX_SUMMARY_CHARS = 4000


def normalize_section_title(title: str) -> str:
    normalized = re.sub(r"\*+", "", title or "").strip().lower()
    normalized = re.sub(r"^[ivx]+\.\s+", "", normalized)
    normalized = re.sub(r"^\d+(?:\.\d+)*\.?\s+", "", normalized)
    return normalized.strip(" :.-—–")


def classify_summary_title(title: str) -> Optional[str]:
    normalized = normalize_section_title(title)
    for kind, aliases in SUMMARY_TITLE_ALIASES.items():
        if normalized in aliases:
            return kind
    return None


def clean_summary_text(text: str) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    cleaned = re.sub(r"^\*+\s*|\s*\*+$", "", cleaned)
    return cleaned.strip(" :.-—–")


def _section_title(section: SectionLike) -> str:
    if isinstance(section, dict):
        return str(section.get("title", "")).strip()
    return str(getattr(section, "title", "")).strip()


def _section_content(section: SectionLike) -> str:
    if isinstance(section, dict):
        return str(section.get("content", "")).strip()
    return str(getattr(section, "content", "")).strip()


def _valid_summary(text: str) -> bool:
    cleaned = clean_summary_text(text)
    return bool(cleaned) and len(cleaned.split()) >= MIN_SUMMARY_WORDS


def _clip_summary(text: str) -> str:
    cleaned = clean_summary_text(text)
    if len(cleaned) <= MAX_SUMMARY_CHARS:
        return cleaned
    clipped = cleaned[:MAX_SUMMARY_CHARS].rsplit(" ", 1)[0]
    return clipped + "..."


def format_section_block(section: SectionLike) -> str:
    title = _section_title(section) or "Section"
    content = _section_content(section)
    return f"### {title}\n{content}".strip()


def extract_summary_from_sections(
    sections: Sequence[SectionLike],
) -> Tuple[str, str]:
    """按 abstract → conclusions → introduction 从章节列表抽取概要。"""
    buckets: Dict[str, List[str]] = {kind: [] for kind in SUMMARY_KINDS}

    for section in sections:
        content = _section_content(section)
        if not content:
            continue
        kind = classify_summary_title(_section_title(section))
        if kind:
            buckets[kind].append(content)
            continue

        lower = content.lower()
        if "abstract:" in lower:
            start = lower.find("abstract:") + len("abstract:")
            end = len(content)
            for marker in [
                "\n\nlessons learned:",
                "\n\nrecommendations:",
                "\n\ndriving event:",
                "\n\nevidence:",
                "\n\nintroduction:",
            ]:
                pos = lower.find(marker)
                if pos != -1 and pos > start:
                    end = min(end, pos)
            abstract = content[start:end].strip()
            if abstract:
                buckets["abstract"].append(abstract)

    for kind in SUMMARY_KINDS:
        if not buckets[kind]:
            continue
        merged = clean_summary_text(" ".join(buckets[kind]))
        if _valid_summary(merged):
            return _clip_summary(merged), kind
    return "", ""


def extract_summary_from_markdown(markdown_text: str) -> Tuple[str, str]:
    """从 Markdown 全文抽取概要。"""
    if not markdown_text:
        return "", ""

    for kind, pattern in INLINE_SECTION_PATTERNS:
        match = re.search(pattern, markdown_text)
        if not match:
            continue
        text = _clip_summary(match.group(1))
        if _valid_summary(text):
            return text, kind

    header_patterns = [
        ("abstract", r"(?ms)^#{1,6}\s+\*?\*?(?:abstract|summary)\*?\*?\s*$.*?(?=^#{1,6}\s+|\Z)"),
        ("conclusions", r"(?ms)^#{1,6}\s+\*?\*?conclusions?\*?\*?\s*$.*?(?=^#{1,6}\s+|\Z)"),
        ("introduction", r"(?ms)^#{1,6}\s+\*?\*?introduction\*?\*?\s*$.*?(?=^#{1,6}\s+|\Z)"),
        ("abstract", r"(?ms)^\*\*(?:abstract|summary)\*\*\s*$.*?(?=^\*\*[^*]+\*\*\s*$|^#{1,6}\s+|\Z)"),
        ("conclusions", r"(?ms)^\*\*(?:conclusions?)\*\*\s*$.*?(?=^\*\*[^*]+\*\*\s*$|^#{1,6}\s+|\Z)"),
        ("introduction", r"(?ms)^\*\*introduction\*\*\s*$.*?(?=^\*\*[^*]+\*\*\s*$|^#{1,6}\s+|\Z)"),
    ]
    for kind, pattern in header_patterns:
        match = re.search(pattern, markdown_text, re.IGNORECASE)
        if not match:
            continue
        text, found_kind = extract_summary_from_text(match.group(0))
        if text:
            return text, found_kind or kind

    return extract_summary_from_text(markdown_text)


def extract_summary_from_text(text: str) -> Tuple[str, str]:
    """从纯文本/Markdown 混合内容中抽取概要。"""
    if not text:
        return "", ""

    block_patterns = [
        (
            "abstract",
            r"(?is)(?:^|\n)\s*(?:\*\*)?(?:abstract|summary)(?:\*\*)?\s*[:\-—–]\s*(.+?)"
            r"(?=\n\s*(?:keywords?|index terms?|\*\*[A-Za-z])|\Z)",
        ),
        (
            "conclusions",
            r"(?is)(?:^|\n)\s*(?:\*\*)?conclusions?(?:\*\*)?\s*[:\-—–]\s*(.+?)"
            r"(?=\n\s*(?:\*\*(?:figure|references?)\b|\*\*[A-Za-z])|\Z)",
        ),
        (
            "introduction",
            r"(?is)(?:^|\n)\s*(?:\*\*)?introduction(?:\*\*)?\s*[:\-—–]\s*(.+?)"
            r"(?=\n\s*(?:\*\*[A-Za-z]|main objectives|methods|results|conclusions?\b)|\Z)",
        ),
    ]
    for kind, pattern in block_patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        summary = _clip_summary(match.group(1))
        if _valid_summary(summary):
            return summary, kind

    lines = [re.sub(r"\s+", " ", ln).strip() for ln in text.splitlines()]
    for kind, label in (
        ("abstract", "abstract"),
        ("abstract", "summary"),
        ("conclusions", "conclusions"),
        ("conclusions", "conclusion"),
        ("introduction", "introduction"),
    ):
        capturing = False
        collected: List[str] = []
        blank_run = 0
        for line in lines:
            if not line:
                if capturing:
                    blank_run += 1
                    if blank_run >= 2 and collected:
                        break
                continue
            blank_run = 0
            if not capturing:
                m = re.match(
                    rf"(?i)^(?:\*{{0,2}})?{label}(?:\*{{0,2}})?\s*[:\-—–]?\s*(.*)$",
                    line,
                )
                if m:
                    capturing = True
                    first = m.group(1).strip()
                    if first:
                        collected.append(first)
                    continue
            else:
                if re.match(r"(?i)^(keywords?|index terms?|references?)\b", line):
                    break
                if re.match(r"^(?:\d+(?:\.\d+)*|[IVX]+\.)\s+[A-Z]", line):
                    break
                collected.append(line)
        summary = _clip_summary(" ".join(collected))
        if _valid_summary(summary):
            return summary, kind

    return "", ""


def resolve_document_summary(
    sections: Sequence[SectionLike],
    stored_abstract: str = "",
    stored_kind: str = "",
) -> Tuple[str, str]:
    """文档级概要：Abstract → Conclusions → Introduction。"""
    summary, kind = extract_summary_from_sections(sections)
    if summary:
        return summary, kind

    stored = clean_summary_text(stored_abstract)
    if _valid_summary(stored):
        kind = stored_kind if stored_kind in SUMMARY_KINDS else "abstract"
        return _clip_summary(stored), kind

    return "", ""


def find_section_index(
    sections: Sequence[SectionLike],
    chunk_title: str,
    chunk_content: str = "",
) -> Optional[int]:
    title = (chunk_title or "").strip()
    content_prefix = (chunk_content or "").strip()[:240]

    if title:
        for i, section in enumerate(sections):
            if _section_title(section) == title:
                return i

    if content_prefix:
        for i, section in enumerate(sections):
            body = _section_content(section)
            if content_prefix in body or body[:240] in content_prefix:
                return i

    return None


def build_adjacent_section_context(
    sections: Sequence[SectionLike],
    section_index: int,
) -> str:
    """取当前片段的前、后相邻章节；缺一侧则只保留存在的一侧。"""
    parts: List[str] = []
    if section_index > 0:
        parts.append(format_section_block(sections[section_index - 1]))
    if section_index + 1 < len(sections):
        parts.append(format_section_block(sections[section_index + 1]))
    return "\n\n".join(p for p in parts if p).strip()


def build_full_document_context(
    sections: Sequence[SectionLike],
    title: str,
) -> str:
    blocks = [format_section_block(section) for section in sections if _section_content(section)]
    if not blocks:
        return f"# {title}".strip()
    return f"# {title}\n\n" + "\n\n".join(blocks)


def format_full_document_text(sections: Sequence[SectionLike]) -> str:
    blocks = [format_section_block(section) for section in sections if _section_content(section)]
    return "\n\n".join(blocks)


def resolve_parent_context_for_chunk(
    sections: Sequence[SectionLike],
    title: str,
    chunk_title: str,
    chunk_content: str,
    stored_abstract: str = "",
    stored_kind: str = "",
    llm_summarize: Optional[Callable[[str, str], str]] = None,
) -> Tuple[str, str]:
    """
    为某个命中子块解析父文档上下文。

    返回 (page_content, summary_source)。
    """
    summary, kind = resolve_document_summary(
        sections, stored_abstract=stored_abstract, stored_kind=stored_kind
    )
    if summary:
        return f"# {title}\n\n{summary}", kind

    if llm_summarize:
        full_text = format_full_document_text(sections)
        if full_text.strip():
            try:
                llm_text = clean_summary_text(llm_summarize(title, full_text))
                if _valid_summary(llm_text):
                    return f"# {title}\n\n{_clip_summary(llm_text)}", "llm_summary"
            except Exception:
                pass

    index = find_section_index(sections, chunk_title, chunk_content)
    if index is not None:
        adjacent = build_adjacent_section_context(sections, index)
        if adjacent:
            return f"# {title}\n\n{adjacent}", "adjacent_sections"

    full_doc = build_full_document_context(sections, title)
    if full_doc.strip() and full_doc.strip() != f"# {title}":
        return full_doc, "full_document"

    return f"# {title}", "outline"
