"""
航空航天知识库数据准备模块
"""

import json
import logging
import hashlib
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd
from langchain_core.documents import Document

from .document_summary import (
    format_full_document_text,
    resolve_document_summary,
    resolve_parent_context_for_chunk,
)

logger = logging.getLogger(__name__)


class DataPreparationModule:
    """加载 NASA 会议论文分块 JSON 与 Lessons Learned CSV。"""

    DOC_CATEGORIES = {
        "technical_document": "会议论文",
        "lessons_learned": "经验教训",
    }
    CATEGORY_LABELS = list(DOC_CATEGORIES.values())

    def __init__(self, data_path: str, source_mode: str = "aerospace"):
        self.data_path = data_path
        self.source_mode = source_mode
        self.documents: List[Document] = []
        self.chunks: List[Document] = []
        self.parent_child_map: Dict[str, str] = {}
        self._parent_sections: Dict[str, List[Dict[str, Any]]] = {}
        self.llm_module = None

    @staticmethod
    def _parent_id_from_key(key: str) -> str:
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    @staticmethod
    def _doc_title_from_meta(meta: dict, fallback: str = "未知文档") -> str:
        return (
            meta.get("doc_title")
            or meta.get("title")
            or meta.get("file_name")
            or fallback
        )

    def load_documents(self) -> List[Document]:
        logger.info(f"正在从 {self.data_path} 加载航空航天知识库...")
        documents = self._load_aerospace_documents()
        self.documents = documents
        logger.info(f"成功加载 {len(documents)} 个父文档")
        return documents

    def _load_aerospace_documents(self) -> List[Document]:
        root = Path(self.data_path).resolve()
        if not root.exists():
            logger.warning(f"数据目录不存在: {root}")
            return []

        parent_sections: Dict[str, List[Dict[str, Any]]] = {}
        parent_abstracts: Dict[str, str] = {}
        parent_summary_kinds: Dict[str, str] = {}
        prebuilt_chunks: List[Document] = []

        for json_file in root.rglob("*.json"):
            if json_file.name.endswith(".partial.json"):
                continue
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                items = data if isinstance(data, list) else [data]
            except Exception as e:
                logger.warning(f"读取 JSON {json_file} 失败: {e}")
                continue

            for item in items:
                if not isinstance(item, dict) or "content" not in item or "title" not in item:
                    continue

                meta = item.get("metadata") or {}
                file_name = meta.get("file_name") or json_file.stem
                parent_key = f"pdf:{file_name}"
                parent_id = self._parent_id_from_key(parent_key)
                chunk_id = item.get("chunk_id") or str(uuid.uuid4())
                abstract = str(item.get("abstract") or meta.get("abstract") or "").strip()
                summary_kind = str(
                    item.get("summary_kind") or meta.get("summary_kind") or ""
                ).strip()
                if abstract and parent_id not in parent_abstracts:
                    parent_abstracts[parent_id] = abstract
                if summary_kind and parent_id not in parent_summary_kinds:
                    parent_summary_kinds[parent_id] = summary_kind

                section = {
                    "title": item.get("title", ""),
                    "content": item.get("content", ""),
                    "page_number": item.get("page_number", ""),
                }
                parent_sections.setdefault(parent_id, []).append(section)

                chunk_meta = {
                    "source": str(json_file),
                    "parent_id": parent_id,
                    "doc_type": "child",
                    "chunk_id": chunk_id,
                    "title": item.get("title", ""),
                    "doc_title": file_name,
                    "category": "technical_document",
                    "section_level": str(item.get("section_level", "未知")),
                    "source_type": "pdf",
                    "file_name": file_name,
                    "page_number": item.get("page_number", ""),
                    "section_level_raw": item.get("section_level", ""),
                    "download_url": meta.get("download_url", ""),
                    "markdown_path": meta.get("markdown_path", ""),
                    "abstract": abstract,
                    "source_mode": "aerospace",
                }
                prebuilt_chunks.append(
                    Document(page_content=item["content"], metadata=chunk_meta)
                )
                self.parent_child_map[chunk_id] = parent_id

        for csv_file in root.rglob("*.csv"):
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                logger.warning(f"读取 CSV {csv_file} 失败: {e}")
                continue

            for _, row in df.iterrows():
                if row.get("url") == "url":
                    continue

                content_parts = []
                for field, label in [
                    ("subject", "Subject"),
                    ("abstract", "Abstract"),
                    ("driving_event", "Driving Event"),
                    ("lessons_learned", "Lessons Learned"),
                    ("recommendations", "Recommendations"),
                    ("evidence", "Evidence"),
                    ("program_relation", "Program Relation"),
                    ("program_phase", "Program Phase"),
                    ("mission_directorate", "Mission Directorate"),
                    ("topics", "Topics"),
                ]:
                    val = row.get(field)
                    if pd.notna(val) and str(val) != "None":
                        content_parts.append(f"{label}: {val}")

                if not content_parts:
                    continue

                url = str(row.get("url", csv_file.name))
                subject = str(row.get("subject", "NASA Lesson Learned"))
                parent_key = f"lesson:{url}"
                parent_id = self._parent_id_from_key(parent_key)
                content = "\n\n".join(content_parts)
                chunk_id = parent_id
                abstract = str(row.get("abstract") or "").strip()
                if abstract and parent_id not in parent_abstracts:
                    parent_abstracts[parent_id] = abstract

                section = {"title": subject, "content": content, "page_number": ""}
                parent_sections.setdefault(parent_id, []).append(section)

                chunk_meta = {
                    "source": str(csv_file),
                    "parent_id": parent_id,
                    "doc_type": "child",
                    "chunk_id": chunk_id,
                    "title": subject,
                    "doc_title": subject,
                    "category": "lessons_learned",
                    "section_level": "未知",
                    "source_type": "lessons_learned",
                    "url": url,
                    "abstract": abstract,
                    "source_mode": "aerospace",
                }
                prebuilt_chunks.append(
                    Document(page_content=content, metadata=chunk_meta)
                )
                self.parent_child_map[chunk_id] = parent_id

        self._parent_sections = parent_sections

        documents = []
        for parent_id, sections in parent_sections.items():
            first_chunk = next(
                (c for c in prebuilt_chunks if c.metadata.get("parent_id") == parent_id),
                None,
            )
            meta = first_chunk.metadata if first_chunk else {}
            title = self._doc_title_from_meta(meta)
            stored_abstract = (parent_abstracts.get(parent_id) or meta.get("abstract") or "").strip()
            stored_kind = (
                parent_summary_kinds.get(parent_id) or meta.get("summary_kind") or ""
            ).strip()
            summary, summary_kind = resolve_document_summary(
                sections,
                stored_abstract=stored_abstract,
                stored_kind=stored_kind,
            )
            if summary:
                page_content = f"# {title}\n\n{summary}"
                summary_ready = True
            else:
                page_content = self._sections_titles_outline(sections, title)
                summary_ready = False
                summary_kind = "deferred"

            documents.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "source": meta.get("source", str(root)),
                        "parent_id": parent_id,
                        "doc_type": "parent",
                        "doc_title": title,
                        "category": meta.get("category", "technical_document"),
                        "section_level": meta.get("section_level", "未知"),
                        "source_type": meta.get("source_type", "pdf"),
                        "source_mode": "aerospace",
                        "abstract": summary or stored_abstract,
                        "summary_ready": summary_ready,
                        "summary_source": summary_kind,
                    },
                )
            )

        for i, chunk in enumerate(prebuilt_chunks):
            chunk.metadata["batch_index"] = i
            chunk.metadata["chunk_size"] = len(chunk.page_content)

        self.chunks = prebuilt_chunks
        logger.info(
            f"航空航天知识库: {len(documents)} 个父文档, {len(prebuilt_chunks)} 个子块"
        )
        return documents

    @staticmethod
    def _sections_titles_outline(sections: List[Dict[str, Any]], title: str) -> str:
        lines = [f"# {title}", "", "Document sections:"]
        for sec in sections[:30]:
            lines.append(f"- {sec.get('title', 'Section')}")
        if len(sections) > 30:
            lines.append(f"- ... and {len(sections) - 30} more sections")
        return "\n".join(lines)

    def _parent_summary_cache_path(self) -> Path:
        return Path(self.data_path).resolve() / "parent_summaries_cache.json"

    def _load_parent_summary_cache(self) -> Dict[str, str]:
        path = self._parent_summary_cache_path()
        if not path.exists():
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"读取父文档概要缓存失败: {e}")
            return {}

    def _save_parent_summary_cache(self, cache: Dict[str, str]) -> None:
        path = self._parent_summary_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)

    def build_parent_documents(self, use_cache: bool = True) -> None:
        """解析父文档：Abstract → Conclusions → Introduction → LLM；其余在检索时动态扩展。"""
        if not self._parent_sections:
            return

        cache = self._load_parent_summary_cache() if use_cache else {}
        updated = False

        for doc in self.documents:
            parent_id = doc.metadata.get("parent_id")
            if not parent_id:
                continue

            if use_cache and parent_id in cache:
                doc.page_content = cache[parent_id]
                doc.metadata["summary_ready"] = True
                continue

            sections = self._parent_sections.get(parent_id, [])
            if not sections:
                continue

            title = self._doc_title_from_meta(doc.metadata, "Document")
            stored_abstract = str(doc.metadata.get("abstract") or "").strip()
            stored_kind = str(doc.metadata.get("summary_source") or "").strip()
            summary, summary_kind = resolve_document_summary(
                sections,
                stored_abstract=stored_abstract,
                stored_kind=stored_kind,
            )

            if summary:
                doc.page_content = f"# {title}\n\n{summary}"
                doc.metadata["summary_ready"] = True
                doc.metadata["summary_source"] = summary_kind
                doc.metadata["abstract"] = summary
                cache[parent_id] = doc.page_content
                updated = True
                logger.info(f"已使用 {summary_kind} 作为父文档概要: {title}")
            elif self.llm_module:
                full_text = format_full_document_text(sections)
                try:
                    llm_summary = self.llm_module.summarize_document(title, full_text).strip()
                    if llm_summary:
                        doc.page_content = f"# {title}\n\n{llm_summary}"
                        doc.metadata["summary_ready"] = True
                        doc.metadata["summary_source"] = "llm_summary"
                        cache[parent_id] = doc.page_content
                        updated = True
                        logger.info(f"已使用 llm_summary 作为父文档概要: {title}")
                        continue
                except Exception as e:
                    logger.warning(f"LLM 父文档概要失败 ({title}): {e}")
                doc.metadata["summary_ready"] = False
                doc.metadata["summary_source"] = "deferred"
                doc.page_content = self._sections_titles_outline(sections, title)
            else:
                doc.metadata["summary_ready"] = False
                doc.metadata["summary_source"] = "deferred"
                doc.page_content = self._sections_titles_outline(sections, title)

        if updated and use_cache:
            self._save_parent_summary_cache(cache)

    @classmethod
    def get_supported_categories(cls) -> List[str]:
        return cls.CATEGORY_LABELS

    @classmethod
    def get_category_filter_map(cls) -> Dict[str, str]:
        return {
            "会议论文": "technical_document",
            "技术文档": "technical_document",
            "论文": "technical_document",
            "经验教训": "lessons_learned",
            "教训": "lessons_learned",
        }

    def chunk_documents(self) -> List[Document]:
        if not self.documents:
            raise ValueError("请先加载文档")
        if not self.chunks:
            raise ValueError("未找到预分块数据，请先运行 PDF 分块流程")
        logger.info(f"使用预分块数据，共 {len(self.chunks)} 个子块")
        return self.chunks

    def filter_documents_by_category(self, category: str) -> List[Document]:
        return [doc for doc in self.documents if doc.metadata.get("category") == category]

    def get_statistics(self) -> Dict[str, Any]:
        if not self.documents:
            return {}

        categories: Dict[str, int] = {}
        source_types: Dict[str, int] = {}
        for doc in self.documents:
            category = doc.metadata.get("category", "未知")
            categories[category] = categories.get(category, 0) + 1
            st = doc.metadata.get("source_type", "未知")
            source_types[st] = source_types.get(st, 0) + 1

        return {
            "total_documents": len(self.documents),
            "total_chunks": len(self.chunks),
            "categories": categories,
            "source_types": source_types,
            "source_mode": "aerospace",
            "avg_chunk_size": (
                sum(chunk.metadata.get("chunk_size", 0) for chunk in self.chunks)
                / len(self.chunks)
                if self.chunks
                else 0
            ),
        }

    def export_metadata(self, output_path: str):
        metadata_list = []
        for doc in self.documents:
            metadata_list.append(
                {
                    "source": doc.metadata.get("source"),
                    "doc_title": doc.metadata.get("doc_title"),
                    "category": doc.metadata.get("category"),
                    "source_type": doc.metadata.get("source_type"),
                    "content_length": len(doc.page_content),
                }
            )

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, ensure_ascii=False, indent=2)

        logger.info(f"元数据已导出到: {output_path}")

    def get_parent_documents(self, child_chunks: List[Document]) -> List[Document]:
        parent_relevance: Dict[str, int] = {}
        parent_docs_map: Dict[str, Document] = {}
        anchor_chunks: Dict[str, Document] = {}

        for chunk in child_chunks:
            parent_id = chunk.metadata.get("parent_id")
            if not parent_id:
                continue
            parent_relevance[parent_id] = parent_relevance.get(parent_id, 0) + 1
            if parent_id not in anchor_chunks:
                anchor_chunks[parent_id] = chunk
            if parent_id not in parent_docs_map:
                for doc in self.documents:
                    if doc.metadata.get("parent_id") == parent_id:
                        parent_docs_map[parent_id] = doc
                        break

        sorted_parent_ids = sorted(
            parent_relevance.keys(),
            key=lambda x: parent_relevance[x],
            reverse=True,
        )

        parent_docs = []
        for parent_id in sorted_parent_ids:
            base_doc = parent_docs_map.get(parent_id)
            if not base_doc:
                continue

            if base_doc.metadata.get("summary_ready"):
                parent_docs.append(base_doc)
                continue

            sections = self._parent_sections.get(parent_id, [])
            anchor = anchor_chunks[parent_id]
            title = self._doc_title_from_meta(base_doc.metadata)
            llm_summarize = (
                (lambda t, s: self.llm_module.summarize_document(t, s))
                if self.llm_module
                else None
            )
            content, source = resolve_parent_context_for_chunk(
                sections,
                title,
                str(anchor.metadata.get("title", "")),
                anchor.page_content,
                stored_abstract=str(base_doc.metadata.get("abstract") or ""),
                stored_kind=str(base_doc.metadata.get("summary_source") or ""),
                llm_summarize=llm_summarize,
            )
            parent_docs.append(
                Document(
                    page_content=content,
                    metadata={
                        **base_doc.metadata,
                        "summary_source": source,
                        "dynamic_parent": True,
                    },
                )
            )

        parent_info = []
        for doc in parent_docs:
            name = self._doc_title_from_meta(doc.metadata)
            parent_id = doc.metadata.get("parent_id")
            relevance_count = parent_relevance.get(parent_id, 0)
            source = doc.metadata.get("summary_source", "unknown")
            parent_info.append(f"{name}({relevance_count}块,{source})")

        logger.info(
            f"从 {len(child_chunks)} 个子块中找到 {len(parent_docs)} 个去重父文档: {', '.join(parent_info)}"
        )
        return parent_docs
