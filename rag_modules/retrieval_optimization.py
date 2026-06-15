"""检索优化：混合检索 + RRF + Cross-Encoder 精排。"""

import logging
from typing import Any, Dict, List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .runtime_accel import gpu_retrieval_lock, resolve_embedding_device

logger = logging.getLogger(__name__)


class RetrievalOptimizationModule:
    def __init__(
        self,
        vectorstore: FAISS,
        chunks: List[Document],
        *,
        serialize_gpu_retrieval: bool = False,
        reranker_model: str = "",
        reranker_enabled: bool = True,
        reranker_device: str = "auto",
    ):
        self.vectorstore = vectorstore
        self.chunks = chunks
        self.serialize_gpu_retrieval = serialize_gpu_retrieval
        self.reranker_model = reranker_model
        self.reranker_enabled = reranker_enabled
        self.reranker_device = reranker_device
        self._cross_encoder = None
        self.bm25_retriever = BM25Retriever.from_documents(chunks, k=20)
        if self.reranker_enabled and self.reranker_model:
            logger.info("Reranker 已启用: %s", self.reranker_model)
        else:
            logger.info("Reranker 已关闭，检索仅使用 hybrid + RRF")

    def hybrid_search(
        self,
        query: str,
        top_k: int = 3,
        vector_query: str = None,
        bm25_query: str = None,
        candidate_k: Optional[int] = None,
    ) -> List[Document]:
        vq = vector_query if vector_query is not None else query
        bq = bm25_query if bm25_query is not None else query
        ck = candidate_k or max(top_k * 4, 20)

        if self.serialize_gpu_retrieval:
            with gpu_retrieval_lock():
                vector_docs = self.vectorstore.similarity_search(vq, k=ck)
        else:
            vector_docs = self.vectorstore.similarity_search(vq, k=ck)

        self.bm25_retriever.k = ck
        bm25_docs = self.bm25_retriever.invoke(bq)

        pool = self._rrf_rerank(vector_docs, bm25_docs)[: max(top_k * 3, 20)]
        return self._cross_encoder_rerank(vq, pool, top_k)

    def metadata_filtered_search(
        self,
        query: str,
        filters: Dict[str, Any],
        top_k: int = 5,
        vector_query: str = None,
        bm25_query: str = None,
        candidate_k: Optional[int] = None,
    ) -> List[Document]:
        ck = candidate_k or max(top_k * 4, 20)
        docs = self.hybrid_search(
            query,
            top_k * 3,
            vector_query=vector_query,
            bm25_query=bm25_query,
            candidate_k=ck,
        )
        filtered = []
        for doc in docs:
            match = True
            for key, value in filters.items():
                if key not in doc.metadata:
                    match = False
                    break
                if isinstance(value, list):
                    if doc.metadata[key] not in value:
                        match = False
                        break
                elif doc.metadata[key] != value:
                    match = False
                    break
            if match:
                filtered.append(doc)
                if len(filtered) >= top_k:
                    break
        return filtered

    def _cross_encoder_rerank(
        self, query: str, docs: List[Document], top_k: int
    ) -> List[Document]:
        if not docs or not self.reranker_enabled or not self.reranker_model:
            return docs[:top_k]
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            device = resolve_embedding_device(self.reranker_device)
            logger.info("加载 reranker: %s (device=%s)", self.reranker_model, device)
            kwargs: Dict[str, Any] = {"device": device}
            if "zerank" in self.reranker_model.lower():
                kwargs["trust_remote_code"] = True
                if device == "cuda":
                    kwargs["model_kwargs"] = {"torch_dtype": "auto"}
            try:
                self._cross_encoder = CrossEncoder(self.reranker_model, **kwargs)
            except Exception as exc:
                raise RuntimeError(
                    f"无法加载 reranker `{self.reranker_model}`: {exc}\n"
                    "zerank-2-reranker 约 4B 参数，Colab 需足够 GPU 显存；"
                    "可改用 RERANKER_MODEL=BAAI/bge-reranker-v2-m3 或 RERANKER_ENABLED=false"
                ) from exc

        pairs = [(query, d.page_content or "") for d in docs]
        scores = self._cross_encoder.predict(pairs)
        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        out = []
        for doc, score in ranked[:top_k]:
            doc.metadata["rerank_score"] = float(score)
            out.append(doc)
        return out

    def _rrf_rerank(
        self, vector_docs: List[Document], bm25_docs: List[Document], k: int = 60
    ) -> List[Document]:
        doc_scores: Dict[Any, float] = {}
        doc_objects: Dict[Any, Document] = {}

        for rank, doc in enumerate(vector_docs):
            doc_id = doc.metadata.get("chunk_id") or hash(doc.page_content)
            doc_objects[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        for rank, doc in enumerate(bm25_docs):
            doc_id = doc.metadata.get("chunk_id") or hash(doc.page_content)
            doc_objects[doc_id] = doc
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        reranked = []
        for doc_id, score in sorted(doc_scores.items(), key=lambda x: x[1], reverse=True):
            if doc_id in doc_objects:
                doc = doc_objects[doc_id]
                doc.metadata["rrf_score"] = score
                reranked.append(doc)
        return reranked
