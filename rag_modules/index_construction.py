"""
索引构建模块
"""

import logging
from typing import List, Optional

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from .runtime_accel import (
    gpu_retrieval_lock,
    move_faiss_index_to_cpu,
    move_faiss_index_to_gpu,
    resolve_embedding_device,
    resolve_faiss_use_gpu,
)

logger = logging.getLogger(__name__)


class IndexConstructionModule:
    """索引构建模块 - 负责向量化和索引构建"""

    def __init__(
        self,
        model_name: str = "BAAI/bge-small-zh-v1.5",
        index_save_path: str = "./vector_index",
        embedding_device: str = "auto",
        faiss_use_gpu: bool = False,
        faiss_gpu_id: int = 0,
        embedding_batch_size: int = 64,
        embedding_encode_batch_size: int = 0,
    ):
        self.model_name = model_name
        self.index_save_path = index_save_path
        self.embedding_device_requested = embedding_device
        self.embedding_device = resolve_embedding_device(embedding_device)
        self.faiss_use_gpu = resolve_faiss_use_gpu(
            faiss_use_gpu, self.embedding_device
        )
        self.faiss_gpu_id = faiss_gpu_id
        self.embedding_batch_size = max(1, embedding_batch_size)
        self.embedding_encode_batch_size = max(0, embedding_encode_batch_size)
        self.embeddings = None
        self.vectorstore = None
        self._faiss_on_gpu = False
        self._resolve_encode_batch_size()
        self.setup_embeddings()

    def _resolve_encode_batch_size(self) -> None:
        if self.embedding_encode_batch_size > 0:
            self.embedding_encode_batch_size = max(1, self.embedding_encode_batch_size)
            return
        if self.embedding_device == "cuda":
            self.embedding_encode_batch_size = min(8, self.embedding_batch_size)
        else:
            self.embedding_encode_batch_size = self.embedding_batch_size
        if (
            self.embedding_device == "cuda"
            and self.embedding_batch_size > self.embedding_encode_batch_size
        ):
            logger.warning(
                "CUDA 嵌入 batch=%d 过大，编码将按 encode_batch=%d 分批"
                "（可通过 EMBEDDING_ENCODE_BATCH_SIZE 调整）",
                self.embedding_batch_size,
                self.embedding_encode_batch_size,
            )

    def setup_embeddings(self, device: Optional[str] = None):
        if device is not None:
            self.embedding_device = resolve_embedding_device(device)
            self.faiss_use_gpu = resolve_faiss_use_gpu(
                self.faiss_use_gpu, self.embedding_device
            )
            self._resolve_encode_batch_size()
        logger.info(
            f"正在初始化嵌入模型: {self.model_name} (device={self.embedding_device})"
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": self.embedding_device},
            encode_kwargs={"normalize_embeddings": True},
        )

        logger.info("嵌入模型初始化完成")

    def _maybe_release_embedding_cache(self) -> None:
        if self.embedding_device != "cuda":
            return
        try:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    def _is_cuda_oom(self, err: BaseException) -> bool:
        msg = str(err).lower()
        return "out of memory" in msg or "cuda error" in msg

    def _embed_texts(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        bs = self.embedding_encode_batch_size
        for i in range(0, len(texts), bs):
            sub = texts[i : i + bs]
            try:
                vectors.extend(self.embeddings.embed_documents(sub))
            except Exception as e:
                if not self._is_cuda_oom(e):
                    raise
                logger.warning(
                    "嵌入 batch OOM (size=%d)，改为逐条编码", len(sub)
                )
                self._maybe_release_embedding_cache()
                for text in sub:
                    vectors.extend(self.embeddings.embed_documents([text]))
                    self._maybe_release_embedding_cache()
            self._maybe_release_embedding_cache()
        return vectors

    def _append_to_faiss(
        self,
        texts: List[str],
        vectors: List[List[float]],
        metadatas: List[dict],
    ) -> None:
        pairs = list(zip(texts, vectors))
        if self.vectorstore is None:
            self.vectorstore = FAISS.from_embeddings(
                text_embeddings=pairs,
                embedding=self.embeddings,
                metadatas=metadatas,
            )
        else:
            self.vectorstore.add_embeddings(
                text_embeddings=pairs,
                metadatas=metadatas,
            )

    def _build_faiss_from_chunks(self, chunks: List[Document]) -> FAISS:
        total = len(chunks)
        encode_bs = self.embedding_encode_batch_size
        logger.info(
            "流式构建 FAISS：共 %d 个文档块，encode_batch=%d",
            total,
            encode_bs,
        )

        self.vectorstore = None
        done = 0
        log_interval = 1000

        for start in range(0, total, encode_bs):
            batch = chunks[start : start + encode_bs]
            texts = [c.page_content for c in batch]
            metadatas = [c.metadata for c in batch]
            vectors = self._embed_texts(texts)
            self._append_to_faiss(texts, vectors, metadatas)

            done += len(batch)
            if done == total or done == len(batch) or done % log_interval == 0:
                logger.info("索引进度: %d / %d", done, total)

        return self.vectorstore

    def _maybe_move_index_to_gpu(self) -> None:
        if not self.faiss_use_gpu or not self.vectorstore:
            return
        if self._faiss_on_gpu:
            return
        try:
            self.vectorstore.index = move_faiss_index_to_gpu(
                self.vectorstore.index, self.faiss_gpu_id
            )
            self._faiss_on_gpu = True
            logger.info("FAISS 索引已加载到 GPU (device %s)", self.faiss_gpu_id)
        except Exception as e:
            logger.warning("FAISS GPU 迁移失败，继续使用 CPU 索引: %s", e)
            self.faiss_use_gpu = False
            self._faiss_on_gpu = False

    def _maybe_move_index_to_cpu_for_save(self) -> None:
        if not self._faiss_on_gpu or not self.vectorstore:
            return
        try:
            self.vectorstore.index = move_faiss_index_to_cpu(self.vectorstore.index)
            self._faiss_on_gpu = False
            logger.info("FAISS 索引已转回 CPU 以便保存")
        except Exception as e:
            logger.warning("FAISS 转 CPU 失败: %s", e)

    def build_vector_index(self, chunks: List[Document]) -> FAISS:
        logger.info("正在构建 FAISS 向量索引...")

        if not chunks:
            raise ValueError("文档块列表不能为空")

        self._maybe_release_embedding_cache()
        try:
            self._build_faiss_from_chunks(chunks)
        except Exception as e:
            if self.embedding_device != "cuda" or not self._is_cuda_oom(e):
                raise
            logger.warning("CUDA 构建索引失败，改用 CPU 重新嵌入并构建 FAISS")
            self.vectorstore = None
            self._faiss_on_gpu = False
            self.faiss_use_gpu = False
            self.setup_embeddings(device="cpu")
            self._maybe_release_embedding_cache()
            self._build_faiss_from_chunks(chunks)

        self._maybe_move_index_to_gpu()

        logger.info(f"向量索引构建完成，包含 {len(chunks)} 个向量")
        return self.vectorstore

    def add_documents(self, new_chunks: List[Document]):
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")

        logger.info(f"正在添加 {len(new_chunks)} 个新文档到索引...")
        self.vectorstore.add_documents(new_chunks)
        logger.info("新文档添加完成")

    def save_index(self):
        if not self.vectorstore:
            raise ValueError("请先构建向量索引")

        self._maybe_move_index_to_cpu_for_save()
        Path(self.index_save_path).mkdir(parents=True, exist_ok=True)
        self.vectorstore.save_local(self.index_save_path)
        logger.info(f"向量索引已保存到: {self.index_save_path}")
        self._maybe_move_index_to_gpu()

    def load_index(self) -> Optional[FAISS]:
        if not self.embeddings:
            self.setup_embeddings()

        if not Path(self.index_save_path).exists():
            logger.info(f"索引路径不存在: {self.index_save_path}，将构建新索引")
            return None

        try:
            self.vectorstore = FAISS.load_local(
                self.index_save_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            index_dim = self.vectorstore.index.d
            embed_dim = self.embeddings.embed_query("test").__len__()
            if index_dim != embed_dim:
                logger.warning(
                    f"向量索引维度 ({index_dim}) 与当前 embedding 模型维度 ({embed_dim}) 不匹配，将重建索引"
                )
                self.vectorstore = None
                return None
            self._maybe_move_index_to_gpu()
            logger.info(f"向量索引已从 {self.index_save_path} 加载")
            return self.vectorstore
        except Exception as e:
            logger.warning(f"加载向量索引失败: {e}，将构建新索引")
            return None

    def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        if not self.vectorstore:
            raise ValueError("请先构建或加载向量索引")

        if self._faiss_on_gpu:
            with gpu_retrieval_lock():
                return self.vectorstore.similarity_search(query, k=k)
        return self.vectorstore.similarity_search(query, k=k)
