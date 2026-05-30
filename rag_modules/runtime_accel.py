"""
运行时加速：设备探测、FAISS GPU、线程池并行（适用于 OpenAI 兼容 API 等 I/O 任务）。
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Any, Callable, Iterable, List, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

# 向量检索在 GPU 上时串行化，避免并发 embed/search 导致 OOM
_gpu_retrieval_lock = Lock()


def cuda_available() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except ImportError:
        return False


def format_gpu_memory(gpu_id: int = 0) -> str:
    """返回 GPU 显存摘要；无 CUDA 时返回空字符串。"""
    if not cuda_available():
        return ""
    try:
        import torch

        free, total = torch.cuda.mem_get_info(gpu_id)
        allocated = torch.cuda.memory_allocated(gpu_id)
        reserved = torch.cuda.memory_reserved(gpu_id)

        def gibi(n: int) -> float:
            return n / (1024**3)

        return (
            f"GPU{gpu_id} 可用 {gibi(free):.1f}/{gibi(total):.1f} GiB "
            f"(PyTorch 已分配 {gibi(allocated):.1f} GiB, 保留 {gibi(reserved):.1f} GiB)"
        )
    except Exception as e:
        return f"GPU{gpu_id} 显存读取失败: {e}"


def faiss_gpu_available() -> bool:
    try:
        import faiss

        return faiss.get_num_gpus() > 0
    except Exception:
        return False


def resolve_embedding_device(requested: str = "auto") -> str:
    """
    解析嵌入模型设备：auto | cpu | cuda | mps
    """
    key = (requested or "auto").strip().lower()
    if key in ("cpu", "cuda", "mps"):
        return key
    if cuda_available():
        return "cuda"
    try:
        import torch

        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def resolve_faiss_use_gpu(requested: bool, embedding_device: str) -> bool:
    """仅 CUDA + faiss-gpu 可用时启用 FAISS GPU 索引。"""
    if not requested:
        return False
    if embedding_device != "cuda":
        return False
    if not faiss_gpu_available():
        logger.warning(
            "已请求 FAISS GPU，但未检测到 GPU 或未安装 faiss-gpu，将使用 CPU 索引。"
            "可执行: pip uninstall faiss-cpu -y && pip install faiss-gpu"
        )
        return False
    return True


def move_faiss_index_to_gpu(index: Any, gpu_id: int = 0) -> Any:
    import faiss

    res = faiss.StandardGpuResources()
    return faiss.index_cpu_to_gpu(res, gpu_id, index)


def move_faiss_index_to_cpu(gpu_index: Any) -> Any:
    import faiss

    return faiss.index_gpu_to_cpu(gpu_index)


def gpu_retrieval_lock():
    """向量检索 / 嵌入 GPU 路径使用的全局锁。"""
    return _gpu_retrieval_lock


def parallel_map(
    func: Callable[[T], R],
    items: Iterable[T],
    *,
    max_workers: int = 1,
    desc: str = "tasks",
) -> List[R]:
    """
    对 I/O 型任务（如 LLM HTTP）使用线程池并行；max_workers<=1 时顺序执行。
    返回结果顺序与 items 一致。
    """
    item_list = list(items)
    if not item_list:
        return []
    workers = max(1, int(max_workers or 1))
    if workers == 1:
        return [func(x) for x in item_list]

    logger.info("并行 %s: %d 项, workers=%d", desc, len(item_list), workers)
    results: List[Optional[R]] = [None] * len(item_list)

    def _indexed(idx: int, item: T) -> tuple[int, R]:
        return idx, func(item)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [
            executor.submit(_indexed, i, item) for i, item in enumerate(item_list)
        ]
        for fut in as_completed(futures):
            idx, value = fut.result()
            results[idx] = value

    return results  # type: ignore[return-value]
