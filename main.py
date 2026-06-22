"""
航空航天知识问答 RAG 系统主程序
"""

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import (
    RAGConfig,
    get_active_config,
    prompt_storage_paths,
)
from langchain_core.documents import Document
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule,
)
from rag_modules.generation_integration import GenerationIntegrationModule as GenModule
from rag_modules.query_utils import is_chinese
from rag_modules.query_understanding import (
    classify_intent,
    resolve_references,
    safety_check,
    verify_answer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class AerospaceRAGSystem:
    """面向航空航天场景的检索增强生成系统"""

    def __init__(self, config: RAGConfig = None, llm_provider: Optional[str] = None):
        self.config = config or get_active_config()
        if llm_provider:
            self.config.llm_provider = llm_provider.lower()
        self.data_module = None
        self.index_module = None
        self.retrieval_module = None
        self.generation_module = None

        data_path = self.config.resolve_data_path()
        if not data_path.exists():
            data_path.mkdir(parents=True, exist_ok=True)
            logger.warning(
                f"已创建空数据目录: {data_path}。"
                f"请先运行 python thesis_pipeline/download_nasa_data.py 采集数据。"
            )

    @staticmethod
    def _doc_title(metadata: dict) -> str:
        return GenModule._doc_title(metadata)

    def initialize_system(self):
        print("正在初始化航空航天 RAG 系统...")

        print("初始化数据准备模块...")
        self.data_module = DataPreparationModule(
            str(self.config.resolve_data_path()),
            source_mode="aerospace",
        )

        print("初始化索引构建模块...")
        self.index_module = IndexConstructionModule(
            model_name=self.config.embedding_model,
            index_save_path=str(self.config.resolve_index_path()),
            embedding_device=self.config.embedding_device,
            faiss_use_gpu=self.config.faiss_use_gpu,
            faiss_gpu_id=self.config.faiss_gpu_id,
            embedding_batch_size=self.config.embedding_batch_size,
            embedding_encode_batch_size=self.config.embedding_encode_batch_size,
        )

        print("初始化生成集成模块...")
        self.generation_module = GenerationIntegrationModule(
            model_name=self.config.llm_model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            base_url=self.config.openai_base_url,
            openai_api_key=self.config.openai_api_key,
            llm_provider=self.config.llm_provider,
            local_llm_base_url=self.config.local_llm_base_url,
            local_llm_model=self.config.local_llm_model,
            local_llm_api_key=self.config.local_llm_api_key,
            deepl_api_key=self.config.deepl_api_key,
            deepl_api_url=self.config.deepl_api_url,
            deepl_api_mode=self.config.deepl_api_mode,
        )
        print(f"当前 LLM: {self.generation_module.provider_label}")
        print("系统初始化完成")

    def build_knowledge_base(self):
        print("\n正在构建航空航天知识库...")

        vectorstore = self.index_module.load_index()

        if vectorstore is not None:
            print("成功加载已保存的向量索引")
            chunks = list(vectorstore.docstore._dict.values())
            print(f"从向量索引恢复 {len(chunks)} 个文档块")
            self.data_module.chunks = chunks
            self.data_module.documents = chunks
            self.data_module.parent_documents = chunks
            self._build_parent_documents()
        else:
            print("未找到已保存的索引（或维度不匹配），开始构建新索引...")
            chunks = self._load_chunks_from_pkl()
            if not chunks:
                print("加载文档...")
                self.data_module.load_documents()
                self._build_parent_documents()
                print("加载预分块...")
                chunks = self.data_module.chunk_documents()
            else:
                print(f"从旧索引恢复 {len(chunks)} 个文档块")
                self.data_module.chunks = chunks
                self.data_module.documents = chunks
                self.data_module.parent_documents = chunks
            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)
            print("保存向量索引...")
            self.index_module.save_index()

        print("初始化检索优化...")
        if self.config.reranker_enabled:
            print(f"  reranker: {self.config.reranker_model} (RRF 之后精排)")
        else:
            print("  reranker: 关闭")
        serialize_gpu = (
            getattr(self.index_module, "_faiss_on_gpu", False)
            or self.index_module.embedding_device == "cuda"
        )
        self.retrieval_module = RetrievalOptimizationModule(
            vectorstore,
            chunks,
            serialize_gpu_retrieval=serialize_gpu,
            reranker_model=self.config.reranker_model,
            reranker_enabled=self.config.reranker_enabled,
            reranker_device=self.config.reranker_device,
        )

        stats = self.data_module.get_statistics()
        if stats:
            print("\n知识库统计:")
            print(f"   父文档总数: {stats.get('total_documents', 'N/A')}")
            print(f"   可检索文本块数: {stats.get('total_chunks', 'N/A')}")
            print(f"   文档类型: {stats.get('categories', {})}")
            print(f"   来源分布: {stats.get('source_types', {})}")
        print("知识库构建完成")

    def _load_chunks_from_pkl(self):
        import pickle
        index_path = Path(self.config.index_save_path)
        pkl_path = index_path / "index.pkl"
        if not pkl_path.exists():
            return []
        try:
            print(f"尝试从 {pkl_path} 恢复文档块...")
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            docstore = data[0]
            return list(docstore._dict.values())
        except Exception as e:
            print(f"从 pkl 恢复失败: {e}")
            return []

    def _build_parent_documents(self):
        print(
            "正在解析父文档（Abstract → Conclusions → Introduction → LLM；"
            "缺失时在检索阶段用相邻章节/全文档扩展）..."
        )
        self.data_module.llm_module = self.generation_module
        self.data_module.build_parent_documents(use_cache=True)

    def switch_llm_provider(self, provider: str) -> None:
        provider = provider.lower()
        if provider not in ("openai", "local"):
            raise ValueError("LLM 提供商仅支持 openai 或 local")
        self.config.llm_provider = provider
        if self.generation_module:
            self.generation_module.set_provider(provider)
        print(f"已切换 LLM: {self.generation_module.provider_label}")

    def _retrieval_queries(self, query: str):
        vector_q = query
        bm25_q = query
        if not is_chinese(query):
            return vector_q, bm25_q

        need_en = (
            self.config.translate_query_for_bm25
            or self.config.translate_query_for_vector
        )
        if not need_en:
            return vector_q, bm25_q

        en = self.generation_module.translate_to_english(query)
        if self.config.translate_query_for_bm25:
            bm25_q = en
        if self.config.translate_query_for_vector:
            vector_q = en
        return vector_q, bm25_q

    def route_and_rewrite(
        self, question: str, *, rewrite: bool = True
    ) -> Tuple[str, str]:
        """先路由，再决定是否重写检索词。"""
        route_type = self.generation_module.query_router(question)
        if route_type == "list" or not rewrite:
            return route_type, question
        rewritten = self.generation_module.query_rewrite(question)
        return route_type, rewritten

    def retrieve_documents(
        self,
        question: str,
        *,
        rewritten_query: Optional[str] = None,
        route_type: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> Tuple[List[Document], Dict[str, Any]]:
        """
        完整检索链路：路由/重写 → 中英文检索词 → 元数据过滤 → 混合检索(RRF)。
        """
        if not self.retrieval_module:
            raise ValueError("请先构建知识库")

        if rewritten_query is None or route_type is None:
            route_type, rewritten_query = self.route_and_rewrite(question)

        top_k = top_k or self.config.top_k
        candidate_k = max(top_k * 4, 20)
        vector_q, bm25_q = self._retrieval_queries(rewritten_query)
        filters = self._extract_filters_from_query(question)

        if filters:
            chunks = self.retrieval_module.metadata_filtered_search(
                rewritten_query,
                filters,
                top_k=top_k,
                vector_query=vector_q,
                bm25_query=bm25_q,
                candidate_k=candidate_k,
            )
        else:
            chunks = self.retrieval_module.hybrid_search(
                rewritten_query,
                top_k=top_k,
                vector_query=vector_q,
                bm25_query=bm25_q,
                candidate_k=candidate_k,
            )

        trace: Dict[str, Any] = {
            "route_type": route_type,
            "search_query": rewritten_query,
            "vector_query": vector_q,
            "bm25_query": bm25_q,
            "metadata_filters": filters,
            "candidate_k": candidate_k,
            "top_k": top_k,
        }
        return chunks, trace

    def generate_answer(
        self,
        question: str,
        chunks: List[Document],
        route_type: str = "general",
        *,
        eval_mode: bool = False,
    ) -> Tuple[str, str, List[Document]]:
        gen = self.generation_module
        if not chunks:
            empty = (
                "The context does not contain enough information to answer this question."
                if eval_mode
                else "抱歉，没有找到相关的航空航天知识。请尝试其他关键词或换一种问法。"
            )
            return empty, "", []

        if route_type == "list":
            parents = self.data_module.get_parent_documents(chunks)
            response = gen.generate_list_answer(question, parents or chunks)
            return response, "", parents or chunks

        parents = self.data_module.get_parent_documents(chunks)
        if eval_mode:
            answer_docs = chunks
        else:
            answer_docs = parents or chunks

        if eval_mode:
            prompt = gen.build_eval_answer_prompt(question, answer_docs)
            response = gen.generate_eval_answer(question, answer_docs)
            return response, prompt, answer_docs

        if route_type == "detail":
            response = gen.generate_step_by_step_answer(question, answer_docs)
        else:
            response = gen.generate_basic_answer(question, answer_docs)
        prompt = gen.build_basic_answer_prompt(question, answer_docs)
        return response, prompt, answer_docs

    # ============================================================
    # 查询理解与路由
    # ============================================================

    def understand_query(
        self,
        question: str,
        history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """
        完整查询理解管线：
        1. 指代消解
        2. 意图识别
        3. 安全过滤
        返回: {"intent": str, "resolved_query": str, "safe": bool, "safety_reason": str}
        """
        llm = self.generation_module.llm

        # 1. 指代消解
        resolved = resolve_references(question, history or [], llm)

        # 2. 意图识别
        intent = classify_intent(resolved, llm)

        # 3. 安全过滤
        is_safe, safety_reason = safety_check(resolved, llm)
        if not is_safe:
            intent = "safety"

        return {
            "intent": intent,
            "original_query": question,
            "resolved_query": resolved,
            "safe": is_safe,
            "safety_reason": safety_reason,
        }

    def generate_direct_answer(self, question: str) -> str:
        """简单问题 / 闲聊：直接用 LLM 回答，不检索知识库。"""
        gen = self.generation_module
        if not gen.llm:
            return "抱歉，LLM 未初始化。"
        try:
            from langchain_core.prompts import ChatPromptTemplate
            from langchain_core.output_parsers import StrOutputParser

            direct_prompt = ChatPromptTemplate.from_template(
                "你是一个友好的航空航天知识助手。请用中文简洁地回答用户的问题。\n\n"
                "用户问题：{question}\n\n回答："
            )
            chain = direct_prompt | gen.llm | StrOutputParser()
            return chain.invoke({"question": question}).strip()
        except Exception as e:
            return f"生成回答时出错: {e}"

    def ask_question(self, question: str, history: Optional[List[Dict[str, str]]] = None):
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        # === 查询理解管线 ===
        understanding = self.understand_query(question, history)
        intent = understanding["intent"]
        resolved = understanding["resolved_query"]

        print(f"\n用户问题: {question}")
        print(f"意图识别: {intent}")

        if not understanding["safe"]:
            print(f"安全过滤: {understanding['safety_reason']}")
            return f"⚠️ 该问题被安全过滤拦截: {understanding['safety_reason']}"

        if resolved != question:
            print(f"指代消解: '{question}' -> '{resolved}'")

        # === 意图路由 ===
        if intent in ("simple", "chitchat"):
            print(f"直接回答（{intent}）...")
            return self.generate_direct_answer(resolved)

        if intent == "safety":
            return "⚠️ 该问题涉及不当内容，无法回答。"

        # === RAG 管线 ===
        route_type, rewritten_query = self.route_and_rewrite(resolved)
        print(f"查询类型: {route_type}")
        if route_type == "list":
            print(f"列表查询保持原样: {resolved}")
        else:
            print("智能分析查询...")
            if rewritten_query != resolved:
                print(f"检索词: {rewritten_query}")

        print("检索相关文档...")
        relevant_chunks, trace = self.retrieve_documents(
            resolved,
            rewritten_query=rewritten_query,
            route_type=route_type,
        )
        if trace["bm25_query"] != rewritten_query:
            print(f"检索翻译: {rewritten_query} -> {trace['bm25_query']}")
        if trace["metadata_filters"]:
            print(f"应用过滤条件: {trace['metadata_filters']}")

        if relevant_chunks:
            chunk_info = []
            for chunk in relevant_chunks:
                doc_title = self._doc_title(chunk.metadata)
                content_preview = chunk.page_content[:100].strip()
                if content_preview.startswith("#"):
                    title_end = (
                        content_preview.find("\n")
                        if "\n" in content_preview
                        else len(content_preview)
                    )
                    section_title = (
                        content_preview[:title_end].replace("#", "").strip()
                    )
                    chunk_info.append(f"{doc_title}({section_title})")
                else:
                    chunk_info.append(f"{doc_title}(内容片段)")
            print(f"找到 {len(relevant_chunks)} 个相关文档块: {', '.join(chunk_info)}")
        else:
            print(f"找到 {len(relevant_chunks)} 个相关文档块")

        if not relevant_chunks:
            return "抱歉，没有找到相关的航空航天知识。请尝试其他关键词或换一种问法。"

        print("生成回答...")
        response, _, _ = self.generate_answer(
            question, relevant_chunks, route_type, eval_mode=False
        )
        return response

    def _extract_filters_from_query(self, query: str) -> dict:
        filters = {}
        for keyword, category in DataPreparationModule.get_category_filter_map().items():
            if keyword in query:
                filters["category"] = category
                break
        return filters

    def search_by_category(self, category_label: str, query: str = "") -> List[str]:
        if not self.retrieval_module:
            raise ValueError("请先构建知识库")

        category_key = DataPreparationModule.get_category_filter_map().get(
            category_label, category_label
        )
        search_query = query if query else category_label
        filters = {"category": category_key}

        vq, bq = self._retrieval_queries(search_query)
        docs = self.retrieval_module.metadata_filtered_search(
            search_query, filters, top_k=10, vector_query=vq, bm25_query=bq
        )

        titles = []
        for doc in docs:
            title = self._doc_title(doc.metadata)
            if title not in titles:
                titles.append(title)
        return titles

    @staticmethod
    def prompt_llm_provider(config: RAGConfig) -> str:
        print("\n请选择大语言模型:")
        print(f"  [1] OpenAI 联网模型 ({config.llm_model})")
        print(f"  [2] 本地 LLaMA 3.3-70B-Instruct-Turbo ({config.local_llm_model})")
        print(f"      服务地址: {config.local_llm_base_url}")
        choice = input("请输入 1 或 2 [默认 1]: ").strip()
        return "local" if choice == "2" else "openai"

    def run_interactive(self, *, pick_paths: bool = True):
        print("=" * 60)
        print("  航空航天知识问答 RAG 系统")
        print("=" * 60)
        print("基于 NASA 会议论文与 Lessons Learned 的检索增强生成")

        if pick_paths:
            self.config = prompt_storage_paths(self.config)

        provider = self.prompt_llm_provider(self.config)
        self.config.llm_provider = provider

        self.initialize_system()
        self.build_knowledge_base()

        print("\n交互式问答 (输入「退出」结束)")
        print(f"当前模型: {self.generation_module.provider_label}")
        print("示例: What thermal control issues arose during spacecraft testing?")
        print("示例: 航天器热真空测试中有哪些热控问题？")

        while True:
            try:
                user_input = input("\n您的问题: ").strip()
                if user_input.lower() in ["退出", "quit", "exit", ""]:
                    break

                switch = input("切换模型? [1=OpenAI / 2=本地 / 回车跳过]: ").strip()
                if switch == "1":
                    self.switch_llm_provider("openai")
                elif switch == "2":
                    self.switch_llm_provider("local")

                answer = self.ask_question(user_input)
                print(f"\n{answer}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")

        print("\n感谢使用航空航天知识问答 RAG 系统")

def main():
    parser = argparse.ArgumentParser(description="航空航天 RAG 交互问答")
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="NASA 数据根目录（覆盖 RAG_DATA_PATH）",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default=None,
        help="FAISS 向量索引目录（覆盖 RAG_INDEX_PATH）",
    )
    parser.add_argument(
        "--no-pick-paths",
        action="store_true",
        help="启动时不交互询问数据/索引路径",
    )
    args = parser.parse_args()

    try:
        config = get_active_config().with_storage_paths(
            data_path=args.data_path,
            index_path=args.index_path,
        )
        rag_system = AerospaceRAGSystem(config=config)
        rag_system.run_interactive(
            pick_paths=not args.no_pick_paths and not (args.data_path or args.index_path)
        )
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")


if __name__ == "__main__":
    main()
