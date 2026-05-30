"""
航空航天知识问答 RAG 系统主程序
"""

import logging
from pathlib import Path
from typing import List, Optional

from config import RAGConfig, get_active_config
from rag_modules import (
    DataPreparationModule,
    IndexConstructionModule,
    RetrievalOptimizationModule,
    GenerationIntegrationModule,
)
from rag_modules.generation_integration import GenerationIntegrationModule as GenModule
from rag_modules.query_utils import is_chinese

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
            index_save_path=self.config.index_save_path,
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
            print("加载文档...")
            self.data_module.load_documents()
            self._build_parent_documents()
            print("加载预分块...")
            chunks = self.data_module.chunk_documents()
        else:
            print("未找到已保存的索引，开始构建新索引...")
            print("加载文档...")
            self.data_module.load_documents()
            self._build_parent_documents()
            print("加载预分块...")
            chunks = self.data_module.chunk_documents()
            print("构建向量索引...")
            vectorstore = self.index_module.build_vector_index(chunks)
            print("保存向量索引...")
            self.index_module.save_index()

        print("初始化检索优化...")
        serialize_gpu = (
            getattr(self.index_module, "_faiss_on_gpu", False)
            or self.index_module.embedding_device == "cuda"
        )
        self.retrieval_module = RetrievalOptimizationModule(
            vectorstore,
            chunks,
            serialize_gpu_retrieval=serialize_gpu,
        )

        stats = self.data_module.get_statistics()
        print("\n知识库统计:")
        print(f"   父文档总数: {stats['total_documents']}")
        print(f"   可检索文本块数: {stats['total_chunks']}")
        print(f"   文档类型: {stats.get('categories', {})}")
        print(f"   来源分布: {stats.get('source_types', {})}")
        print("知识库构建完成")

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

    def ask_question(self, question: str, stream: bool = False):
        if not all([self.retrieval_module, self.generation_module]):
            raise ValueError("请先构建知识库")

        print(f"\n用户问题: {question}")

        route_type = self.generation_module.query_router(question)
        print(f"查询类型: {route_type}")

        if route_type == "list":
            rewritten_query = question
            print(f"列表查询保持原样: {question}")
        else:
            print("智能分析查询...")
            rewritten_query = self.generation_module.query_rewrite(question)

        print("检索相关文档...")
        vector_q, bm25_q = self._retrieval_queries(rewritten_query)
        if bm25_q != rewritten_query:
            print(f"检索翻译: {rewritten_query} -> {bm25_q}")

        filters = self._extract_filters_from_query(question)
        if filters:
            print(f"应用过滤条件: {filters}")
            relevant_chunks = self.retrieval_module.metadata_filtered_search(
                rewritten_query,
                filters,
                top_k=self.config.top_k,
                vector_query=vector_q,
                bm25_query=bm25_q,
            )
        else:
            relevant_chunks = self.retrieval_module.hybrid_search(
                rewritten_query,
                top_k=self.config.top_k,
                vector_query=vector_q,
                bm25_query=bm25_q,
            )

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

        if route_type == "list":
            print("生成文档列表...")
            relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
            return self.generation_module.generate_list_answer(question, relevant_docs)

        print("获取完整文档...")
        relevant_docs = self.data_module.get_parent_documents(relevant_chunks)
        doc_names = [self._doc_title(doc.metadata) for doc in relevant_docs]
        if doc_names:
            print(f"找到文档: {', '.join(doc_names)}")

        print("生成回答...")
        if route_type == "detail":
            if stream:
                return self.generation_module.generate_step_by_step_answer_stream(
                    question, relevant_docs
                )
            return self.generation_module.generate_step_by_step_answer(
                question, relevant_docs
            )

        if stream:
            return self.generation_module.generate_basic_answer_stream(
                question, relevant_docs
            )
        return self.generation_module.generate_basic_answer(question, relevant_docs)

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

    def run_interactive(self):
        print("=" * 60)
        print("  航空航天知识问答 RAG 系统")
        print("=" * 60)
        print("基于 NASA 会议论文与 Lessons Learned 的检索增强生成")

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

                stream_choice = input("是否使用流式输出? (y/n, 默认y): ").strip().lower()
                use_stream = stream_choice != "n"

                print("\n回答:")
                if use_stream:
                    for chunk in self.ask_question(user_input, stream=True):
                        print(chunk, end="", flush=True)
                    print("\n")
                else:
                    answer = self.ask_question(user_input, stream=False)
                    print(f"{answer}\n")

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"处理问题时出错: {e}")

        print("\n感谢使用航空航天知识问答 RAG 系统")

def main():
    try:
        rag_system = AerospaceRAGSystem(config=get_active_config())
        rag_system.run_interactive()
    except Exception as e:
        logger.error(f"系统运行出错: {e}")
        print(f"系统错误: {e}")


if __name__ == "__main__":
    main()
