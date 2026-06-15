"""生成集成模块"""

import logging
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from .query_utils import is_chinese, deepl_translate

logger = logging.getLogger(__name__)

AEROSPACE_SYSTEM_HINT = (
    "You are an aerospace engineering knowledge assistant. "
    "Answer based on NASA technical documents, conference papers, and lessons learned."
)

BASIC_ANSWER_TEMPLATE = (
    AEROSPACE_SYSTEM_HINT
    + """
Answer the user's question using ONLY the context below.
Think step by step: identify the relevant facts in the context, then answer.
Be direct and concise (2-5 sentences). Lead with the final answer, not disclaimers.
If the context supports a partial answer, give what the context states.
Only say the context is insufficient when the question truly cannot be answered at all.
Answer in the same language as the user's question.

User question: {question}

Context:
{context}

Answer:"""
)

EVAL_ANSWER_TEMPLATE = (
    AEROSPACE_SYSTEM_HINT
    + """
You are in evaluation mode. Answer the user's question using ONLY the context below.

Rules:
1. Read all context blocks. Collect every sentence that directly answers the question.
2. Lead with the direct answer. Numbers, names, and units must match the context exactly.
3. If the question asks how something was done, measured, or characterized, list every method or technique named in the context.
4. If the question asks for a value, include related values in the same passage (e.g. units, secondary measurements, conditions).
5. Do NOT invent facts absent from the context.
6. Say "The context does not contain enough information to answer this question." only when no block contains relevant facts.
7. Answer in the same language as the user's question. Be complete; do not omit listed items to stay brief.

User question: {question}

Context:
{context}

Answer:"""
)


class GenerationIntegrationModule:
    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 2048,
        base_url: str = "https://api.openai.com/v1",
        openai_api_key: str = "",
        llm_provider: str = "openai",
        local_llm_base_url: str = "http://localhost:8000/v1",
        local_llm_model: str = "meta-llama/Llama-3.3-70B-Instruct-Turbo",
        local_llm_api_key: str = "not-needed",
        deepl_api_key: str = "",
        deepl_api_url: str = "https://api-free.deepl.com/v2/translate",
        deepl_api_mode: str = "deepl",
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.openai_api_key = openai_api_key
        self.llm_provider = (llm_provider or "openai").lower()
        self.local_llm_base_url = local_llm_base_url
        self.local_llm_model = local_llm_model
        self.local_llm_api_key = local_llm_api_key or "not-needed"
        self.deepl_api_key = deepl_api_key or ""
        self.deepl_api_url = deepl_api_url
        self.deepl_api_mode = deepl_api_mode
        self.llm = None
        self.setup_llm()

    @property
    def provider_label(self) -> str:
        if self.llm_provider == "local":
            return f"本地 LLaMA ({self.local_llm_model})"
        return f"OpenAI 联网 ({self.model_name})"

    def set_provider(self, provider: str) -> None:
        provider = (provider or "openai").lower()
        if provider not in ("openai", "local"):
            raise ValueError(f"不支持的 LLM 提供商: {provider}")
        if provider != self.llm_provider:
            self.llm_provider = provider
            self.setup_llm()

    def setup_llm(self):
        if self.llm_provider == "local":
            model = self.local_llm_model
            base_url = self.local_llm_base_url
            api_key = self.local_llm_api_key
            if not base_url:
                raise ValueError("未配置 LOCAL_LLM_BASE_URL")
        else:
            model = self.model_name
            base_url = self.base_url
            api_key = self.openai_api_key
            if not api_key:
                raise ValueError("未配置 OPENAI_API_KEY")

        self.llm = ChatOpenAI(
            model=model,
            base_url=base_url,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            api_key=api_key,
        )
        logger.info("LLM 初始化完成 (%s)", self.provider_label)

    def translate_to_english(self, query: str) -> str:
        if not is_chinese(query):
            return query

        prompt = PromptTemplate(
            template="""Translate the following aerospace-related user question into natural English for document search.
Return ONLY the English translation, no explanation.

Question: {query}

English:""",
            input_variables=["query"],
        )
        chain = prompt | self.llm | StrOutputParser()
        try:
            result = chain.invoke({"query": query}).strip()
            if result:
                return result
        except Exception as e:
            logger.warning(f"LLM 翻译失败，尝试 DeepL: {e}")

        deepl_result = deepl_translate(
            query,
            api_key=self.deepl_api_key,
            api_url=self.deepl_api_url,
            api_mode=self.deepl_api_mode,
        )
        return deepl_result or query

    def summarize_document(self, title: str, sections_text: str) -> str:
        prompt = ChatPromptTemplate.from_template("""
You are summarizing an aerospace technical document for retrieval context.

Document title: {title}

Document content:
{sections}

Write a concise English overview (350 words or fewer) covering main topic, key findings, and scope.
Output only the summary paragraph.""")
        chain = prompt | self.llm | StrOutputParser()
        return chain.invoke({"title": title, "sections": sections_text}).strip()

    def build_eval_answer_prompt(
        self, query: str, context_docs: List[Document], *, max_length: Optional[int] = None
    ) -> str:
        context = self._build_context(context_docs, max_length=max_length)
        prompt = ChatPromptTemplate.from_template(EVAL_ANSWER_TEMPLATE)
        messages = prompt.format_messages(question=query, context=context)
        return "\n\n".join(str(m.content) for m in messages).strip()

    def generate_eval_answer(
        self, query: str, context_docs: List[Document], *, max_length: Optional[int] = None
    ) -> str:
        context = self._build_context(context_docs, max_length=max_length)
        prompt = ChatPromptTemplate.from_template(EVAL_ANSWER_TEMPLATE)
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query)

    def build_basic_answer_prompt(
        self, query: str, context_docs: List[Document], *, max_length: int = 2000
    ) -> str:
        context = self._build_context(context_docs, max_length=max_length)
        prompt = ChatPromptTemplate.from_template(BASIC_ANSWER_TEMPLATE)
        messages = prompt.format_messages(question=query, context=context)
        return "\n\n".join(str(m.content) for m in messages).strip()

    def generate_basic_answer(
        self, query: str, context_docs: List[Document], *, max_length: int = 2000
    ) -> str:
        context = self._build_context(context_docs, max_length=max_length)
        prompt = ChatPromptTemplate.from_template(BASIC_ANSWER_TEMPLATE)
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query)

    def generate_step_by_step_answer(self, query: str, context_docs: List[Document]) -> str:
        context = self._build_context(context_docs)
        prompt = ChatPromptTemplate.from_template("""
""" + AEROSPACE_SYSTEM_HINT + """
Provide a clear, structured answer using the context.
Answer in the same language as the user's question.

User question: {question}

Context:
{context}

Structure your answer with headings and numbered steps where appropriate.

Answer:""")
        chain = (
            {"question": RunnablePassthrough(), "context": lambda _: context}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        return chain.invoke(query)

    def query_rewrite(self, query: str) -> str:
        prompt = PromptTemplate(
            template="""
Analyze the aerospace-related user query for search. If it is vague, rewrite it to be more specific for retrieval.
If it is already specific, return it unchanged.
Return ONLY the final query text.

Original: {query}

Final query:""",
            input_variables=["query"],
        )
        chain = {"query": RunnablePassthrough()} | prompt | self.llm | StrOutputParser()
        response = chain.invoke(query).strip()
        return response or query

    def query_router(self, query: str) -> str:
        prompt = ChatPromptTemplate.from_template("""
Classify the user question into one of:
- list: user wants a list of documents, topics, or recommendations
- detail: user wants detailed steps or in-depth explanation
- general: everything else

Return only: list, detail, or general

Question: {query}

Classification:""")
        chain = {"query": RunnablePassthrough()} | prompt | self.llm | StrOutputParser()
        result = chain.invoke(query).strip().lower()
        if result in ("list", "detail", "general"):
            return result
        return "general"

    def generate_list_answer(self, query: str, context_docs: List[Document]) -> str:
        if not context_docs:
            return "抱歉，没有找到相关的航空航天知识条目。"
        names = []
        for doc in context_docs:
            name = self._doc_title(doc.metadata)
            if name not in names:
                names.append(name)
        if len(names) == 1:
            return f"相关文档：{names[0]}"
        lines = "\n".join(f"{i+1}. {n}" for i, n in enumerate(names[:10]))
        extra = f"\n\n另有 {len(names)-10} 条未列出。" if len(names) > 10 else ""
        return f"相关文档：\n{lines}{extra}"

    @staticmethod
    def _doc_title(metadata: dict) -> str:
        return (
            metadata.get("doc_title")
            or metadata.get("title")
            or metadata.get("file_name")
            or "未知文档"
        )

    def _build_context(
        self, docs: List[Document], max_length: Optional[int] = 2000
    ) -> str:
        if not docs:
            return "暂无相关信息。"
        context_parts = []
        current_length = 0
        for doc in docs:
            name = self._doc_title(doc.metadata)
            meta = f"【{name}】"
            if doc.metadata.get("category"):
                meta += f" | {doc.metadata['category']}"
            doc_text = f"{meta}\n{doc.page_content}\n"
            if max_length is not None and current_length + len(doc_text) > max_length:
                break
            context_parts.append(doc_text)
            current_length += len(doc_text)
        return "\n" + "=" * 50 + "\n".join(context_parts)
