"""
查询理解模块
- 意图识别：简单问题直答 vs RAG 检索 vs 闲聊
- 指代消解：利用对话历史解析代词
- 安全过滤：过滤有害/不当内容
- 答案验证：生成答案后与源文档交叉验证
"""

import logging
import re
from typing import List, Dict, Any, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

logger = logging.getLogger(__name__)

# ============================================================
# 1. 意图识别
# ============================================================

INTENT_PROMPT = ChatPromptTemplate.from_template("""
你是航空航天领域的查询意图分类器。根据用户问题判断其意图类别。

类别说明：
- chitchat: 问候、闲聊、自我介绍、打招呼等非知识性问题（如"你好"、"你是谁"、"今天天气怎么样"）
- simple: 简单事实性问题，不需要检索文档即可回答（如"1+1等于几"、"什么是牛顿第一定律"）
- rag: 需要检索 NASA 文档/论文/经验教训才能回答的专业问题（如"航天器热真空测试有哪些问题"、"推进系统故障案例"）
- safety: 涉及有害、违法、暴力、色情等不当内容的问题

请只返回一个词：chitchat / simple / rag / safety

用户问题：{query}
意图类别：""")

SAFETY_KEYWORDS = [
    "如何制造炸弹", "怎么伤害", "如何入侵", "如何破解密码",
    "如何偷窃", "如何诈骗", "如何投毒", "如何纵火",
    "hack", "bomb", "kill", "attack", "exploit",
    "色情", "裸露", "暴力", "自杀方法",
]


def classify_intent(query: str, llm) -> str:
    """使用 LLM 对查询进行意图分类。"""
    try:
        chain = INTENT_PROMPT | llm | StrOutputParser()
        result = chain.invoke({"query": query}).strip().lower()
        if result in ("chitchat", "simple", "rag", "safety"):
            return result
        # 兜底：关键词匹配
        return _fallback_intent(query)
    except Exception as e:
        logger.warning(f"意图分类失败: {e}")
        return _fallback_intent(query)


def _fallback_intent(query: str) -> str:
    """基于关键词的兜底意图分类。"""
    q = query.lower().strip()
    # 安全检查
    for kw in SAFETY_KEYWORDS:
        if kw in q:
            return "safety"
    # 闲聊检测
    chitchat_patterns = [
        r"^(你好|hi|hello|hey|嗨|哈喽|早|晚安|再见|拜拜)",
        r"^(你是谁|你叫什么|你的名字)",
        r"^(谢谢|感谢|多谢|辛苦了)",
        r"^(今天天气|现在几点|星期几)",
        r"^(1\+1|1加1|1+1等于)",
    ]
    for pat in chitchat_patterns:
        if re.search(pat, q):
            return "chitchat"
    # 简单事实问题（不涉及航空航天专业词汇）
    aerospace_keywords = [
        "nasa", "航天", "太空", "火箭", "卫星", "轨道", "发射",
        "推进", "热控", "真空", "发动机", "推进器", "飞行器",
        "太空站", "空间站", "月球", "火星", "航天器", "载人",
        "无人", "探测器", "着陆", "返回", "再入", "隔热",
        "经验教训", "故障", "事故", "异常", "测试", "试验",
        "会议论文", "技术报告", "technical", "paper", "conference",
    ]
    for kw in aerospace_keywords:
        if kw in q:
            return "rag"
    return "simple"


# ============================================================
# 2. 指代消解
# ============================================================

REFERENCE_PROMPT = PromptTemplate(
    template="""你是一个指代消解助手。根据对话历史，将用户问题中的代词和指代替换为具体名称。

规则：
- "它"、"那个"、"这个"、"该系统" 等指代需要替换为对话中提到的具体名词
- 如果对话历史为空或无法确定指代对象，保持原样返回
- 只替换代词部分，保持句子其余内容不变
- 返回改写后的问题，不要解释

对话历史：
{history}

用户问题：{query}

改写后的问题：""",
    input_variables=["history", "query"],
)

# 代词模式
PRONOUN_PATTERNS = re.compile(
    r"(它|那个|这个|该|此|其|这些|那些|上面的|下面的|刚才的|之前的|那个文档|该系统|该方法|该技术)"
)


def resolve_references(
    query: str, history: List[Dict[str, str]], llm
) -> str:
    """
    利用对话历史解析指代词。
    history: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """
    if not history:
        return query
    if not PRONOUN_PATTERNS.search(query):
        return query

    # 取最近 3 轮对话作为上下文
    recent = history[-6:] if len(history) > 6 else history
    history_text = "\n".join(
        f"{'用户' if m['role'] == 'user' else '助手'}: {m['content'][:200]}"
        for m in recent
    )

    try:
        chain = REFERENCE_PROMPT | llm | StrOutputParser()
        result = chain.invoke({"history": history_text, "query": query}).strip()
        if result and len(result) > 5:
            logger.info(f"指代消解: '{query}' -> '{result}'")
            return result
        return query
    except Exception as e:
        logger.warning(f"指代消解失败: {e}")
        return query


# ============================================================
# 3. 安全过滤
# ============================================================

SAFETY_CHECK_PROMPT = PromptTemplate(
    template="""请判断以下用户问题是否安全、合规。

判断标准：
1. 不涉及制造武器、爆炸物、毒品等危险物品
2. 不涉及伤害他人或自己的方法
3. 不涉及违法活动（黑客攻击、诈骗、盗窃等）
4. 不涉及色情、暴力等不当内容
5. 不涉及侵犯他人隐私

只返回 "safe" 或 "unsafe"，不要解释。

用户问题：{query}
判断结果：""",
    input_variables=["query"],
)


def safety_check(query: str, llm) -> Tuple[bool, str]:
    """
    安全过滤检查。
    返回 (is_safe, reason)
    """
    q = query.lower().strip()

    # 快速关键词检查
    for kw in SAFETY_KEYWORDS:
        if kw in q:
            return False, f"检测到不安全关键词: {kw}"

    # LLM 深度检查
    try:
        chain = SAFETY_CHECK_PROMPT | llm | StrOutputParser()
        result = chain.invoke({"query": query}).strip().lower()
        if "unsafe" in result or "不安全" in result:
            return False, "LLM 判定该问题可能涉及不当内容"
        return True, ""
    except Exception as e:
        logger.warning(f"安全检查失败: {e}")
        return True, ""


# ============================================================
# 4. 答案验证 (HiT)
# ============================================================

VERIFY_PROMPT = ChatPromptTemplate.from_template("""
你是一个答案验证专家。请检查生成的答案是否与参考文档内容一致。

验证标准：
1. 答案中的事实是否都能在参考文档中找到依据
2. 答案是否包含文档中没有的信息（幻觉）
3. 答案是否准确回答了用户的问题
4. 数字、日期、名称等关键信息是否正确

用户问题：{question}
生成答案：{answer}
参考文档片段：
{context}

请返回一个 JSON 对象，包含：
- "score": 1-10 的评分（10 = 完全准确）
- "issues": 问题列表（如果没有问题返回空数组）
- "verified_answer": 如果需要修正，返回修正后的答案；如果不需要修正，返回原答案

只返回 JSON，不要其他内容。""")


def verify_answer(
    question: str,
    answer: str,
    source_docs: List[Any],
    llm,
) -> Dict[str, Any]:
    """
    验证生成的答案与源文档的一致性。
    返回 {"score": int, "issues": list, "verified_answer": str}
    """
    if not source_docs:
        return {"score": 5, "issues": ["无参考文档可验证"], "verified_answer": answer}

    # 构建上下文
    context_parts = []
    for i, doc in enumerate(source_docs[:3], 1):
        title = doc.metadata.get("title", doc.metadata.get("doc_title", f"文档{i}"))
        content = doc.page_content[:500]
        context_parts.append(f"【{title}】\n{content}")
    context = "\n\n".join(context_parts)

    try:
        chain = VERIFY_PROMPT | llm | StrOutputParser()
        raw = chain.invoke({
            "question": question,
            "answer": answer,
            "context": context,
        }).strip()

        # 尝试解析 JSON
        import json
        # 清理可能的 markdown 代码块包裹
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)

        result = json.loads(raw)
        return {
            "score": min(10, max(1, result.get("score", 5))),
            "issues": result.get("issues", []),
            "verified_answer": result.get("verified_answer", answer),
        }
    except Exception as e:
        logger.warning(f"答案验证失败: {e}")
        return {"score": 5, "issues": [f"验证过程出错: {e}"], "verified_answer": answer}
