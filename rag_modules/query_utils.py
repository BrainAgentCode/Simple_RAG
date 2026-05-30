"""
查询语言检测与 DeepL 翻译兜底
"""

import logging
import re
from typing import Optional

import requests

logger = logging.getLogger(__name__)

_CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿]")


def is_chinese(text: str, min_cjk_ratio: float = 0.08) -> bool:
    """粗略判断文本是否以中文为主（用于决定是否翻译检索词）。"""
    if not text or not text.strip():
        return False
    cjk = len(_CJK_RE.findall(text))
    if cjk == 0:
        return False
    if cjk >= 2:
        return True
    return cjk / max(len(text), 1) >= min_cjk_ratio


def deepl_translate(
    text: str,
    *,
    target_lang: str = "EN",
    source_lang: str = "ZH",
    api_key: Optional[str] = None,
    api_url: str = "https://api-free.deepl.com/v2/translate",
    api_mode: str = "deepl",
    timeout: int = 30,
) -> Optional[str]:
    """翻译兜底。默认按 DeepL 接口协议调用；支持自定义兼容接口。"""
    if not api_key:
        logger.warning("未配置 DEEPL_API_KEY，跳过翻译兜底接口")
        return None

    try:
        mode = (api_mode or "deepl").strip().lower()

        if mode == "openai":
            resp = requests.post(
                api_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Translate the user query into natural English for document retrieval. Return only the translation.",
                        },
                        {"role": "user", "content": text},
                    ],
                    "temperature": 0,
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            data = resp.json()
            return (
                data.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
                .strip()
                or None
            )

        resp = requests.post(
            api_url,
            data={
                "auth_key": api_key,
                "text": text,
                "target_lang": target_lang,
                "source_lang": source_lang,
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        translations = data.get("translations") or []
        if translations:
            return translations[0].get("text", "").strip() or None
    except Exception as e:
        logger.warning(f"翻译兜底接口失败: {e}")
    return None
