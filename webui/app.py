"""
航空航天知识问答 RAG 系统 - WebUI
所有配置项均可在前端设置，未设置时使用硬编码默认值
"""

import streamlit as st
import sys
import json
import time
import requests
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import RAGConfig, DEFAULT_DATA_PATH, DEFAULT_INDEX_PATH
from main import AerospaceRAGSystem

SETTINGS_PATH = Path(__file__).resolve().parent / "app_settings.json"
CONFIGS_PATH = Path(__file__).resolve().parent / "model_configs.json"

DEFAULTS = {
    "llm_provider": "openai",
    "openai_api_key": "",
    "openai_base_url": "https://api.openai.com/v1",
    "llm_model": "gpt-4o-mini",
    "local_llm_base_url": "http://localhost:8000/v1",
    "local_llm_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo",
    "local_llm_api_key": "not-needed",
    "embedding_model": "BAAI/bge-small-zh-v1.5",
    "embedding_device": "cpu",
    "top_k": 3,
    "temperature": 0.1,
    "max_tokens": 2048,
    "reranker_enabled": True,
    "reranker_model": "BAAI/bge-reranker-v2-m3",
    "reranker_device": "auto",
    "translate_for_bm25": True,
    "translate_for_vector": True,
    "data_path": DEFAULT_DATA_PATH,
    "index_path": DEFAULT_INDEX_PATH,
}

st.set_page_config(
    page_title="航空航天知识问答 - NASA RAG",
    page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><circle cx='50' cy='50' r='48' fill='%230B3D91'/><text x='50' y='58' text-anchor='middle' fill='white' font-size='28' font-weight='bold'>N</text></svg>",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ─── CSS Variables (light/dark) ──────────────────────────────────────── */
:root {
  --bg: #f8f9fa;
  --surface: #ffffff;
  --surface-chat: #ffffff;
  --surface-secondary: #f7f7f8;
  --surface-hover: #f0f0f0;
  --border: #e2e8f0;
  --border-medium: #cdcdcd;
  --code-bg: #f1f5f9;
  --text-primary: #1a202c;
  --text-secondary: #64748b;
  --accent: #0B3D91;
  --accent-hover: #09326e;
  --accent-bg: rgba(11, 61, 145, 0.08);
  --brand-green: #047857;
  --brand-red: #FC3D21;
  --error: #ef4444;
  --error-bg: rgba(239, 68, 68, 0.1);
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0f172a;
    --surface: #1e293b;
    --surface-chat: #1e293b;
    --surface-secondary: #1a2332;
    --surface-hover: #283548;
    --border: #334155;
    --border-medium: #475569;
    --code-bg: #1a2535;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --accent: #3b82f6;
    --accent-hover: #2563eb;
    --accent-bg: rgba(59, 130, 246, 0.1);
    --brand-green: #22c55e;
    --brand-red: #f87171;
    --error: #f87171;
    --error-bg: rgba(248, 113, 113, 0.1);
  }
}

/* ─── Global Reset & Base ─────────────────────────────────────────────── */
.stApp {
  font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif;
  background: var(--bg);
}

/* ─── Sidebar ─────────────────────────────────────────────────────────── */
section[data-testid="stSidebar"] {
  background: var(--surface);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
  display: flex !important;
  flex-direction: column !important;
  align-items: center !important;
  justify-content: center !important;
  padding: 0 !important;
}
section[data-testid="stSidebar"] [data-testid="stLogoSpacer"] {
  display: none !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarCollapseButton"] {
  position: absolute !important;
  right: 8px !important;
  top: 8px !important;
}
section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
  text-align: center !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
  text-align: center !important;
}
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
  color: var(--text-secondary);
  text-align: center !important;
}

/* ─── Layout ──────────────────────────────────────────────────────────── */
section.main .block-container {
  max-width: 55rem !important;
  margin: 0 auto !important;
  padding-top: 1rem !important;
}
header[data-testid="stHeader"] {
  justify-content: center !important;
}
small { display: none !important; height: 0 !important; min-height: 0 !important; }
[data-testid="stFormHelp"] { display: none !important; }
.sidebar-section { display: none !important; }
h3[data-anchor] { text-align: center !important; }
h3[data-anchor] [data-testid="stHeaderActionElements"] { display: none !important; }

/* ─── Chat Messages ───────────────────────────────────────────────────── */
div[data-testid="stChatMessage"] {
  border: none !important;
  border-radius: 0 !important;
  padding: 12px 0 !important;
  background: transparent !important;
  max-width: 78%;
  margin: 0 !important;
}
div[data-testid="stChatMessage"][data-testid]:has(div[data-testid="chatAvatarIcon-user"]) {
  margin-left: auto !important;
}
div[data-testid="stChatMessage"]:has(div[data-testid="chatAvatarIcon-assistant"]) {
  margin-left: 0 !important;
}

/* ─── Chat Input ──────────────────────────────────────────────────────── */
div[data-testid="stChatInput"] {
  background: var(--surface-chat) !important;
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  box-shadow: 0 2px 12px rgba(0, 0, 0, 0.06) !important;
  max-width: 48rem;
  margin: 0 auto !important;
  padding: 0 !important;
}
div[data-testid="stChatInput"] textarea {
  border: none !important;
  box-shadow: none !important;
  font-family: inherit !important;
}

/* ─── Buttons ─────────────────────────────────────────────────────────── */
.stButton > button {
  border-radius: 8px !important;
  font-weight: 500 !important;
  transition: background 0.15s, opacity 0.15s !important;
  font-family: inherit !important;
}
.stButton > button[kind="primary"] {
  background-color: transparent !important;
  border: 1px solid var(--border) !important;
  color: var(--text-primary) !important;
}
.stButton > button[kind="primary"]:hover {
  background: var(--surface-hover) !important;
  border-color: var(--border-medium) !important;
}

/* ─── Sidebar Buttons ─────────────────────────────────────────────────── */
div[data-testid="stSidebar"] .stButton > button {
  width: 100%;
  border-radius: 8px;
  background: transparent;
  border: 1px solid var(--border);
  color: var(--text-primary);
  font-weight: 500;
  text-align: left;
  padding: 8px 12px;
}
div[data-testid="stSidebar"] .stButton > button:hover {
  background: var(--surface-hover);
  border-color: var(--border-medium);
}
div[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: transparent !important;
  border: 1px solid var(--border) !important;
  color: var(--text-primary) !important;
}
div[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
  background: var(--surface-hover) !important;
  border-color: var(--border-medium) !important;
}

/* ─── Source Cards (BESSER-style) ─────────────────────────────────────── */
.source-card {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: 8px;
  background: var(--accent-bg);
  color: var(--accent);
  font-size: .82rem;
  font-weight: 500;
  border: 1px solid var(--border);
  margin: 2px;
  text-decoration: none;
  transition: background 0.15s, color 0.15s;
}
.source-card:hover {
  background: var(--accent);
  color: #fff;
}

/* ─── Route/Intent Tags ───────────────────────────────────────────────── */
.route-tag {
  display: inline-block;
  padding: 3px 12px;
  border-radius: 20px;
  font-size: .72rem;
  font-weight: 600;
  margin-bottom: 8px;
  letter-spacing: 0.02em;
}
.route-list { background: #fef3c7; color: #92400e; }
.route-detail { background: #dbeafe; color: #1e40af; }
.route-general { background: #e0e7ff; color: #3730a3; }
.route-rag { background: var(--accent-bg); color: var(--accent); }
.route-simple { background: #fef9c3; color: #854d0e; }
.route-chitchat { background: #f3e8ff; color: #6b21a8; }
.route-safety { background: var(--error-bg); color: var(--error); }

/* ─── RAG Details Card (BESSER-style collapsible) ────────────────────── */
.rag-details {
  border: 1px solid var(--border);
  border-radius: 10px;
  overflow: hidden;
  margin: 8px 0;
}
.rag-details__summary {
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 8px 14px;
  width: 100%;
  font-size: .85rem;
  font-weight: 500;
  color: var(--text-secondary);
  background: var(--code-bg);
  border: none;
  text-align: left;
  user-select: none;
}
.rag-details__summary:hover {
  background: var(--surface-hover);
}
.rag-details__arrow {
  display: inline-block;
  font-size: .65rem;
  transition: transform 0.25s ease;
}
.rag-details__arrow--open {
  transform: rotate(90deg);
}
.rag-details__body {
  display: grid;
  grid-template-rows: 0fr;
  transition: grid-template-rows 0.3s ease;
}
.rag-details__body--open {
  grid-template-rows: 1fr;
}
.rag-details__body-inner {
  overflow: hidden;
}
.rag-details__content {
  padding: 12px 14px;
  display: flex;
  flex-direction: column;
  gap: 10px;
}
.rag-details__label {
  font-size: .75rem;
  font-weight: 700;
  color: var(--text-secondary);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.rag-doc-card {
  border: 1px solid var(--border);
  border-radius: 8px;
  overflow: hidden;
  font-size: .84rem;
}
.rag-doc-card__header {
  padding: 6px 12px;
  font-size: .78rem;
  font-weight: 600;
  color: var(--text-secondary);
  background: var(--code-bg);
  border-bottom: 1px solid var(--border);
}
.rag-doc-card__row {
  padding: 6px 12px;
  border-bottom: 1px solid var(--border);
  line-height: 1.5;
}
.rag-doc-card__row:last-child { border-bottom: none; }
.rag-doc-card__key {
  font-size: .75rem;
  font-weight: 600;
  color: var(--text-secondary);
  display: block;
  margin-bottom: 2px;
}
.rag-doc-card__value { word-break: break-word; }

/* ─── Verification Badge ──────────────────────────────────────────────── */
.verification-badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 14px;
  border-radius: 8px;
  font-size: .82rem;
  font-weight: 600;
}
.verification-badge--high {
  background: var(--accent-bg);
  color: var(--accent);
  border: 1px solid var(--border);
}
.verification-badge--mid {
  background: rgba(245, 158, 11, 0.1);
  color: #d97706;
  border: 1px solid rgba(245, 158, 11, 0.3);
}
.verification-badge--low {
  background: var(--error-bg);
  color: var(--error);
  border: 1px solid rgba(239, 68, 68, 0.3);
}

/* ─── Status Dot ──────────────────────────────────────────────────────── */
.status-dot {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  flex-shrink: 0;
}
.status-dot--connected { background: var(--accent); }
.status-dot--loading { background: #f59e0b; }
.status-dot--error { background: #ef4444; }
.status-dot--disconnected { background: #94a3b8; }

/* ─── Session List ────────────────────────────────────────────────────── */
.session-item {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 8px 12px;
  border-radius: 8px;
  border-left: 3px solid transparent;
  font-size: .85rem;
  cursor: pointer;
  transition: background 0.15s, border-color 0.15s;
  text-align: left;
  width: 100%;
}
.session-item:hover {
  background: var(--surface-hover);
}
.session-item--active {
  border-left-color: var(--accent);
  background: var(--accent-bg);
  font-weight: 600;
}
.session-item__title {
  flex: 1;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/* ─── Empty State ─────────────────────────────────────────────────────── */
.empty-state {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 12px;
  padding: 48px 24px;
  text-align: center;
}
.empty-state__icon {
  font-size: 2.5rem;
  line-height: 1;
  opacity: 0.6;
}
.empty-state__title {
  font-size: 1.1rem;
  font-weight: 600;
  color: var(--text-primary);
}
.empty-state__desc {
  font-size: .88rem;
  color: var(--text-secondary);
  max-width: 360px;
}

/* ─── Header Gradient ─────────────────────────────────────────────────── */
.header-gradient {
  background: linear-gradient(to bottom, var(--surface-primary, #fff) 60%, transparent);
  padding: 1rem 0 2rem;
  text-align: center;
  margin-bottom: -1rem;
}

/* ─── Feature Cards ───────────────────────────────────────────────────── */
.feature-card {
  padding: 20px;
  border: 1px solid var(--border);
  border-radius: 12px;
  text-align: center;
  transition: box-shadow 0.2s, border-color 0.2s;
}
.feature-card:hover {
  border-color: var(--accent);
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
}
.feature-card__icon { font-size: 1.5rem; margin-bottom: 8px; }
.feature-card__title { font-weight: 600; margin-bottom: 6px; }
.feature-card__desc { font-size: .8rem; color: var(--text-secondary); white-space: pre-line; }

/* ─── Expanders ───────────────────────────────────────────────────────── */
.streamlit-expanderHeader {
  font-weight: 500 !important;
  font-size: .88rem !important;
}

/* ─── Spinner override ────────────────────────────────────────────────── */
.stSpinner > div {
  border-top-color: var(--accent) !important;
}
</style>
""", unsafe_allow_html=True)


# ===== Settings Management =====
def load_settings() -> Dict:
    if SETTINGS_PATH.exists():
        try:
            return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def save_settings(settings: Dict):
    SETTINGS_PATH.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")

def get_setting(key: str):
    s = load_settings()
    return s.get(key, DEFAULTS.get(key))

def get_config_from_settings() -> RAGConfig:
    s = load_settings()
    d = DEFAULTS.copy()
    d.update({k: v for k, v in s.items() if v != "" and v is not None})
    return RAGConfig(
        data_path=d["data_path"],
        index_save_path=d["index_path"],
        llm_provider=d["llm_provider"],
        openai_api_key=d.get("openai_api_key", ""),
        openai_base_url=d["openai_base_url"],
        llm_model=d["llm_model"],
        local_llm_base_url=d["local_llm_base_url"],
        local_llm_model=d["local_llm_model"],
        local_llm_api_key=d.get("local_llm_api_key", "not-needed"),
        embedding_model=d["embedding_model"],
        embedding_device=d["embedding_device"],
        top_k=d["top_k"],
        temperature=d["temperature"],
        max_tokens=d["max_tokens"],
        reranker_enabled=d["reranker_enabled"],
        reranker_model=d["reranker_model"],
        reranker_device=d["reranker_device"],
        translate_query_for_bm25=d["translate_for_bm25"],
        translate_query_for_vector=d["translate_for_vector"],
        deepl_api_key=d.get("deepl_api_key", ""),
        deepl_api_url=d.get("deepl_api_url", "https://api-free.deepl.com/v2/translate"),
        deepl_api_mode=d.get("deepl_api_mode", "deepl"),
    )


# ===== Model Config Management =====
def load_model_configs() -> List[Dict]:
    if CONFIGS_PATH.exists():
        try:
            return json.loads(CONFIGS_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return []

def save_model_configs(configs: List[Dict]):
    CONFIGS_PATH.write_text(json.dumps(configs, ensure_ascii=False, indent=2), encoding="utf-8")

def fetch_models_from_api(base_url: str, api_key: str) -> List[str]:
    url = base_url.rstrip("/")
    if not url.endswith("/models"):
        url = url.rstrip("/") + "/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        models = []
        if isinstance(data, dict) and "data" in data:
            for m in data["data"]:
                mid = m.get("id", "")
                if mid:
                    models.append(mid)
        return sorted(models)
    except Exception:
        return []


# ===== Session State =====
def init_session_state():
    for k, v in {
        "rag_system": None, "system_ready": False,
        "conversations": {}, "current_conv_id": None,
        "messages": [], "show_settings": False,
        "fetched_models": None, "fetched_base_url": "",
        "fetched_api_key": "", "fetched_name": "",
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v


def init_rag_system():
    if st.session_state.system_ready and st.session_state.rag_system:
        return st.session_state.rag_system
    try:
        config = get_config_from_settings()
        rag = AerospaceRAGSystem(config=config)
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
        st.stop()
    try:
        with st.spinner("正在初始化系统模块..."):
            rag.initialize_system()
        with st.spinner("正在构建知识库..."):
            rag.build_knowledge_base()
    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")
        st.stop()
    st.session_state.rag_system = rag
    st.session_state.system_ready = True
    return rag


def new_conversation():
    cid = f"conv_{int(time.time()*1000)}"
    st.session_state.conversations[cid] = {"title": "新对话", "messages": []}
    st.session_state.current_conv_id = cid
    st.session_state.messages = []
    st.rerun()

def switch_conv(cid):
    st.session_state.current_conv_id = cid
    st.session_state.messages = st.session_state.conversations[cid]["messages"]
    st.rerun()

def delete_conv(cid):
    st.session_state.conversations.pop(cid, None)
    if st.session_state.current_conv_id == cid:
        if st.session_state.conversations:
            k = next(iter(st.session_state.conversations))
            st.session_state.current_conv_id = k
            st.session_state.messages = st.session_state.conversations[k]["messages"]
        else:
            new_conversation()
    st.rerun()


# ===== RAG Processing =====
def format_sources(chunks):
    sources, seen = [], set()
    for c in chunks:
        m = c.metadata
        title = m.get("source_file", m.get("title", "未知来源"))
        if title not in seen:
            seen.add(title)
            sources.append({"title": title, "url": m.get("url", "")})
    return sources

def render_sources_html(sources):
    if not sources:
        return ""
    tags = []
    for s in sources:
        if s["url"]:
            tags.append(f'<a class="source-card" href="{s["url"]}" target="_blank">{s["title"]}</a>')
        else:
            tags.append(f'<span class="source-card">{s["title"]}</span>')
    return " ".join(tags)


def render_rag_details_html(extra):
    """BESSER-style collapsible RAG details card."""
    parts = []

    # Sources
    sources = extra.get("sources", [])
    if sources:
        doc_cards = []
        for i, s in enumerate(sources):
            url_part = f'<a href="{s["url"]}" target="_blank" style="color:var(--accent);text-decoration:none;">{s["title"]}</a>' if s["url"] else s["title"]
            doc_cards.append(f'''
            <div class="rag-doc-card">
              <div class="rag-doc-card__header">Document {i+1}/{len(sources)}</div>
              <div class="rag-doc-card__row">
                <span class="rag-doc-card__key">Source</span>
                <span class="rag-doc-card__value">{url_part}</span>
              </div>
            </div>''')

        parts.append(f'''
        <div class="rag-details">
          <div class="rag-details__content">
            <div class="rag-details__label">References ({len(sources)} documents)</div>
            {"".join(doc_cards)}
          </div>
        </div>''')

    # Verification
    verification = extra.get("verification")
    if verification:
        score = verification.get("score", 5)
        level = "high" if score >= 7 else "mid" if score >= 4 else "low"
        score_color = "#16a34a" if score >= 7 else "#d97706" if score >= 4 else "#ef4444"
        issues_html = ""
        for issue in verification.get("issues", []):
            issues_html += f'<div style="font-size:.82rem;color:var(--text-secondary);padding:2px 0;">- {issue}</div>'
        parts.append(f'''
        <div class="rag-details">
          <div class="rag-details__content">
            <div class="rag-details__label">Answer Verification</div>
            <div class="verification-badge verification-badge--{level}">Score: {score}/10</div>
            {issues_html}
          </div>
        </div>''')

    return "".join(parts)


def process_query(question, rag, history=None):
    try:
        understanding = rag.understand_query(question, history)
        intent = understanding["intent"]
        resolved = understanding["resolved_query"]
        if not understanding["safe"]:
            return f"⚠️ {understanding['safety_reason']}", {"intent": intent, "original": question, "rewritten": resolved, "sources": []}
        if intent in ("simple", "chitchat"):
            response = rag.generate_direct_answer(resolved)
            return response, {"intent": intent, "original": question, "rewritten": resolved, "sources": []}
        route_type, rewritten = rag.route_and_rewrite(resolved)
        chunks, trace = rag.retrieve_documents(resolved, rewritten_query=rewritten, route_type=route_type)
        response, _, docs = rag.generate_answer(resolved, chunks, route_type, eval_mode=False)
        llm = rag.generation_module.llm
        from rag_modules.query_understanding import verify_answer
        verification = verify_answer(resolved, response, docs, llm)
        sources = format_sources(docs if docs else chunks)
        return response, {"intent": intent, "route_type": route_type, "original": question, "rewritten": rewritten, "resolved": resolved, "sources": sources, "verification": verification}
    except Exception as e:
        return f"{type(e).__name__}: {e}", {}


NASA_SVG = '<svg width="32" height="32" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0"><circle cx="50" cy="50" r="48" fill="#0B3D91"/><text x="50" y="58" text-anchor="middle" fill="white" font-family="Arial,sans-serif" font-size="28" font-weight="bold">N</text><ellipse cx="50" cy="50" rx="46" ry="18" fill="none" stroke="#FC3D21" stroke-width="3" transform="rotate(-20 50 50)"/></svg>'


# ===== Sidebar =====
def render_sidebar_not_ready():
    st.markdown(
        f'<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:8px 0 4px;">'
        f'{NASA_SVG}'
        f'<span style="font-size:.95rem;font-weight:600;color:var(--text-primary);">航空航天知识问答</span></div>',
        unsafe_allow_html=True,
    )

    configs = load_model_configs()
    if configs:
        all_models = []
        for cfg in configs:
            for m in cfg.get("models", []):
                all_models.append(f"{cfg.get('name', '未知')}/{m}")
        if all_models:
            sel = st.selectbox("快捷选择模型", all_models, label_visibility="collapsed")
            if sel:
                name, model = sel.split("/", 1)
                for cfg in configs:
                    if cfg.get("name") == name and model in cfg.get("models", []):
                        settings = load_settings()
                        settings["llm_provider"] = "openai"
                        settings["openai_api_key"] = cfg.get("api_key", "")
                        settings["openai_base_url"] = cfg.get("base_url", "")
                        settings["llm_model"] = model
                        save_settings(settings)
                        break
    else:
        st.caption("暂无已配置的模型")

    if st.button("⚙️ 高级设置", use_container_width=True):
        st.session_state.show_settings = True
        st.rerun()

    if st.button("🚀 启动系统", type="primary", use_container_width=True):
        init_rag_system()
        st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-size:.8rem;color:var(--text-secondary);line-height:1.6;text-align:left;">
    <b>功能特点</b><br>
    <span style="display:flex;align-items:center;gap:6px;margin:4px 0;">🔍 混合检索 + RRF 融合</span>
    <span style="display:flex;align-items:center;gap:6px;margin:4px 0;">🎯 Cross-Encoder 精排</span>
    <span style="display:flex;align-items:center;gap:6px;margin:4px 0;">🌐 中英文双语支持</span>
    <span style="display:flex;align-items:center;gap:6px;margin:4px 0;">🧠 智能查询路由</span>
    <span style="display:flex;align-items:center;gap:6px;margin:4px 0;">🛡️ 意图识别与安全过滤</span>
    </div>""", unsafe_allow_html=True)


def render_sidebar_ready():
    rag = st.session_state.rag_system
    model_label = rag.generation_module.provider_label
    st.markdown(
        f'<div style="display:flex;flex-direction:column;align-items:center;justify-content:center;padding:8px 0 4px;">'
        f'{NASA_SVG}'
        f'<span style="font-size:.95rem;font-weight:600;color:var(--text-primary);">航空航天知识问答</span></div>'
        f'<div style="font-size:.8rem;color:var(--text-secondary);margin:0 0 8px;text-align:center;display:flex;align-items:center;justify-content:center;gap:6px;">'
        f'<span class="status-dot status-dot--connected"></span>'
        f'当前模型: <b>{model_label}</b></div>',
        unsafe_allow_html=True,
    )
    if st.button("+ 新建对话", use_container_width=True):
        new_conversation()

    if not st.session_state.conversations:
        st.markdown(
            '<div class="empty-state" style="padding:20px 0;">'
            '<div style="font-size:.82rem;color:var(--text-secondary);">暂无对话</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        for cid, conv in reversed(list(st.session_state.conversations.items())):
            is_active = cid == st.session_state.current_conv_id
            if is_active:
                st.markdown(
                    f'<div class="session-item session-item--active">'
                    f'<span class="session-item__title">{conv["title"]}</span></div>',
                    unsafe_allow_html=True,
                )
            else:
                if st.button(f"💬 {conv['title']}", key=f"c_{cid}", use_container_width=True):
                    switch_conv(cid)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("⚙️ 设置", use_container_width=True):
            st.session_state.show_settings = True
            st.rerun()
    with col2:
        if st.button("🔄 切换模型", use_container_width=True):
            st.session_state.system_ready = False
            st.session_state.rag_system = None
            st.rerun()


# ===== Settings Page =====
def render_settings_page():
    st.markdown("### ⚙️ 系统设置")
    st.caption("所有配置均可在此修改，未填写的项将使用默认值")

    saved = load_settings()

    with st.expander("🤖 大语言模型 (LLM)", expanded=True):
        provider = st.selectbox("模型提供商", ["openai", "local"],
            index=0 if saved.get("llm_provider", DEFAULTS["llm_provider"]) == "openai" else 1)
        if provider == "openai":
            api_key = st.text_input("OpenAI API Key", value=saved.get("openai_api_key", ""), type="password", placeholder="sk-...")
            base_url = st.text_input("OpenAI Base URL", value=saved.get("openai_base_url", DEFAULTS["openai_base_url"]))
        else:
            api_key = st.text_input("本地模型 API Key", value=saved.get("local_llm_api_key", DEFAULTS["local_llm_api_key"]), type="password")
            base_url = st.text_input("本地模型 Base URL", value=saved.get("local_llm_base_url", DEFAULTS["local_llm_base_url"]), placeholder="http://localhost:8000/v1")

        api_models = []
        if base_url:
            with st.spinner("正在加载模型列表..."):
                api_models = fetch_models_from_api(base_url, api_key)

        saved_model = saved.get("llm_model", DEFAULTS["llm_provider"] == "openai" and DEFAULTS["llm_model"] or DEFAULTS["local_llm_model"])
        if provider == "local":
            saved_model = saved.get("local_llm_model", DEFAULTS["local_llm_model"])
        else:
            saved_model = saved.get("llm_model", DEFAULTS["llm_model"])

        if api_models:
            use_custom = saved_model not in api_models
            model_choice = st.selectbox("模型名称", api_models + ["自定义..."],
                index=len(api_models) if use_custom else (api_models.index(saved_model) if saved_model in api_models else 0))
            if model_choice == "自定义...":
                model = st.text_input("自定义模型名称", value=saved_model if use_custom else "", placeholder="输入模型名称")
            else:
                model = model_choice
        else:
            model = st.text_input("模型名称", value=saved_model, placeholder="输入模型名称（API 未返回模型列表）")

        test_url = base_url.rstrip("/")
        if not test_url.endswith("/chat/completions"):
            test_url = test_url.rstrip("/") + "/chat/completions"
        if st.button("🔗 测试连接", use_container_width=True):
            try:
                headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
                payload = {"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}
                resp = requests.post(test_url, json=payload, headers=headers, timeout=30)
                if resp.status_code == 200:
                    st.success(f"✅ 连接成功！模型 {model} 可用")
                else:
                    st.error(f"❌ 请求失败 (HTTP {resp.status_code}): {resp.text[:200]}")
            except Exception as e:
                st.error(f"❌ 连接失败: {e}")

        temperature = st.slider("Temperature", 0.0, 2.0, saved.get("temperature", DEFAULTS["temperature"]), 0.1)
        max_tokens = st.number_input("最大 Token 数", 256, 16384, saved.get("max_tokens", DEFAULTS["max_tokens"]), 256)

    with st.expander("📦 已保存的模型配置", expanded=False):
        configs = load_model_configs()
        if configs:
            for i, cfg in enumerate(configs):
                name = cfg.get('name', '未知')
                models = cfg.get('models', [])
                st.markdown(f"**{name}** — {len(models)} 个模型")
                st.caption(f"地址: `{cfg.get('base_url', '')}`")
                st.caption(f"模型: {', '.join(models[:5])}")
                if st.button("删除", key=f"del_{i}", type="secondary"):
                    configs.pop(i)
                    save_model_configs(configs)
                    st.rerun()
                st.divider()
        with st.form("add_model_form", clear_on_submit=True):
            st.markdown("**添加新模型配置**")
            m_name = st.text_input("名称", placeholder="我的 OpenAI")
            m_url = st.text_input("API 地址", placeholder="https://api.openai.com/v1")
            m_key = st.text_input("API 密钥", type="password", placeholder="sk-...")
            if st.form_submit_button("获取模型列表"):
                if m_url:
                    models = fetch_models_from_api(m_url, m_key)
                    if models:
                        st.session_state.fetched_models = models
                        st.session_state.fetched_url = m_url
                        st.session_state.fetched_key = m_key
                        st.session_state.fetched_name = m_name or m_url
                        st.success(f"找到 {len(models)} 个模型")
                    else:
                        st.warning("未获取到模型列表")
        if st.session_state.get("fetched_models"):
            selected = st.multiselect("选择模型", st.session_state.fetched_models,
                default=st.session_state.fetched_models[:5], key="msel")
            if st.button("保存配置"):
                new_cfg = {"name": st.session_state.fetched_name, "base_url": st.session_state.fetched_url,
                           "api_key": st.session_state.fetched_key, "models": selected}
                existing = load_model_configs()
                existing = [c for c in existing if c.get("base_url") != new_cfg["base_url"]]
                existing.append(new_cfg)
                save_model_configs(existing)
                st.session_state.fetched_models = None
                st.success("已保存")
                st.rerun()

    with st.expander("📐 向量嵌入", expanded=False):
        emb_model = st.text_input("嵌入模型", value=saved.get("embedding_model", DEFAULTS["embedding_model"]))
        emb_device = st.selectbox("嵌入设备", ["auto", "cpu", "cuda"], index=["auto","cpu","cuda"].index(saved.get("embedding_device", DEFAULTS["embedding_device"])))

    with st.expander("🔍 检索设置", expanded=False):
        top_k = st.number_input("检索数量 (Top K)", 1, 20, saved.get("top_k", DEFAULTS["top_k"]))
        trans_bm25 = st.checkbox("查询翻译用于 BM25", value=saved.get("translate_for_bm25", DEFAULTS["translate_for_bm25"]))
        trans_vec = st.checkbox("查询翻译用于向量检索", value=saved.get("translate_for_vector", DEFAULTS["translate_for_vector"]))

    with st.expander("🎯 Reranker 精排", expanded=False):
        rerank_enabled = st.checkbox("启用 Reranker", value=saved.get("reranker_enabled", DEFAULTS["reranker_enabled"]))
        rerank_model = st.text_input("Reranker 模型", value=saved.get("reranker_model", DEFAULTS["reranker_model"]))
        rerank_device = st.selectbox("Reranker 设备", ["auto", "cpu", "cuda"],
            index=["auto","cpu","cuda"].index(saved.get("reranker_device", DEFAULTS["reranker_device"])))

    with st.expander("📂 数据路径", expanded=False):
        data_path = st.text_input("数据目录", value=saved.get("data_path", DEFAULTS["data_path"]))
        index_path = st.text_input("向量索引目录", value=saved.get("index_path", DEFAULTS["index_path"]))

        if st.button("🔍 测试数据路径", use_container_width=True):
            from pathlib import Path
            dp = Path(data_path).resolve()
            ip = Path(index_path).resolve()
            if dp.exists():
                json_files = list(dp.rglob("*.json"))
                csv_files = list(dp.rglob("*.csv"))
                total = len(json_files) + len(csv_files)
                if total > 0:
                    st.success(f"✅ 数据目录正常: {len(json_files)} 个 JSON + {len(csv_files)} 个 CSV")
                else:
                    st.error(f"❌ 数据目录为空: {dp}")
            else:
                st.error(f"❌ 数据目录不存在: {dp}")
            if ip.exists():
                faiss_files = list(ip.glob("*.faiss"))
                if faiss_files:
                    st.success(f"✅ 向量索引存在: {len(faiss_files)} 个索引文件")
                else:
                    st.warning(f"⚠️ 向量索引目录存在但无 .faiss 文件，启动系统时会自动构建")
            else:
                st.info(f"ℹ️ 向量索引目录不存在: {ip}（启动系统时会自动创建）")

    st.markdown("---")
    if st.button("💾 保存设置", type="primary", use_container_width=True):
        settings = {
            "llm_provider": provider,
            "openai_api_key": api_key if provider == "openai" else "",
            "openai_base_url": base_url if provider == "openai" else DEFAULTS["openai_base_url"],
            "llm_model": model if provider == "openai" else DEFAULTS["llm_model"],
            "local_llm_base_url": base_url if provider == "local" else DEFAULTS["local_llm_base_url"],
            "local_llm_model": model if provider == "local" else DEFAULTS["local_llm_model"],
            "local_llm_api_key": api_key if provider == "local" else DEFAULTS["local_llm_api_key"],
            "temperature": temperature, "max_tokens": max_tokens,
            "embedding_model": emb_model, "embedding_device": emb_device,
            "top_k": top_k, "translate_for_bm25": trans_bm25, "translate_for_vector": trans_vec,
            "reranker_enabled": rerank_enabled, "reranker_model": rerank_model, "reranker_device": rerank_device,
            "data_path": data_path, "index_path": index_path,
        }
        save_settings(settings)
        st.success("设置已保存")
        st.session_state.show_settings = False
        if st.session_state.system_ready:
            st.session_state.system_ready = False
            st.session_state.rag_system = None
        st.rerun()

    if st.button("返回", use_container_width=True):
        st.session_state.show_settings = False
        st.rerun()


# ===== Main Interface =====
def render_main_not_ready():
    st.markdown('<div class="header-gradient">', unsafe_allow_html=True)
    st.markdown(
        f'<div style="display:flex;justify-content:center;margin-bottom:12px;">'
        f'<svg width="56" height="56" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">'
        f'<circle cx="50" cy="50" r="48" fill="#0B3D91"/><text x="50" y="58" text-anchor="middle" fill="white" font-family="Arial,sans-serif" font-size="28" font-weight="bold">N</text>'
        f'<ellipse cx="50" cy="50" rx="46" ry="18" fill="none" stroke="#FC3D21" stroke-width="3" transform="rotate(-20 50 50)"/></svg></div>',
        unsafe_allow_html=True,
    )
    st.markdown("### 航空航天知识问答系统")
    st.markdown("基于 NASA 会议论文与 Lessons Learned 的检索增强生成")
    st.markdown('</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    for col, (icon, title, desc) in zip(cols, [
        ("🔍", "智能检索", "向量相似度 + BM25 关键词\nRRF 融合 + Cross-Encoder 精排"),
        ("🧠", "查询理解", "意图识别 + 指代消解\n简单问题直答 + 安全过滤"),
        ("📊", "答案验证", "生成后与源文档交叉验证\n可信度评分 1-10"),
    ]):
        with col:
            st.markdown(
                f'<div class="feature-card">'
                f'<div class="feature-card__icon">{icon}</div>'
                f'<div class="feature-card__title">{title}</div>'
                f'<div class="feature-card__desc">{desc}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    st.info("请在左侧点击「⚙️ 高级设置」配置模型，然后点击「🚀 启动系统」开始使用。")


def _render_assistant_extras(extra):
    """Render metadata tags and collapsible details for an assistant message."""
    if not extra:
        return

    # Intent tag
    intent = extra.get("intent")
    if intent:
        il = {"rag": "RAG检索", "simple": "直接回答", "chitchat": "闲聊", "safety": "安全过滤"}
        ic = {"rag": "rag", "simple": "simple", "chitchat": "chitchat", "safety": "safety"}
        st.markdown(
            f'<span class="route-tag route-{ic.get(intent, "general")}">{il.get(intent, intent)}</span>',
            unsafe_allow_html=True,
        )

    # Route type tag
    rt = extra.get("route_type")
    if rt:
        rl = {"list": "列表查询", "detail": "详细分析", "general": "通用查询"}
        st.markdown(
            f'<span class="route-tag route-{rt}">{rl.get(rt, "通用查询")}</span>',
            unsafe_allow_html=True,
        )

    # Reference resolution
    if extra.get("resolved") and extra["resolved"] != extra.get("original"):
        with st.expander("🔄 指代消解", expanded=False):
            st.caption(f"原始: {extra['original']}")
            st.caption(f"消解: {extra['resolved']}")

    # Query rewrite
    if extra.get("rewritten") and extra["rewritten"] != extra.get("resolved", extra.get("original")):
        with st.expander("🔍 检索优化", expanded=False):
            st.caption(f"检索词: {extra['rewritten']}")

    # BESSER-style RAG details (sources + verification in one card)
    details_html = render_rag_details_html(extra)
    if details_html:
        st.markdown(details_html, unsafe_allow_html=True)


def render_main_ready():
    if st.session_state.current_conv_id is None:
        if st.session_state.conversations:
            cid = next(iter(st.session_state.conversations))
            st.session_state.current_conv_id = cid
            st.session_state.messages = st.session_state.conversations[cid]["messages"]
        else:
            new_conversation()

    # Render existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("extra"):
                _render_assistant_extras(msg["extra"])
            st.markdown(msg["content"])

    # Input
    if prompt := st.chat_input("输入您的航空航天相关问题..."):
        conv = st.session_state.conversations[st.session_state.current_conv_id]
        if not conv["messages"]:
            conv["title"] = prompt[:30] + ("..." if len(prompt) > 30 else "")
        st.session_state.messages.append({"role": "user", "content": prompt})
        conv["messages"] = st.session_state.messages

        with st.chat_message("user"):
            st.markdown(prompt)

        rag = st.session_state.rag_system
        with st.chat_message("assistant"):
            with st.spinner("正在分析问题并检索知识库..."):
                response, extra = process_query(prompt, rag, st.session_state.messages)
            _render_assistant_extras(extra)
            st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response, "extra": extra})
        conv["messages"] = st.session_state.messages


def main():
    init_session_state()
    with st.sidebar:
        if st.session_state.show_settings:
            render_settings_page()
        elif st.session_state.system_ready:
            render_sidebar_ready()
        else:
            render_sidebar_not_ready()
    if not st.session_state.show_settings:
        if st.session_state.system_ready:
            render_main_ready()
        else:
            render_main_not_ready()


if __name__ == "__main__":
    main()
