"""
航空航天知识问答 RAG 系统 - WebUI (UltraRAG Style)
"""
import streamlit as st
import streamlit.components.v1 as components
import sys
import json
import time
import hashlib
import requests
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAGConfig, DEFAULT_DATA_PATH, DEFAULT_INDEX_PATH
from main import AerospaceRAGSystem

SETTINGS_PATH = Path(__file__).resolve().parent / "app_settings.json"
CONFIGS_PATH = Path(__file__).resolve().parent / "model_configs.json"

DEFAULTS = {
    "llm_provider": "openai", "openai_api_key": "", "openai_base_url": "https://api.openai.com/v1",
    "llm_model": "gpt-4o-mini", "local_llm_base_url": "http://localhost:8000/v1",
    "local_llm_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo", "local_llm_api_key": "not-needed",
    "embedding_model": "BAAI/bge-small-zh-v1.5", "embedding_device": "auto",
    "faiss_use_gpu": False, "faiss_gpu_id": 0, "embedding_batch_size": 64, "embedding_encode_batch_size": 0,
    "top_k": 3, "temperature": 0.1, "max_tokens": 2048, "llm_max_workers": 4,
    "reranker_enabled": True, "reranker_model": "BAAI/bge-reranker-v2-m3", "reranker_device": "auto",
    "translate_for_bm25": True, "translate_for_vector": True,
    "deepl_api_key": "", "deepl_api_url": "https://api-free.deepl.com/v2/translate", "deepl_api_mode": "deepl",
    "data_path": DEFAULT_DATA_PATH, "index_path": DEFAULT_INDEX_PATH,
}

st.set_page_config(page_title="NASA RAG", page_icon="data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 100 100'><circle cx='50' cy='50' r='48' fill='%230B3D91'/><text x='50' y='58' text-anchor='middle' fill='white' font-size='28' font-weight='bold'>N</text></svg>", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""<style>
header, footer, nav, .stAppHeader, .stAppToolbar, #MainMenu, .stDeployButton, #stDecoration, [data-testid="stStatusWidget"], [data-testid="stSidebar"] { display: none !important; }
html, body, #root, .stApp, .stAppViewContainer, .stMain, .stMainBlockContainer, .stVerticalBlock, .stElementContainer { margin: 0 !important; padding: 0 !important; height: 100vh !important; max-height: 100vh !important; overflow: hidden !important; width: 100vw !important; max-width: 100vw !important; }
.stCustomComponentV1, iframe[data-testid="stCustomComponentV1"] { width: 100vw !important; height: 100vh !important; border: none !important; margin: 0 !important; padding: 0 !important; }
.stSkeleton, [data-testid="stSkeleton"] { display: none !important; }
.stVerticalBlock > * { margin: 0 !important; padding: 0 !important; }
</style>""", unsafe_allow_html=True)


def load_settings() -> Dict:
    if SETTINGS_PATH.exists():
        try: return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        except: pass
    return {}

def save_settings(s: Dict):
    SETTINGS_PATH.write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")

def get_config_from_settings() -> RAGConfig:
    s, d = load_settings(), DEFAULTS.copy()
    d.update({k: v for k, v in s.items() if v != "" and v is not None})
    return RAGConfig(data_path=d["data_path"], index_save_path=d["index_path"], llm_provider=d["llm_provider"],
        openai_api_key=d.get("openai_api_key", ""), openai_base_url=d["openai_base_url"], llm_model=d["llm_model"],
        local_llm_base_url=d["local_llm_base_url"], local_llm_model=d["local_llm_model"], local_llm_api_key=d.get("local_llm_api_key", "not-needed"),
        embedding_model=d["embedding_model"], embedding_device=d["embedding_device"],
        faiss_use_gpu=d.get("faiss_use_gpu", False), faiss_gpu_id=d.get("faiss_gpu_id", 0),
        embedding_batch_size=d.get("embedding_batch_size", 64), embedding_encode_batch_size=d.get("embedding_encode_batch_size", 0),
        top_k=d["top_k"], temperature=d["temperature"], max_tokens=d["max_tokens"], llm_max_workers=d.get("llm_max_workers", 4),
        reranker_enabled=d["reranker_enabled"], reranker_model=d["reranker_model"], reranker_device=d["reranker_device"],
        translate_query_for_bm25=d["translate_for_bm25"], translate_query_for_vector=d["translate_for_vector"],
        deepl_api_key=d.get("deepl_api_key", ""), deepl_api_url=d.get("deepl_api_url", ""), deepl_api_mode=d.get("deepl_api_mode", "deepl"))


def init_session_state():
    for k, v in {"rag_system": None, "system_ready": False, "conversations": {}, "current_conv_id": None, "messages": [], "pending_action": None}.items():
        if k not in st.session_state: st.session_state[k] = v

def init_rag_system():
    if st.session_state.system_ready and st.session_state.rag_system: return st.session_state.rag_system
    try: rag = AerospaceRAGSystem(config=get_config_from_settings())
    except Exception as e: st.error(f"{type(e).__name__}: {e}"); st.stop()
    try:
        with st.spinner("初始化系统模块..."): rag.initialize_system()
        with st.spinner("构建知识库..."): rag.build_knowledge_base()
    except Exception as e: st.error(f"{type(e).__name__}: {e}"); st.stop()
    st.session_state.rag_system = rag; st.session_state.system_ready = True; return rag

def new_conversation():
    cid = f"conv_{int(time.time()*1000)}"; st.session_state.conversations[cid] = {"title": "新对话", "messages": []}
    st.session_state.current_conv_id = cid; st.session_state.messages = []

def switch_conv(cid):
    if cid in st.session_state.conversations:
        st.session_state.current_conv_id = cid; st.session_state.messages = st.session_state.conversations[cid]["messages"]

def delete_conv(cid):
    st.session_state.conversations.pop(cid, None)
    if st.session_state.current_conv_id == cid:
        if st.session_state.conversations:
            k = next(iter(st.session_state.conversations)); st.session_state.current_conv_id = k; st.session_state.messages = st.session_state.conversations[k]["messages"]
        else: new_conversation()

def format_sources(chunks):
    sources, seen = [], set()
    for c in chunks:
        title = c.metadata.get("source_file", c.metadata.get("title", "未知来源"))
        if title not in seen: seen.add(title); sources.append({"title": title, "url": c.metadata.get("url", "")})
    return sources

def render_rag_details(extra):
    parts = []
    for i, s in enumerate(extra.get("sources", [])):
        url = f'<a href="{s["url"]}" target="_blank" style="color:#2563eb;text-decoration:none;">{s["title"]}</a>' if s["url"] else s["title"]
        parts.append(f'<div class="rag-doc-card"><div class="rag-doc-card__header">Document {i+1}/{len(extra["sources"])}</div><div class="rag-doc-card__row"><span class="rag-doc-card__key">Source</span><span class="rag-doc-card__value">{url}</span></div></div>')
    v = extra.get("verification")
    if v:
        sc, lv = v.get("score", 5), "high" if v.get("score", 5) >= 7 else "mid" if v.get("score", 5) >= 4 else "low"
        issues = "".join(f'<div style="font-size:.82rem;color:#6e6e80;padding:2px 0;">- {i}</div>' for i in v.get("issues", []))
        parts.append(f'<div class="rag-details"><div class="rag-details__content"><div class="rag-details__label">Answer Verification</div><div class="verification-badge verification-badge--{lv}">Score: {sc}/10</div>{issues}</div></div>')
    return "".join(parts)

def build_extras_html(extra):
    parts = []
    intent = extra.get("intent")
    if intent:
        il = {"rag": "RAG检索", "simple": "直接回答", "chitchat": "闲聊", "safety": "安全过滤"}
        ic = {"rag": "rag", "simple": "simple", "chitchat": "chitchat", "safety": "safety"}
        parts.append(f'<span class="route-tag route-{ic.get(intent, "general")}">{il.get(intent, intent)}</span>')
    rt = extra.get("route_type")
    if rt:
        rl = {"list": "列表查询", "detail": "详细分析", "general": "通用查询"}
        parts.append(f'<span class="route-tag route-{rt}">{rl.get(rt, "通用查询")}</span>')
    details = render_rag_details(extra)
    if details: parts.append(details)
    return "".join(parts)

def process_query(question, rag, history=None):
    try:
        u = rag.understand_query(question, history); intent, resolved = u["intent"], u["resolved_query"]
        if not u["safe"]: return f"⚠️ {u['safety_reason']}", {"intent": intent, "sources": []}
        if intent in ("simple", "chitchat"): return rag.generate_direct_answer(resolved), {"intent": intent, "sources": []}
        rt, rw = rag.route_and_rewrite(resolved); chunks, _ = rag.retrieve_documents(resolved, rewritten_query=rw, route_type=rt)
        response, _, docs = rag.generate_answer(resolved, chunks, rt)
        from rag_modules.query_understanding import verify_answer
        verification = verify_answer(resolved, response, docs, rag.generation_module.llm)
        return response, {"intent": intent, "route_type": rt, "sources": format_sources(docs if docs else chunks), "verification": verification, "original": question, "rewritten": rw, "resolved": resolved}
    except Exception as e: return f"{type(e).__name__}: {e}", {}


def build_ultrarag_html():
    s = load_settings(); msgs = st.session_state.messages; convs = st.session_state.conversations
    cur_id = st.session_state.current_conv_id or ""; ready = st.session_state.system_ready
    import html as h

    msgs_html = ""
    for m in msgs:
        extras = build_extras_html(m["extra"]) if m["role"] == "assistant" and m.get("extra") else ""
        msgs_html += f'<div class="chat-bubble {m["role"]}"><div class="msg-content">{extras}{h.escape(m["content"]).replace(chr(10), "<br>")}</div></div>'

    sessions_html = ""
    if convs:
        for k in reversed(list(convs.keys())):
            t = h.escape(convs[k].get("title", "新对话"))
            a = " active" if k == cur_id else ""
            sessions_html += f'<div class="chat-session-item{a}" onclick="switchSession(\'{k}\')"><span class="session-title">{t}</span><button class="chat-session-delete" onclick="event.stopPropagation();deleteSession(\'{k}\')" title="删除">✕</button></div>'
    else: sessions_html = '<div style="padding:8px 12px;font-size:.82rem;color:#94a3b8;">No history</div>'

    ph = "向 NASA RAG 提问" if ready else "请先在设置中配置模型..."
    dis = "" if ready else "disabled"
    sj = h.escape(json.dumps(s, ensure_ascii=False))

    scroll_js = "" if not msgs else "setTimeout(function(){var el=document.getElementById('chat-history');if(el)el.scrollTop=el.scrollHeight;},100);"

    JS_CODE = (
        "var stReady=" + str(ready).lower() + ";var stSettings=" + sj + ";\n"
        "function toggleProvider(v){var o=(v==='openai');['f_llm_model','f_openai_key','f_openai_url'].forEach(function(i){var e=document.getElementById(i);if(e)e.style.display=o?'':'none'});['f_local_url','f_local_key','f_local_model'].forEach(function(i){var e=document.getElementById(i);if(e)e.style.display=o?'none':''})}\n"
        "function openSettings(){document.getElementById('settingsOverlay').classList.add('show')}\n"
        "function closeSettings(){document.getElementById('settingsOverlay').classList.remove('show')}\n"
        "document.getElementById('settings-menu-trigger').addEventListener('click',function(e){e.stopPropagation();document.getElementById('settings-menu').classList.toggle('open')});\n"
        "document.addEventListener('click',function(e){if(!document.getElementById('settings-menu').contains(e.target))document.getElementById('settings-menu').classList.remove('open')});\n"
        "document.getElementById('settings-model').addEventListener('click',function(){document.getElementById('settings-menu').classList.remove('open');openSettings()});\n"
        "function submitSettings(){var fd=new FormData(document.getElementById('settingsForm'));var d={};fd.forEach(function(v,k){d[k]=v});window.parent.postMessage({type:'settings_submit',data:JSON.stringify(d)},'*');closeSettings()}\n"
        "function submitChat(){var i=document.getElementById('chat-input');if(!i||i.disabled)return;var v=i.value.trim();if(!v)return;i.value='';window.parent.postMessage({type:'chat_submit',data:v},'*')}\n"
        "document.getElementById('chat-form').addEventListener('submit',function(e){e.preventDefault();submitChat()});\n"
        "document.getElementById('chat-input').addEventListener('keydown',function(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();submitChat()}});\n"
        "document.getElementById('chat-input').addEventListener('input',function(){this.style.height='auto';this.style.height=Math.min(this.scrollHeight,200)+'px'});\n"
        "document.getElementById('chat-new-btn').addEventListener('click',function(){window.parent.postMessage({type:'new_conv'},'*')});\n"
        "function switchSession(id){window.parent.postMessage({type:'switch_conv',data:id},'*')}\n"
        "function deleteSession(id){window.parent.postMessage({type:'delete_conv',data:id},'*')}\n"
        "function loadSettings(s){if(!s)return;['llm_provider','llm_model','openai_api_key','openai_base_url','local_llm_base_url','local_llm_model','local_llm_api_key','temperature','max_tokens','llm_max_workers','embedding_model','embedding_device','faiss_use_gpu','faiss_gpu_id','embedding_batch_size','embedding_encode_batch_size','top_k','translate_for_bm25','translate_for_vector','reranker_enabled','reranker_model','reranker_device','deepl_api_key','deepl_api_url','deepl_api_mode','data_path'].forEach(function(f){var e=document.getElementById('s_'+f);if(e&&s[f]!==undefined)e.value=s[f]});toggleProvider(s.llm_provider||'openai')}\n"
        "loadSettings(stSettings);\n"
        + scroll_js
    )

    CSS = """,*::before,*::after{box-sizing:border-box;margin:0;padding:0}:root{--bg-body:#fff;--bg-surface:#f9f9fa;--bg-card:#fff;--text-primary:#1a1a1a;--text-secondary:#6e6e80;--text-tertiary:#9ca3af;--border-subtle:#e5e5e5;--accent-blue:#2563eb;--gradient-ai:linear-gradient(135deg,#667eea 0%,#764ba2 50%,#f093fb 100%);--radius-sm:6px;--radius-md:12px;--radius-lg:16px;--shadow-sm:0 1px 2px rgba(0,0,0,.05);--font-sans:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;--font-mono:'SFMono-Regular',Consolas,monospace;--bg-input:#eff1f5;--bg-sidebar:#f9f9fa;--error:#ef4444;--error-bg:rgba(239,68,68,.1)}html,body{font-family:var(--font-sans);background:var(--bg-body);color:var(--text-primary);-webkit-font-smoothing:antialiased;line-height:1.6;height:100vh;overflow:hidden}.view-overlay{position:fixed;inset:0;background:var(--bg-surface);z-index:1000;overflow:hidden}.chat-layout{display:flex;width:100%;height:100%;overflow:hidden}.chat-sidebar{width:260px;background:var(--bg-sidebar);border-right:1px solid var(--border-subtle);display:flex;flex-direction:column;flex-shrink:0}.sidebar-header{padding:1.2rem .8rem;display:flex;flex-direction:column;gap:4px}.sidebar-toggle-wrapper{display:flex;justify-content:space-between;align-items:center;margin-bottom:.8rem;min-height:32px;padding-left:20px!important;padding-right:20px!important}.sidebar-logo-btn{border:none;background:transparent;padding:0;margin:0;display:flex;align-items:center;cursor:pointer;gap:8px}.btn-nav{width:100%;display:flex;align-items:center;gap:10px;background:transparent;border:none;padding:9px 12px 9px 20px;font-family:inherit;font-size:.88rem;font-weight:500;color:#0d0d0d;cursor:pointer;border-radius:0}.btn-nav .icon{color:#0d0d0d;display:flex;align-items:center}.btn-nav:hover{background:rgba(0,0,0,.05)}.btn-nav .btn-text{flex:1;text-align:left}.sidebar-list{flex:1;overflow-y:auto;padding:0}.sidebar-label{font-size:.68rem;font-weight:500;color:#94a3b8;margin:1rem 20px .5rem 20px;text-transform:uppercase;letter-spacing:.06em}.chat-session-bar{display:flex;align-items:center;justify-content:space-between;padding:12px 12px 12px 20px;margin-top:8px}.chat-session-bar-title{font-size:.7rem;font-weight:600;color:#8f8f8f;text-transform:uppercase;letter-spacing:.05em}.chat-session-bar-actions{display:flex;gap:4px}.chat-session-bar-btn{background:transparent;border:none;padding:6px;border-radius:6px;cursor:pointer;color:#8f8f8f;transition:all .15s;display:flex;align-items:center;justify-content:center}.chat-session-bar-btn:hover{background:rgba(239,68,68,.1);color:#ef4444}.session-list{display:flex;flex-direction:column;gap:.25rem;padding:0 8px}.chat-session-item{border:none;width:100%;text-align:left;background:transparent;border-radius:8px;padding:10px 12px;display:flex;align-items:center;justify-content:space-between;gap:8px;color:var(--text-primary);cursor:pointer;font-size:.9rem;font-family:inherit;transition:background .15s}.chat-session-item:hover{background:rgba(0,0,0,.05)}.chat-session-item.active{background:#e0e7ff;color:var(--accent-blue);font-weight:500}.chat-session-item .session-title{flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.chat-session-delete{opacity:0;visibility:hidden;background:transparent;border:none;padding:4px 6px;border-radius:6px;cursor:pointer;color:var(--text-tertiary);transition:all .15s;font-size:.8rem}.chat-session-item:hover .chat-session-delete{opacity:1;visibility:visible}.chat-session-delete:hover{background:rgba(239,68,68,.1);color:#ef4444}.sidebar-footer{margin-top:auto;padding:.8rem 0 1.2rem;border-top:1px solid var(--border-subtle);position:relative;z-index:100}.settings-menu{position:relative}.settings-trigger{width:100%;display:flex;align-items:center;gap:10px;background:transparent;border:none;padding:9px 12px 9px 20px;font-family:inherit;font-size:.88rem;font-weight:500;color:#0d0d0d;cursor:pointer;border-radius:0}.settings-trigger:hover{background:rgba(0,0,0,.05)}.settings-user-avatar{width:24px;height:24px;border-radius:999px;background:linear-gradient(135deg,#7c3aed,#9333ea);color:#fff;font-size:.72rem;font-weight:700;display:inline-flex;align-items:center;justify-content:center;flex-shrink:0}.settings-user-label{min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}.settings-dropdown{position:absolute;left:10px;right:10px;bottom:calc(100% + 4px);background:#fff;border:1px solid rgba(0,0,0,.1);border-radius:12px;padding:6px;box-shadow:0 10px 30px -10px rgba(0,0,0,.15),0 4px 12px rgba(0,0,0,.08);display:none;flex-direction:column;gap:2px;z-index:1000;min-width:200px}.settings-menu.open .settings-dropdown{display:flex}.settings-divider{height:1px;background:#e5e7eb;margin:4px 6px}.settings-item{width:100%;display:flex;align-items:center;gap:12px;padding:10px 12px;border-radius:8px;border:none;background:transparent;text-align:left;cursor:pointer;color:#0d0d0d;font-size:.9rem;font-family:inherit;transition:background .15s;white-space:nowrap}.settings-item:hover{background:#f5f5f5}.settings-item .item-icon{width:20px;height:20px;display:inline-flex;align-items:center;justify-content:center;color:#0d0d0d;flex-shrink:0}.settings-item .item-text{flex:1;font-weight:500}.settings-item-has-submenu{display:flex;flex-direction:row;align-items:center;position:relative;padding:10px 12px;border-radius:8px;cursor:pointer;transition:background .15s}.settings-item-has-submenu:hover{background:#f5f5f5}.settings-item-main{display:flex;flex-direction:row;align-items:center;gap:12px;width:100%;white-space:nowrap}.submenu-caret{margin-left:auto;color:#888;display:flex;align-items:center;flex-shrink:0}.settings-submenu{display:none;position:absolute;left:calc(100% + 8px);top:0;background:#fff;border:1px solid rgba(0,0,0,.1);border-radius:12px;padding:6px;box-shadow:0 10px 30px -10px rgba(0,0,0,.15);flex-direction:column;gap:2px;min-width:150px;z-index:2000}.settings-item-has-submenu:hover .settings-submenu{display:flex}.settings-submenu-item{border:none;background:transparent;border-radius:6px;padding:8px 12px;text-align:left;font-size:.88rem;color:#0d0d0d;cursor:pointer;transition:background .15s;display:flex;align-items:center;gap:10px;white-space:nowrap;font-family:inherit}.settings-submenu-item:hover{background:#f5f5f5}.settings-submenu-item.active{color:var(--accent-blue);font-weight:600;background:#f0f7ff}.settings-user-entry{cursor:default}.settings-user-avatar-sm{width:20px;height:20px;font-size:.66rem}.chat-main{flex:1;display:flex;flex-direction:column;position:relative;background:#fff;min-width:0;height:100%;overflow:hidden}.chat-container{flex:1;display:flex;flex-direction:column;min-height:0;overflow:hidden}.chat-container.empty-state{justify-content:center;align-items:center;padding-bottom:15vh}.chat-container.empty-state .chat-scroll-area{flex:0;display:flex;justify-content:center;align-items:flex-end;padding-bottom:1.5rem;overflow:visible}.chat-container.empty-state .chat-input-wrapper{padding-bottom:1rem}.chat-scroll-area{flex:1;overflow-y:auto;padding:2rem max(1rem,calc(50% - 400px));display:flex;flex-direction:column;gap:1.5rem;scroll-behavior:smooth}.chat-scroll-area::-webkit-scrollbar{width:8px}.chat-scroll-area::-webkit-scrollbar-track{background:transparent}.chat-scroll-area::-webkit-scrollbar-thumb{background:transparent;border-radius:4px;border:2px solid transparent;background-clip:content-box}.chat-scroll-area:hover::-webkit-scrollbar-thumb{background-color:rgba(0,0,0,.1);background-clip:content-box}.empty-state-wrapper{max-width:800px;width:100%;margin:0 auto;display:flex;flex-direction:column;justify-content:center;align-items:center;padding-bottom:1rem}.greeting-section{text-align:center;width:100%;max-width:800px;padding:0 1rem}.greeting-gradient{font-size:2.2rem;font-weight:600;background:var(--gradient-ai);-webkit-background-clip:text;background-clip:text;color:transparent;letter-spacing:-.01em;display:block;line-height:1.25}.chat-bubble{max-width:85%;padding:0;line-height:1.6;font-size:1rem}.chat-bubble.user{align-self:flex-end;background:#f3f4f6;color:#0d0d0d;padding:.75rem 1.25rem;border-radius:20px 20px 4px 20px}.chat-bubble.assistant{align-self:stretch;width:100%;max-width:100%;padding-right:0}.chat-bubble .msg-content{font-size:.95rem;line-height:1.7;color:#0d0d0d;text-align:justify;text-align-last:left}.chat-bubble .msg-content p{margin:0 0 .85rem}.chat-bubble .msg-content p:last-child{margin-bottom:0}.route-tag{display:inline-block;padding:3px 12px;border-radius:20px;font-size:.72rem;font-weight:600;margin-bottom:8px}.route-rag{background:rgba(11,61,145,.08);color:#0B3D91}.route-simple{background:#fef9c3;color:#854d0e}.route-chitchat{background:#f3e8ff;color:#6b21a8}.route-safety{background:var(--error-bg);color:var(--error)}.route-list{background:#fef3c7;color:#92400e}.route-detail{background:#dbeafe;color:#1e40af}.route-general{background:#e0e7ff;color:#3730a3}.rag-details{border:1px solid var(--border-subtle);border-radius:10px;overflow:hidden;margin:8px 0}.rag-details__content{padding:12px 14px;display:flex;flex-direction:column;gap:10px}.rag-details__label{font-size:.75rem;font-weight:700;color:var(--text-secondary);text-transform:uppercase;letter-spacing:.06em}.rag-doc-card{border:1px solid var(--border-subtle);border-radius:8px;overflow:hidden;font-size:.84rem}.rag-doc-card__header{padding:6px 12px;font-size:.78rem;font-weight:600;color:var(--text-secondary);background:#f1f5f9;border-bottom:1px solid var(--border-subtle)}.rag-doc-card__row{padding:6px 12px;border-bottom:1px solid var(--border-subtle);line-height:1.5}.rag-doc-card__row:last-child{border-bottom:none}.rag-doc-card__key{font-size:.75rem;font-weight:600;color:var(--text-secondary);display:block;margin-bottom:2px}.rag-doc-card__value{word-break:break-word}.verification-badge{display:inline-flex;align-items:center;gap:6px;padding:6px 14px;border-radius:8px;font-size:.82rem;font-weight:600}.verification-badge--high{background:rgba(11,61,145,.08);color:#0B3D91;border:1px solid var(--border-subtle)}.verification-badge--mid{background:rgba(245,158,11,.1);color:#d97706;border:1px solid rgba(245,158,11,.3)}.verification-badge--low{background:var(--error-bg);color:var(--error);border:1px solid rgba(239,68,68,.3)}.chat-input-wrapper{display:flex;justify-content:center;padding:0 1rem 2rem;flex-shrink:0;background:#fff;z-index:10;width:100%}.chat-input-container{width:100%;max-width:800px;background:#fff;border:1px solid #e5e7eb;box-shadow:0 4px 12px rgba(0,0,0,.08);border-radius:28px;padding:12px 12px 12px 20px;display:flex;flex-direction:column;transition:all .2s}.chat-input-container:focus-within{border-color:var(--accent-blue);box-shadow:0 6px 16px rgba(0,0,0,.12)}.chat-input{background:transparent;border:none;padding:4px 0 4px 12px;font-size:1.05rem;color:#0d0d0d;width:100%;outline:none;box-shadow:none;resize:none;min-height:24px;max-height:200px;line-height:1.5;font-family:var(--font-sans)}.chat-input::placeholder{color:#8f8f8f}.chat-input:disabled{background:transparent!important;color:var(--text-secondary);opacity:1;cursor:not-allowed}.actions-row{display:flex;justify-content:space-between;align-items:center;width:100%;margin-top:4px}.left-actions{display:flex;align-items:center;gap:8px}.right-actions{display:flex;align-items:center}.kb-dropdown-trigger{display:flex;align-items:center;gap:8px;padding:8px 12px;border-radius:99px;cursor:pointer;transition:all .2s;color:#0d0d0d;font-size:.85rem;font-weight:500;background:transparent;border:1px solid transparent}.kb-dropdown-trigger:hover{background:rgba(0,0,0,.06)}.kb-icon-svg{flex-shrink:0;color:inherit}.kb-chevron{transition:transform .2s;opacity:.6}.btn-send{width:40px;height:40px;border-radius:50%;background:#1a1a1a;color:#fff;border:none;display:flex;align-items:center;justify-content:center;transition:transform .2s,background .2s;padding:0;cursor:pointer;flex-shrink:0}.btn-send:hover{transform:scale(1.05);background:#000}.btn-send:disabled{background:#f3f4f6;color:#d1d5db;cursor:not-allowed;opacity:1}.settings-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.4);backdrop-filter:blur(4px);z-index:9999;justify-content:center;align-items:center}.settings-overlay.show{display:flex}.settings-panel{background:#fff;border-radius:16px;box-shadow:0 25px 60px -12px rgba(0,0,0,.3);width:680px;max-width:95vw;max-height:85vh;display:flex;flex-direction:column;overflow:hidden}.settings-header{display:flex;align-items:center;justify-content:space-between;padding:16px 24px;border-bottom:1px solid var(--border-subtle)}.settings-title{font-size:1.1rem;font-weight:700}.settings-close{background:transparent;border:none;font-size:1.2rem;cursor:pointer;color:var(--text-secondary);padding:4px 8px;border-radius:6px}.settings-close:hover{background:#f3f4f6}.settings-body{flex:1;overflow-y:auto;padding:16px 24px}.settings-section{margin-bottom:20px}.settings-section-title{font-size:.85rem;font-weight:700;margin-bottom:12px;padding-bottom:8px;border-bottom:1px solid var(--border-subtle)}.settings-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.settings-field{display:flex;flex-direction:column;gap:4px}.settings-field.full-width{grid-column:1/-1}.settings-label{font-size:.75rem;font-weight:500;color:var(--text-secondary);margin-bottom:2px}.settings-input{width:100%;padding:8px 12px;border:1px solid var(--border-subtle);border-radius:var(--radius-sm);font-size:.85rem;color:var(--text-primary);background:var(--bg-input);font-family:var(--font-mono)}.settings-input:focus{outline:none;border-color:var(--accent-blue);box-shadow:0 0 0 3px rgba(37,99,235,.15)}.settings-select{width:100%;padding:8px 12px;border:1px solid var(--border-subtle);border-radius:var(--radius-sm);font-size:.85rem;color:var(--text-primary);background:var(--bg-input);font-family:var(--font-sans);cursor:pointer}.settings-select:focus{outline:none;border-color:var(--accent-blue);box-shadow:0 0 0 3px rgba(37,99,235,.15)}.settings-actions{display:flex;align-items:center;justify-content:flex-end;gap:12px;padding:16px 24px;border-top:1px solid var(--border-subtle);background:#fff}.btn-settings{padding:8px 20px;border-radius:8px;font-weight:500;font-size:.9rem;cursor:pointer;border:1px solid var(--border-subtle);background:#fff;color:var(--text-primary);font-family:var(--font-sans)}.btn-settings:hover{background:#f3f4f6}.btn-settings.primary{background:var(--accent-blue);color:#fff;border-color:var(--accent-blue)}.btn-settings.primary:hover{background:#1d4ed8}@keyframes fadeInUp{from{opacity:0;transform:translateY(10px)}to{opacity:1;transform:translateY(0)}}.fade-in-up{animation:fadeInUp .5s ease-out forwards}"""

    return f'''<!DOCTYPE html><html lang="zh-CN"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1.0"><title>NASA RAG</title>
<style>{CSS}</style></head><body>
<section id="chat-view" class="view-overlay"><main class="chat-layout">
<aside class="chat-sidebar" id="chatSidebar">
  <header class="sidebar-header">
    <div class="sidebar-toggle-wrapper">
      <button type="button" class="sidebar-logo-btn"><svg width="28" height="28" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg"><circle cx="50" cy="50" r="48" fill="#0B3D91"/><text x="50" y="60" text-anchor="middle" fill="white" font-size="32" font-weight="bold" font-family="Arial,sans-serif">N</text><ellipse cx="50" cy="50" rx="46" ry="18" fill="none" stroke="#FC3D21" stroke-width="3" transform="rotate(-20 50 50)"/></svg><span style="font-size:1rem;font-weight:700;color:var(--text-primary);margin-left:8px">Simple_RAG</span></button>
    </div>
    <button type="button" id="chat-new-btn" class="btn-nav"><span class="icon"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" width="20" height="20"><path stroke-linecap="round" stroke-linejoin="round" d="M16.862 4.487l1.687-1.688a1.875 1.875 0 112.652 2.652L10.582 16.07a4.5 4.5 0 01-1.897 1.13L6 18l.8-2.685a4.5 4.5 0 011.13-1.897l8.932-8.931zm0 0L19.5 7.125M18 14v4.75A2.25 2.25 0 0115.75 21H5.25A2.25 2.25 0 013 18.75V8.25A2.25 2.25 0 015.25 6H10"/></svg></span><span class="btn-text">新建对话</span></button>
    <button type="button" id="kb-btn" class="btn-nav"><span class="icon"><svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke-width="1.5" stroke="currentColor" width="20" height="20"><path stroke-linecap="round" stroke-linejoin="round" d="M12 6.042A8.967 8.967 0 006 3.75c-1.052 0-2.062.18-3 .512v14.25A8.987 8.987 0 016 18c2.305 0 4.408.867 6 2.292m0-14.25a8.966 8.966 0 016-2.292c1.052 0 2.062.18 3 .512v14.25A8.987 8.987 0 0018 18a8.967 8.967 0 00-6 2.292m0-14.25v14.25"/></svg></span><span class="btn-text">知识库</span></button>
  </header>
  <section class="sidebar-list">
    <div class="chat-session-bar"><div class="chat-session-bar-title"><span>最近</span></div><div class="chat-session-bar-actions"><button class="chat-session-bar-btn" id="clear-all-chats" title="清空全部对话"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/></svg></button></div></div>
    <div class="session-list">{sessions_html}</div>
  </section>
  <footer class="sidebar-footer">
    <div class="settings-menu" id="settings-menu">
      <button type="button" class="settings-trigger" id="settings-menu-trigger"><span class="settings-user-avatar">D</span><span class="settings-user-label">default</span></button>
      <div class="settings-dropdown">
        <button type="button" class="settings-item settings-user-entry"><span class="item-icon"><span class="settings-user-avatar settings-user-avatar-sm">D</span></span><div class="item-text"><span>default</span></div></button>
        <div class="settings-divider" role="separator"></div>
        <button type="button" class="settings-item" id="settings-model"><span class="item-icon"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10 16h.01"/><path d="M2.212 11.577a2 2 0 0 0-.212.896V18a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-5.527a2 2 0 0 0-.212-.896L18.55 5.11A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"/><path d="M21.946 12.013H2.054"/><path d="M6 16h.01"/></svg></span><div class="item-text">模型设置</div></button>
        <div class="settings-item settings-item-has-submenu" id="settings-language"><div class="settings-item-main"><span class="item-icon"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg></span><div class="item-text">语言</div><span class="submenu-caret"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><polyline points="9 18 15 12 9 6"/></svg></span></div>
          <div class="settings-submenu"><button type="button" class="settings-submenu-item active"><span>中文</span></button><button type="button" class="settings-submenu-item"><span>English</span></button></div>
        </div>
        <div class="settings-divider" role="separator"></div>
        <button type="button" class="settings-item" id="settings-login"><span class="item-icon"><svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="m16 17 5-5-5-5"/><path d="M21 12H9"/><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"/></svg></span><div class="item-text">登录</div></button>
      </div>
    </div>
  </footer>
</aside>
<section class="chat-main">
  <div class="chat-container" id="chat-container">
    <div class="chat-scroll-area" id="chat-history">{"".join(msgs_html) if msgs else '<div class="empty-state-wrapper fade-in-up"><div class="greeting-section"><span class="greeting-gradient">你好。今天想探索什么？</span></div></div>'}</div>
    <div class="chat-input-wrapper">
      <form id="chat-form" class="chat-input-container">
        <textarea id="chat-input" placeholder="{ph}" rows="1" class="chat-input" {dis}></textarea>
        <div class="actions-row">
          <div class="left-actions">
            <button type="button" class="kb-dropdown-trigger"><svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" class="kb-icon-svg"><path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20"/><path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z"/></svg><span>知识库</span><svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="kb-chevron"><polyline points="6 9 12 15 18 9"/></svg></button>
          </div>
          <div class="right-actions">
            <button class="btn-send" type="submit" id="chat-send" title="发送" {dis}><svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg></button>
          </div>
        </div>
      </form>
    </div>
  </div>
</section>
</main></section>

<!-- Settings Modal -->
<div class="settings-overlay" id="settingsOverlay"><div class="settings-panel">
  <div class="settings-header"><div class="settings-title">⚙️ 系统设置</div><button class="settings-close" onclick="closeSettings()">✕</button></div>
  <div class="settings-body"><form id="settingsForm">
    <div class="settings-section"><div class="settings-section-title">🤖 大语言模型 (LLM)</div><div class="settings-grid">
      <div class="settings-field"><label class="settings-label">模型提供商</label><select class="settings-select" name="llm_provider" id="s_llm_provider" onchange="toggleProvider(this.value)"><option value="openai">OpenAI API</option><option value="local">本地模型</option></select></div>
      <div class="settings-field" id="f_llm_model"><label class="settings-label">模型名称</label><input class="settings-input" name="llm_model" id="s_llm_model"></div>
      <div class="settings-field" id="f_openai_key"><label class="settings-label">OpenAI API Key</label><input class="settings-input" type="password" name="openai_api_key" id="s_openai_api_key" placeholder="sk-..."></div>
      <div class="settings-field" id="f_openai_url"><label class="settings-label">OpenAI Base URL</label><input class="settings-input" name="openai_base_url" id="s_openai_base_url"></div>
      <div class="settings-field" id="f_local_url" style="display:none"><label class="settings-label">本地模型 Base URL</label><input class="settings-input" name="local_llm_base_url" id="s_local_llm_base_url"></div>
      <div class="settings-field" id="f_local_key" style="display:none"><label class="settings-label">本地模型 API Key</label><input class="settings-input" type="password" name="local_llm_api_key" id="s_local_llm_api_key"></div>
      <div class="settings-field" id="f_local_model" style="display:none"><label class="settings-label">本地模型名称</label><input class="settings-input" name="local_llm_model" id="s_local_llm_model"></div>
      <div class="settings-field"><label class="settings-label">Temperature</label><input class="settings-input" type="number" name="temperature" id="s_temperature" step="0.1" min="0" max="2"></div>
      <div class="settings-field"><label class="settings-label">最大 Token 数</label><input class="settings-input" type="number" name="max_tokens" id="s_max_tokens" step="256" min="256" max="16384"></div>
      <div class="settings-field"><label class="settings-label">最大并发数</label><input class="settings-input" type="number" name="llm_max_workers" id="s_llm_max_workers" min="1" max="32"></div>
    </div></div>
    <div class="settings-section"><div class="settings-section-title">📐 向量嵌入</div><div class="settings-grid">
      <div class="settings-field"><label class="settings-label">嵌入模型</label><input class="settings-input" name="embedding_model" id="s_embedding_model"></div>
      <div class="settings-field"><label class="settings-label">嵌入设备</label><select class="settings-select" name="embedding_device" id="s_embedding_device"><option value="auto">auto</option><option value="cpu">cpu</option><option value="cuda">cuda</option></select></div>
      <div class="settings-field"><label class="settings-label">FAISS 使用 GPU</label><select class="settings-select" name="faiss_use_gpu" id="s_faiss_use_gpu"><option value="false">否</option><option value="true">是</option></select></div>
      <div class="settings-field"><label class="settings-label">FAISS GPU ID</label><input class="settings-input" type="number" name="faiss_gpu_id" id="s_faiss_gpu_id" min="0"></div>
      <div class="settings-field"><label class="settings-label">嵌入批次大小</label><input class="settings-input" type="number" name="embedding_batch_size" id="s_embedding_batch_size" min="1"></div>
      <div class="settings-field"><label class="settings-label">编码批次大小 (0=自动)</label><input class="settings-input" type="number" name="embedding_encode_batch_size" id="s_embedding_encode_batch_size" min="0"></div>
    </div></div>
    <div class="settings-section"><div class="settings-section-title">🔍 检索设置</div><div class="settings-grid">
      <div class="settings-field"><label class="settings-label">检索数量 (Top K)</label><input class="settings-input" type="number" name="top_k" id="s_top_k" min="1" max="20"></div>
      <div class="settings-field"><label class="settings-label">查询翻译用于 BM25</label><select class="settings-select" name="translate_for_bm25" id="s_translate_for_bm25"><option value="true">是</option><option value="false">否</option></select></div>
      <div class="settings-field"><label class="settings-label">查询翻译用于向量检索</label><select class="settings-select" name="translate_for_vector" id="s_translate_for_vector"><option value="true">是</option><option value="false">否</option></select></div>
    </div></div>
    <div class="settings-section"><div class="settings-section-title">🎯 Reranker 精排</div><div class="settings-grid">
      <div class="settings-field"><label class="settings-label">启用 Reranker</label><select class="settings-select" name="reranker_enabled" id="s_reranker_enabled"><option value="true">是</option><option value="false">否</option></select></div>
      <div class="settings-field"><label class="settings-label">Reranker 模型</label><input class="settings-input" name="reranker_model" id="s_reranker_model"></div>
      <div class="settings-field"><label class="settings-label">Reranker 设备</label><select class="settings-select" name="reranker_device" id="s_reranker_device"><option value="auto">auto</option><option value="cpu">cpu</option><option value="cuda">cuda</option></select></div>
    </div></div>
    <div class="settings-section"><div class="settings-section-title">🌐 翻译 / DeepL</div><div class="settings-grid">
      <div class="settings-field"><label class="settings-label">DeepL API Key</label><input class="settings-input" type="password" name="deepl_api_key" id="s_deepl_api_key"></div>
      <div class="settings-field"><label class="settings-label">DeepL API URL</label><input class="settings-input" name="deepl_api_url" id="s_deepl_api_url"></div>
      <div class="settings-field"><label class="settings-label">翻译模式</label><select class="settings-select" name="deepl_api_mode" id="s_deepl_api_mode"><option value="deepl">DeepL</option><option value="openai">OpenAI 翻译</option></select></div>
    </div></div>
    <div class="settings-section"><div class="settings-section-title">📂 数据路径</div><div class="settings-grid">
      <div class="settings-field full-width"><label class="settings-label">数据目录</label><input class="settings-input" name="data_path" id="s_data_path"></div>
    </div></div>
  </form></div>
  <div class="settings-actions"><button class="btn-settings" onclick="closeSettings()">取消</button><button class="btn-settings primary" onclick="submitSettings()">💾 保存设置</button></div>
</div></div>

<script>
{JS_CODE}
</script></body></html>'''


def main():
    init_session_state()

    # Process pending actions from query params
    qp = dict(st.query_params)
    if "action" in qp:
        action = qp["action"]
        data = qp.get("data", "")
        st.query_params.clear()
        if action == "chat_submit" and data:
            st.session_state.messages.append({"role": "user", "content": data})
            conv = st.session_state.conversations.get(st.session_state.current_conv_id, {})
            if not conv.get("messages"): conv["title"] = data[:30] + ("..." if len(data) > 30 else "")
            conv["messages"] = st.session_state.messages
            if st.session_state.system_ready:
                rag = st.session_state.rag_system
                resp, extra = process_query(data, rag, st.session_state.messages)
                st.session_state.messages.append({"role": "assistant", "content": resp, "extra": extra})
                conv["messages"] = st.session_state.messages
            st.rerun()
        elif action == "new_conv":
            new_conversation(); st.rerun()
        elif action == "switch_conv" and data:
            switch_conv(data); st.rerun()
        elif action == "delete_conv" and data:
            delete_conv(data); st.rerun()
        elif action == "settings_submit" and data:
            try:
                settings = json.loads(data)
                bool_keys = ["faiss_use_gpu", "reranker_enabled", "translate_for_bm25", "translate_for_vector"]
                float_keys = ["temperature"]
                int_keys = ["top_k", "max_tokens", "llm_max_workers", "faiss_gpu_id", "embedding_batch_size", "embedding_encode_batch_size"]
                processed = {}
                for k, v in settings.items():
                    if k in bool_keys: processed[k] = v.lower() in ("true", "1", "yes")
                    elif k in float_keys: processed[k] = float(v)
                    elif k in int_keys: processed[k] = int(v)
                    else: processed[k] = v
                save_settings(processed)
                if st.session_state.system_ready: st.session_state.system_ready = False; st.session_state.rag_system = None
            except: pass
            st.rerun()

    # Build and render the component
    html = build_ultrarag_html()
    html_path = Path(__file__).resolve().parent / "_page.html"
    html_path.write_text(html, encoding="utf-8")
    components.html(open(html_path).read(), height=900, scrolling=False)


if __name__ == "__main__": main()
