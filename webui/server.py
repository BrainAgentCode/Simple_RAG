"""
航空航天知识问答 RAG 系统 - FastAPI Backend + UltraRAG Frontend
"""
import sys
import json
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel

from config import RAGConfig, DEFAULT_DATA_PATH, DEFAULT_INDEX_PATH
from main import AerospaceRAGSystem

app = FastAPI(title="NASA RAG")

SETTINGS_PATH = Path(__file__).resolve().parent / "app_settings.json"
STATIC_DIR = Path(__file__).resolve().parent / "static"

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

# ===== State =====
class AppState:
    def __init__(self):
        self.lock = threading.Lock()
        self.rag_system: Optional[AerospaceRAGSystem] = None
        self.system_ready = False
        self.conversations: Dict[str, dict] = {}
        self.current_conv_id: str = ""
        self.messages: List[dict] = []

state = AppState()

# ===== Settings =====
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

def init_rag():
    with state.lock:
        if state.system_ready: return
    try:
        config = get_config_from_settings()
        rag = AerospaceRAGSystem(config=config)
        rag.initialize_system()
        rag.build_knowledge_base()
        with state.lock:
            state.rag_system = rag
            state.system_ready = True
        print("RAG system ready!")
    except Exception as e:
        print(f"RAG init failed: {e}")

def process_query(question, rag, history=None):
    try:
        u = rag.understand_query(question, history)
        intent, resolved = u["intent"], u["resolved_query"]
        if not u["safe"]:
            return f"⚠️ {u['safety_reason']}", {"intent": intent, "sources": []}
        if intent in ("simple", "chitchat"):
            return rag.generate_direct_answer(resolved), {"intent": intent, "sources": []}
        rt, rw = rag.route_and_rewrite(resolved)
        chunks, _ = rag.retrieve_documents(resolved, rewritten_query=rw, route_type=rt)
        response, _, docs = rag.generate_answer(resolved, chunks, rt)
        from rag_modules.query_understanding import verify_answer
        verification = verify_answer(resolved, response, docs, rag.generation_module.llm)
        sources = []
        seen = set()
        for c in (docs if docs else chunks):
            t = c.metadata.get("source_file", c.metadata.get("title", "未知来源"))
            if t not in seen:
                seen.add(t)
                sources.append({"title": t, "url": c.metadata.get("url", "")})
        return response, {"intent": intent, "route_type": rt, "sources": sources, "verification": verification}
    except Exception as e:
        return f"{type(e).__name__}: {e}", {}

def format_extras_html(extra):
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
    import html as html_mod
    for i, s in enumerate(extra.get("sources", [])):
        safe_title = html_mod.escape(s["title"])
        safe_url = html_mod.escape(s["url"]) if s["url"] else ""
        url_html = f'<a href="{safe_url}" target="_blank" style="color:#2563eb;text-decoration:none;">{safe_title}</a>' if s["url"] else safe_title
        parts.append(f'<div class="rag-doc-card"><div class="rag-doc-card__header">Document {i+1}/{len(extra["sources"])}</div><div class="rag-doc-card__row"><span class="rag-doc-card__key">Source</span><span class="rag-doc-card__value">{url_html}</span></div></div>')
    v = extra.get("verification")
    if v:
        sc, lv = v.get("score", 5), "high" if v.get("score", 5) >= 7 else "mid" if v.get("score", 5) >= 4 else "low"
        issues = "".join(f'<div style="font-size:.82rem;color:#6e6e80;padding:2px 0;">- {html_mod.escape(str(i))}</div>' for i in v.get("issues", []))
        parts.append(f'<div class="rag-details"><div class="rag-details__content"><div class="rag-details__label">Answer Verification</div><div class="verification-badge verification-badge--{lv}">Score: {sc}/10</div>{issues}</div></div>')
    return "".join(parts)


# ===== API =====
class ChatRequest(BaseModel):
    message: str

class SettingsRequest(BaseModel):
    settings: dict

@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/api/state")
async def get_state():
    with state.lock:
        return {
            "system_ready": state.system_ready,
            "conversations": state.conversations,
            "current_conv_id": state.current_conv_id,
            "settings": load_settings(),
        }

@app.post("/api/chat")
async def chat(req: ChatRequest):
    with state.lock:
        if not state.system_ready:
            return {"error": "系统未启动，请先配置模型"}
        rag = state.rag_system
        # Auto-create conversation on first message
        if not state.current_conv_id or state.current_conv_id not in state.conversations:
            cid = f"conv_{int(time.time()*1000)}"
            state.conversations[cid] = {"title": req.message[:30] + ("..." if len(req.message) > 30 else ""), "messages": []}
            state.current_conv_id = cid
            state.messages = state.conversations[cid]["messages"]
        conv = state.conversations[state.current_conv_id]
        if not conv.get("messages"):
            conv["title"] = req.message[:30] + ("..." if len(req.message) > 30 else "")
        state.messages.append({"role": "user", "content": req.message})
        conv["messages"] = state.messages
        response, extra = process_query(req.message, rag, state.messages)
        extras_html = format_extras_html(extra)
        state.messages.append({"role": "assistant", "content": response, "extras_html": extras_html})
        conv["messages"] = state.messages
    return {"response": response, "extras_html": extras_html}

@app.post("/api/clear_all")
async def clear_all():
    with state.lock:
        state.conversations = {}
        state.current_conv_id = ""
        state.messages = []
    return {"ok": True}

@app.post("/api/switch_conv")
async def switch_conv(req: Request):
    data = await req.json()
    cid = data.get("conv_id", "")
    with state.lock:
        if cid in state.conversations:
            state.current_conv_id = cid
            state.messages = state.conversations[cid]["messages"]
    return {"ok": True}

@app.post("/api/delete_conv")
async def delete_conv(req: Request):
    data = await req.json()
    cid = data.get("conv_id", "")
    with state.lock:
        state.conversations.pop(cid, None)
        if state.current_conv_id == cid:
            if state.conversations:
                k = next(iter(state.conversations))
                state.current_conv_id = k
                state.messages = state.conversations[k]["messages"]
            else:
                state.current_conv_id = ""
                state.messages = []
    return {"ok": True}

@app.post("/api/settings")
async def save_settings_api(req: SettingsRequest):
    settings = req.settings
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
    with state.lock:
        state.system_ready = False
        state.rag_system = None
    return {"ok": True}

@app.get("/api/kb/files")
async def kb_files():
    settings = load_settings()
    data_path = Path(settings.get("data_path", DEFAULT_DATA_PATH)).resolve()
    index_path = Path(settings.get("index_path", DEFAULT_INDEX_PATH)).resolve()
    files = []
    if data_path.exists():
        for f in sorted(data_path.rglob("*")):
            if f.is_file() and f.suffix.lower() in (".json", ".csv", ".txt", ".md", ".pdf"):
                files.append({
                    "name": f.name,
                    "path": str(f.relative_to(data_path)),
                    "size": f.stat().st_size,
                    "type": f.suffix.lower(),
                })
    has_index = index_path.exists() and any(index_path.glob("*.faiss"))
    return {"files": files, "has_index": has_index, "data_path": str(data_path), "index_path": str(index_path)}

@app.post("/api/kb/rebuild")
async def kb_rebuild():
    with state.lock:
        state.system_ready = False
        state.rag_system = None
    threading.Thread(target=init_rag, daemon=True).start()
    return {"ok": True, "message": "知识库正在重建..."}

@app.post("/api/test_llm")
async def test_llm(req: Request):
    data = await req.json()
    import requests as req_lib
    try:
        if data.get("provider") == "local":
            url = data.get("local_url", "").rstrip("/") + "/models"
            api_key = data.get("local_key", "")
        else:
            url = data.get("base_url", "").rstrip("/") + "/models"
            api_key = data.get("api_key", "")
        if not url:
            return {"ok": False, "message": "URL 不能为空"}
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        resp = req_lib.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        models = [m.get("id", "") for m in resp.json().get("data", []) if m.get("id")]
        model = data.get("model") or data.get("local_model", "")
        if model and model in models:
            return {"ok": True, "message": f"连接成功，模型 {model} 可用（共 {len(models)} 个模型）"}
        elif models:
            return {"ok": True, "message": f"连接成功（共 {len(models)} 个模型）"}
        else:
            return {"ok": True, "message": "连接成功"}
    except Exception as e:
        return {"ok": False, "message": str(e)}

@app.get("/api/kb/test")
async def kb_test(data_path: str = "", index_path: str = ""):
    dp = Path(data_path).resolve() if data_path else Path(DEFAULT_DATA_PATH).resolve()
    ip = Path(index_path).resolve() if index_path else Path(DEFAULT_INDEX_PATH).resolve()
    msgs = []
    all_ok = True
    if dp.exists():
        count = sum(1 for f in dp.rglob("*") if f.is_file() and f.suffix.lower() in (".json", ".csv", ".txt", ".md", ".pdf"))
        msgs.append(f"数据目录存在，包含 {count} 个文件")
    else:
        msgs.append(f"数据目录不存在: {dp}")
        all_ok = False
    if ip.exists():
        has_idx = any(ip.glob("*.faiss"))
        msgs.append("向量索引目录存在" + ("，已索引" if has_idx else "，未索引"))
    else:
        msgs.append(f"向量索引目录不存在: {ip}")
        all_ok = False
    return {"ok": all_ok, "message": "；".join(msgs)}

@app.post("/api/kb/save_settings")
async def kb_save_settings(req: Request):
    data = await req.json()
    settings = load_settings()
    if data.get("data_path"):
        settings["data_path"] = data["data_path"]
    if data.get("index_path"):
        settings["index_path"] = data["index_path"]
    save_settings(settings)
    return {"ok": True}

@app.on_event("startup")
async def startup():
    threading.Thread(target=init_rag, daemon=True).start()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8501)
