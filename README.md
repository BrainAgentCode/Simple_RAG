# 航空航天知识问答 RAG 系统

面向 **NASA 会议论文 + Lessons Learned** 的检索增强生成（RAG）问答系统。

知识来源：NTRS 会议论文 PDF、NASA 经验教训 CSV。  
流水线：**PDF → Markdown 缓存 → 章节分块 → 向量索引 → 混合检索 → 父文档聚合 → LLM 生成**。

---

## 快速开始

```bash
cp .env.example .env   # 填入 OPENAI_API_KEY 和/或 LOCAL_LLM_BASE_URL
pip install -r requirements.txt

# 1. 采集数据（PDF / CSV / 分块）
python thesis_pipeline/download_nasa_data.py

# 2. 启动问答（首次会建 vector_index）
python main.py
```

启动后可选择：
- **[1] OpenAI 联网模型**（如 `gpt-4o-mini`）
- **[2] 本地 LLaMA 3.3-70B-Instruct-Turbo**（需 OpenAI 兼容推理服务）

---

## 双 LLM 配置

| 模式 | 环境变量 | 说明 |
|------|----------|------|
| 联网 | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` | 默认 |
| 本地 | `LOCAL_LLM_BASE_URL`, `LOCAL_LLM_MODEL` | 如 vLLM 端口 8000 |

`.env` 中 `LLM_PROVIDER=openai|local` 可设默认；交互时也可随时切换。

**vLLM 启动示例：**

```bash
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Llama-3.3-70B-Instruct-Turbo \
  --port 8000
```

---

## 目录结构

```
./
├── main.py                    # 航空航天 RAG 主程序
├── config.py                  # 配置加载
├── rag_modules/               # 检索、索引、生成
├── thesis_pipeline/           # NASA 爬虫 + PDF 分块
├── evaluation/                # RAGAS 评测
└── data/nasa/
    ├── pdfs/conference/
    ├── markdown/conference/
    ├── chunks/
    └── lessons_learned/
```

---

## 数据处理流水线

```
PDF → Markdown 缓存 → 章节分块 JSON → vector_index
```

各阶段支持**断点续跑**（已有文件自动跳过）：

| 阶段 | 跳过条件 |
|------|----------|
| PDF 下载 | 目标目录已有同名 PDF |
| Lessons CSV | CSV 中已有相同 url |
| Markdown | `.md` 存在且不比 PDF 旧 |
| 分块 | `chunks/per_pdf/*.json` 存在且不比 PDF/Markdown 旧 |

**父文档构建策略：**

1. **Abstract → Conclusions → Introduction**（含 LPSC `**Introduction:**` 格式）
2. 若都没有：**LLM 概要**（失败则继续下一步）
3. 检索命中时，用该片段的**前、后相邻章节**拼成父文档（缺一侧则用现有的一侧）
4. 仍不够：**全文档章节**作为父文档

---

## 环境变量（完整版见 `.env.example`）

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | OpenAI 联网模型密钥 |
| `LOCAL_LLM_BASE_URL` | 本地 LLaMA OpenAI 兼容地址 |
| `LOCAL_LLM_MODEL` | 默认 `meta-llama/Llama-3.3-70B-Instruct-Turbo` |
| `LLM_PROVIDER` | `openai` / `local` |
| `DEEPL_API_KEY` | 中文检索翻译兜底（建议） |
| `EMBEDDING_DEVICE` | `cuda` / `cpu` / `auto` |
| `FAISS_USE_GPU` | FAISS GPU 加速 |
| `LLM_MAX_WORKERS` | 并行 API 调用（评测跑批等） |

---

## 示例问题

- `What unexpected thermal control issues arose during spacecraft testing?`
- `航天器热真空测试中有哪些意外的热控问题？`
- `List lessons learned about propulsion system anomalies`

---

## 程序化调用

```python
from config import get_active_config
from main import AerospaceRAGSystem

config = get_active_config()
config.llm_provider = "openai"  # 或 "local"

system = AerospaceRAGSystem(config=config)
system.initialize_system()
system.build_knowledge_base()
answer = system.ask_question("What are common thermal vacuum test issues?")
```

---

## 评估

```bash
# 1. LLM 生成评测集（或 thesis 导入，见 evaluation/README.md）
python evaluation/generate_dataset.py llm --max-docs 5 --per-doc 2

# 2. 跑 RAG
python evaluation/run_eval.py --dataset evaluation/datasets/eval_all.json

# 3. RAGAS
python evaluation/run_ragas.py --predictions evaluation/results/run_*/predictions.jsonl
```

---

## 常见问题

**未配置 API Key**  
至少配置 `OPENAI_API_KEY` 或 `LOCAL_LLM_BASE_URL` 之一。

**加载 0 个文档**  
先运行 `thesis_pipeline/download_nasa_data.py` 采集并分块。

**本地模型连不上**  
确认 vLLM/Ollama 已启动，且 `LOCAL_LLM_BASE_URL` 指向 `/v1` 端点。

**更换数据后回答不变**  
删除 `vector_index/` 与 `data/nasa/parent_summaries_cache.json` 后重启。

---

## 与 Master_Thesis_RAG 的关系

| 能力 | 来源 |
|------|------|
| NASA 爬虫、PDF 分块 | `thesis_pipeline/` |
| 检索、索引、双 LLM 生成 | `rag_modules/` + `main.py` |
| 评测 | RAGAS（`evaluation/`） |
