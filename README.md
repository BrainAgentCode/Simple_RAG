# 航空航天知识问答 RAG 系统

面向 **NASA 会议论文 + Lessons Learned** 的检索增强生成（RAG）问答系统。

知识来源：NTRS 会议论文 PDF、NASA 经验教训 CSV。  
流水线：**PDF → Markdown 缓存 → 章节分块 → 向量索引 → 混合检索 → 父文档聚合 → LLM 生成**。

---

## Google Colab 部署

### 方式一：Colab 内置代理（最简单）

```python
# 安装依赖
!pip install -r requirements.txt

# 直接启动，Colab 右侧会自动生成访问 URL
!streamlit run webui/app.py --server.port 8501 --server.address 0.0.0.0
```

Colab 会自动在右侧显示 `https://xxxx-8501.colab.googleusercontent.com`，点击即可访问。

### 方式二：Cloudflare Tunnel（免费稳定）

```python
!pip install -r requirements.txt

# 安装 cftunnel
!curl -fsSL https://raw.githubusercontent.com/qingchencloud/cftunnel/main/install.sh | bash

# 后台启动 Streamlit
!streamlit run webui/app.py --server.port 8501 --server.address 0.0.0.0 &

# 快速启动免域名隧道
!cftunnel quick http://localhost:8501
```

输出类似 `https://xxx-xxx.trycloudflare.com`，直接访问即可。无需注册，完全免费。

### 方式三：ngrok

```python
!pip install -r requirements.txt pyngrok

!streamlit run webui/app.py --server.port 8501 --server.address 0.0.0.0 &

from pyngrok import ngrok
public_url = ngrok.connect(8501)
print(f"访问地址: {public_url}")
```

> 💡 **推荐方式一**，零配置最省事。如果 Colab 代理不稳定，用方式二 Cloudflare Tunnel。

---

## 快速开始

```bash
pip install -r requirements.txt

# 1. 采集数据（PDF / CSV / 分块）
python thesis_pipeline/download_nasa_data.py

# 2. 启动 WebUI（推荐）
bash run_webui.sh

# 或启动命令行交互
python main.py
```

**WebUI 使用：**
1. 打开 `http://localhost:8501`
2. 侧边栏点击「⚙️ 高级设置」配置模型
3. 填写 API 地址、密钥，选择模型
4. 点击「🔗 测试连接」验证模型可用
5. 点击「🚀 启动系统」开始使用

**命令行使用：**
```bash
python main.py
# 选择 [1] OpenAI 联网模型 或 [2] 本地模型
```

---

## 功能特点

### 查询理解与路由

| 模块 | 功能 |
|------|------|
| 意图识别 | 自动分类为 `rag`（检索）/ `simple`（直答）/ `chitchat`（闲聊）/ `safety`（安全过滤） |
| 指代消解 | 利用对话历史解析"它""那个""这个"等代词 |
| 安全过滤 | 关键词 + LLM 双重检查有害内容 |
| 查询改写 | 模糊问题自动改写为更具体的检索词 |
| 查询路由 | 分类为列表/详细/通用，决定生成格式 |

### 检索与生成

| 能力 | 说明 |
|------|------|
| 混合检索 | 向量相似度 + BM25 关键词 |
| RRF 融合 | Reciprocal Rank Fusion 融合排序 |
| Cross-Encoder 精排 | BAAI/bge-reranker-v2-m3 重排序 |
| 中英文双语 | 自动检测语言，智能翻译优化检索词 |
| 父文档聚合 | 检索命中时自动关联摘要、结论等上下文 |
| 答案验证 | 生成后与源文档交叉验证，可信度评分 1-10 |

### 双 LLM 配置

| 模式 | 环境变量 | 说明 |
|------|----------|------|
| 联网 | `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` | 默认 |
| 本地 | `LOCAL_LLM_BASE_URL`, `LOCAL_LLM_MODEL` | 如 vLLM 端口 8000 |

WebUI 中可直接配置 API 地址和密钥，支持从 `/v1/models` 自动获取模型列表。

---

## 目录结构

```
Simple_RAG/
├── main.py                    # 航空航天 RAG 主程序
├── config.py                  # 配置加载
├── run_webui.sh               # WebUI 启动脚本
├── .env.example              # 环境变量模板
├── .gitignore
├── requirements.txt
├── rag_modules/
│   ├── query_understanding.py # 查询理解（意图/指代/安全/验证）
│   ├── query_utils.py         # 中文检测、翻译
│   ├── generation_integration.py  # LLM 生成、路由
│   ├── index_construction.py  # FAISS 向量索引
│   ├── retrieval_optimization.py  # 混合检索、RRF、精排
│   ├── data_preparation.py    # 数据加载、父文档构建
│   ├── document_summary.py    # 文档摘要提取
│   └── runtime_accel.py       # GPU/并行加速
├── webui/
│   ├── app.py                 # Streamlit WebUI
│   └── __init__.py
├── thesis_pipeline/           # NASA 爬虫 + PDF 分块
├── evaluation/                # RAGAS 评测
└── data/nasa/                 # 数据目录（gitignore）
```

---

## WebUI 配置

所有配置项均可在 WebUI 前端设置，未填写的项使用硬编码默认值：

| 设置分组 | 包含内容 |
|----------|----------|
| 🤖 大语言模型 | 提供商、API Key、Base URL、模型名称、Temperature、Max Tokens |
| 📦 模型快捷配置 | 保存/删除模型配置，支持从 API 自动获取模型列表 |
| 📐 向量嵌入 | 嵌入模型、设备(auto/cpu/cuda) |
| 🔍 检索设置 | Top K、查询翻译开关 |
| 🎯 Reranker | 启用开关、模型、设备 |
| 📂 数据路径 | 数据目录、向量索引目录 |

配置保存在 `webui/app_settings.json`（已 gitignore）。

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

1. **Abstract → Conclusions → Introduction**
2. 若都没有：**LLM 概要**（失败则继续下一步）
3. 检索命中时，用该片段的**前、后相邻章节**拼成父文档
4. 仍不够：**全文档章节**作为父文档

---

## 环境变量

详见 `.env.example`：

| 变量 | 说明 |
|------|------|
| `OPENAI_API_KEY` | OpenAI 联网模型密钥 |
| `LOCAL_LLM_BASE_URL` | 本地 LLaMA OpenAI 兼容地址 |
| `EMBEDDING_MODEL` | 嵌入模型（默认 `BAAI/bge-small-zh-v1.5`） |
| `EMBEDDING_DEVICE` | `cuda` / `cpu` / `auto` |
| `RERANKER_MODEL` | 精排模型（默认 `BAAI/bge-reranker-v2-m3`） |

---

## 示例问题

- `What unexpected thermal control issues arose during spacecraft testing?`
- `航天器热真空测试中有哪些意外的热控问题？`
- `List lessons learned about propulsion system anomalies`
- `你好`（直接回答，不检索）
- `它有什么问题？`（指代消解后检索）

---

## 程序化调用

```python
from config import RAGConfig
from main import AerospaceRAGSystem

config = RAGConfig.from_env(llm_provider="local")
system = AerospaceRAGSystem(config=config)
system.initialize_system()
system.build_knowledge_base()

# 带查询理解的问答
answer = system.ask_question(
    "航天器热真空测试中有哪些热控问题？",
    history=[{"role": "user", "content": "上一个问题"}]
)
```

---

## 评估

```bash
# 1. LLM 生成评测集
python evaluation/generate_dataset.py llm --max-docs 5 --per-doc 2

# 2. 跑 RAG
python evaluation/run_eval.py --dataset evaluation/datasets/eval_all.json

# 3. RAGAS
python evaluation/run_ragas.py --predictions evaluation/results/run_*/predictions.jsonl
```

---

## 常见问题

**启动报 CUDA 错误**  
将 `.env` 中 `EMBEDDING_DEVICE=cpu`（默认已设为 cpu）。

**模型无法连接**  
在 WebUI 设置中点击「🔗 测试连接」检查 API 地址和密钥。

**加载 0 个文档**  
先运行 `python thesis_pipeline/download_nasa_data.py` 采集并分块。

**更换数据后回答不变**  
删除 `vector_index/` 与 `data/nasa/parent_summaries_cache.json` 后重启。
