# RAGAS 评测

评测流程：**生成数据集 → 跑 RAG → RAGAS 打分**。

## 1. 生成评测集

### LLM 自动生成（推荐）

将整篇文档（每个 chunk 带 `[chunk_id: ...]` 标注）发给 LLM，生成 question / expected_answer / relevant_chunk_ids：

```bash
python evaluation/generate_dataset.py llm \
  --chunks-dir data/nasa/chunks \
  --max-docs 5 --per-doc 2 \
  -o evaluation/datasets/eval_all.json
```

指定文档（PDF 编号）：

```bash
python evaluation/generate_dataset.py llm \
  --stem 20170001706 --per-doc 3 \
  -o evaluation/datasets/eval_all.json
```

环境变量 `EVAL_GEN_MODEL` 可覆盖生成用的 LLM（默认 `OPENAI_MODEL`）。

### 导入 Master_Thesis_RAG 人工标注

```bash
python evaluation/generate_dataset.py thesis \
  --chunks-dir data/nasa/chunks \
  --skip-missing \
  -o evaluation/datasets/eval_all.json
```

## 2. 跑 RAG

与交互式 `main.py` 使用**同一套检索链路**：

`query_router` → `query_rewrite` → 中英文检索词翻译 → 元数据过滤 → `hybrid_search`（向量 + BM25 + **RRF 重排**）

**数据 / 索引路径**（优先级：命令行 > .env > 默认）：

```bash
# .env
RAG_DATA_PATH=./data/nasa
RAG_INDEX_PATH=./vector_index

# 或命令行指定
python evaluation/run_eval.py \
  --dataset evaluation/datasets/eval_all.json \
  --data-path /path/to/nasa/data \
  --index-path /path/to/vector_index \
  --limit 5 -v
```

生成默认：**eval 专用 prompt + temperature=0**

```bash
python evaluation/run_eval.py \
  --dataset evaluation/datasets/eval_all.json \
  --limit 5 -v
```

检索链路：`hybrid(BM25+向量)` → **RRF** → **Cross-Encoder rerank**（`RERANKER_MODEL`）

## 3. RAGAS

```bash
python evaluation/run_ragas.py \
  --predictions evaluation/results/run_*/predictions.jsonl

# 快速试跑
python evaluation/run_ragas.py \
  --predictions evaluation/results/run_*/predictions.jsonl \
  --max-samples 10

# 含 answer_correctness（更慢）
python evaluation/run_ragas.py \
  --predictions evaluation/results/run_*/predictions.jsonl \
  --full
```

| 环境变量 | 说明 |
|---------|------|
| `OPENAI_API_KEY` | RAGAS judge LLM |
| `RAGAS_LLM_MODEL` | judge 模型（默认 `OPENAI_MODEL`） |
| `RAGAS_MAX_WORKERS` | 并行数（默认 2，慢模型建议保持 2） |
| `RAGAS_TIMEOUT` | 单 job 超时秒数（默认 3600） |
| `RAGAS_EMBEDDING_BACKEND` | `local`（默认）或 `openai` |

## 评测集格式

```json
{
  "id": "llm_20170001706_q1",
  "question": "...",
  "ground_truth": "...",
  "ground_truth_contexts": ["相关 chunk 原文"],
  "relevant_chunk_ids": ["20170001706.pdf_3"]
}
```

`ground_truth_contexts` 由生成脚本自动填充，供 RAGAS `context_recall` 使用。

## 查看每题的 prompt 与回答

`run_eval.py` 会在结果目录生成 `review.md`，包含每题的 question、ground truth、**response**、**generation prompt**（实际送入 LLM 的完整 prompt）。

```bash
python evaluation/run_eval.py --dataset evaluation/datasets/eval_all.json
# 打开 evaluation/results/run_*/review.md
```
