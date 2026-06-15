# AGENTS.md

## What This Is

RAG (Retrieval-Augmented Generation) system for aerospace knowledge Q&A over NASA conference papers and Lessons Learned. Chinese and English bilingual.

## Quick Commands

```bash
# Install deps
pip install -r requirements.txt

# Download + chunk NASA data (must run before anything works)
python thesis_pipeline/download_nasa_data.py

# Start web UI (Streamlit on port 8501)
bash run_webui.sh

# Start CLI interactive mode
python main.py
```

## Data Pipeline Must Run First

The system loads from `data/nasa/` and `vector_index/`. Both are gitignored. If you see "加载 0 个文档" or index errors, run:

```bash
python thesis_pipeline/download_nasa_data.py
```

Pipeline is resumable — existing files are skipped automatically.

## Environment

Copy `.env.example` to `.env` and set at least one of:
- `OPENAI_API_KEY` (for OpenAI API)
- `LOCAL_LLM_BASE_URL` (for local vLLM/Ollama)

Key defaults: `EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5`, `EMBEDDING_DEVICE=cpu`, `RERANKER_MODEL=BAAI/bge-reranker-v2-m3`.

## Architecture

```
main.py                    — AerospaceRAGSystem: orchestrator, CLI entrypoint
config.py                  — RAGConfig dataclass, loads from .env
rag_modules/
  data_preparation.py      — loads PDFs/chunks, builds parent documents
  index_construction.py    — FAISS vector index build/load
  retrieval_optimization.py — hybrid search (vector + BM25 + RRF + reranker)
  generation_integration.py — LLM calls, prompt templates, query routing
  query_understanding.py   — intent classification, reference resolution, safety
  query_utils.py           — Chinese detection, translation helpers
  runtime_accel.py         — GPU/parallel acceleration utilities
  document_summary.py      — LLM-based document summarization
webui/app.py               — Streamlit frontend (config UI + chat)
thesis_pipeline/           — NASA data scraping + PDF chunking
evaluation/                — RAGAS evaluation pipeline
```

## Key Conventions

- **Language**: Code comments and UI strings are in Chinese. Keep it that way.
- **Dual LLM**: Supports OpenAI API and local models (vLLM). Provider is set via `LLM_PROVIDER` env var or WebUI.
- **Query flow**: intent classification → reference resolution → safety check → query rewrite → hybrid retrieval (vector + BM25) → RRF fusion → reranker → LLM generation.
- **Parent documents**: Retrieved chunks map back to parent docs (abstract/conclusions/introduction) for generation context.
- **Bilingual**: Chinese queries are auto-translated to English for BM25/vector retrieval. Controlled by `TRANSLATE_QUERY_FOR_BM25` and `TRANSLATE_QUERY_FOR_VECTOR`.

## Evaluation

```bash
# Generate eval dataset from chunks
python evaluation/generate_dataset.py llm --max-docs 5 --per-doc 2

# Run RAG on dataset
python evaluation/run_eval.py --dataset evaluation/datasets/eval_all.json

# Score with RAGAS
python evaluation/run_ragas.py --predictions evaluation/results/run_*/predictions.jsonl
```

See `evaluation/README.md` for full details.

## Gotchas

- `data/`, `vector_index/`, `webui/app_settings.json`, `webui/model_configs.json` are all gitignored — never commit these.
- FAISS GPU is auto-detected but defaults to CPU. Set `FAISS_USE_GPU=true` in `.env` if you have CUDA + faiss-gpu.
- The reranker loads on first retrieval call, not at startup — don't be surprised by a delay on first query.
- `requirements.txt` installs `faiss-cpu` by default. For GPU: `pip uninstall faiss-cpu -y && pip install faiss-gpu`.
- After changing data, delete `vector_index/` and `data/nasa/parent_summaries_cache.json` to force rebuild.
