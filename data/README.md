# 数据目录

航空航天知识库数据根目录：`data/nasa/`（大文件默认 gitignore）。

```
data/nasa/
├── pdfs/conference/       # NTRS 会议论文 PDF
├── markdown/conference/   # PDF → Markdown 缓存
├── chunks/
│   ├── section_chunks.json
│   └── per_pdf/           # 单篇分块缓存（断点续跑）
└── lessons_learned/       # Lessons Learned CSV
```

采集与分块：

```bash
python thesis_pipeline/download_nasa_data.py
```
