#!/bin/bash
cd "$(dirname "$0")"
exec streamlit run webui/app.py --server.port 8501 --server.address 0.0.0.0
