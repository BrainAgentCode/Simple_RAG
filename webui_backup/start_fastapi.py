#!/usr/bin/env python3
"""Start the FastAPI server as a daemon."""
import subprocess, sys, os, signal

os.system("fuser -k 8501/tcp 2>/dev/null")

proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "webui.server:app",
     "--host", "0.0.0.0", "--port", "8501"],
    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    stdout=open("/tmp/fastapi.log", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True,
    env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
)
signal.signal(signal.SIGHUP, signal.SIG_IGN)
print(f"Server started as PID={proc.pid}, detached.")
sys.exit(0)
