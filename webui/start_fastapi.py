#!/usr/bin/env python3
"""Start the FastAPI server as a daemon."""
import subprocess, sys, os, time, signal

# Kill any existing process on port 8501
os.system("fuser -k 8501/tcp 2>/dev/null")
time.sleep(1)

proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "webui.server:app",
     "--host", "0.0.0.0", "--port", "8501"],
    cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    stdout=open("/tmp/fastapi.log", "w"),
    stderr=subprocess.STDOUT,
    start_new_session=True
)
print(f"Started PID={proc.pid}")

for i in range(30):
    time.sleep(1)
    try:
        import urllib.request
        r = urllib.request.urlopen("http://localhost:8501", timeout=2)
        if r.status == 200:
            print(f"Server ready after {i+1}s")
            break
    except:
        pass
else:
    print("Server did not start in time")
    sys.exit(1)

print("Server is running. Press Ctrl+C to stop.")
try:
    proc.wait()
except KeyboardInterrupt:
    proc.terminate()
