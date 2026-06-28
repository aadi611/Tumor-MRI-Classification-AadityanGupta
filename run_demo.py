"""
run_demo.py — one-command launcher for the live demo.

    python run_demo.py

Starts the FastAPI server and opens the browser automatically.
Models load lazily (fast startup); the first time you pick a model it warms up,
then every prediction is instant.
"""
import threading
import webbrowser

import uvicorn

HOST, PORT = "127.0.0.1", 8000


def _open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    print(f"\n  Brain Tumor MRI Classifier — live demo")
    print(f"  → opening http://{HOST}:{PORT}  (Ctrl+C to stop)\n")
    threading.Timer(1.8, _open_browser).start()
    uvicorn.run("app.main:app", host=HOST, port=PORT, log_level="warning")
