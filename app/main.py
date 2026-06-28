"""
main.py
=======
FastAPI backend for the live demo.

Endpoints
---------
GET  /              -> the single-page UI
GET  /api/models    -> models whose checkpoints exist (drives the dropdown)
POST /api/load      -> warm-load a model into memory (called when the dropdown changes)
POST /api/predict   -> classify an uploaded image with the selected model
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from . import inference

app = FastAPI(title="Brain Tumor MRI Classifier", version="1.0")
STATIC = Path(__file__).resolve().parent / "static"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/models")
def list_models() -> dict:
    models = inference.available_models()
    if not models:
        return {"models": [], "error": "No trained checkpoints found in models/."}
    return {"models": models, "device": inference.DEVICE.type}


@app.post("/api/load")
def load(model: str = Form(...)) -> dict:
    if model not in {m["id"] for m in inference.available_models()}:
        raise HTTPException(404, f"Unknown or unavailable model: {model}")
    try:
        inference.load_model(model)
    except Exception as e:                       # surface load errors cleanly in the UI
        raise HTTPException(500, f"Failed to load '{model}': {e}")
    return {"status": "ready", "model": model}


@app.post("/api/predict")
async def predict(model: str = Form(...), file: UploadFile = File(...)) -> dict:
    if model not in {m["id"] for m in inference.available_models()}:
        raise HTTPException(404, f"Unknown or unavailable model: {model}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "Empty file.")
    try:
        return inference.predict(model, data)
    except Exception as e:
        raise HTTPException(500, f"Prediction failed: {e}")
