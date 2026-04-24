"""
plrs.api.app
============
FastAPI application — see full implementation in the complete repo zip.
This stub exists so imports work in the sandbox / CI check.
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

DEV_MODE = os.getenv("PLRS_DEV_MODE", "0") == "1"

app = FastAPI(title="PLRS API", version="0.4.0", docs_url="/docs")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_pipelines = {}

def register_pipeline(domain: str, pipeline) -> None:
    _pipelines[domain] = pipeline

def get_pipeline(domain: str):
    from fastapi import HTTPException
    if domain not in _pipelines:
        raise HTTPException(status_code=404, detail=f"Domain '{domain}' not found.")
    return _pipelines[domain]

@app.get("/health")
def health():
    return {"status": "ok", "version": "0.4.0", "loaded_domains": list(_pipelines.keys()), "auth_enabled": not DEV_MODE}

@app.get("/")
def root():
    return {"name": "PLRS API", "docs": "/docs", "health": "/health"}
