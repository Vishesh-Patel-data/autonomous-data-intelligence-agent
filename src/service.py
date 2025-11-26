"""
service.py

FastAPI service exposing your data pipeline as HTTP endpoints.

For now it supports:
- GET  /health               → simple health check
- POST /run-pipeline/local   → run pipeline on a local CSV/Excel path

Later we will:
- Add PDF support
- Add a /chat endpoint
- Connect this to an ADK agent + UI
"""

from __future__ import annotations

import os
from typing import Optional, List


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .pipeline import run_full_pipeline, FullPipelineResult



app = FastAPI(
    title="Data Intelligence Pipeline Service",
    description="Backend service wrapping the autonomous data pipeline.",
    version="0.1.0",
)

# ------------------------------------------------------------
# Request / Response models
# ------------------------------------------------------------

class LocalPipelineRequest(BaseModel):
    tabular_path: str
    pdf_folder: Optional[str] = None


class PipelineSummaryResponse(BaseModel):
    raw_rows: int
    raw_cols: int
    cleaned_rows: int
    cleaned_cols: int
    columns: List[str]
    eda_report_markdown: str



# ------------------------------------------------------------
# Health check
# ------------------------------------------------------------

@app.get("/health")
def health_check():
    return {"status": "ok", "message": "Service is running."}


# ------------------------------------------------------------
# Run pipeline on a local path (dev only)
# ------------------------------------------------------------

@app.post("/run-pipeline/local", response_model=PipelineSummaryResponse)
def run_pipeline_local(req: LocalPipelineRequest):
    """
    Development-only endpoint:
    - Takes a local path to CSV/Excel
    - Runs the full pipeline
    - Returns a compact summary suitable for an agent to consume
    """
    if not os.path.exists(req.tabular_path):
        raise HTTPException(
            status_code=400,
            detail=f"tabular_path does not exist: {req.tabular_path}",
        )

    try:
        result: FullPipelineResult = run_full_pipeline(
            tabular_path=req.tabular_path,
            pdf_folder=req.pdf_folder,
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Pipeline execution failed: {e}",
        )

    if result.tabular is None:
        raise HTTPException(
            status_code=500,
            detail="Pipeline returned no tabular result.",
        )

    tab = result.tabular

    return PipelineSummaryResponse(
        raw_rows=tab.raw_df.shape[0],
        raw_cols=tab.raw_df.shape[1],
        cleaned_rows=tab.cleaned_df.shape[0],
        cleaned_cols=tab.cleaned_df.shape[1],
        columns=[str(c) for c in tab.cleaned_df.columns],
        eda_report_markdown=tab.eda_report_markdown,
    )
