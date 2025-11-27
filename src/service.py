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
from typing import Optional, List, Dict, Any


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


class PipelineRequest(BaseModel):
    tabular_path: Optional[str] = None
    pdf_paths: Optional[List[str]] = None



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

@app.post("/run-pipeline/local")
async def run_pipeline_local(request: PipelineRequest):
    """
    Run the full pipeline on:
    - an optional tabular path (CSV/Excel)
    - optional list of PDF paths

    Returns a combined JSON with:
    - Tabular summary + EDA report (if tabular_path set)
    - PDF summaries + combined PDF summary (if pdf_paths set)
    """
    try:
        result: FullPipelineResult = run_full_pipeline(
            tabular_path=request.tabular_path,
            pdf_paths=request.pdf_paths,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if result.tabular is None and result.pdfs is None:
        raise HTTPException(
            status_code=400,
            detail="No valid inputs provided. Expected tabular_path and/or pdf_paths."
        )

    response: Dict[str, Any] = {}

    # Tabular output (backwards compatible)
    if result.tabular is not None:
        t = result.tabular
        response.update(
            {
                "raw_rows": int(t.raw_df.shape[0]),
                "raw_cols": int(t.raw_df.shape[1]),
                "cleaned_rows": int(t.cleaned_df.shape[0]),
                "cleaned_cols": int(t.cleaned_df.shape[1]),
                "columns": list(t.cleaned_df.columns),
                "eda_report_markdown": t.eda_report_markdown,
            }
        )

    # PDF output (new)
    if result.pdfs is not None:
        p = result.pdfs
        response["pdf_summaries"] = p.pdf_summaries
        response["pdf_combined_summary_markdown"] = p.combined_summary_markdown

    return response

