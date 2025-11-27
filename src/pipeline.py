"""
pipeline.py

This module hosts the *core data pipeline* logic for your project:
- Load tabular data (CSV/Excel)
- Run cleaning / EDA pipeline
- (Later) Run PDF summarization
- (Later) Produce an executive summary

Later, we'll call these functions from an ADK Agent / API.
For now, we just define clean, reusable Python functions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import fitz  # PyMuPDF

from .llm_utils import generate_eda_report_markdown


# NOTE:
# We'll plug in Gemini / LLM logic later from a separate config/util.
# This file currently focuses on data + basic PDF helpers.


# ------------------------------------------------------------
# 1. Simple dataclasses to describe outputs
# ------------------------------------------------------------

@dataclass
class TabularPipelineResult:
    raw_df: pd.DataFrame
    cleaned_df: pd.DataFrame
    eda_summary: Dict[str, Any]
    eda_report_markdown: str  # will be LLM-generated later


@dataclass
class PdfPipelineResult:
    pdf_summaries: Dict[str, str]
    combined_summary_markdown: str


@dataclass
class FullPipelineResult:
    tabular: Optional[TabularPipelineResult]
    pdfs: Optional[PdfPipelineResult]
    executive_report_markdown: Optional[str]


# ------------------------------------------------------------
# 2. Basic file utilities (tabular + PDF)
# ------------------------------------------------------------

def load_tabular_file(path: str) -> pd.DataFrame:
    """
    Load a CSV or Excel file into a pandas DataFrame.
    This is the simple, generic loader that other functions will use.
    """
    ext = path.lower().split(".")[-1]
    if ext == "csv":
        return pd.read_csv(path)
    elif ext in ("xlsx", "xls"):
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported tabular extension for path: {path}")


def detect_file_type(path: str) -> str:
    """Identify CSV / Excel / PDF / Unknown based on extension."""
    ext = path.lower().split(".")[-1]
    if ext == "csv":
        return "csv"
    if ext in ("xlsx", "xls"):
        return "excel"
    if ext == "pdf":
        return "pdf"
    return "unknown"


def extract_pdf_text(pdf_path: str) -> str:
    """Extract full plain text from a PDF file."""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        return f"[ERROR] Could not open PDF: {e}"

    text = ""
    for page in doc:
        text += page.get_text()

    return text.strip()


def list_pdf_files(folder: str) -> List[str]:
    """Return full paths to all PDFs inside a folder (recursive)."""
    pdfs: List[str] = []
    for root, _, files in os.walk(folder):
        for fname in files:
            if fname.lower().endswith(".pdf"):
                pdfs.append(os.path.join(root, fname))
    return pdfs


# ------------------------------------------------------------
# 3. EDA summary helper (generic, no LLM)
# ------------------------------------------------------------

def compute_eda_summary(df: pd.DataFrame, max_categories: int = 10) -> Dict[str, Any]:
    """
    Create a compact, structured EDA summary of a cleaned dataframe.
    This is similar to your Kaggle version, but kept generic.
    """
    summary: Dict[str, Any] = {
        "shape": {
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
        },
        "columns": list(df.columns),
        "missing_values": {},
        "numeric_stats": {},
        "categorical_top_values": {},
        "correlations": {},
    }

    # Missing values
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100

    summary["missing_values"] = {
        col: {"count": int(missing_counts[col]), "pct": float(missing_pct[col])}
        for col in df.columns
        if missing_counts[col] > 0
    }

    # Numeric stats
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_cols:
        summary["numeric_stats"] = df[numeric_cols].describe().to_dict()

    # Categorical top values
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in categorical_cols:
        summary["categorical_top_values"][col] = (
            df[col].value_counts().head(max_categories).to_dict()
        )

    # Correlations
    if len(numeric_cols) >= 2:
        corr_matrix = df[numeric_cols].corr()
        summary["correlations"] = corr_matrix.to_dict()

    return summary


# ------------------------------------------------------------
# 4. Simple generic cleaning (no agents yet)
# ------------------------------------------------------------

def simple_clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    A simple, generic cleaning routine:
    - Drop columns with > 80% missing
    - Fill numeric NaNs with median
    - Fill object NaNs with mode
    - Drop duplicate rows

    This mirrors the spirit of your orchestrated pipeline
    but in a compact form for backend use.
    """
    df_clean = df.copy()

    # 1) Drop columns that are mostly missing
    missing_ratio = df_clean.isnull().mean()
    cols_to_drop = missing_ratio[missing_ratio > 0.8].index.tolist()
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)

    # 2) Fill numeric NaNs with median
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        median_val = df_clean[col].median()
        df_clean[col] = df_clean[col].fillna(median_val)

    # 3) Fill object NaNs with mode
    obj_cols = df_clean.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        if df_clean[col].isnull().any():
            mode_vals = df_clean[col].mode()
            fill_val = mode_vals.iloc[0] if not mode_vals.empty else "Unknown"
            df_clean[col] = df_clean[col].fillna(fill_val)

    # 4) Drop duplicate rows
    df_clean = df_clean.drop_duplicates()

    return df_clean


# ------------------------------------------------------------
# 5. High-level pipeline entry point
# ------------------------------------------------------------

def run_full_pipeline(
    tabular_path: Optional[str] = None,
    pdf_folder: Optional[str] = None,
) -> FullPipelineResult:
    """
    High-level entry point that will be called by:
    - ADK agent tools
    - FastAPI endpoints
    - Notebooks, etc.

    CURRENT IMPLEMENTATION (STEP 5):
    - If tabular_path is provided:
        - load the file
        - run simple cleaning
        - compute EDA summary
        - (EDA report text will be filled in later with LLM)
    - PDF + executive report will be wired in later steps.
    """
    tabular_result: Optional[TabularPipelineResult] = None
    pdf_result: Optional[PdfPipelineResult] = None
    executive_report_md: Optional[str] = None

    # ---- Tabular pipeline (basic version) ----
    if tabular_path is not None:
        raw_df = load_tabular_file(tabular_path)
        cleaned_df = simple_clean_dataframe(raw_df)
        eda_summary = compute_eda_summary(cleaned_df)

        # Placeholder EDA report text (LLM will replace this later)
        eda_report_md = generate_eda_report_markdown(eda_summary)

        tabular_result = TabularPipelineResult(
            raw_df=raw_df,
            cleaned_df=cleaned_df,
            eda_summary=eda_summary,
            eda_report_markdown=eda_report_md,
        )

    # ---- PDF pipeline + executive report will be implemented later ----

    return FullPipelineResult(
        tabular=tabular_result,
        pdfs=pdf_result,
        executive_report_markdown=executive_report_md,
    )


if __name__ == "__main__":
    # Small manual test hook – this won't run in ADK,
    # but is useful if you run `python src/pipeline.py` locally.
    print("pipeline.py loaded. Implementations will be added step-by-step.")




