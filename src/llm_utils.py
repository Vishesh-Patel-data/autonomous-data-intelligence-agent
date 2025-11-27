"""
llm_utils.py

Utility functions for calling Gemini to generate natural-language reports,
using the raw HTTP API (no google-generativeai client).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict

import requests


GEMINI_API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-2.5-flash-lite:generateContent"
)




def _ensure_api_key() -> str:
    """Fetch GOOGLE_API_KEY from environment or raise a clear error."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY environment variable is not set. "
            "Set it before running the pipeline."
        )
    return api_key


def _call_gemini_http(prompt: str) -> str:
    """
    Call Gemini 1.5 Flash via the public HTTP endpoint.

    Uses the "contents" format described in the Generative Language API docs.
    """
    api_key = _ensure_api_key()

    url = f"{GEMINI_API_URL}?key={api_key}"
    headers = {"Content-Type": "application/json"}

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
        resp.raise_for_status()
        data = resp.json()

        # Typical structure:
        # {
        #   "candidates": [
        #     {
        #       "content": {
        #         "parts": [{"text": "..."}]
        #       },
        #       ...
        #     }
        #   ]
        # }
        candidates = data.get("candidates", [])
        if not candidates:
            return f"LLM EDA report generation failed: no candidates in response: {data}"

        content = candidates[0].get("content", {})
        parts = content.get("parts", [])
        if not parts:
            return f"LLM EDA report generation failed: no parts in response: {data}"

        text = parts[0].get("text", "")
        if not text:
            return f"LLM EDA report generation failed: empty text in response: {data}"

        return text

    except requests.HTTPError as e:
        return f"LLM EDA report generation failed (HTTP error): {e}, body={resp.text}"
    except Exception as e:
        return f"LLM EDA report generation failed (unexpected error): {e}"


def generate_eda_report_markdown(eda_summary: Dict[str, Any]) -> str:
    """
    Use Gemini (via HTTP) to convert a structured EDA summary
    into a clean, professional Markdown report.
    """
    prompt = f"""
You are a senior data analyst. Convert this structured EDA summary
into a clean, professional Markdown report.

Summary (JSON):
{json.dumps(eda_summary, indent=2)}

Include sections:

1. Dataset Overview
2. Missing Value Analysis
3. Numeric Insights
4. Categorical Patterns
5. Correlation Highlights (if any)
6. Key Takeaways

Write clearly and concisely.
"""
    return _call_gemini_http(prompt)

# ------------------------------------------------------------
# PDF summarization helpers (single + combined)
# ------------------------------------------------------------
import os
import json
import requests


def summarize_pdf_text_with_gemini(pdf_text: str) -> str:
    """
    Summarize a single PDF's raw text into a concise Markdown summary.
    Uses the Gemini HTTP API (same style as app_gradio.py).
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "PDF summary error: GOOGLE_API_KEY environment variable is not set."

    model = "gemini-2.5-flash-lite"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    prompt = f"""
You are a senior analyst.

Summarize the following PDF content.

Include:
- A concise overview
- Key themes/topics
- Important insights
- Any high-level structure (sections, phases, etc.)
- Write in clean Markdown

PDF content:
---
{pdf_text[:20000]}
---
"""

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            return "PDF summary error: LLM returned no candidates."

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return "PDF summary error: LLM returned no content parts."

        return parts[0].get("text", "") or "PDF summary error: LLM returned empty text."

    except requests.exceptions.HTTPError as http_err:
        return f"PDF summary HTTP error: {http_err}\nResponse: {resp.text}"
    except Exception as e:
        return f"PDF summary error: {e}"


def summarize_multiple_pdfs_with_gemini(pdf_summaries: dict) -> str:
    """
    Given a dict {pdf_name: single_summary}, produce a combined meta-summary.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return "Combined PDF summary error: GOOGLE_API_KEY environment variable is not set."

    model = "gemini-2.5-flash-lite"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    prompt = f"""
You are an AI analyst. Below are summaries of multiple PDF documents:

{json.dumps(pdf_summaries, indent=2)}

Produce a single clear combined summary including:
- What all documents are about overall
- Key shared themes and differences
- Any implicit structure or phases
- High-level strategic takeaways
- Write in business-ready Markdown
"""

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    body = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    try:
        resp = requests.post(url, headers=headers, json=body, timeout=300)
        resp.raise_for_status()
        data = resp.json()

        candidates = data.get("candidates", [])
        if not candidates:
            return "Combined PDF summary error: LLM returned no candidates."

        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return "Combined PDF summary error: LLM returned no content parts."

        return parts[0].get("text", "") or "Combined PDF summary error: LLM returned empty text."

    except requests.exceptions.HTTPError as http_err:
        return f"Combined PDF summary HTTP error: {http_err}\nResponse: {resp.text}"
    except Exception as e:
        return f"Combined PDF summary error: {e}"

