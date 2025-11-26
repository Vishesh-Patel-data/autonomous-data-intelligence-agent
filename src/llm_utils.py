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
