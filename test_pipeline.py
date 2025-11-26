"""
test_pipeline.py

Small script to test src/pipeline.py using a local CSV/Excel file.
"""

import os
import sys

# Make sure Python can see the src/ folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(CURRENT_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from pipeline import run_full_pipeline, FullPipelineResult


# TODO: 👉 UPDATE THIS PATH TO POINT TO YOUR REAL CSV/EXCEL FILE
TABULAR_PATH = r"C:\Users\91727\Downloads\TUVYproperty_details.xlsx"


def main() -> None:
    if not os.path.exists(TABULAR_PATH):
        print("❌ Tabular file not found at path:")
        print(f"   {TABULAR_PATH}")
        print("➡️  Please edit TABULAR_PATH in test_pipeline.py and try again.")
        return

    print("✅ Found tabular file, running pipeline...")
    result: FullPipelineResult = run_full_pipeline(tabular_path=TABULAR_PATH)

    if result.tabular is None:
        print("❌ No tabular result returned from pipeline.")
        return

    tab = result.tabular

    print("\n=== TABULAR PIPELINE RESULT ===")
    print(f"- Raw shape:     {tab.raw_df.shape}")
    print(f"- Cleaned shape: {tab.cleaned_df.shape}")
    print(f"- Columns:       {tab.cleaned_df.columns.tolist()}")

    print("\n=== MISSING VALUES SUMMARY (after cleaning) ===")
    mv = tab.eda_summary.get("missing_values", {})
    if not mv:
        print("- No missing values reported.")
    else:
        for col, info in mv.items():
            print(f"  • {col}: {info['count']} missing ({info['pct']:.2f}%)")

    print("\n=== EDA REPORT PLACEHOLDER ===")
    print(tab.eda_report_markdown)


if __name__ == "__main__":
    main()
