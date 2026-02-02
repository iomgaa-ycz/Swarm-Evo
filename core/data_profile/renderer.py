from __future__ import annotations

from typing import Any


def _truncate(s: str, max_chars: int) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= max_chars else (s[:max_chars] + "\n... [truncated]")


def render_data_profile_text(data_profile: dict[str, Any], max_chars: int = 4000) -> str:
    """
    Render a compact, token-friendly text summary for prompt.
    Keep it stable & structured.
    """
    bg = data_profile.get("background", {})
    dd = data_profile.get("data_description", {})
    ev = data_profile.get("evaluation", {})
    sub = data_profile.get("submission", {})
    ds = data_profile.get("dataset", {})
    ov = data_profile.get("task_overview", {})

    # submission columns
    cols = sub.get("columns", [])
    col_names = [c.get("name") for c in cols if isinstance(c, dict)]
    if len(col_names) > 30:
        col_names = col_names[:30] + ["..."]

    # train columns
    train_csv = ds.get("train_csv", {})
    train_cols = train_csv.get("columns", [])
    train_col_names = [c.get("name") for c in train_cols if isinstance(c, dict)]
    if len(train_col_names) > 50:
        train_col_names = train_col_names[:50] + ["..."]

    # train sample rows (first 3 shown)
    sample_rows = train_csv.get("sample_rows", [])
    sample_rows_show = sample_rows[:3] if isinstance(sample_rows, list) else []

    text = f"""
# Data Profile (Auto-Extracted)

## Competition Description (from description.md)
{_truncate(bg.get("raw_text", ""), 1200)}

## Dataset Description (from description.md)
{_truncate(dd.get("raw_text", ""), 1200)}

## Task Overview
- task_type: {ov.get("task_type")}
- supervision: {ov.get("supervision")}
- input_modalities: {ov.get("input_modalities")}


## Evaluation (from description.md)
{_truncate(ev.get("raw_text", ""), 1200)}

## Submission Format (from description.md)
{_truncate(sub.get("format_description", ""), 800)}

## sample_submission.csv (shape only)
- num_columns: {sub.get("num_columns")}
- columns: {col_names}
- has_id_column: {sub.get("has_id_column")}
- label_column_guess: {sub.get("label_column_guess")}

## Training Data Organization
- organization_type: {ds.get("organization_type")}
- modality_guess: {ds.get("modality_guess")}
- train_csv: {train_csv.get("path")}
- train_folder: {ds.get("train_folder", {}).get("path")}
- label_files: {ds.get("label_files")}

## train.csv
- columns: {train_col_names}
- label_column_guess: {train_csv.get("label_column_guess")}
- sample_rows (first 3 of {train_csv.get("num_rows_previewed")} previewed):
{sample_rows_show}
""".strip()

    return _truncate(text, max_chars)
