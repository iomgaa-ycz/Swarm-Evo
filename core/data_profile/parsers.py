from __future__ import annotations

import csv
import os
import re
from typing import Optional, Iterable

from .schema import (
    EvaluationInfo, SubmissionInfo, SubmissionColumn,
    CSVColumnInfo, TrainCSVInfo, DataFolderInfo,
    DTypeGuess
)
#两层规则机制，第一层宽松提取需要的标题段落，第二层严格删除不需要的标题段落
#可以以较好的鲁棒性适配于不同的kaggle竞赛说明的形式
# ----------------------------
# Denylist for unwanted sections
# ----------------------------

DENY_SECTION_TITLES = [
    "acknowledgments",
    "acknowledgements",
    "citation",
    "timeline",
    "prizes",
    "rules",
    "code requirements",
    "submission limits",
    "team merger",
    "organizers",
]
# ----------------------------
# Markdown section extraction
# ----------------------------

def extract_markdown_section(md_text: str, section_titles: Iterable[str]) -> Optional[str]:
    def _normalize(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    titles = [_normalize(t) for t in section_titles]
    lines = md_text.splitlines()

    collecting = False
    buf = []

    for line in lines:
        m = re.match(r"^(#{1,6})\s*(.+?)\s*$", line)
        if m:
            title = _normalize(m.group(2))

            # start
            if any(t in title for t in titles):
                collecting = True
                buf = []
                continue

            # stop at ANY new heading
            if collecting:
                break

        if collecting:
            buf.append(line)

    content = "\n".join(buf).strip()
    return content if content else None


def extract_markdown_sections_multi(
    md_text: str,
    section_titles: Iterable[str],
) -> list[str]:
    """
    Extract ALL markdown sections whose heading matches any of section_titles.
    Flat extraction: each matched heading is treated independently.
    Stops each section at the next heading (any level).
    """

    def _normalize(s: str) -> str:
        return re.sub(r"\s+", " ", s.strip().lower())

    titles = [_normalize(t) for t in section_titles]
    lines = md_text.splitlines()

    sections: list[str] = []
    collecting = False
    buf: list[str] = []

    for line in lines:
        m = re.match(r"^(#{1,6})\s*(.+?)\s*$", line)
        if m:
            title = _normalize(m.group(2))

            # ---- start a new section ----
            if any(t in title for t in titles):
                if collecting and buf:
                    sections.append("\n".join(buf).strip())
                collecting = True
                buf = []
                continue

            # ---- end current section ----
            if collecting:
                sections.append("\n".join(buf).strip())
                collecting = False
                buf = []

        if collecting:
            buf.append(line)

    # flush last
    if collecting and buf:
        sections.append("\n".join(buf).strip())

    # remove empties
    return [s for s in sections if s]



def strip_deny_sections(text: str) -> str:
    if not text:
        return ""

    lines = text.splitlines()
    out = []
    deny_pattern = re.compile(
        r"^(#{1,6})\s*(acknowledg|timeline|citation|prize|rule)",
        re.IGNORECASE
    )

    for line in lines:
        if deny_pattern.match(line):
            break
        out.append(line)

    return "\n".join(out).strip()



def infer_metrics_from_text(text: str) -> list[str]:
    """
    Lightweight keyword-based metric extraction.
    Not perfect; good enough for data_profile.
    """
    if not text:
        return []
    t = text.lower()
    keys = [
        ("auc", ["auc", "roc auc", "area under"]),
        ("accuracy", ["accuracy", "acc"]),
        ("logloss", ["log loss", "logloss", "cross entropy"]),
        ("rmse", ["rmse", "root mean squared"]),
        ("mae", ["mae", "mean absolute"]),
        ("f1", ["f1", "f-1"]),
        ("map", ["mAP", "mean average precision".lower()]),
        ("bleu", ["bleu"]),
    ]
    found = []
    for name, variants in keys:
        if any(v in t for v in variants):
            found.append(name)
    return sorted(list(set(found)))


def parse_description_md(md_text: str):
    md_text = md_text or ""

    # 1. Competition Description (multi parser)
    background_text = "\n\n".join(
        strip_deny_sections(s)
        for s in extract_markdown_sections_multi(md_text, ["description"])
    )

    # 2. Data Description (multi parser)
    data_text = "\n\n".join(
        strip_deny_sections(s)
        for s in extract_markdown_sections_multi(md_text, [
            "dataset description",
            "data",
            "files",
            "file descriptions",
            "data fields",
        ])
    )

    # 3. Evaluation (single, strict)
    eval_text = extract_markdown_section(
        md_text,
        ["evaluation", "evaluation metric", "metrics", "metric"]
    ) or ""

    ev = EvaluationInfo(
        raw_text=eval_text,
        metrics=infer_metrics_from_text(eval_text),
        ranking_rule=None,
        notes=None,
    )

    # 4. Submission (single, strict)
    submission_text = extract_markdown_section(
        md_text,
        ["submission file", "submission format", "submission"]
    ) or ""

    submission_text = strip_deny_sections(submission_text)

    return background_text, data_text, ev, submission_text




# ----------------------------
# CSV shape parsers
# ----------------------------

def read_csv_header(csv_path: str, encoding: str = "utf-8") -> list[str]:
    with open(csv_path, "r", encoding=encoding, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, [])
    return header


def read_csv_sample_rows(
    csv_path: str,
    n: int = 10,
    encoding: str = "utf-8",
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(csv_path, "r", encoding=encoding, newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= n:
                break
            # stringify values to avoid leaking "numbers" semantics—still shows shape
            rows.append({k: ("" if v is None else str(v)) for k, v in row.items()})
    return rows


def guess_dtype_from_values(values: Iterable[str]) -> DTypeGuess:
    """
    Very lightweight dtype guess:
    - numeric: most values parse as float and are short
    - text: long strings or high alphabetic ratio
    - categorical: short strings but not numeric
    """
    vals = [v for v in values if v is not None and str(v).strip() != ""]
    if not vals:
        return "unknown"

    # numeric check
    numeric_cnt = 0
    for v in vals[:50]:
        try:
            float(str(v))
            numeric_cnt += 1
        except Exception:
            pass

    if numeric_cnt / max(1, len(vals[:50])) > 0.8:
        return "numeric"

    # text check
    long_cnt = sum(1 for v in vals[:50] if len(str(v)) >= 30)
    if long_cnt / max(1, len(vals[:50])) > 0.3:
        return "text"

    return "categorical"


def build_submission_info(sample_submission_path: str, format_description: str) -> SubmissionInfo:
    # header only, no values
    header = read_csv_header(sample_submission_path)
    cols = [SubmissionColumn(name=c, index=i) for i, c in enumerate(header)]
    lower_cols = [c.lower() for c in header]

    has_id = any(c in lower_cols for c in ["id", "image_id", "audio_id", "uid", "filename", "file", "path"])

    label_guess = None
    if len(header) == 1:
        label_guess = header[0]
    elif len(header) >= 2:
        # often first is id, second is target
        # choose a non-id as target if exists
        non_id = [c for c in header if c.lower() not in ["id", "image_id", "audio_id", "uid", "filename", "file", "path"]]
        if non_id:
            label_guess = non_id[-1]

    return SubmissionInfo(
        file_name="submission.csv",
        format_description=format_description or "",
        columns=cols,
        num_columns=len(cols),
        has_id_column=has_id,
        label_column_guess=label_guess,
    )


def build_train_csv_info(train_csv_path: str, preview_rows: int = 10) -> TrainCSVInfo:
    header = read_csv_header(train_csv_path)
    sample_rows = read_csv_sample_rows(train_csv_path, n=preview_rows)

    # dtype guess per column using sampled values
    columns: list[CSVColumnInfo] = []
    for c in header:
        col_values = [r.get(c, "") for r in sample_rows]
        dtype = guess_dtype_from_values(col_values)
        columns.append(CSVColumnInfo(name=c, dtype_guess=dtype))

    # label guess: common names, or last column if seems like classification/regression
    lower = [c.lower() for c in header]
    candidates = ["target", "label", "y", "class", "score", "outcome"]
    label_guess = None
    for cand in candidates:
        if cand in lower:
            label_guess = header[lower.index(cand)]
            break
    if label_guess is None and len(header) >= 2:
        # often last col is label for train.csv tasks
        label_guess = header[-1]

    return TrainCSVInfo(
        exists=True,
        path=train_csv_path,
        num_rows_previewed=len(sample_rows),
        columns=columns,
        label_column_guess=label_guess,
        sample_rows=sample_rows,
    )


# ----------------------------
# Folder inference
# ----------------------------

IMAGE_EXTS = {"jpg", "jpeg", "png", "gif", "bmp", "tiff", "webp"}
AUDIO_EXTS = {"wav", "mp3", "flac", "ogg", "m4a", "aac"}


def detect_folder_type_and_stats(folder_path: str, max_count: int = 5000) -> DataFolderInfo:
    """
    Scan up to max_count files to infer file type and extensions.
    """
    exts = {}
    num = 0

    for root, _, files in os.walk(folder_path):
        for fn in files:
            num += 1
            if num > max_count:
                break
            ext = os.path.splitext(fn)[1].lstrip(".").lower()
            if not ext:
                continue
            exts[ext] = exts.get(ext, 0) + 1
        if num > max_count:
            break

    ext_list = sorted(exts.keys(), key=lambda e: exts[e], reverse=True)
    file_type = "unknown"
    if ext_list:
        top_ext = ext_list[0]
        if top_ext in IMAGE_EXTS:
            file_type = "image"
        elif top_ext in AUDIO_EXTS:
            file_type = "audio"

    return DataFolderInfo(
        exists=True,
        path=folder_path,
        file_type=file_type,
        num_files=num if num <= max_count else None,
        file_extensions=ext_list[:10],
    )
