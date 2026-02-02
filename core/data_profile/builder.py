from __future__ import annotations

import os
import glob
from datetime import datetime, timezone
from typing import Optional

from .schema import (
    DataProfile,
    MetaInfo,
    TaskOverview,
    DatasetInfo,
    FilesEvidence,
    BackgroundInfo,       
    DataDescriptionInfo, 
)
from .parsers import (
    parse_description_md,
    build_submission_info,
    build_train_csv_info,
    detect_folder_type_and_stats,
)
from utils.directory_tree_generator import DirectoryTreeGenerator
 


def _iso_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def _read_text_file(path: str, encodings=("utf-8", "utf-8-sig", "latin-1")) -> str:
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except Exception:
            continue
    return ""


def _find_first_existing(paths: list[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _find_by_glob(workspace_root: str, patterns: list[str]) -> list[str]:
    hits = []
    for pat in patterns:
        hits.extend(glob.glob(os.path.join(workspace_root, pat)))
    # unique + stable
    out = []
    seen = set()
    for h in hits:
        h2 = os.path.abspath(h)
        if h2 not in seen:
            seen.add(h2)
            out.append(h2)
    return out


class DataProfileBuilder:
    def __init__(
        self,
        workspace_root: str,
        max_depth: int = 6,
        collapse_threshold: int = 5,
        preview_files: Optional[list[str]] = None,
        preview_rows_train: int = 10,
    ):
        self.workspace_root = os.path.abspath(workspace_root)
        self.max_depth = max_depth
        self.collapse_threshold = collapse_threshold
        self.preview_files = preview_files or ["sample_submission.csv", "train.csv", "description.md"]
        self.preview_rows_train = preview_rows_train

    def build(self) -> dict:
        # 1) Evidence layer from directory tree
        tree_md, previews = DirectoryTreeGenerator(
            root_path=self.workspace_root,
            max_depth=self.max_depth,
            collapse_threshold=self.collapse_threshold,
            preview_files=self.preview_files,
        ).generate()

        files_evidence = FilesEvidence(
            directory_tree_markdown=tree_md,
            previews=previews,
        )

        # 2) Parse description.md 
        description_path = self._locate_description_md()
        description_text = _read_text_file(description_path) if description_path else ""
        background_text, data_text, evaluation_info, submission_format_text = parse_description_md(description_text)
        background_info = BackgroundInfo(raw_text=background_text, source_path=description_path)
        data_desc_info = DataDescriptionInfo(raw_text=data_text, source_path=description_path)


        # 3) Parse sample_submission.csv (shape)
        sample_sub_path = self._locate_sample_submission()
        submission_info = None
        if sample_sub_path:
            submission_info = build_submission_info(sample_sub_path, submission_format_text)
        else:
            # fallback empty submission info but keep description text
            from .schema import SubmissionInfo
            submission_info = SubmissionInfo(format_description=submission_format_text)

        # 4) Parse training data organization
        dataset_info = self._build_dataset_info()

        # 5) Infer task_overview from dataset + submission
        task_overview = self._infer_task_overview(dataset_info, submission_info)

        profile = DataProfile(
            meta=MetaInfo(
                workspace_root=self.workspace_root,
                generated_at=_iso_now(),
            ),
            task_overview=task_overview,

            background=background_info,
            data_description=data_desc_info,

            evaluation=evaluation_info,
            submission=submission_info,
            dataset=dataset_info,
            files=files_evidence,
        )
        return profile.to_dict()

    # -----------------------
    # Locators
    # -----------------------
    def _locate_description_md(self) -> Optional[str]:
        candidates = [
            os.path.join(self.workspace_root, "description.md"),
            os.path.join(self.workspace_root, "DESCRIPTION.md"),
            os.path.join(self.workspace_root, "data", "description.md"),
            os.path.join(self.workspace_root, "README.md"),
            os.path.join(self.workspace_root, "readme.md"),
        ]
        hit = _find_first_existing(candidates)
        if hit:
            return os.path.abspath(hit)

        # fallback: search shallow
        hits = _find_by_glob(self.workspace_root, ["**/description.md", "**/README.md"])
        return hits[0] if hits else None

    def _locate_sample_submission(self) -> Optional[str]:
        candidates = [
            os.path.join(self.workspace_root, "sample_submission.csv"),
            os.path.join(self.workspace_root, "submission", "sample_submission.csv"),
            os.path.join(self.workspace_root, "data", "sample_submission.csv"),
            os.path.join(self.workspace_root, "sampleSubmission.csv"),
        ]
        hit = _find_first_existing(candidates)
        if hit:
            return os.path.abspath(hit)

        hits = _find_by_glob(self.workspace_root, ["**/sample_submission.csv"])
        return hits[0] if hits else None

    def _locate_train_csv(self) -> Optional[str]:
        candidates = [
            os.path.join(self.workspace_root, "train.csv"),
            os.path.join(self.workspace_root, "data", "train.csv"),
            os.path.join(self.workspace_root, "dataset", "train.csv"),
        ]
        hit = _find_first_existing(candidates)
        if hit:
            return os.path.abspath(hit)

        hits = _find_by_glob(self.workspace_root, ["**/train.csv"])
        return hits[0] if hits else None

    def _locate_train_folder(self) -> Optional[str]:
        candidates = [
            os.path.join(self.workspace_root, "train"),
            os.path.join(self.workspace_root, "data", "train"),
        ]
        for c in candidates:
            if os.path.isdir(c):
                return os.path.abspath(c)

        # fallback: common variants (images, audio)
        # but avoid huge recursive matches; pick shallow first
        for name in ["images", "imgs", "audio", "wav", "data_train", "train_data"]:
            c = os.path.join(self.workspace_root, name)
            if os.path.isdir(c):
                return os.path.abspath(c)

        # last resort: glob
        hits = _find_by_glob(self.workspace_root, ["**/train"])
        return hits[0] if hits else None

    def _locate_label_files(self) -> list[str]:
        patterns = [
            "**/label.txt",
            "**/labels.txt",
            "**/labels.csv",
            "**/train_labels.csv",
            "**/train_label.csv",
        ]
        return _find_by_glob(self.workspace_root, patterns)

    def _locate_test_files(self) -> list[str]:
        patterns = [
            "**/test.csv",
            "**/test/*",
            "**/data/test.csv",
            "**/data/test/*",
        ]
        return _find_by_glob(self.workspace_root, patterns)

    # -----------------------
    # Dataset builder
    # -----------------------
    def _build_dataset_info(self) -> DatasetInfo:
        dataset = DatasetInfo()

        train_csv = self._locate_train_csv()
        train_folder = self._locate_train_folder()
        label_files = self._locate_label_files()
        test_files = self._locate_test_files()

        if train_csv:
            dataset.train_csv = build_train_csv_info(train_csv, preview_rows=self.preview_rows_train)
        else:
            dataset.train_csv.exists = False
            dataset.train_csv.path = None

        if train_folder:
            dataset.train_folder = detect_folder_type_and_stats(train_folder)
        else:
            dataset.train_folder.exists = False
            dataset.train_folder.path = None

        dataset.label_files = label_files
        dataset.test_files = test_files
        dataset.has_test_set = len(test_files) > 0

        # organization_type inference
        dataset.organization_type = self._infer_org_type(dataset)

        # modality_guess inference
        dataset.modality_guess = self._infer_modality_guess(dataset)

        return dataset

    def _infer_org_type(self, dataset: DatasetInfo) -> str:
        # Your 4 categories:
        # - single_csv: train.csv contains features + label (tabular or nlp)
        # - csv_plus_folder: train folder exists + train.csv exists (CV)
        # - folder_plus_labels: folder exists + label.txt/labels... exists (Audio variant)
        if dataset.train_csv.exists and not dataset.train_folder.exists and not dataset.label_files:
            return "single_csv"
        if dataset.train_csv.exists and dataset.train_folder.exists:
            return "csv_plus_folder"
        if dataset.train_folder.exists and dataset.label_files:
            return "folder_plus_labels"
        return "unknown"

    def _infer_modality_guess(self, dataset: DatasetInfo) -> str:
        if dataset.train_folder.exists:
            if dataset.train_folder.file_type == "image":
                return "cv"
            if dataset.train_folder.file_type == "audio":
                return "audio"
            return "multimodal"

        # no folder; rely on train.csv column dtype guesses
        if dataset.train_csv.exists:
            dtypes = [c.dtype_guess for c in dataset.train_csv.columns]
            if any(dt == "text" for dt in dtypes):
                return "nlp"
            return "tabular"
        return "unknown"

    def _infer_task_overview(self, dataset: DatasetInfo, submission_info) -> TaskOverview:
        ov = TaskOverview()
        ov.task_type = dataset.modality_guess  # align TaskType enum
        # input modalities
        mods = []
        if dataset.modality_guess in ["tabular", "nlp"]:
            mods.append("tabular")
            if dataset.modality_guess == "nlp":
                mods.append("text")
        elif dataset.modality_guess == "cv":
            mods.append("image")
            # often also tabular labels in CSV
            if dataset.train_csv.exists:
                mods.append("tabular")
        elif dataset.modality_guess == "audio":
            mods.append("audio")
            if dataset.label_files:
                mods.append("tabular")
        else:
            # best effort
            if dataset.train_csv.exists:
                mods.append("tabular")
            if dataset.train_folder.exists:
                if dataset.train_folder.file_type == "image":
                    mods.append("image")
                elif dataset.train_folder.file_type == "audio":
                    mods.append("audio")
        ov.input_modalities = sorted(list(set(mods)))

        # supervision: assume supervised if we see train csv or label files
        if dataset.train_csv.exists or dataset.label_files:
            ov.supervision = "supervised"
        else:
            ov.supervision = "unknown"

        return ov
