from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Literal


TaskType = Literal["tabular", "nlp", "cv", "audio", "multimodal", "unknown"]
OutputType = Literal["binary", "multiclass", "multilabel", "regression", "unknown"]
OrgType = Literal["single_csv", "csv_plus_folder", "folder_plus_labels", "unknown"]
DTypeGuess = Literal["numeric", "text", "categorical", "unknown"]


@dataclass
class BackgroundInfo:
    raw_text: str = ""
    source_path: Optional[str] = None  # 从哪个文件抽取（description.md）


@dataclass
class DataDescriptionInfo:
    raw_text: str = ""
    source_path: Optional[str] = None


@dataclass
class MetaInfo:
    workspace_root: str
    generated_at: str  # ISO string
    generator_version: str = "data_profile_v1"


@dataclass
class EvaluationInfo:
    raw_text: str = ""
    metrics: list[str] = field(default_factory=list)
    ranking_rule: Optional[str] = None
    notes: Optional[str] = None


@dataclass
class SubmissionColumn:
    name: str
    index: int


@dataclass
class SubmissionInfo:
    file_name: str = "submission.csv"
    format_description: str = ""          # from description.md
    columns: list[SubmissionColumn] = field(default_factory=list)
    num_columns: int = 0
    has_id_column: bool = False
    label_column_guess: Optional[str] = None


@dataclass
class CSVColumnInfo:
    name: str
    dtype_guess: DTypeGuess = "unknown"


@dataclass
class TrainCSVInfo:
    exists: bool = False
    path: Optional[str] = None
    num_rows_previewed: int = 0
    columns: list[CSVColumnInfo] = field(default_factory=list)
    label_column_guess: Optional[str] = None
    sample_rows: list[dict[str, str]] = field(default_factory=list)  # stringify values


@dataclass
class DataFolderInfo:
    exists: bool = False
    path: Optional[str] = None
    file_type: Literal["image", "audio", "unknown"] = "unknown"
    num_files: Optional[int] = None
    file_extensions: list[str] = field(default_factory=list)


@dataclass
class DatasetInfo:
    organization_type: OrgType = "unknown"
    modality_guess: TaskType = "unknown"
    train_csv: TrainCSVInfo = field(default_factory=TrainCSVInfo)
    train_folder: DataFolderInfo = field(default_factory=DataFolderInfo)
    label_files: list[str] = field(default_factory=list)   # e.g., label.txt, labels.csv
    has_test_set: bool = False
    test_files: list[str] = field(default_factory=list)


@dataclass
class TaskOverview:
    task_type: TaskType = "unknown"
    supervision: Literal["supervised", "unsupervised", "unknown"] = "unknown"
    input_modalities: list[str] = field(default_factory=list)  # ["tabular","text","image","audio"]


@dataclass
class FilesEvidence:
    directory_tree_markdown: str = ""
    previews: dict[str, str] = field(default_factory=dict)  # abs_path -> preview text


@dataclass
class DataProfile:
    meta: MetaInfo
    task_overview: TaskOverview = field(default_factory=TaskOverview)
    background: BackgroundInfo = field(default_factory=BackgroundInfo)
    data_description: DataDescriptionInfo = field(default_factory=DataDescriptionInfo)
    evaluation: EvaluationInfo = field(default_factory=EvaluationInfo)
    submission: SubmissionInfo = field(default_factory=SubmissionInfo)
    dataset: DatasetInfo = field(default_factory=DatasetInfo)
    files: FilesEvidence = field(default_factory=FilesEvidence)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # dataclass nested lists of dataclasses become dicts already via asdict()
        return d
