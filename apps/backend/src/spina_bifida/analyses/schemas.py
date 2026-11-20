from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class NewFrame:
    frame_id: str
    path: Path
    checksum: str
    position: int


@dataclass(frozen=True, slots=True)
class NewAnalysisJob:
    job_id: str
    batch_id: str
    batch_position: int
    model_version: Literal["v1", "v2"]
    frame: NewFrame


@dataclass(frozen=True, slots=True)
class QueuedAnalysis:
    id: str
    state: str
    frames: tuple[str, ...]
    polling_url: str
