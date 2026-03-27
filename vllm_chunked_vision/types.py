"""Shared data models for visual inputs, chunk plans, and benchmark results."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

VisualKind = Literal["image", "video_frame"]


@dataclass(slots=True, frozen=True)
class VisualInput:
    """Describes one image or extracted video frame destined for the VLM."""

    identifier: str
    kind: VisualKind = "image"
    source_path: Path | None = None
    source_uri: str | None = None
    bytes_data: bytes | None = None
    media_type: str | None = None
    width: int | None = None
    height: int | None = None
    frame_index: int | None = None
    estimated_patch_tokens: int | None = None
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True, frozen=True)
class VisualChunk:
    """Logical grouping of visual items that are encoded together."""

    chunk_index: int
    items: tuple[VisualInput, ...]
    planned_tokens: int


@dataclass(slots=True, frozen=True)
class EncodedVisualItem:
    """Model-specific embedding payload for one visual input."""

    identifier: str
    kind: VisualKind
    payload: dict[str, Any]
    token_count: int
    width: int | None = None
    height: int | None = None


@dataclass(slots=True, frozen=True)
class BudgetEstimate:
    """Represents a runtime override for vLLM multimodal token budgeting."""

    requested_mm_tokens: int
    prompt_token_reserve: int
    total_token_budget: int
    item_token_counts: dict[str, int]
    requested_item_counts: dict[str, int]
    rationale: str = ""


@dataclass(slots=True, frozen=True)
class BudgetPatchResult:
    """Records which engine attributes were modified by the interceptor."""

    applied_limit_per_prompt: dict[str, int]
    patched_paths: tuple[str, ...]


@dataclass(slots=True)
class EmbeddingChunk:
    """Carries one chunk of encoded vision embeddings back to the caller."""

    chunk_index: int
    item_ids: tuple[str, ...]
    items: tuple[EncodedVisualItem, ...]
    total_tokens: int
    started_at: float
    finished_at: float
    completed: bool = False


@dataclass(slots=True, frozen=True)
class PreparedPrompt:
    """Prepared multimodal request ready for vLLM or the OpenAI-compatible server."""

    prompt: str
    inputs: tuple[VisualInput, ...]
    encoded_items: tuple[EncodedVisualItem, ...]
    budget: BudgetEstimate
    patch_result: BudgetPatchResult | None = None
    system_prompt: str | None = None


@dataclass(slots=True, frozen=True)
class MetricSnapshot:
    """One sampled CPU/GPU utilization point collected during benchmarking."""

    timestamp_s: float
    cpu_percent: float
    gpu_percent: float | None = None


@dataclass(slots=True, frozen=True)
class BenchmarkRun:
    """Metrics collected for one benchmark run."""

    label: str
    ttft_ms: float
    peak_cpu_percent: float
    peak_gpu_percent: float | None
    samples: tuple[MetricSnapshot, ...]
    notes: tuple[str, ...] = ()


@dataclass(slots=True, frozen=True)
class BenchmarkComparison:
    """Comparison between the control and experimental runs."""

    control: BenchmarkRun
    experimental: BenchmarkRun
    ttft_improvement_ms: float
    ttft_improvement_ratio: float

