"""Shared data models for visual inputs, chunk plans, and budget estimates."""

from __future__ import annotations

from dataclasses import dataclass
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
    width: int | None = None
    height: int | None = None
    frame_index: int | None = None
    estimated_patch_tokens: int | None = None


@dataclass(slots=True, frozen=True)
class VisualChunk:
    """Logical grouping of visual items that will later be encoded together."""

    chunk_index: int
    items: tuple[VisualInput, ...]


@dataclass(slots=True, frozen=True)
class BudgetEstimate:
    """Represents a future runtime override for vLLM multimodal token budgeting."""

    requested_mm_tokens: int
    prompt_token_reserve: int
    total_token_budget: int
    rationale: str = ""


@dataclass(slots=True)
class EmbeddingChunk:
    """Carries one future chunk of vision embeddings back to the caller."""

    chunk_index: int
    item_ids: tuple[str, ...]
    embeddings: Any | None = None
    completed: bool = False

