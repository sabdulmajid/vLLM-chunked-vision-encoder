"""Configuration objects for the chunked vision encoder runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True, frozen=True)
class ChunkedVisionConfig:
    """Controls how visual inputs will eventually be chunked for asynchronous encoding."""

    max_images_per_chunk: int = 4
    max_video_frames_per_chunk: int = 16
    device: str = "cuda"
    dtype: str = "bfloat16"


@dataclass(slots=True, frozen=True)
class BudgetOverrideConfig:
    """Toggles and safety margins for runtime multimodal token budget overrides."""

    enable_registry_patch: bool = True
    extra_token_margin: int = 256
    max_override_tokens: int | None = None


@dataclass(slots=True, frozen=True)
class MCPServerConfig:
    """Configuration for the future MCP transport layer."""

    host: str = "127.0.0.1"
    port: int = 8765
    transport: Literal["stdio", "sse"] = "stdio"


@dataclass(slots=True, frozen=True)
class RuntimeConfig:
    """Aggregates the public configuration surfaces for the library."""

    encoder: ChunkedVisionConfig = field(default_factory=ChunkedVisionConfig)
    budget: BudgetOverrideConfig = field(default_factory=BudgetOverrideConfig)
    mcp: MCPServerConfig = field(default_factory=MCPServerConfig)

