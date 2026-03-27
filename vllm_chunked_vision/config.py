"""Configuration objects for the chunked vision encoder runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass(slots=True, frozen=True)
class ChunkedVisionConfig:
    """Controls chunk planning and concurrent vision encoding."""

    max_images_per_chunk: int = 4
    max_video_frames_per_chunk: int = 16
    max_tokens_per_chunk: int | None = 8192
    max_concurrent_chunks: int = 2
    preserve_input_order: bool = True
    load_image_metadata: bool = True
    device: str = "cuda"
    dtype: str = "bfloat16"


@dataclass(slots=True, frozen=True)
class BudgetOverrideConfig:
    """Safety margins and heuristics for runtime multimodal token budgeting."""

    enable_registry_patch: bool = True
    patch_size: int = 14
    spatial_merge_size: int = 2
    temporal_patch_size: int = 2
    default_image_size: tuple[int, int] = (1024, 1024)
    default_video_frame_size: tuple[int, int] = (1024, 1024)
    extra_token_margin: int = 256
    max_override_tokens: int | None = None
    engine_limit_attr_candidates: tuple[str, ...] = (
        "model_config.multimodal_config.limit_per_prompt",
        "model_config.mm_config.limit_per_prompt",
        "llm_engine.model_config.multimodal_config.limit_per_prompt",
        "llm_engine.model_config.mm_config.limit_per_prompt",
        "engine.model_config.multimodal_config.limit_per_prompt",
        "engine.model_config.mm_config.limit_per_prompt",
    )


@dataclass(slots=True, frozen=True)
class MCPServerConfig:
    """Configuration for the MCP transport layer."""

    host: str = "127.0.0.1"
    port: int = 8765
    mount_path: str = "/mcp"
    transport: Literal["stdio", "sse", "streamable-http"] = "stdio"


@dataclass(slots=True, frozen=True)
class RuntimeConfig:
    """Aggregates the public configuration surfaces for the library."""

    encoder: ChunkedVisionConfig = field(default_factory=ChunkedVisionConfig)
    budget: BudgetOverrideConfig = field(default_factory=BudgetOverrideConfig)
    mcp: MCPServerConfig = field(default_factory=MCPServerConfig)

