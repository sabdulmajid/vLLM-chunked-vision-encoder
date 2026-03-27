"""Interfaces for asynchronous chunked vision encoding."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from .config import ChunkedVisionConfig
from .types import EmbeddingChunk, VisualChunk, VisualInput


class ChunkedVisionEncoder:
    """High-level interface for streaming multimodal embeddings into vLLM."""

    def __init__(
        self,
        config: ChunkedVisionConfig | None = None,
        *,
        vit_executor: object | None = None,
    ) -> None:
        self.config = config or ChunkedVisionConfig()
        self._vit_executor = vit_executor

    def plan_chunks(self, inputs: Sequence[VisualInput]) -> tuple[VisualChunk, ...]:
        """Return the future chunk plan for a batch of visual inputs."""

        raise NotImplementedError(
            "Chunk planning will be implemented after the repository bootstrap phase."
        )

    async def stream_embeddings(
        self,
        inputs: Sequence[VisualInput],
    ) -> AsyncIterator[EmbeddingChunk]:
        """Yield encoded chunks without blocking the main request path."""

        raise NotImplementedError(
            "Asynchronous embedding streaming will be implemented after the scaffold push."
        )

