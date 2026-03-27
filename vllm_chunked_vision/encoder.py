"""Asynchronous chunk planning and embedding streaming."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from time import perf_counter

from .backends import DeterministicVisionBackend, VisionEncoderBackend
from .config import ChunkedVisionConfig
from .media import hydrate_visual_inputs
from .types import EmbeddingChunk, EncodedVisualItem, VisualChunk, VisualInput


class ChunkedVisionEncoder:
    """High-level interface for streaming multimodal embeddings into vLLM."""

    def __init__(
        self,
        config: ChunkedVisionConfig | None = None,
        *,
        backend: VisionEncoderBackend | None = None,
    ) -> None:
        self.config = config or ChunkedVisionConfig()
        self._backend = backend or DeterministicVisionBackend()

    def plan_chunks(self, inputs: Sequence[VisualInput]) -> tuple[VisualChunk, ...]:
        """Build chunk boundaries that cap image/frame count and token pressure."""

        hydrated_inputs = hydrate_visual_inputs(inputs, load_metadata=self.config.load_image_metadata)
        chunks: list[VisualChunk] = []
        current_items: list[VisualInput] = []
        current_tokens = 0

        for item in hydrated_inputs:
            item_tokens = item.estimated_patch_tokens or 0
            max_items = (
                self.config.max_video_frames_per_chunk
                if item.kind == "video_frame"
                else self.config.max_images_per_chunk
            )
            would_exceed_items = len(current_items) >= max_items
            would_exceed_tokens = (
                self.config.max_tokens_per_chunk is not None
                and current_items
                and current_tokens + item_tokens > self.config.max_tokens_per_chunk
            )
            if would_exceed_items or would_exceed_tokens:
                chunks.append(
                    VisualChunk(
                        chunk_index=len(chunks),
                        items=tuple(current_items),
                        planned_tokens=current_tokens,
                    )
                )
                current_items = []
                current_tokens = 0

            if item.estimated_patch_tokens is None:
                item = replace(item, estimated_patch_tokens=item_tokens or 1)
            current_items.append(item)
            current_tokens += item.estimated_patch_tokens or 0

        if current_items:
            chunks.append(
                VisualChunk(
                    chunk_index=len(chunks),
                    items=tuple(current_items),
                    planned_tokens=current_tokens,
                )
            )

        return tuple(chunks)

    async def stream_embeddings(
        self,
        inputs: Sequence[VisualInput],
    ) -> AsyncIterator[EmbeddingChunk]:
        """Yield encoded chunks as soon as they complete."""

        chunks = self.plan_chunks(inputs)
        if not chunks:
            return

        semaphore = asyncio.Semaphore(self.config.max_concurrent_chunks)
        queue: asyncio.Queue[EmbeddingChunk] = asyncio.Queue()

        async def run_chunk(chunk: VisualChunk) -> None:
            async with semaphore:
                started_at = perf_counter()
                items = await self._backend.encode_chunk(chunk)
                finished_at = perf_counter()
                total_tokens = sum(item.token_count for item in items)
                await queue.put(
                    EmbeddingChunk(
                        chunk_index=chunk.chunk_index,
                        item_ids=tuple(item.identifier for item in items),
                        items=tuple(items),
                        total_tokens=total_tokens,
                        started_at=started_at,
                        finished_at=finished_at,
                        completed=True,
                    )
                )

        tasks = [asyncio.create_task(run_chunk(chunk)) for chunk in chunks]
        emitted = 0
        buffered: dict[int, EmbeddingChunk] = {}
        next_index = 0
        try:
            while emitted < len(chunks):
                chunk = await queue.get()
                if not self.config.preserve_input_order:
                    emitted += 1
                    yield chunk
                    continue
                buffered[chunk.chunk_index] = chunk
                while next_index in buffered:
                    emitted += 1
                    yield buffered.pop(next_index)
                    next_index += 1
        finally:
            await asyncio.gather(*tasks)

    async def materialize(self, inputs: Sequence[VisualInput]) -> tuple[EncodedVisualItem, ...]:
        """Collect every embedding chunk and flatten them back into input order."""

        items: list[EncodedVisualItem] = []
        async for chunk in self.stream_embeddings(inputs):
            items.extend(chunk.items)
        return tuple(items)

