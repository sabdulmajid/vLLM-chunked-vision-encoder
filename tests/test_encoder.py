from __future__ import annotations

import asyncio

from vllm_chunked_vision import ChunkedVisionConfig, ChunkedVisionEncoder, VisualInput
from vllm_chunked_vision.backends import DeterministicVisionBackend


def test_chunk_planning_respects_limits() -> None:
    encoder = ChunkedVisionEncoder(
        ChunkedVisionConfig(max_images_per_chunk=3, max_concurrent_chunks=2),
        backend=DeterministicVisionBackend(encode_delay_s=0.0),
    )
    inputs = [VisualInput(identifier=str(index), width=1024, height=1024) for index in range(7)]
    chunks = encoder.plan_chunks(inputs)
    assert len(chunks) == 3
    assert [len(chunk.items) for chunk in chunks] == [3, 3, 1]


def test_stream_embeddings_preserves_order() -> None:
    encoder = ChunkedVisionEncoder(
        ChunkedVisionConfig(max_images_per_chunk=2, max_concurrent_chunks=2, preserve_input_order=True),
        backend=DeterministicVisionBackend(encode_delay_s=0.01),
    )
    inputs = [VisualInput(identifier=str(index), width=1024, height=1024) for index in range(5)]

    async def collect() -> list[str]:
        item_ids: list[str] = []
        async for chunk in encoder.stream_embeddings(inputs):
            item_ids.extend(chunk.item_ids)
        return item_ids

    assert asyncio.run(collect()) == ["0", "1", "2", "3", "4"]

