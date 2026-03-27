"""Backend interfaces and a deterministic fallback encoder."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import math
from collections.abc import Awaitable, Callable, Sequence
from typing import Protocol, runtime_checkable

import torch

from .types import EncodedVisualItem, VisualChunk, VisualInput


@runtime_checkable
class VisionEncoderBackend(Protocol):
    """Backend that can encode one `VisualChunk` into model-specific embeddings."""

    async def encode_chunk(self, chunk: VisualChunk) -> tuple[EncodedVisualItem, ...]:
        """Encode the provided visual chunk."""


class CallableVisionEncoderBackend:
    """Adapter for user-supplied sync or async encoding callables."""

    def __init__(
        self,
        encoder: Callable[
            [VisualChunk],
            Sequence[EncodedVisualItem] | Awaitable[Sequence[EncodedVisualItem]],
        ],
    ) -> None:
        self._encoder = encoder

    async def encode_chunk(self, chunk: VisualChunk) -> tuple[EncodedVisualItem, ...]:
        result = self._encoder(chunk)
        if inspect.isawaitable(result):
            resolved = await result
        else:
            resolved = await asyncio.to_thread(lambda: tuple(result))
        return tuple(resolved)


class DeterministicVisionBackend:
    """Synthetic backend that emits stable tensors for tests, demos, and dry-runs."""

    def __init__(
        self,
        *,
        hidden_size: int = 3584,
        patch_size: int = 14,
        spatial_merge_size: int = 2,
        dtype: torch.dtype = torch.float32,
        encode_delay_s: float = 0.03,
        qwen_style: bool = True,
    ) -> None:
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.dtype = dtype
        self.encode_delay_s = encode_delay_s
        self.qwen_style = qwen_style

    def _seed_for(self, item: VisualInput) -> int:
        digest = hashlib.sha256(item.identifier.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="little", signed=False)

    def _grid(self, item: VisualInput) -> tuple[int, int, int]:
        width = item.width or 1024
        height = item.height or 1024
        grid_h = math.ceil(height / self.patch_size / self.spatial_merge_size)
        grid_w = math.ceil(width / self.patch_size / self.spatial_merge_size)
        grid_t = 1 if item.kind == "image" else 2
        return (grid_t, max(1, grid_h), max(1, grid_w))

    async def encode_chunk(self, chunk: VisualChunk) -> tuple[EncodedVisualItem, ...]:
        if self.encode_delay_s > 0:
            await asyncio.sleep(self.encode_delay_s)
        items: list[EncodedVisualItem] = []
        for item in chunk.items:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self._seed_for(item))
            token_count = item.estimated_patch_tokens or max(1, self._grid(item)[1] * self._grid(item)[2])
            image_embeds = torch.randn(
                (token_count, self.hidden_size),
                generator=generator,
                dtype=self.dtype,
            )
            payload: dict[str, torch.Tensor] = {"image_embeds": image_embeds}
            if self.qwen_style:
                payload["image_grid_thw"] = torch.tensor(self._grid(item), dtype=torch.int64)
            items.append(
                EncodedVisualItem(
                    identifier=item.identifier,
                    kind=item.kind,
                    payload=payload,
                    token_count=token_count,
                    width=item.width,
                    height=item.height,
                )
            )
        return tuple(items)
