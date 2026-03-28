"""Qwen2.5-VL visual-only backend for generating `image_embeds` externally."""

from __future__ import annotations

import asyncio
import os
from io import BytesIO
from pathlib import Path
from threading import Lock

import torch
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors import safe_open
from transformers import AutoConfig, AutoProcessor
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
)

from .types import EncodedVisualItem, VisualChunk


class Qwen2_5_VLVisualBackend:
    """Runs the Qwen2.5-VL vision tower outside vLLM and returns image embeddings."""

    def __init__(
        self,
        *,
        model_id: str = "Qwen/Qwen2.5-VL-72B-Instruct",
        cache_dir: str | Path | None = None,
        device: str = "cuda:0",
        dtype: torch.dtype = torch.bfloat16,
        use_fast_processor: bool = False,
        local_files_only: bool | None = None,
    ) -> None:
        self.model_id = model_id
        self.cache_dir = str(cache_dir) if cache_dir is not None else None
        self.device = device
        self.dtype = dtype
        self.use_fast_processor = use_fast_processor
        self.local_files_only = local_files_only
        self._lock = Lock()
        self._processor = None
        self._visual: Qwen2_5_VisionTransformerPretrainedModel | None = None

    def _should_use_local_files_only(self) -> bool:
        if self.local_files_only is not None:
            return self.local_files_only
        offline_values = {"1", "on", "true", "yes"}
        return (
            os.getenv("HF_HUB_OFFLINE", "").strip().lower() in offline_values
            or os.getenv("TRANSFORMERS_OFFLINE", "").strip().lower() in offline_values
        )

    def _resolve_visual_shard(self) -> str:
        return hf_hub_download(
            repo_id=self.model_id,
            filename="model-00001-of-00038.safetensors",
            cache_dir=self.cache_dir,
            local_files_only=self._should_use_local_files_only(),
        )

    def _ensure_loaded(self) -> None:
        if self._visual is not None:
            return
        with self._lock:
            if self._visual is not None:
                return

            local_files_only = self._should_use_local_files_only()
            config = AutoConfig.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                local_files_only=local_files_only,
            )
            processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=self.cache_dir,
                use_fast=self.use_fast_processor,
                local_files_only=local_files_only,
            )
            visual = Qwen2_5_VisionTransformerPretrainedModel(config.vision_config)
            shard_path = self._resolve_visual_shard()
            state_dict: dict[str, torch.Tensor] = {}
            with safe_open(shard_path, framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    if key.startswith("visual."):
                        state_dict[key.removeprefix("visual.")] = handle.get_tensor(key)
            visual.load_state_dict(state_dict, strict=True)
            visual.eval()
            visual.to(device=self.device, dtype=self.dtype)
            self._processor = processor
            self._visual = visual

    def _load_images(self, chunk: VisualChunk) -> list[Image.Image]:
        images: list[Image.Image] = []
        for item in chunk.items:
            if item.source_path is not None:
                with Image.open(item.source_path) as image:
                    images.append(image.convert("RGB"))
                continue
            if item.bytes_data is not None:
                with Image.open(BytesIO(item.bytes_data)) as image:
                    images.append(image.convert("RGB"))
                continue
            raise ValueError(
                f"Visual input {item.identifier!r} does not have a readable image source."
            )
        return images

    def _encode_chunk_sync(self, chunk: VisualChunk) -> tuple[EncodedVisualItem, ...]:
        self._ensure_loaded()
        assert self._visual is not None
        assert self._processor is not None

        images = self._load_images(chunk)
        processor_outputs = self._processor.image_processor(images=images, return_tensors="pt")
        pixel_values = processor_outputs["pixel_values"].to(device=self.device, dtype=self.dtype)
        image_grid_thw = processor_outputs["image_grid_thw"].to(device=self.device)

        with torch.inference_mode():
            image_embeds = self._visual(pixel_values, grid_thw=image_grid_thw)
        split_sizes = (
            image_grid_thw.prod(-1) // self._visual.spatial_merge_size**2
        ).tolist()
        split_embeds = image_embeds.split(split_sizes)

        encoded_items: list[EncodedVisualItem] = []
        for item, embeds, grid in zip(chunk.items, split_embeds, image_grid_thw):
            encoded_items.append(
                EncodedVisualItem(
                    identifier=item.identifier,
                    kind=item.kind,
                    payload={
                        "image_embeds": embeds.detach().cpu(),
                        "image_grid_thw": grid.detach().cpu(),
                    },
                    token_count=int(embeds.shape[0]),
                    width=item.width,
                    height=item.height,
                )
            )
        return tuple(encoded_items)

    async def encode_chunk(self, chunk: VisualChunk) -> tuple[EncodedVisualItem, ...]:
        return await asyncio.to_thread(self._encode_chunk_sync, chunk)
