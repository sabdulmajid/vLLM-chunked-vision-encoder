"""Serialization and aggregation helpers for precomputed multimodal embeddings."""

from __future__ import annotations

import base64
import io
from collections.abc import Sequence
from typing import Any

import torch

from .types import EncodedVisualItem, PreparedPrompt


def tensor_to_base64(tensor: torch.Tensor) -> str:
    """Serialize a PyTorch tensor into a base64 string using `torch.save`."""

    buffer = io.BytesIO()
    torch.save(tensor.detach().cpu(), buffer)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def serialize_payload(payload: dict[str, Any]) -> Any:
    """Recursively encode tensor payloads for transport over OpenAI-compatible JSON."""

    result: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, torch.Tensor):
            result[key] = tensor_to_base64(value)
        else:
            result[key] = value
    return result


def build_openai_image_embed_part(items: Sequence[EncodedVisualItem]) -> dict[str, Any]:
    """Build one aggregated `image_embeds` part for the OpenAI-compatible server."""

    aggregated = aggregate_vllm_multi_modal_data(items).get("image")
    if aggregated is None:
        raise ValueError("No image embeddings were provided.")

    if isinstance(aggregated, torch.Tensor):
        image_embeds: Any = tensor_to_base64(aggregated)
    elif isinstance(aggregated, dict):
        image_embeds = serialize_payload(aggregated)
    else:
        raise TypeError(
            "OpenAI-compatible `image_embeds` requests must serialize to a tensor or dict payload."
        )

    return {"type": "image_embeds", "image_embeds": image_embeds}


def aggregate_vllm_multi_modal_data(items: Sequence[EncodedVisualItem]) -> dict[str, Any]:
    """Aggregate per-item embedding payloads back into the offline vLLM format."""

    if not items:
        return {}

    first_payload_keys = set(items[0].payload)
    if first_payload_keys == {"image_embeds"}:
        tensors = [item.payload["image_embeds"] for item in items]
        if all(isinstance(tensor, torch.Tensor) and tensor.shape == tensors[0].shape for tensor in tensors):
            return {"image": torch.stack(tensors, dim=0)}
        return {"image": [tensor for tensor in tensors]}

    if first_payload_keys == {"image_embeds", "image_grid_thw"}:
        return {
            "image": {
                "image_embeds": torch.cat(
                    [item.payload["image_embeds"] for item in items],
                    dim=0,
                ),
                "image_grid_thw": torch.stack(
                    [item.payload["image_grid_thw"] for item in items],
                    dim=0,
                ),
            }
        }

    payload: dict[str, list[Any]] = {}
    for item in items:
        for key, value in item.payload.items():
            payload.setdefault(key, []).append(value)
    return {"image": payload}


def build_openai_messages(prepared: PreparedPrompt) -> list[dict[str, Any]]:
    """Construct OpenAI-compatible chat messages from a prepared prompt."""

    messages: list[dict[str, Any]] = []
    if prepared.system_prompt is not None:
        messages.append({"role": "system", "content": prepared.system_prompt})

    content: list[dict[str, Any]] = []
    if prepared.prompt:
        content.append({"type": "text", "text": prepared.prompt})
    if prepared.encoded_items:
        content.append(build_openai_image_embed_part(prepared.encoded_items))
    messages.append({"role": "user", "content": content})
    return messages
