from __future__ import annotations

import asyncio

import torch

from vllm_chunked_vision import VLLMChunkedVisionProxy, VisualInput, build_openai_messages
from vllm_chunked_vision.types import BudgetEstimate, EncodedVisualItem, PreparedPrompt


def test_build_openai_messages_uses_image_embeds() -> None:
    prepared = PreparedPrompt(
        prompt="reason about the image",
        inputs=(VisualInput(identifier="img-0", width=256, height=256),),
        encoded_items=(
            EncodedVisualItem(
                identifier="img-0",
                kind="image",
                token_count=4,
                payload={
                    "image_embeds": torch.randn(4, 8),
                    "image_grid_thw": torch.tensor([1, 2, 2]),
                },
            ),
        ),
        budget=BudgetEstimate(
            requested_mm_tokens=4,
            prompt_token_reserve=128,
            total_token_budget=388,
            item_token_counts={"img-0": 4},
            requested_item_counts={"image": 1, "video": 0},
        ),
    )

    messages = build_openai_messages(prepared)
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][1]["type"] == "image_embeds"
    assert isinstance(messages[0]["content"][1]["image_embeds"], dict)


def test_proxy_prepares_openai_messages() -> None:
    proxy = VLLMChunkedVisionProxy()
    inputs = [VisualInput(identifier="img-0", width=256, height=256)]

    messages = asyncio.run(proxy.prepare_openai_messages("hello", inputs))
    assert messages[0]["role"] == "user"
    assert messages[0]["content"][0]["type"] == "text"
    assert messages[0]["content"][1]["type"] == "image_embeds"

