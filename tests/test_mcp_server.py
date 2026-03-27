from __future__ import annotations

import asyncio
from pathlib import Path

from PIL import Image

from vllm_chunked_vision import ChunkedVisionMCPServer


def test_mcp_server_tools_work_without_sdk(tmp_path: Path) -> None:
    image_path = tmp_path / "example.png"
    Image.new("RGB", (320, 240), color=(255, 255, 255)).save(image_path)

    server = ChunkedVisionMCPServer()
    description = server.describe()
    assert description["service"] == "vllm-chunked-vision-encoder"

    budget = asyncio.run(server.estimate_budget([str(image_path)], prompt_token_reserve=256))
    assert budget["requested_item_counts"]["image"] == 1

    payload = asyncio.run(
        server.encode_images(
            [str(image_path)],
            prompt="describe the image",
            prompt_token_reserve=256,
        )
    )
    assert payload["messages"][0]["content"][1]["type"] == "image_embeds"
