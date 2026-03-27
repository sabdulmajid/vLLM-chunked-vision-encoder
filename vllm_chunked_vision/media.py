"""Utilities for hydrating visual metadata from image paths or inline bytes."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from io import BytesIO

from PIL import Image

from .types import VisualInput


def _infer_size(item: VisualInput) -> tuple[int, int] | None:
    if item.width is not None and item.height is not None:
        return (item.width, item.height)
    if item.source_path is not None and item.source_path.exists():
        with Image.open(item.source_path) as image:
            return image.size
    if item.bytes_data is not None:
        with Image.open(BytesIO(item.bytes_data)) as image:
            return image.size
    return None


def hydrate_visual_inputs(
    inputs: Sequence[VisualInput],
    *,
    load_metadata: bool,
) -> tuple[VisualInput, ...]:
    """Populate missing image dimensions from disk or in-memory bytes."""

    hydrated: list[VisualInput] = []
    for item in inputs:
        if not load_metadata:
            hydrated.append(item)
            continue
        size = _infer_size(item)
        if size is None:
            hydrated.append(item)
            continue
        hydrated.append(replace(item, width=size[0], height=size[1]))
    return tuple(hydrated)

