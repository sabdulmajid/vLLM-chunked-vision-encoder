"""Runtime multimodal token budget calculation and reflective patching."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from .config import BudgetOverrideConfig
from .media import hydrate_visual_inputs
from .types import BudgetEstimate, BudgetPatchResult, VisualInput


def _resolve_attr_path(root: object, path: str) -> tuple[object, str] | None:
    current = root
    segments = path.split(".")
    for segment in segments[:-1]:
        if not hasattr(current, segment):
            return None
        current = getattr(current, segment)
    if not hasattr(current, segments[-1]):
        return None
    return current, segments[-1]


class DynamicBudgetInterceptor:
    """Coordinates runtime budget estimation and non-invasive vLLM overrides."""

    def __init__(self, config: BudgetOverrideConfig | None = None) -> None:
        self.config = config or BudgetOverrideConfig()
        self._installed_engine: object | None = None

    def estimate_item_tokens(self, item: VisualInput) -> int:
        """Estimate the number of visual tokens contributed by one input item."""

        if item.estimated_patch_tokens is not None:
            return item.estimated_patch_tokens

        default_size = (
            self.config.default_video_frame_size
            if item.kind == "video_frame"
            else self.config.default_image_size
        )
        width = item.width or default_size[0]
        height = item.height or default_size[1]
        patch_size = max(1, self.config.patch_size)
        spatial_merge = max(1, self.config.spatial_merge_size)
        temporal_patch = max(1, self.config.temporal_patch_size)

        grid_h = math.ceil(height / patch_size)
        grid_w = math.ceil(width / patch_size)
        merged_h = math.ceil(grid_h / spatial_merge)
        merged_w = math.ceil(grid_w / spatial_merge)
        grid_t = 1 if item.kind == "image" else temporal_patch
        return max(1, merged_h * merged_w * grid_t)

    def hydrate(self, inputs: Sequence[VisualInput]) -> tuple[VisualInput, ...]:
        """Fill missing input metadata and token estimates."""

        hydrated = hydrate_visual_inputs(inputs, load_metadata=True)
        return tuple(
            replace(item, estimated_patch_tokens=self.estimate_item_tokens(item))
            if item.estimated_patch_tokens is None
            else item
            for item in hydrated
        )

    def estimate_budget(
        self,
        inputs: Sequence[VisualInput],
        *,
        prompt_token_reserve: int = 0,
    ) -> BudgetEstimate:
        """Compute the multimodal token budget needed for the given visual inputs."""

        hydrated_inputs = self.hydrate(inputs)
        item_token_counts = {
            item.identifier: item.estimated_patch_tokens or 0 for item in hydrated_inputs
        }
        requested_item_counts = {
            "image": sum(1 for item in hydrated_inputs if item.kind == "image"),
            "video": sum(1 for item in hydrated_inputs if item.kind == "video_frame"),
        }
        requested_mm_tokens = sum(item_token_counts.values())
        total_token_budget = requested_mm_tokens + prompt_token_reserve + self.config.extra_token_margin
        if self.config.max_override_tokens is not None:
            total_token_budget = min(total_token_budget, self.config.max_override_tokens)
        rationale = (
            "Calculated from hydrated visual dimensions, patch size, spatial merge size, "
            "and a fixed safety margin so vLLM does not reserve worst-case multimodal tokens."
        )
        return BudgetEstimate(
            requested_mm_tokens=requested_mm_tokens,
            prompt_token_reserve=prompt_token_reserve,
            total_token_budget=total_token_budget,
            item_token_counts=item_token_counts,
            requested_item_counts=requested_item_counts,
            rationale=rationale,
        )

    def install(self, engine: object) -> None:
        """Register a local engine for later per-request budget overrides."""

        self._installed_engine = engine
        setattr(engine, "_chunked_vision_budget_interceptor", self)

    def apply_budget(
        self,
        inputs: Sequence[VisualInput],
        *,
        engine: object | None = None,
    ) -> BudgetPatchResult:
        """Patch common `limit_per_prompt` locations on a local engine object."""

        target_engine = engine or self._installed_engine
        if target_engine is None:
            return BudgetPatchResult(applied_limit_per_prompt={"image": 0, "video": 0}, patched_paths=())

        hydrated_inputs = self.hydrate(inputs)
        image_count = sum(1 for item in hydrated_inputs if item.kind == "image")
        video_count = sum(1 for item in hydrated_inputs if item.kind == "video_frame")
        override = {"image": image_count, "video": video_count}

        patched_paths: list[str] = []
        for path in self.config.engine_limit_attr_candidates:
            resolved = _resolve_attr_path(target_engine, path)
            if resolved is None:
                continue
            owner, attribute = resolved
            current_value = getattr(owner, attribute)
            if not isinstance(current_value, dict):
                current_value = {}
            updated = dict(current_value)
            updated.update(override)
            setattr(owner, attribute, updated)
            patched_paths.append(path)

        return BudgetPatchResult(applied_limit_per_prompt=override, patched_paths=tuple(patched_paths))

    def recommended_server_flags(self) -> dict[str, Any]:
        """Return the recommended startup flags for the proxy-based embeddings path."""

        return {
            "enable_mm_embeds": True,
            "limit_mm_per_prompt": {"image": 0, "video": 0},
        }

