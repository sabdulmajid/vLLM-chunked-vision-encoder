from __future__ import annotations

from dataclasses import dataclass

from vllm_chunked_vision import DynamicBudgetInterceptor, VisualInput


@dataclass
class _MMConfig:
    limit_per_prompt: dict[str, int]


@dataclass
class _ModelConfig:
    multimodal_config: _MMConfig


@dataclass
class _Engine:
    model_config: _ModelConfig


def test_budget_estimate_and_reflective_patch() -> None:
    interceptor = DynamicBudgetInterceptor()
    inputs = [
        VisualInput(identifier="a", width=1024, height=1024),
        VisualInput(identifier="b", width=512, height=512),
    ]

    budget = interceptor.estimate_budget(inputs, prompt_token_reserve=2048)
    assert budget.requested_mm_tokens > 0
    assert budget.total_token_budget > budget.requested_mm_tokens
    assert budget.requested_item_counts["image"] == 2

    engine = _Engine(model_config=_ModelConfig(multimodal_config=_MMConfig(limit_per_prompt={})))
    interceptor.install(engine)
    patch_result = interceptor.apply_budget(inputs)
    assert patch_result.applied_limit_per_prompt["image"] == 2
    assert "model_config.multimodal_config.limit_per_prompt" in patch_result.patched_paths
    assert engine.model_config.multimodal_config.limit_per_prompt["image"] == 2

