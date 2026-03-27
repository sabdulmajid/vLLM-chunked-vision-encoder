"""Interfaces for runtime multimodal token budget calculation and patching."""

from __future__ import annotations

from collections.abc import Sequence

from .config import BudgetOverrideConfig
from .types import BudgetEstimate, VisualInput


class DynamicBudgetInterceptor:
    """Coordinates future runtime budget estimation and vLLM override hooks."""

    def __init__(self, config: BudgetOverrideConfig | None = None) -> None:
        self.config = config or BudgetOverrideConfig()

    def estimate_budget(
        self,
        inputs: Sequence[VisualInput],
        *,
        prompt_token_reserve: int = 0,
    ) -> BudgetEstimate:
        """Compute a future multimodal token budget for the given visual inputs."""

        raise NotImplementedError(
            "Dynamic budget estimation will be implemented after the scaffold push."
        )

    def install(self, engine: object) -> None:
        """Install future monkey patches or proxy hooks against a vLLM engine."""

        raise NotImplementedError(
            "Runtime vLLM integration will be implemented after the scaffold push."
        )

