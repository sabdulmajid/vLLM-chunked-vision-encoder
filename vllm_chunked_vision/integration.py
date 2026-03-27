"""High-level wrappers that prepare embeddings and call vLLM."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from .backends import VisionEncoderBackend
from .budgeting import DynamicBudgetInterceptor
from .config import RuntimeConfig
from .encoder import ChunkedVisionEncoder
from .serialize import aggregate_vllm_multi_modal_data, build_openai_messages
from .types import PreparedPrompt, VisualInput


class VLLMChunkedVisionProxy:
    """Wrap a local vLLM engine and feed it precomputed multimodal embeddings."""

    def __init__(
        self,
        engine: object | None = None,
        *,
        runtime: RuntimeConfig | None = None,
        encoder: ChunkedVisionEncoder | None = None,
        backend: VisionEncoderBackend | None = None,
        interceptor: DynamicBudgetInterceptor | None = None,
    ) -> None:
        self.runtime = runtime or RuntimeConfig()
        self.encoder = encoder or ChunkedVisionEncoder(self.runtime.encoder, backend=backend)
        self.interceptor = interceptor or DynamicBudgetInterceptor(self.runtime.budget)
        self.engine = engine
        if self.engine is not None:
            self.interceptor.install(self.engine)

    async def prepare_prompt(
        self,
        prompt: str,
        inputs: Sequence[VisualInput],
        *,
        prompt_token_reserve: int = 0,
        system_prompt: str | None = None,
    ) -> PreparedPrompt:
        """Encode visual inputs and compute the matching runtime token budget."""

        hydrated_inputs = self.interceptor.hydrate(inputs)
        encoded_items = await self.encoder.materialize(hydrated_inputs)
        order = {item.identifier: index for index, item in enumerate(hydrated_inputs)}
        ordered_items = tuple(sorted(encoded_items, key=lambda item: order[item.identifier]))
        budget = self.interceptor.estimate_budget(
            hydrated_inputs,
            prompt_token_reserve=prompt_token_reserve,
        )
        patch_result = self.interceptor.apply_budget(hydrated_inputs, engine=self.engine)
        return PreparedPrompt(
            prompt=prompt,
            system_prompt=system_prompt,
            inputs=tuple(hydrated_inputs),
            encoded_items=ordered_items,
            budget=budget,
            patch_result=patch_result,
        )

    async def agenerate(
        self,
        prompt: str,
        inputs: Sequence[VisualInput],
        *,
        sampling_params: object | None = None,
        prompt_token_reserve: int = 0,
        system_prompt: str | None = None,
        **generate_kwargs: Any,
    ) -> Any:
        """Prepare the request and invoke `engine.generate` in a worker thread."""

        if self.engine is None:
            raise RuntimeError("No local vLLM engine was provided to the proxy.")

        prepared = await self.prepare_prompt(
            prompt,
            inputs,
            prompt_token_reserve=prompt_token_reserve,
            system_prompt=system_prompt,
        )
        payload = {
            "prompt": prepared.prompt,
            "multi_modal_data": aggregate_vllm_multi_modal_data(prepared.encoded_items),
        }
        return await asyncio.to_thread(
            self.engine.generate,
            payload,
            sampling_params,
            **generate_kwargs,
        )

    def generate(
        self,
        prompt: str,
        inputs: Sequence[VisualInput],
        *,
        sampling_params: object | None = None,
        prompt_token_reserve: int = 0,
        system_prompt: str | None = None,
        **generate_kwargs: Any,
    ) -> Any:
        """Synchronous wrapper around `agenerate` for standard `LLM.generate` flows."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(
                self.agenerate(
                    prompt,
                    inputs,
                    sampling_params=sampling_params,
                    prompt_token_reserve=prompt_token_reserve,
                    system_prompt=system_prompt,
                    **generate_kwargs,
                )
            )
        raise RuntimeError(
            "An event loop is already running. Use `await proxy.agenerate(...)` instead."
        )

    async def prepare_openai_messages(
        self,
        prompt: str,
        inputs: Sequence[VisualInput],
        *,
        prompt_token_reserve: int = 0,
        system_prompt: str | None = None,
    ) -> list[dict[str, Any]]:
        """Encode inputs and return OpenAI-compatible `image_embeds` messages."""

        prepared = await self.prepare_prompt(
            prompt,
            inputs,
            prompt_token_reserve=prompt_token_reserve,
            system_prompt=system_prompt,
        )
        return build_openai_messages(prepared)
