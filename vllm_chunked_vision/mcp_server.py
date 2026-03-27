"""MCP gateway exposing chunked vision ingestion and benchmark helpers."""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from .benchmarking import run_benchmark
from .budgeting import DynamicBudgetInterceptor
from .config import RuntimeConfig
from .encoder import ChunkedVisionEncoder
from .integration import VLLMChunkedVisionProxy
from .types import VisualInput

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - exercised through fallback behavior.
    FastMCP = None  # type: ignore[assignment]


class ChunkedVisionMCPServer:
    """Owns MCP tool registration for chunked vision ingestion workflows."""

    def __init__(
        self,
        runtime: RuntimeConfig | None = None,
        *,
        encoder: ChunkedVisionEncoder | None = None,
        interceptor: DynamicBudgetInterceptor | None = None,
    ) -> None:
        self.runtime = runtime or RuntimeConfig()
        self.encoder = encoder or ChunkedVisionEncoder(self.runtime.encoder)
        self.interceptor = interceptor or DynamicBudgetInterceptor(self.runtime.budget)
        self.proxy = VLLMChunkedVisionProxy(
            engine=None,
            runtime=self.runtime,
            encoder=self.encoder,
            interceptor=self.interceptor,
        )
        self._mcp = self._build_fastmcp() if FastMCP is not None else None

    def describe(self) -> dict[str, object]:
        """Return minimal process metadata for the MCP service."""

        return {
            "service": "vllm-chunked-vision-encoder",
            "status": "ready" if self._mcp is not None else "degraded-no-mcp-sdk",
            "transport": self.runtime.mcp.transport,
            "host": self.runtime.mcp.host,
            "port": self.runtime.mcp.port,
            "mount_path": self.runtime.mcp.mount_path,
        }

    async def estimate_budget(
        self,
        image_paths: list[str],
        *,
        prompt_token_reserve: int = 0,
    ) -> dict[str, Any]:
        """Return the computed budget for the provided image paths."""

        inputs = [VisualInput(identifier=path, source_path=Path(path)) for path in image_paths]
        budget = self.interceptor.estimate_budget(inputs, prompt_token_reserve=prompt_token_reserve)
        return {
            "requested_mm_tokens": budget.requested_mm_tokens,
            "prompt_token_reserve": budget.prompt_token_reserve,
            "total_token_budget": budget.total_token_budget,
            "item_token_counts": budget.item_token_counts,
            "requested_item_counts": budget.requested_item_counts,
            "rationale": budget.rationale,
        }

    async def encode_images(
        self,
        image_paths: list[str],
        *,
        prompt: str,
        prompt_token_reserve: int = 0,
        system_prompt: str | None = None,
    ) -> dict[str, Any]:
        """Encode images and return OpenAI-compatible image_embeds messages."""

        inputs = [VisualInput(identifier=path, source_path=Path(path)) for path in image_paths]
        messages = await self.proxy.prepare_openai_messages(
            prompt,
            inputs,
            prompt_token_reserve=prompt_token_reserve,
            system_prompt=system_prompt,
        )
        return {
            "messages": messages,
            "recommended_server_flags": self.interceptor.recommended_server_flags(),
        }

    async def run_benchmark_tool(
        self,
        *,
        mode: str = "simulated",
        image_count: int = 20,
    ) -> dict[str, Any]:
        """Run the packaged TTFT benchmark in simulated or remote mode."""

        comparison = await run_benchmark(mode=mode, image_count=image_count)
        return {
            "control": asdict(comparison.control),
            "experimental": asdict(comparison.experimental),
            "ttft_improvement_ms": comparison.ttft_improvement_ms,
            "ttft_improvement_ratio": comparison.ttft_improvement_ratio,
        }

    def _build_fastmcp(self) -> FastMCP | None:
        """Construct the FastMCP app when the optional dependency is present."""

        if FastMCP is None:
            return None

        mcp = FastMCP(
            "vLLM Chunked Vision Encoder",
            host=self.runtime.mcp.host,
            port=self.runtime.mcp.port,
            sse_path=self.runtime.mcp.mount_path,
            streamable_http_path=self.runtime.mcp.mount_path,
            json_response=True,
            stateless_http=True,
        )

        @mcp.tool()
        async def describe_service() -> dict[str, object]:
            """Describe the available chunked-vision service."""

            return self.describe()

        @mcp.tool()
        async def estimate_budget(
            image_paths: list[str],
            prompt_token_reserve: int = 0,
        ) -> dict[str, Any]:
            """Compute the runtime visual token budget for a prompt."""

            return await self.estimate_budget(
                image_paths,
                prompt_token_reserve=prompt_token_reserve,
            )

        @mcp.tool()
        async def encode_images(
            image_paths: list[str],
            prompt: str,
            prompt_token_reserve: int = 0,
            system_prompt: str | None = None,
        ) -> dict[str, Any]:
            """Encode images into OpenAI-compatible `image_embeds` content parts."""

            return await self.encode_images(
                image_paths,
                prompt=prompt,
                prompt_token_reserve=prompt_token_reserve,
                system_prompt=system_prompt,
            )

        @mcp.tool()
        async def benchmark(mode: str = "simulated", image_count: int = 20) -> dict[str, Any]:
            """Run the benchmark harness and return the comparison summary."""

            return await self.run_benchmark_tool(mode=mode, image_count=image_count)

        return mcp

    async def serve(self) -> None:
        """Start the configured MCP transport."""

        if self._mcp is None:
            raise RuntimeError(
                "The optional `mcp` package is not installed. Install `vllm-chunked-vision-encoder[server]`."
            )

        await asyncio.to_thread(self._mcp.run, transport=self.runtime.mcp.transport)


def main() -> None:
    """Console entry point for the packaged MCP service."""

    parser = argparse.ArgumentParser(description="Run the vLLM chunked vision MCP server.")
    parser.add_argument(
        "--transport",
        choices=("stdio", "sse", "streamable-http"),
        default="stdio",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--mount-path", default="/mcp")
    args = parser.parse_args()

    runtime = RuntimeConfig()
    runtime = RuntimeConfig(
        encoder=runtime.encoder,
        budget=runtime.budget,
        mcp=replace(
            runtime.mcp,
            transport=args.transport,
            host=args.host,
            port=args.port,
            mount_path=args.mount_path,
        ),
    )
    server = ChunkedVisionMCPServer(runtime)
    asyncio.run(server.serve())
