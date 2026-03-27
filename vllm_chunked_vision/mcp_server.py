"""Scaffold for the MCP gateway exposing chunked vision ingestion."""

from __future__ import annotations

from .budgeting import DynamicBudgetInterceptor
from .config import RuntimeConfig
from .encoder import ChunkedVisionEncoder


class ChunkedVisionMCPServer:
    """Owns future MCP tool registration for chunked vision ingestion workflows."""

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

    def describe(self) -> dict[str, object]:
        """Return minimal process metadata for the current scaffold."""

        return {
            "service": "vllm-chunked-vision-encoder",
            "status": "scaffold",
            "transport": self.runtime.mcp.transport,
            "host": self.runtime.mcp.host,
            "port": self.runtime.mcp.port,
        }

    async def serve(self) -> None:
        """Start the MCP transport once the tool handlers are implemented."""

        raise NotImplementedError(
            "MCP transport wiring will be implemented after the scaffold push."
        )


def main() -> None:
    """Console entry point for quick inspection of the scaffolded MCP surface."""

    server = ChunkedVisionMCPServer()
    description = server.describe()
    for key, value in description.items():
        print(f"{key}: {value}")
