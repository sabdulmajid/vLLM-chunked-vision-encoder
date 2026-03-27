"""Public package surface for the vLLM chunked vision encoder."""

from .backends import CallableVisionEncoderBackend, DeterministicVisionBackend, VisionEncoderBackend
from .benchmarking import run_benchmark
from .budgeting import DynamicBudgetInterceptor
from .config import BudgetOverrideConfig, ChunkedVisionConfig, MCPServerConfig, RuntimeConfig
from .encoder import ChunkedVisionEncoder
from .integration import VLLMChunkedVisionProxy
from .mcp_server import ChunkedVisionMCPServer
from .serialize import aggregate_vllm_multi_modal_data, build_openai_messages
from .types import (
    BenchmarkComparison,
    BenchmarkRun,
    BudgetEstimate,
    BudgetPatchResult,
    EmbeddingChunk,
    EncodedVisualItem,
    MetricSnapshot,
    PreparedPrompt,
    VisualChunk,
    VisualInput,
    VisualKind,
)

__all__ = [
    "BenchmarkComparison",
    "BenchmarkRun",
    "BudgetEstimate",
    "BudgetOverrideConfig",
    "BudgetPatchResult",
    "CallableVisionEncoderBackend",
    "ChunkedVisionConfig",
    "ChunkedVisionEncoder",
    "ChunkedVisionMCPServer",
    "DeterministicVisionBackend",
    "DynamicBudgetInterceptor",
    "EmbeddingChunk",
    "EncodedVisualItem",
    "MCPServerConfig",
    "MetricSnapshot",
    "PreparedPrompt",
    "RuntimeConfig",
    "VLLMChunkedVisionProxy",
    "VisionEncoderBackend",
    "VisualChunk",
    "VisualInput",
    "VisualKind",
    "aggregate_vllm_multi_modal_data",
    "build_openai_messages",
    "run_benchmark",
]
