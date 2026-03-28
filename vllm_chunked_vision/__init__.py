"""Public package surface for the vLLM chunked vision encoder."""

from .backends import CallableVisionEncoderBackend, DeterministicVisionBackend, VisionEncoderBackend
from .benchmarking import measure_openai_run, run_benchmark
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

try:  # Optional HF-heavy backend.
    from .qwen2_5_vl_backend import Qwen2_5_VLVisualBackend
except ImportError:  # pragma: no cover - exercised when HF extras are absent.
    Qwen2_5_VLVisualBackend = None  # type: ignore[assignment]

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
    "Qwen2_5_VLVisualBackend",
    "RuntimeConfig",
    "VLLMChunkedVisionProxy",
    "VisionEncoderBackend",
    "VisualChunk",
    "VisualInput",
    "VisualKind",
    "aggregate_vllm_multi_modal_data",
    "build_openai_messages",
    "measure_openai_run",
    "run_benchmark",
]
