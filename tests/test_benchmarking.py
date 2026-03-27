from __future__ import annotations

import asyncio

from vllm_chunked_vision import run_benchmark


def test_simulated_benchmark_improves_ttft() -> None:
    comparison = asyncio.run(run_benchmark(mode="simulated", image_count=20))
    assert comparison.control.ttft_ms > comparison.experimental.ttft_ms
    assert comparison.ttft_improvement_ms > 0
    assert comparison.experimental.peak_gpu_percent is not None

