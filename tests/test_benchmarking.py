from __future__ import annotations

import asyncio
from pathlib import Path

from PIL import Image

from vllm_chunked_vision import measure_openai_run, run_benchmark
from vllm_chunked_vision.types import BenchmarkRun


def _write_image(path: Path) -> None:
    Image.new("RGB", (32, 32), color=(240, 240, 240)).save(path)


def test_simulated_benchmark_improves_ttft() -> None:
    comparison = asyncio.run(run_benchmark(mode="simulated", image_count=20))
    assert comparison.control.ttft_ms > comparison.experimental.ttft_ms
    assert comparison.ttft_improvement_ms > 0
    assert comparison.experimental.peak_gpu_percent is not None


def test_measure_openai_run_raw_builds_image_url_payload(tmp_path: Path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_measure(**kwargs: object) -> BenchmarkRun:
        captured.update(kwargs)
        return BenchmarkRun(
            label="test",
            ttft_ms=123.0,
            peak_cpu_percent=50.0,
            peak_gpu_percent=70.0,
            samples=(),
        )

    image_path = tmp_path / "frame.png"
    _write_image(image_path)

    monkeypatch.setattr(
        "vllm_chunked_vision.benchmarking._measure_streaming_ttft",
        fake_measure,
    )

    run = asyncio.run(
        measure_openai_run(
            model="test-model",
            prompt="compare the image",
            image_paths=(image_path,),
            base_url="http://127.0.0.1:8000/v1",
            input_mode="raw",
        )
    )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["messages"][0]["content"][1]["type"] == "image_url"
    assert run.notes == ("input_mode=raw",)


def test_measure_openai_run_image_embeds_builds_embed_payload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, object] = {}

    async def fake_measure(**kwargs: object) -> BenchmarkRun:
        captured.update(kwargs)
        return BenchmarkRun(
            label="test",
            ttft_ms=98.0,
            peak_cpu_percent=40.0,
            peak_gpu_percent=60.0,
            samples=(),
        )

    image_path = tmp_path / "frame.png"
    _write_image(image_path)

    monkeypatch.setattr(
        "vllm_chunked_vision.benchmarking._measure_streaming_ttft",
        fake_measure,
    )

    run = asyncio.run(
        measure_openai_run(
            model="test-model",
            prompt="compare the image",
            image_paths=(image_path,),
            base_url="http://127.0.0.1:8000/v1",
            input_mode="image_embeds",
            experimental_backend="deterministic",
        )
    )

    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["messages"][0]["content"][1]["type"] == "image_embeds"
    assert run.notes == ("input_mode=image_embeds",)
