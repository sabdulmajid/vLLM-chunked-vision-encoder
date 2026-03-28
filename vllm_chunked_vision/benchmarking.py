"""Benchmark helpers for real and simulated TTFT comparisons."""

from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
import psutil
from PIL import Image, ImageDraw

from .backends import DeterministicVisionBackend
from .config import ChunkedVisionConfig, RuntimeConfig
from .encoder import ChunkedVisionEncoder
from .integration import VLLMChunkedVisionProxy
from .types import BenchmarkComparison, BenchmarkRun, MetricSnapshot, VisualInput


def _generate_synthetic_images(
    directory: Path,
    *,
    image_count: int,
    width: int,
    height: int,
) -> tuple[Path, ...]:
    paths: list[Path] = []
    for index in range(image_count):
        image = Image.new("RGB", (width, height), color=(245, 241, 232))
        draw = ImageDraw.Draw(image)
        draw.rectangle((40, 40, width - 40, height - 40), outline=(24, 53, 77), width=8)
        draw.text((80, 80), f"Frame {index:02d}", fill=(24, 53, 77))
        path = directory / f"frame_{index:02d}.png"
        image.save(path)
        paths.append(path)
    return tuple(paths)


def _data_url(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def _query_gpu_utilization() -> float | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    values: list[float] = []
    for line in completed.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            values.append(float(line))
        except ValueError:
            continue
    return max(values) if values else None


def _prime_process_tree_cpu(root_pid: int) -> None:
    try:
        root = psutil.Process(root_pid)
    except psutil.Error:
        return

    processes = [root, *root.children(recursive=True)]
    for process in processes:
        try:
            process.cpu_percent(interval=None)
        except psutil.Error:
            continue


def _sample_process_tree_cpu(root_pid: int) -> float:
    try:
        root = psutil.Process(root_pid)
    except psutil.Error:
        return 0.0

    total = 0.0
    for process in [root, *root.children(recursive=True)]:
        try:
            total += process.cpu_percent(interval=None)
        except psutil.Error:
            continue
    return total


async def _sample_metrics(
    stop_event: asyncio.Event,
    *,
    server_pid: int | None,
    interval_s: float = 0.1,
) -> tuple[MetricSnapshot, ...]:
    samples: list[MetricSnapshot] = []
    started_at = perf_counter()
    if server_pid is not None:
        _prime_process_tree_cpu(server_pid)
    else:
        psutil.cpu_percent(interval=None)
    while not stop_event.is_set():
        cpu_percent = (
            _sample_process_tree_cpu(server_pid)
            if server_pid is not None
            else psutil.cpu_percent(interval=None)
        )
        samples.append(
            MetricSnapshot(
                timestamp_s=perf_counter() - started_at,
                cpu_percent=cpu_percent,
                gpu_percent=_query_gpu_utilization(),
            )
        )
        await asyncio.sleep(interval_s)
    return tuple(samples)


async def _measure_streaming_ttft(
    *,
    base_url: str,
    payload: dict[str, Any],
    api_key: str = "EMPTY",
    server_pid: int | None = None,
) -> BenchmarkRun:
    stop_event = asyncio.Event()
    sampler = asyncio.create_task(_sample_metrics(stop_event, server_pid=server_pid))
    started_at = perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0, connect=30.0)) as client:
        async with client.stream(
            "POST",
            f"{base_url.rstrip('/')}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {api_key}"},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if not data or data == "[DONE]":
                    continue
                ttft_ms = (perf_counter() - started_at) * 1000.0
                stop_event.set()
                samples = await sampler
                return BenchmarkRun(
                    label=base_url,
                    ttft_ms=ttft_ms,
                    peak_cpu_percent=max((sample.cpu_percent for sample in samples), default=0.0),
                    peak_gpu_percent=max(
                        (sample.gpu_percent for sample in samples if sample.gpu_percent is not None),
                        default=None,
                    ),
                    samples=samples,
                )
    stop_event.set()
    samples = await sampler
    raise RuntimeError(
        f"No streamed token events were received from {base_url}. Collected {len(samples)} samples."
    )


def _messages_for_raw_images(prompt: str, image_paths: Sequence[Path]) -> list[dict[str, Any]]:
    content = [{"type": "text", "text": prompt}]
    for path in image_paths:
        content.append({"type": "image_url", "image_url": {"url": _data_url(path)}})
    return [{"role": "user", "content": content}]


async def _build_openai_payload(
    *,
    model: str,
    prompt: str,
    image_paths: Sequence[Path],
    input_mode: str,
    experimental_backend: str | None,
    cache_dir: str | None,
    device: str,
) -> dict[str, Any]:
    if input_mode == "raw":
        return {
            "model": model,
            "stream": True,
            "messages": _messages_for_raw_images(prompt, image_paths),
            "max_tokens": 128,
        }

    if input_mode != "image_embeds":
        raise ValueError("input_mode must be 'raw' or 'image_embeds'")

    if experimental_backend == "qwen2.5-vl":
        from .qwen2_5_vl_backend import Qwen2_5_VLVisualBackend

        backend = Qwen2_5_VLVisualBackend(
            model_id=model,
            cache_dir=cache_dir,
            device=device,
        )
    else:
        backend = DeterministicVisionBackend(encode_delay_s=0.0)

    proxy = VLLMChunkedVisionProxy(
        runtime=RuntimeConfig(
            encoder=ChunkedVisionConfig(
                max_images_per_chunk=4,
                max_concurrent_chunks=2,
                max_tokens_per_chunk=8192,
            )
        ),
        backend=backend,
    )
    messages = await proxy.prepare_openai_messages(
        prompt,
        [VisualInput(identifier=str(path), source_path=path) for path in image_paths],
        prompt_token_reserve=2048,
    )
    return {
        "model": model,
        "stream": True,
        "messages": messages,
        "max_tokens": 128,
    }


async def measure_openai_run(
    *,
    model: str,
    prompt: str,
    image_paths: Sequence[Path],
    base_url: str,
    input_mode: str = "raw",
    experimental_backend: str | None = None,
    cache_dir: str | None = None,
    device: str = "cuda:0",
    server_pid: int | None = None,
    label: str | None = None,
) -> BenchmarkRun:
    """Measure one OpenAI-compatible run and return TTFT plus sampled CPU/GPU metrics."""

    payload = await _build_openai_payload(
        model=model,
        prompt=prompt,
        image_paths=image_paths,
        input_mode=input_mode,
        experimental_backend=experimental_backend,
        cache_dir=cache_dir,
        device=device,
    )
    benchmark_run = await _measure_streaming_ttft(
        base_url=base_url,
        payload=payload,
        server_pid=server_pid,
    )
    notes = (f"input_mode={input_mode}",)
    return BenchmarkRun(
        label=label or benchmark_run.label,
        ttft_ms=benchmark_run.ttft_ms,
        peak_cpu_percent=benchmark_run.peak_cpu_percent,
        peak_gpu_percent=benchmark_run.peak_gpu_percent,
        samples=benchmark_run.samples,
        notes=notes,
    )


async def _run_simulated_benchmark(image_count: int) -> BenchmarkComparison:
    prompt = (
        "Compare the geometry, text, and semantic overlap across all inputs and explain the "
        "strongest cross-frame evidence for the final answer."
    )
    inputs = tuple(
        VisualInput(
            identifier=f"image-{index:02d}",
            width=1536,
            height=1536,
        )
        for index in range(image_count)
    )
    runtime = RuntimeConfig(
        encoder=ChunkedVisionConfig(
            max_images_per_chunk=4,
            max_concurrent_chunks=2,
            max_tokens_per_chunk=4096,
        )
    )
    proxy = VLLMChunkedVisionProxy(
        engine=None,
        runtime=runtime,
        backend=DeterministicVisionBackend(encode_delay_s=0.12),
    )

    control_started = perf_counter()
    await asyncio.sleep(image_count * 0.12)
    control_ttft_ms = (perf_counter() - control_started) * 1000.0

    experimental_started = perf_counter()
    async for _chunk in proxy.encoder.stream_embeddings(inputs):
        break
    experimental_ttft_ms = (perf_counter() - experimental_started) * 1000.0

    control_samples = tuple(
        MetricSnapshot(timestamp_s=index * 0.1, cpu_percent=93.0, gpu_percent=11.0)
        for index in range(max(1, int(control_ttft_ms / 100.0)))
    )
    experimental_samples = tuple(
        MetricSnapshot(timestamp_s=index * 0.1, cpu_percent=64.0, gpu_percent=74.0)
        for index in range(max(1, int(experimental_ttft_ms / 100.0)))
    )
    control = BenchmarkRun(
        label="control-simulated-vllm",
        ttft_ms=control_ttft_ms,
        peak_cpu_percent=93.0,
        peak_gpu_percent=11.0,
        samples=control_samples,
        notes=("Synthetic control: serial CPU-bound multimodal prefill.",),
    )
    experimental = BenchmarkRun(
        label="experimental-simulated-chunked-encoder",
        ttft_ms=experimental_ttft_ms,
        peak_cpu_percent=64.0,
        peak_gpu_percent=74.0,
        samples=experimental_samples,
        notes=(
            "Synthetic experimental run: chunked embeddings prepared concurrently before vLLM.",
        ),
    )
    improvement_ms = control.ttft_ms - experimental.ttft_ms
    return BenchmarkComparison(
        control=control,
        experimental=experimental,
        ttft_improvement_ms=improvement_ms,
        ttft_improvement_ratio=improvement_ms / control.ttft_ms if control.ttft_ms else 0.0,
    )


async def _run_remote_openai_benchmark(
    *,
    model: str,
    prompt: str,
    image_paths: Sequence[Path],
    control_base_url: str,
    experimental_base_url: str,
    experimental_input_mode: str,
    experimental_backend: str | None,
    cache_dir: str | None,
    device: str,
    control_server_pid: int | None,
    experimental_server_pid: int | None,
) -> BenchmarkComparison:
    control = await measure_openai_run(
        model=model,
        prompt=prompt,
        image_paths=image_paths,
        base_url=control_base_url,
        input_mode="raw",
        server_pid=control_server_pid,
        label="control-raw-images",
    )
    experimental = await measure_openai_run(
        model=model,
        prompt=prompt,
        image_paths=image_paths,
        base_url=experimental_base_url,
        input_mode=experimental_input_mode,
        experimental_backend=experimental_backend,
        cache_dir=cache_dir,
        device=device,
        server_pid=experimental_server_pid,
        label=f"experimental-{experimental_input_mode}",
    )
    improvement_ms = control.ttft_ms - experimental.ttft_ms
    return BenchmarkComparison(
        control=control,
        experimental=experimental,
        ttft_improvement_ms=improvement_ms,
        ttft_improvement_ratio=improvement_ms / control.ttft_ms if control.ttft_ms else 0.0,
    )


async def run_benchmark(
    *,
    mode: str = "simulated",
    image_count: int = 20,
    model: str = "Qwen/Qwen2.5-VL-72B-Instruct",
    prompt: str | None = None,
    control_base_url: str | None = None,
    experimental_base_url: str | None = None,
    image_paths: Sequence[str] | None = None,
    experimental_input_mode: str = "raw",
    experimental_backend: str | None = None,
    cache_dir: str | None = None,
    device: str = "cuda:0",
    image_width: int = 1536,
    image_height: int = 1536,
    control_server_pid: int | None = None,
    experimental_server_pid: int | None = None,
) -> BenchmarkComparison:
    """Run the benchmark in simulated mode or against two OpenAI-compatible endpoints."""

    if mode == "simulated":
        return await _run_simulated_benchmark(image_count=image_count)

    if mode != "remote-openai":
        raise ValueError("mode must be 'simulated' or 'remote-openai'")

    if control_base_url is None or experimental_base_url is None:
        raise ValueError("remote-openai mode requires both control_base_url and experimental_base_url")

    benchmark_prompt = prompt or (
        "You are given twenty high-resolution images. Identify the strongest multi-image evidence, "
        "explain contradictions, and produce a final answer with confidence."
    )
    if image_paths is None:
        with tempfile.TemporaryDirectory(prefix="chunked-vision-benchmark-") as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            paths = _generate_synthetic_images(
                temp_dir,
                image_count=image_count,
                width=image_width,
                height=image_height,
            )
            return await _run_remote_openai_benchmark(
                model=model,
                prompt=benchmark_prompt,
                image_paths=paths,
                control_base_url=control_base_url,
                experimental_base_url=experimental_base_url,
                experimental_input_mode=experimental_input_mode,
                experimental_backend=experimental_backend,
                cache_dir=cache_dir,
                device=device,
                control_server_pid=control_server_pid,
                experimental_server_pid=experimental_server_pid,
            )

    return await _run_remote_openai_benchmark(
        model=model,
        prompt=benchmark_prompt,
        image_paths=tuple(Path(path) for path in image_paths),
        control_base_url=control_base_url,
        experimental_base_url=experimental_base_url,
        experimental_input_mode=experimental_input_mode,
        experimental_backend=experimental_backend,
        cache_dir=cache_dir,
        device=device,
        control_server_pid=control_server_pid,
        experimental_server_pid=experimental_server_pid,
    )


async def run_benchmark_matrix(
    *,
    scenarios: Sequence[dict[str, Any]],
    mode: str,
    model: str,
    control_base_url: str | None = None,
    experimental_base_url: str | None = None,
    experimental_input_mode: str = "raw",
    experimental_backend: str | None = None,
    cache_dir: str | None = None,
    device: str = "cuda:0",
    control_server_pid: int | None = None,
    experimental_server_pid: int | None = None,
) -> dict[str, BenchmarkComparison]:
    """Run a named benchmark matrix and return results keyed by scenario name."""

    results: dict[str, BenchmarkComparison] = {}
    for scenario in scenarios:
        name = str(scenario["name"])
        results[name] = await run_benchmark(
            mode=mode,
            model=model,
            prompt=scenario.get("prompt"),
            image_count=int(scenario["image_count"]),
            control_base_url=control_base_url,
            experimental_base_url=experimental_base_url,
            experimental_input_mode=experimental_input_mode,
            experimental_backend=experimental_backend,
            cache_dir=cache_dir,
            device=device,
            control_server_pid=control_server_pid,
            experimental_server_pid=experimental_server_pid,
            image_width=int(scenario.get("image_width", 1536)),
            image_height=int(scenario.get("image_height", 1536)),
        )
    return results


def comparison_to_json(comparison: BenchmarkComparison) -> str:
    """Serialize a benchmark comparison into pretty JSON."""

    return json.dumps(
        {
            "control": {
                "label": comparison.control.label,
                "ttft_ms": comparison.control.ttft_ms,
                "peak_cpu_percent": comparison.control.peak_cpu_percent,
                "peak_gpu_percent": comparison.control.peak_gpu_percent,
                "notes": list(comparison.control.notes),
                "samples": [asdict(sample) for sample in comparison.control.samples],
            },
            "experimental": {
                "label": comparison.experimental.label,
                "ttft_ms": comparison.experimental.ttft_ms,
                "peak_cpu_percent": comparison.experimental.peak_cpu_percent,
                "peak_gpu_percent": comparison.experimental.peak_gpu_percent,
                "notes": list(comparison.experimental.notes),
                "samples": [asdict(sample) for sample in comparison.experimental.samples],
            },
            "ttft_improvement_ms": comparison.ttft_improvement_ms,
            "ttft_improvement_ratio": comparison.ttft_improvement_ratio,
        },
        indent=2,
        sort_keys=True,
    )
