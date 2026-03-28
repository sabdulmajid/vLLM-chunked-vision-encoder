"""Benchmark the real Qwen2.5-VL vision tower with monolithic vs chunked ingestion."""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageDraw

from vllm_chunked_vision import ChunkedVisionConfig, ChunkedVisionEncoder, VisualInput
from vllm_chunked_vision.benchmarking import _sample_metrics  # noqa: SLF001
from vllm_chunked_vision.qwen2_5_vl_backend import Qwen2_5_VLVisualBackend


def _generate_images(directory: Path, image_count: int, width: int, height: int) -> list[VisualInput]:
    inputs: list[VisualInput] = []
    for index in range(image_count):
        image = Image.new("RGB", (width, height), color=(244, 242, 236))
        draw = ImageDraw.Draw(image)
        draw.rectangle((64, 64, width - 64, height - 64), outline=(32, 64, 91), width=10)
        draw.text((96, 96), f"Benchmark {index:02d}", fill=(32, 64, 91))
        path = directory / f"bench_{index:02d}.png"
        image.save(path)
        inputs.append(VisualInput(identifier=path.name, source_path=path, width=width, height=height))
    return inputs


async def _measure_encoder(encoder: ChunkedVisionEncoder, inputs: list[VisualInput]) -> dict[str, object]:
    stop_event = asyncio.Event()
    sampler = asyncio.create_task(_sample_metrics(stop_event, server_pid=None))
    started_at = perf_counter()
    first_chunk_ms: float | None = None
    chunk_count = 0
    total_tokens = 0
    async for chunk in encoder.stream_embeddings(inputs):
        chunk_count += 1
        total_tokens += chunk.total_tokens
        if first_chunk_ms is None:
            first_chunk_ms = (perf_counter() - started_at) * 1000.0
    total_ms = (perf_counter() - started_at) * 1000.0
    stop_event.set()
    samples = await sampler
    return {
        "time_to_first_chunk_ms": first_chunk_ms,
        "total_encode_ms": total_ms,
        "chunk_count": chunk_count,
        "total_tokens": total_tokens,
        "peak_cpu_percent": max((sample.cpu_percent for sample in samples), default=0.0),
        "peak_gpu_percent": max(
            (sample.gpu_percent for sample in samples if sample.gpu_percent is not None),
            default=None,
        ),
        "samples": [asdict(sample) for sample in samples],
    }


async def _run_scenario(
    *,
    backend: Qwen2_5_VLVisualBackend,
    image_count: int,
    width: int,
    height: int,
) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix="qwen-vision-benchmark-") as temp_dir_name:
        inputs = _generate_images(Path(temp_dir_name), image_count, width, height)
        monolithic_encoder = ChunkedVisionEncoder(
            ChunkedVisionConfig(
                max_images_per_chunk=image_count,
                max_concurrent_chunks=1,
                max_tokens_per_chunk=None,
            ),
            backend=backend,
        )
        chunked_encoder = ChunkedVisionEncoder(
            ChunkedVisionConfig(
                max_images_per_chunk=4,
                max_concurrent_chunks=2,
                max_tokens_per_chunk=8192,
            ),
            backend=backend,
        )
        control = await _measure_encoder(monolithic_encoder, inputs)
        experimental = await _measure_encoder(chunked_encoder, inputs)
        control_ttff = float(control["time_to_first_chunk_ms"] or 0.0)
        experimental_ttff = float(experimental["time_to_first_chunk_ms"] or 0.0)
        return {
            "control": control,
            "experimental": experimental,
            "ttff_improvement_ms": control_ttff - experimental_ttff,
            "ttff_improvement_ratio": (
                (control_ttff - experimental_ttff) / control_ttff if control_ttff else 0.0
            ),
        }


async def _warm_backend(backend: Qwen2_5_VLVisualBackend) -> None:
    with tempfile.TemporaryDirectory(prefix="qwen-vision-warmup-") as temp_dir_name:
        inputs = _generate_images(Path(temp_dir_name), 1, 512, 512)
        encoder = ChunkedVisionEncoder(
            ChunkedVisionConfig(
                max_images_per_chunk=1,
                max_concurrent_chunks=1,
                max_tokens_per_chunk=None,
            ),
            backend=backend,
        )
        async for _chunk in encoder.stream_embeddings(inputs):
            break


async def main_async(args: argparse.Namespace) -> dict[str, object]:
    backend = Qwen2_5_VLVisualBackend(
        model_id=args.model,
        cache_dir=args.cache_dir,
        device=args.device,
    )
    await _warm_backend(backend)
    scenarios = {
        "images-4-1024": await _run_scenario(
            backend=backend,
            image_count=4,
            width=1024,
            height=1024,
        ),
        "images-10-1536": await _run_scenario(
            backend=backend,
            image_count=10,
            width=1536,
            height=1536,
        ),
        "images-20-1536": await _run_scenario(
            backend=backend,
            image_count=20,
            width=1536,
            height=1536,
        ),
    }
    return scenarios


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the real Qwen2.5-VL vision pipeline.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--cache-dir", default="/pub7/neel2/hf-cache")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    payload = asyncio.run(main_async(args))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
