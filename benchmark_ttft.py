"""CLI entry point for the packaged TTFT benchmark."""

from __future__ import annotations

import argparse
import asyncio

from vllm_chunked_vision.benchmarking import comparison_to_json, run_benchmark


def main() -> None:
    """Parse CLI arguments and print the benchmark comparison as JSON."""

    parser = argparse.ArgumentParser(description="Benchmark TTFT for chunked vision ingestion.")
    parser.add_argument(
        "--mode",
        choices=("simulated", "remote-openai"),
        default="simulated",
    )
    parser.add_argument("--image-count", type=int, default=20)
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--control-base-url", default=None)
    parser.add_argument("--experimental-base-url", default=None)
    parser.add_argument("--image-path", action="append", dest="image_paths", default=None)
    args = parser.parse_args()

    comparison = asyncio.run(
        run_benchmark(
            mode=args.mode,
            image_count=args.image_count,
            model=args.model,
            prompt=args.prompt,
            control_base_url=args.control_base_url,
            experimental_base_url=args.experimental_base_url,
            image_paths=args.image_paths,
        )
    )
    print(comparison_to_json(comparison))


if __name__ == "__main__":
    main()

