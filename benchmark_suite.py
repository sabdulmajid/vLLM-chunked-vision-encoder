"""Run a small benchmark matrix for multiple multimodal scenarios."""

from __future__ import annotations

import argparse
import asyncio
import json

from vllm_chunked_vision.benchmarking import comparison_to_json, run_benchmark_matrix


DEFAULT_SCENARIOS = (
    {
        "name": "images-4-1024",
        "image_count": 4,
        "image_width": 1024,
        "image_height": 1024,
        "prompt": "Compare all four images and identify the strongest shared evidence.",
    },
    {
        "name": "images-10-1536",
        "image_count": 10,
        "image_width": 1536,
        "image_height": 1536,
        "prompt": "Reason jointly over ten high-resolution images and explain the final answer.",
    },
    {
        "name": "images-20-1536",
        "image_count": 20,
        "image_width": 1536,
        "image_height": 1536,
        "prompt": "Analyze twenty high-resolution images, reconcile contradictions, and give a final answer.",
    },
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a benchmark matrix for chunked vision ingestion.")
    parser.add_argument(
        "--mode",
        choices=("simulated", "remote-openai"),
        default="simulated",
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--control-base-url", default=None)
    parser.add_argument("--experimental-base-url", default=None)
    parser.add_argument(
        "--experimental-input-mode",
        choices=("raw", "image_embeds"),
        default="raw",
    )
    parser.add_argument(
        "--experimental-backend",
        choices=("deterministic", "qwen2.5-vl"),
        default=None,
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--control-server-pid", type=int, default=None)
    parser.add_argument("--experimental-server-pid", type=int, default=None)
    args = parser.parse_args()

    results = asyncio.run(
        run_benchmark_matrix(
            scenarios=DEFAULT_SCENARIOS,
            mode=args.mode,
            model=args.model,
            control_base_url=args.control_base_url,
            experimental_base_url=args.experimental_base_url,
            experimental_input_mode=args.experimental_input_mode,
            experimental_backend=args.experimental_backend,
            cache_dir=args.cache_dir,
            device=args.device,
            control_server_pid=args.control_server_pid,
            experimental_server_pid=args.experimental_server_pid,
        )
    )

    payload = {name: json.loads(comparison_to_json(result)) for name, result in results.items()}
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
