"""Managed end-to-end benchmark for a single 2x96 GB TP=2 Qwen2.5-VL deployment."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import httpx
from PIL import Image, ImageDraw

from benchmark_suite import DEFAULT_SCENARIOS
from vllm_chunked_vision import BenchmarkRun, measure_openai_run


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


def _tail_log(path: Path, *, lines: int = 80) -> str:
    if not path.exists():
        return "<log file not found>"
    entries = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(entries[-lines:])


async def _wait_for_server_ready(
    *,
    base_url: str,
    process: subprocess.Popen[str],
    log_path: Path,
    timeout_s: float,
) -> None:
    deadline = perf_counter() + timeout_s
    last_error: str | None = None
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
        while perf_counter() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    f"Server exited with code {process.returncode}.\n{_tail_log(log_path)}"
                )
            try:
                response = await client.get(f"{base_url.rstrip('/')}/models")
                if response.is_success:
                    return
                last_error = f"HTTP {response.status_code}: {response.text[:400]}"
            except httpx.HTTPError as exc:
                last_error = str(exc)
            await asyncio.sleep(5.0)

    raise TimeoutError(
        f"Timed out waiting for {base_url} to become ready. Last error: {last_error}\n"
        f"{_tail_log(log_path)}"
    )


def _launch_server(
    *,
    script_path: Path,
    env: dict[str, str],
    log_path: Path,
    cwd: Path,
) -> subprocess.Popen[str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        ["bash", str(script_path)],
        cwd=cwd,
        env=env,
        stdout=handle,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )


def _stop_server(process: subprocess.Popen[str], *, shutdown_timeout_s: float) -> None:
    if process.poll() is not None:
        return

    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=shutdown_timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=30.0)


def _resolve_scenarios(selected_names: tuple[str, ...]) -> tuple[dict[str, object], ...]:
    if not selected_names:
        return DEFAULT_SCENARIOS
    selected = [scenario for scenario in DEFAULT_SCENARIOS if scenario["name"] in selected_names]
    missing = sorted(set(selected_names) - {str(item["name"]) for item in selected})
    if missing:
        raise ValueError(f"Unknown scenario names: {', '.join(missing)}")
    return tuple(selected)


def _serialize_run(run: BenchmarkRun) -> dict[str, object]:
    return {
        "label": run.label,
        "ttft_ms": run.ttft_ms,
        "peak_cpu_percent": run.peak_cpu_percent,
        "peak_gpu_percent": run.peak_gpu_percent,
        "notes": list(run.notes),
        "samples": [asdict(sample) for sample in run.samples],
    }


async def main_async(args: argparse.Namespace) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parent
    assets_root = Path(args.assets_dir).resolve()
    logs_root = Path(args.log_dir).resolve()
    assets_root.mkdir(parents=True, exist_ok=True)
    logs_root.mkdir(parents=True, exist_ok=True)

    scenarios = _resolve_scenarios(tuple(args.scenario))
    scenario_images: dict[str, tuple[Path, ...]] = {}
    for scenario in scenarios:
        name = str(scenario["name"])
        scenario_dir = assets_root / name
        scenario_dir.mkdir(parents=True, exist_ok=True)
        scenario_images[name] = _generate_synthetic_images(
            scenario_dir,
            image_count=int(scenario["image_count"]),
            width=int(scenario["image_width"]),
            height=int(scenario["image_height"]),
        )

    base_url = f"http://{args.host}:{args.port}/v1"
    base_env = os.environ.copy()
    base_env.update(
        {
            "MODEL": args.model,
            "CACHE_DIR": args.cache_dir,
            "PORT": str(args.port),
            "TP": str(args.tensor_parallel_size),
            "MAX_MODEL_LEN": str(args.max_model_len),
            "ENFORCE_EAGER": "1",
            "COMPILATION_MODE": "0",
        }
    )

    control_env = dict(base_env)
    control_env["GPU_MEMORY_UTILIZATION"] = str(args.control_gpu_memory_utilization)
    experimental_env = dict(base_env)
    experimental_env["GPU_MEMORY_UTILIZATION"] = str(args.experimental_gpu_memory_utilization)

    control_log = logs_root / "control.log"
    experimental_log = logs_root / "experimental.log"

    control_process = _launch_server(
        script_path=Path(args.control_script).resolve(),
        env=control_env,
        log_path=control_log,
        cwd=repo_root,
    )
    try:
        await _wait_for_server_ready(
            base_url=base_url,
            process=control_process,
            log_path=control_log,
            timeout_s=args.startup_timeout_s,
        )
        control_runs = {}
        for scenario in scenarios:
            name = str(scenario["name"])
            control_runs[name] = _serialize_run(
                await measure_openai_run(
                    model=args.model,
                    prompt=str(scenario["prompt"]),
                    image_paths=scenario_images[name],
                    base_url=base_url,
                    input_mode="raw",
                    server_pid=control_process.pid,
                    label="control-raw-images",
                )
            )
    finally:
        _stop_server(control_process, shutdown_timeout_s=args.shutdown_timeout_s)

    await asyncio.sleep(args.cooldown_s)

    experimental_process = _launch_server(
        script_path=Path(args.experimental_script).resolve(),
        env=experimental_env,
        log_path=experimental_log,
        cwd=repo_root,
    )
    try:
        await _wait_for_server_ready(
            base_url=base_url,
            process=experimental_process,
            log_path=experimental_log,
            timeout_s=args.startup_timeout_s,
        )
        experimental_runs = {}
        for scenario in scenarios:
            name = str(scenario["name"])
            experimental_runs[name] = _serialize_run(
                await measure_openai_run(
                    model=args.model,
                    prompt=str(scenario["prompt"]),
                    image_paths=scenario_images[name],
                    base_url=base_url,
                    input_mode="image_embeds",
                    experimental_backend="qwen2.5-vl",
                    cache_dir=args.cache_dir,
                    device=args.device,
                    server_pid=experimental_process.pid,
                    label="experimental-image-embeds",
                )
            )
    finally:
        _stop_server(experimental_process, shutdown_timeout_s=args.shutdown_timeout_s)

    comparisons: dict[str, dict[str, object]] = {}
    for scenario in scenarios:
        name = str(scenario["name"])
        control_payload = control_runs[name]
        experimental_payload = experimental_runs[name]
        control_ttft = float(control_payload["ttft_ms"])
        experimental_ttft = float(experimental_payload["ttft_ms"])
        improvement_ms = control_ttft - experimental_ttft
        comparisons[name] = {
            "control": control_payload,
            "experimental": experimental_payload,
            "ttft_improvement_ms": improvement_ms,
            "ttft_improvement_ratio": improvement_ms / control_ttft if control_ttft else 0.0,
        }

    return {
        "metadata": {
            "model": args.model,
            "base_url": base_url,
            "cache_dir": args.cache_dir,
            "device": args.device,
            "tensor_parallel_size": args.tensor_parallel_size,
            "max_model_len": args.max_model_len,
            "enforce_eager": True,
            "compilation_mode": 0,
            "control_gpu_memory_utilization": args.control_gpu_memory_utilization,
            "experimental_gpu_memory_utilization": args.experimental_gpu_memory_utilization,
            "control_script": str(Path(args.control_script).resolve()),
            "experimental_script": str(Path(args.experimental_script).resolve()),
            "control_log": str(control_log),
            "experimental_log": str(experimental_log),
        },
        "comparisons": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a TP=2 Qwen2.5-VL deployment by rotating control and experimental servers."
    )
    parser.add_argument("--model", default="Qwen/Qwen2.5-VL-72B-Instruct")
    parser.add_argument("--cache-dir", default="/pub7/neel2/hf-cache")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=65536)
    parser.add_argument("--control-gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--experimental-gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--startup-timeout-s", type=float, default=5400.0)
    parser.add_argument("--shutdown-timeout-s", type=float, default=120.0)
    parser.add_argument("--cooldown-s", type=float, default=15.0)
    parser.add_argument("--assets-dir", default=".bench-assets/managed")
    parser.add_argument("--log-dir", default=".bench-assets/logs")
    parser.add_argument("--control-script", default="scripts/serve_qwen_control.sh")
    parser.add_argument("--experimental-script", default="scripts/serve_qwen_experimental.sh")
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        help="Repeat to select specific scenarios. Defaults to all built-in scenarios.",
    )
    payload = asyncio.run(main_async(parser.parse_args()))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
