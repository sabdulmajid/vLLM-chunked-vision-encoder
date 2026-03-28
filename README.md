# vLLM Chunked Vision Encoder

`vllm-chunked-vision-encoder` is a Python library and MCP server for moving the
vision encoder out of vLLM's monolithic multimodal prefill path.

The package is built for large multimodal deployments such as
`Qwen/Qwen2.5-VL-72B-Instruct` running with tensor parallelism on a 2x96 GB RTX
PRO 6000 host. Its goal is to reduce time-to-first-token by:

1. Chunking image or video-frame ingestion into bounded asynchronous batches.
2. Computing the exact multimodal token budget per request instead of relying on
   static worst-case `--limit-mm-per-prompt` estimates.
3. Forwarding precomputed `image_embeds` payloads into vLLM's multimodal inputs
   path so the serving engine no longer performs one giant CPU-bound visual
   prefill step.

## Why This Exists

Standard raw-image serving pushes all visual preprocessing into the same request
critical path that already has to tokenize and schedule generation. On prompts
with dozens of high-resolution images, that leads to three predictable failure
modes:

- TTFT spikes because the CPU is busy materializing visual tokens before the GPU
  can start meaningful prefill work.
- GPU utilization stays low during that prefill window.
- Large models can crash or reserve too aggressively during startup when static
  multimodal limits are sized for worst-case prompts.

This package keeps the user-facing vLLM deployment mostly stock. The primary
deployment path is a proxy-style flow that sends precomputed `image_embeds` into
vLLM with `enable_mm_embeds=True` and `limit_mm_per_prompt={"image": 0, "video":
0}`. For local `LLM` usage, a reflective budget interceptor also patches common
`limit_per_prompt` config locations at runtime.

## Architecture

The package is split into four layers:

- `ChunkedVisionEncoder`: plans bounded chunks and streams encoded embedding
  chunks concurrently.
- `DynamicBudgetInterceptor`: estimates per-request multimodal token budgets and
  patches common vLLM config locations without requiring a fork.
- `VLLMChunkedVisionProxy`: wraps a local engine or prepares OpenAI-compatible
  `image_embeds` chat messages for a remote vLLM server.
- `ChunkedVisionMCPServer`: exposes encode, budget, and benchmark operations as
  MCP tools.

The default backend is deterministic and synthetic so the repository can be
tested without downloading a large vision encoder. For production use, replace
it with a backend that emits model-correct embeddings for your target VLM.

## Installation

Core package:

```bash
pip install vllm-chunked-vision-encoder
```

With MCP server support:

```bash
pip install 'vllm-chunked-vision-encoder[server]'
```

With development tooling:

```bash
pip install -e '.[dev,server]'
```

With optional integrations:

```bash
pip install 'vllm-chunked-vision-encoder[hf,vllm,server]'
```

## Quick Start

### 1. Wrap a Local vLLM Engine

```python
from pathlib import Path

from vllm import LLM, SamplingParams

from vllm_chunked_vision import VLLMChunkedVisionProxy, VisualInput

engine = LLM(
    model="Qwen/Qwen2.5-VL-72B-Instruct",
    tensor_parallel_size=2,
    enable_mm_embeds=True,
    limit_mm_per_prompt={"image": 0, "video": 0},
)
proxy = VLLMChunkedVisionProxy(engine)

inputs = [
    VisualInput(identifier="img-0", source_path=Path("frame0.png")),
    VisualInput(identifier="img-1", source_path=Path("frame1.png")),
]
outputs = proxy.generate(
    "Compare the two frames and explain the strongest evidence.",
    inputs,
    sampling_params=SamplingParams(max_tokens=128),
    prompt_token_reserve=2048,
)
```

### 2. Prepare OpenAI-Compatible `image_embeds`

```python
import asyncio
from pathlib import Path

from vllm_chunked_vision import VLLMChunkedVisionProxy, VisualInput

proxy = VLLMChunkedVisionProxy()
messages = asyncio.run(
    proxy.prepare_openai_messages(
        "Reason over all images jointly.",
        [VisualInput(identifier="img-0", source_path=Path("frame0.png"))],
        prompt_token_reserve=2048,
    )
)
```

### 3. Run the MCP Server

```bash
vllm-chunked-vision-mcp --transport stdio
```

Available tools:

- `describe_service`
- `estimate_budget`
- `encode_images`
- `benchmark`

### 4. Use the Real Qwen2.5-VL Vision Tower

```python
import asyncio
from pathlib import Path

from vllm_chunked_vision import Qwen2_5_VLVisualBackend, VLLMChunkedVisionProxy, VisualInput

backend = Qwen2_5_VLVisualBackend(
    model_id="Qwen/Qwen2.5-VL-72B-Instruct",
    cache_dir="/pub7/neel2/hf-cache",
    device="cuda:0",
)
proxy = VLLMChunkedVisionProxy(backend=backend)

messages = asyncio.run(
    proxy.prepare_openai_messages(
        "Compare all images and explain the strongest evidence.",
        [VisualInput(identifier="frame0", source_path=Path("frame0.png"))],
        prompt_token_reserve=2048,
    )
)
```

## Benchmarking

The repository ships with `benchmark_ttft.py`. It supports two modes:

- `simulated`: deterministic local validation with no vLLM or GPUs required.
- `remote-openai`: benchmark two OpenAI-compatible endpoints, typically a raw
  control vLLM server and an experimental proxy or embeddings-enabled server.

Simulated benchmark:

```bash
python benchmark_ttft.py --mode simulated --image-count 20
```

Remote benchmark:

```bash
python benchmark_ttft.py \
  --mode remote-openai \
  --control-base-url http://127.0.0.1:8000/v1 \
  --experimental-base-url http://127.0.0.1:8100/v1 \
  --model Qwen/Qwen2.5-VL-72B-Instruct \
  --experimental-input-mode image_embeds \
  --experimental-backend qwen2.5-vl \
  --cache-dir /pub7/neel2/hf-cache \
  --device cuda:0 \
  --image-count 20
```

The default remote benchmark generates twenty synthetic 1536x1536 images and
measures streamed TTFT from each endpoint. It also samples CPU utilization and,
when `nvidia-smi` is available, GPU utilization during prefill.

Benchmark matrix:

```bash
python benchmark_suite.py \
  --mode remote-openai \
  --control-base-url http://127.0.0.1:8000/v1 \
  --experimental-base-url http://127.0.0.1:8100/v1 \
  --experimental-input-mode image_embeds \
  --experimental-backend qwen2.5-vl \
  --cache-dir /pub7/neel2/hf-cache \
  --device cuda:0
```

Managed single-host benchmark on one 2x96 GB machine:

```bash
scripts/run_qwen_benchmark_matrix.sh
```

That wrapper uses `benchmark_deployment.py` to rotate the control and
experimental TP=2 servers on the same GPUs instead of assuming two full 72B
servers can stay resident at once.

Real encoder-pipeline benchmark:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python benchmark_encoder_pipeline.py \
  --model Qwen/Qwen2.5-VL-72B-Instruct \
  --cache-dir /pub7/neel2/hf-cache \
  --device cuda:0
```

Validated results from the real Qwen2.5-VL visual tower on this host:

- `images-4-1024`: control TTFF `1792 ms`, experimental TTFF `1778 ms`, improvement `15 ms` (`0.8%`)
- `images-10-1536`: control TTFF `10173 ms`, experimental TTFF `7472 ms`, improvement `2701 ms` (`26.6%`)
- `images-20-1536`: control TTFF `19690 ms`, experimental TTFF `7081 ms`, improvement `12609 ms` (`64.0%`)

Those numbers come from `.bench-assets/qwen_encoder_pipeline.json` and isolate the
external Qwen vision tower itself: monolithic ingestion versus chunked,
concurrent ingestion using the same model-correct embeddings.

## Startup Notes

On this Ubuntu 24.04 host, `vllm 0.12.0` tries to compile Triton launchers
during multimodal startup but the base Python install does not ship
`/usr/include/python3.12/Python.h`. The repository now ships two mitigations:

- `scripts/ensure_python_headers.sh` downloads and extracts the `python3.12-dev`
  headers into `.deps/` when the system headers are missing.
- The `serve_qwen_*.sh` scripts default to `--enforce-eager` and
  `-cc.mode=0`, which disables the failing compile path on minimal hosts.

This keeps the benchmark and serving scripts reproducible without requiring
root or a custom vLLM fork.

## Public Deployment Notes

For a production Qwen2.5-VL deployment on a 2x96 GB host, use a serving command
in this shape:

```bash
vllm serve Qwen/Qwen2.5-VL-72B-Instruct \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --enforce-eager \
  -cc.mode=0 \
  --enable-mm-embeds \
  --limit-mm-per-prompt '{"image":0,"video":0}'
```

The library then becomes the request-side controller that:

- loads and chunks images
- emits model-correct `image_embeds`
- computes exact prompt-side budget requirements
- keeps the raw vLLM server free of heavyweight image preprocessing

## Limitations

- The bundled deterministic backend is for validation and CI, not production
  model accuracy.
- A production deployment still needs a backend that emits embeddings matching
  the target VLM's expected vision projection format.
- The remote benchmark compares endpoints, not internal vLLM scheduler traces.
- The full TP=2 raw-image versus `image_embeds` API benchmark against the live
  Qwen2.5-VL-72B server is scaffolded here but was not completed in this push.

## Development

```bash
pytest
python benchmark_ttft.py --mode simulated --image-count 20
```
