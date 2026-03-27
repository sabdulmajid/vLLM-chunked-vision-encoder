# vLLM Chunked Vision Encoder

`vllm-chunked-vision-encoder` is a standalone Python library and future MCP server
for asynchronously ingesting large image and video batches into vLLM without the
monolithic multimodal prefill bottlenecks that currently dominate TTFT.

## Status

This repository currently contains the initial scaffold only. The package surface,
configuration objects, and integration boundaries are in place so the next phase
can implement:

- chunked asynchronous vision encoding
- runtime multimodal token budget overrides
- an MCP gateway for agent-driven ingestion workflows

## Target Environment

The design target is a tensor-parallel vLLM deployment running on 2x96 GB NVIDIA
RTX PRO 6000 GPUs, with large VLMs such as `Qwen/Qwen2.5-VL-72B-Instruct`.

## Bootstrap

```bash
pip install -e .[dev]
```

