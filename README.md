# llm-serving-lab

Measuring how LLM inference servers behave under load: time to first token, inter-token latency, throughput and goodput against an SLO, from a single request up to the point where latency breaks.

> Status: work in progress. Results appear below as each stage lands; nothing here is claimed until it is measured.

## Why

Serving LLMs is a reliability problem with unusual shape. Requests stream, their cost depends on prompt and output length, and two phases compete for the same GPU: **prefill** (compute-bound, sets time to first token) and **decode** (memory-bandwidth-bound, sets time per output token). This lab builds the tooling to measure that behaviour honestly and to check the numbers against simple models (roofline, Little's law).

## What it measures

| Metric | Meaning |
|---|---|
| TTFT | Time from sending a request to receiving the first token |
| ITL / TPOT | Gap between consecutive tokens; mean time per output token |
| Throughput | Output tokens per second across all requests |
| Goodput | Requests per second that meet the SLO (e.g. TTFT < 500 ms and TPOT < 50 ms) |
| Tail latency | p50 / p95 / p99 of the above, per load level |

## Servers under test

- **llama.cpp** (`llama-server`) on Apple Silicon (M4 Max, 36 GB), 4-bit GGUF models
- **MLX** on the same machine
- **vLLM** on rented NVIDIA GPUs (L4 for long sweeps, short H100 runs for comparison)

All servers are driven through the same OpenAI-compatible streaming API, so one client measures all of them.

## Roadmap

- [ ] **Stage 0** Local server streaming; project skeleton with CI
- [ ] **Stage 1** Async streaming client that timestamps every chunk; tests against a fake SSE server
- [ ] **Stage 2** Open-loop load generator (Poisson, bursty, long-prompt traffic), avoiding coordinated omission
- [ ] **Stage 3** Analysis: percentiles, goodput, latency-versus-load knee, Little's law check
- [ ] **Stage 4** vLLM sweeps over `max_num_seqs`, prompt length, prefix caching, quantisation and speculative decoding
- [ ] **Stage 5** Prometheus and GPU (DCGM) metrics in Grafana, correlated with client-side latency

## Quick start

```bash
uv sync
llama-server -hf bartowski/Qwen2.5-7B-Instruct-GGUF:Q4_K_M --port 8080
uv run pytest -q
```

## Results

_To be added as stages complete, with hardware, model, quantisation and server version recorded for every run._

## Licence

MIT
