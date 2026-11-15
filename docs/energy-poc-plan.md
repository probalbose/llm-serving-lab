# Energy POC plan, least energy that still meets the SLO

Status is plan only, nothing built or measured yet. Every number marked *(to measure)* is a target, not a result.

## 1. The question

> Given a latency SLO and a confidence level chosen by a human, which serving configuration meets the SLO for the fewest joules per good request, and how few benchmark runs do we need to prove it?

This is the Microstructure Stack chain (KPI → required rigor → compute profile → resources) applied to LLM serving instead of Monte Carlo risk.

| Chain step | In this lab |
|---|---|
| KPI | Goodput SLO, for example TTFT p99 < 500 ms and TPOT p99 < 50 ms |
| Required rigor | Confidence that the SLO holds, for example 95% |
| Compute profile | Model, quantisation, batch limit, prompt mix, request rate |
| Resources | Hardware (M4 Max, L4, H100), GPU power cap, server (llama.cpp, MLX, vLLM) |

### Why inference and not training first

The repo already measures serving, so the POC builds on code that exists. Serving is also where most LLM energy is spent once a model is deployed, and every benchmark run is cheap, so we can afford the exhaustive grid needed to check the agent's answer. Training a small model is kept as Phase 2 (section 9), using the same metering and stopping code.

### What is new and what is prior art

Prior art that we use but do not claim includes GPU power capping and its energy effect (Zeus, You et al., NSDI 2023), hyperparameter search with early stopping (successive halving, Hyperband), vLLM and llama.cpp tuning guides, and the anytime valid confidence sequences we rely on (Howard et al. 2021, Waudby-Smith and Ramdas 2023).

What we test is the coupling. A human states an SLO and a confidence level, the system decides how much benchmarking is needed to certify each configuration, stops as soon as the evidence is sufficient, and returns the cheapest certified configuration with an energy receipt. No serving benchmark we know of reports a statistical guarantee on the SLO or the energy spent reaching it.

## 2. Hypotheses

- **H1 (search saving).** The agent loop finds a configuration whose energy per good request is within 10% of the best found by an exhaustive grid, while spending at most 50% of the grid's benchmark energy. *(to measure)*
- **H2 (certification saving).** Sequential stopping certifies or rejects each configuration with fewer requests than a fixed length benchmark of the same confidence, on average. *(to measure)*
- **H3 (energy spread).** Across configurations that all meet the SLO, energy per good request varies enough to matter, say by 1.5x or more. If it does not, the product story is weak on this workload and we say so. *(to measure)*

H3 is the kill criterion. Check it first with a small grid before building agents.

## 3. Metrics

Existing: TTFT, ITL, TPOT, throughput, goodput (see README).

New:

| Metric | Definition |
|---|---|
| Energy per good request | Joules measured over a run divided by requests that met the SLO |
| Benchmark energy | Joules spent running benchmarks to reach a decision, including rejected configurations |
| SLO violation rate | Fraction of requests with TTFT or TPOT above the SLO |
| Certified | Upper confidence bound on violation rate is below the allowed rate |

Every energy number carries a provenance tag, following Avishek's Strand A protocol.

- `[MEASURED]` read from a hardware counter (NVML, DCGM, `powermetrics`)
- `[MODELLED]` derived from a measured quantity, for example idle power subtracted
- `[ESTIMATED]` from a datasheet or a published figure

## 4. Statistics, certifying an SLO with as few requests as possible

Let $X_i = 1$ if request $i$ violates the SLO. With allowed violation rate $p_0$ (0.01 for a p99 SLO) and error level $\alpha$ (0.05 for 95% confidence):

- **Certify** when the upper confidence bound on $P(X=1)$ falls below $p_0$.
- **Reject** when the lower confidence bound rises above $p_0$.
- Otherwise keep sending requests, up to a cap.

A fixed length test gives a useful floor. With zero violations, certifying $p \le 0.01$ at 95% needs $n \ge \ln(0.05)/\ln(0.99) \approx 299$ requests. A bad configuration, in contrast, can often be rejected after a few dozen requests, which is where most of the saving should come from.

Because the decision is checked after every request, a fixed sample bound is not valid here (peeking inflates the error). Use an anytime valid confidence sequence for a Bernoulli mean, for example the betting construction of Waudby-Smith and Ramdas (2023), which keeps the error at $\alpha$ however often we look.

Caveats to handle explicitly, not to hide:

1. **Requests under load are not independent.** Queueing makes neighbouring latencies correlated. To mitigate, discard a warm up window, then test on block means or on every $k$th request, and report the effective sample size.
2. **Open loop load only.** The Stage 2 generator must avoid coordinated omission, otherwise the violation rate is biased low.
3. **Joint SLO.** TTFT and TPOT are tested together as one violation indicator, so no multiple testing correction is needed for the pair. Comparing many configurations does need one; split $\alpha$ across configurations or use a procedure based on e values.

## 5. Energy metering

| Hardware | Tool | Notes |
|---|---|---|
| M4 Max | `sudo powermetrics --samplers cpu_power,gpu_power` | Package level, about 1 s resolution, needs sudo |
| NVIDIA L4, H100 | NVML via `pynvml`, or Zeus `ZeusMonitor` | Use the energy counter (`nvmlDeviceGetTotalEnergyConsumption`) where supported, not sampled power |
| NVIDIA, Stage 5 | DCGM exporter to Prometheus | Same numbers in Grafana next to latency |

Rules:

- Record idle power before each sweep and report both gross and idle subtracted energy.
- Record which part is measured. GPU counters miss CPU, memory and host power, so label GPU only numbers as such.
- Store hardware, driver, server version, model and quantisation with every run, as the README already requires.

## 6. Agent loop

Three roles, plus a human who owns the goal (Wall 2).

1. **Human.** Sets the SLO, $p_0$, $\alpha$, the energy budget for the search and the allowed configuration space. Can override any decision.
2. **Planner agent.** Proposes the next configuration to test, using results so far. Start with a simple strategy (successive halving over a fixed grid) and only add an LLM planner if it beats that baseline.
3. **Runner.** Deterministic code, not an agent. Starts the server with the configuration, runs the load, streams results.
4. **Judge.** Deterministic code. Applies the sequential test from section 4 after every request and stops the run as soon as it can certify or reject.

The planner never touches prompts or outputs, only configuration and measurements. This mirrors the alpha blind wall.

Every decision is written to a run manifest (JSON lines) with `resources_used`, `rigor_assumed`, `who_assented` and `intent`, as in the main design.

## 7. Configuration space

Start small so the exhaustive grid is affordable.

| Knob | Values |
|---|---|
| Server and hardware | llama.cpp on M4 Max, vLLM on L4 |
| Model | Qwen2.5 7B Instruct (already in the quick start) |
| Quantisation | Q4_K_M and Q8_0 (llama.cpp), FP16 and AWQ 4 bit (vLLM) |
| Batch limit | `max_num_seqs` 8, 16, 32, 64 (vLLM), `--parallel` 1, 2, 4, 8 (llama.cpp) |
| GPU power cap (L4) | 72 W (default), 60 W, 50 W, via `nvidia-smi -pl` |
| Request rate | 3 to 4 levels below and around the latency knee from Stage 3 |
| Prompt mix | Short chat and long prompt traffic from Stage 2 |

## 8. Work plan

Builds on the existing roadmap. Stages 0 to 3 are needed first.

| Step | Work | Output | Done when |
|---|---|---|---|
| E0 | Finish Stages 1 to 3 | Load generator and analysis | Goodput and latency knee reported for llama.cpp on M4 Max |
| E1 | Energy meter module (`serving_lab/energy.py`) with `powermetrics` and NVML backends | Joules per run with provenance tag | Idle and loaded readings repeatable within 5% across 3 runs |
| E2 | Small grid on M4 Max, about 12 configurations | Energy per good request table | H3 checked. Stop here if the spread is small |
| E3 | Sequential judge (`serving_lab/sequential.py`) with unit tests on simulated Bernoulli streams | Certify or reject with request count | Empirical error rate at or below $\alpha$ over 10,000 simulated runs |
| E4 | Full grid on rented L4 | Ground truth best configuration | All configurations measured at fixed length |
| E5 | Planner and runner loop | Agent's chosen configuration and its benchmark energy | H1 and H2 evaluated against E4 |
| E6 | Write up | Results section in README, one plot of energy versus SLO | Every number tagged and reproducible from stored manifests |

Rough budget: E4 and E5 need tens of GPU hours on an L4. Check current rental prices before starting. No H100 time is needed for the POC.

## 9. Phase 2: training a small model with the same method

Once the serving POC works, repeat the idea for training, where the question becomes the least energy to reach a target validation loss.

- **Model and code.** GPT 2 small (124M) with nanoGPT or llm.c, on FineWeb Edu or TinyStories.
- **Target.** Validation loss below a chosen value, with confidence over seeds or evaluation batches.
- **Knobs.** Model size versus tokens (Chinchilla, Hoffmann et al. 2022), batch size, learning rate schedule, GPU power cap.
- **Stopping.** Cut runs whose loss curve cannot reach the target, and stop evaluation once the loss estimate is tight enough.
- **Link to Strand A.** Fine tuning several variants from one base model gives the model derivation DAG that Avishek's Shapley attribution works on, so the energy receipts can be split fairly across variants.

This phase belongs in a separate repo (`llm-training-lab` or similar), sharing the energy and sequential modules.

## 10. Risks

| Risk | Mitigation |
|---|---|
| Energy differences are small (H3 fails) | Found at E2, before agents are built. Report it honestly and change workload |
| `powermetrics` too coarse for short runs | Make runs at least 60 s, or measure on NVIDIA only |
| Correlated latencies break the test | Block means and effective sample size, section 4 |
| Agent planner adds no value over successive halving | Keep the simple planner, that is still a valid result |
| Rented GPU variance (noisy neighbours, thermal) | Repeat the best 3 configurations on a second instance |

## 11. Deliverable for Tony, EI and Avishek

One page with the plot of energy per good request against SLO, the agent's choice versus the grid's best, and benchmark energy saved. It demonstrates the core claim (same required quality, less energy, with proof) on a public, reproducible workload that does not reveal any unfiled method detail.
