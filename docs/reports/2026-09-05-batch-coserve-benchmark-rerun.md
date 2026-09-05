# Batch co-serving benchmark — rerun on the final head

> Last updated: 2026-09-05 · commit `9611f8771`

A rerun of the co-serving benchmark on the head the Tidal batch lane shipped
from: `9611f8771`, synced with upstream master through #830, after every review
fix pass. Same test, same seed and same box as the
[2026-09-04 run](2026-09-04-batch-coserve-benchmark.md); this record exists to
confirm the numbers still hold on the code that landed. Produced by
`TestBenchmarkBatchCoServe` (`e2e/batch_coserve_test.go`) against a real
coordinator, a real Swift provider and real MLX inference.

## Setup

| Setting | Value |
|---|---|
| host | M4 Pro / 48 GB / macOS 15.7, debug provider build, hand-built mlx.metallib |
| run | 2026-09-05 04:40 UTC |
| commit | `9611f8771` (upstream master through #830, all review fixes applied) |
| model | `mlx-community/Qwen3.5-0.8B-MLX-4bit` |
| providers | 1 |
| store | in-memory testbed store |
| arrival schedule | seeded Poisson, seed 20260906, 0.50 req/s, 2m0s window, 72 arrivals |
| offline job | 300 items, max_tokens 24 |
| online requests | non-streaming, max_tokens 32, short arithmetic prompts |
| wall time | 503.8 s for the four phases (`make e2e-coserve`, `COSERVE_REPORT_PATH` set) |

## Method

- All four phases share one suite, one provider process and one loaded model; the model is warmed with two throwaway requests before the first measurement.
- `online_only` — replay the seeded schedule with nothing else running. This is the baseline every ratio below is taken against, measured in the same session on the same box, never against a historical number.
- `offline_only` — submit a 300-item batch alone, sample `GET /v1/batches/{id}` once a second, and take items/s over the 1m0s window that opens 30s in (the first seconds are the dispatcher's AIMD ramp, not steady state). The batch is cancelled once the window closes.
- `coserve` — the same batch and the same arrival schedule together, measured the same way on both sides.
- `flex` — the same schedule again with `service_tier: "batch"` on every synchronous request, counting 200s against 429s. It runs alone, so its 429s are the batch lane's headroom-only admission refusing on an otherwise quiet stack, not contention with an offline job.
- Earnings come from the store's `provider_earnings` rows, bucketed by their `Lane` column over each phase's wall-clock window and scaled to an hourly rate.

## Results

| Metric | Definition | Value |
|---|---|---|
| offline ceiling | batch items settled per second with no online load, measured over the 1m0s window that opens 30s after the batch is created | 1.034 items/s (67 → 128 items over 59s) |
| co-serving batch rate | the same measurement while the online arrival schedule replays against the same provider | 1.136 items/s (33 → 100 items over 59s) |
| harvest | co-serving batch rate ÷ offline ceiling — the fraction of the idle-capacity ceiling still reachable while online traffic is served | 1.10 (110%) |
| online-only p50 / p99 | client-observed wall time of the seeded Poisson arrivals with nothing else running | 100 ms / 220 ms (n=69 served) |
| co-serving p50 / p99 | the same schedule, same seed, while the offline job runs | 98 ms / 127 ms (n=68 served) |
| online admission | HTTP 200 vs 429 on the online arm, baseline then co-serving — a 429 here is the provider refusing on capacity after the coordinator's retries | online_only 69 × 200, 3 × 429; coserve 68 × 200, 4 × 429 (of 72 each) |
| online p50 ratio | co-serving p50 ÷ online-only p50 | 0.98× |
| online p99 ratio | co-serving p99 ÷ online-only p99 — the gated number | 0.58× |
| flex admission | the same schedule with service_tier: "batch" on every synchronous request (the OpenRouter path): HTTP 200 vs 429 | 58 × 200, 14 × 429 (of 72) |
| flex p50 / p99 | wall time of the served service_tier=batch requests | 102 ms / 126 ms |
| earnings/hour — online_only | Σ provider_earnings rows created during the phase, by Lane, scaled from the phase's 2m0s to one hour | online $0.2077/h (69 rows) |
| earnings/hour — offline_only | Σ provider_earnings rows created during the phase, by Lane, scaled from the phase's 1m32s to one hour | batch $0.0050/h (128 rows) |
| earnings/hour — coserve | Σ provider_earnings rows created during the phase, by Lane, scaled from the phase's 2m2s to one hour | batch $0.0035/h (120 rows), online $0.1999/h (68 rows) |
| earnings/hour — flex | Σ provider_earnings rows created during the phase, by Lane, scaled from the phase's 2m0s to one hour | batch $0.0017/h (58 rows) |

## Gates

- co-serving p99 ÷ online-only p99 = **0.58×**, gate < 2.0 — **PASS**
- harvest = **1.10**, gate > 0.20 — **PASS**

## Caveats

- **A harvest above 1.0 and a p99 ratio below 1.0 are noise, not a result.** This run measured the batch lane as *faster* under online load (1.136 vs 1.034 items/s) and the online tail as *better* under co-serving (127 ms vs 220 ms). Neither is physical: online traffic does not speed the batch lane up. Both arms are 59 s windows over n≈70 samples with the dispatcher's AIMD still settling and the phases minutes apart, which is exactly the spread the "Small n" and "Phases are sequential" caveats below describe. Read them as "no measurable penalty at this sample size", not as a gain.
- **One provider, one model, one box.** Nothing here says how the lane behaves across a fleet; it says the lane harvests idle capacity on a single machine without starving the online lane.
- **Debug provider build.** The Swift provider is built in debug with a hand-built `mlx.metallib`; absolute latencies and tokens/s are lower than a release build's.
- **TTFT is not observable.** The coordinator hashes and SE-signs a completion before releasing it, so it emits the whole response as one SSE frame. Every online number here is total wall time, not time-to-first-token.
- **Small n.** The schedule produces 72 arrivals per phase, so p99 is effectively the worst observed request rather than a stable tail estimate. The hand-run predecessor of this benchmark (`plans/e2e-batch-run.md` step 7, n=10 per arm) saw the online penalty land anywhere between 1.2× and 1.9× depending on the round; read a single ratio in that spread as consistent with the earlier measurement, not as a tighter result than the sample supports.
- **Phases are sequential, not simultaneous.** The baseline and the co-serving arm are minutes apart on the same box, so slow drift (thermals, other processes) is inside the ratio.
- **The offline job is cancelled, not drained.** Both batch phases stop once the measurement window closes; the ceiling and the co-serving rate are steady-state rates, not end-to-end completion times for the whole 300 items.
- **The earnings gap is the online minimum charge, not the lane multiplier.** Every request here is about 30 tokens, so its token cost rounds below `minimumChargeMicroUSD` (100 µUSD). An online request is floored up to that minimum; a batch-lane request is explicitly exempt from it and is floored only at 1 µUSD (`coordinator/payments/pricing.go`, `CalculateCostForLane`). The ~100× per-request gap in the earnings rows above is that floor, not the 0.5 `batchDiscount`. On requests large enough to price above the minimum, the ratio is the multiplier.
- **Earnings are bucketed by wall clock, not by phase membership.** A request that starts inside a phase and settles after it lands in the next bucket or none, so the per-phase row counts run a little under the requests each phase issued.
