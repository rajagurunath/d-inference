# Single-Mac dev loop

> Last updated: 2026-09-05 · commit `897e32333`

Run a coordinator and one provider on one Mac for manual testing, without
touching the shared dev environment or writing a throwaway launch script. For
developers working on coordinator or provider changes who want a live
`http://127.0.0.1:18080` endpoint to `curl` or point an SDK at.

Skip to the [Morning checklist](#morning-checklist) if the box is already set
up.

## Prerequisites

- Xcode selected without `sudo`:

  ```bash
  export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
  ```

- Swift 6.3 (the provider needs it; a stock Xcode 16.4 ships 6.1) and Go
  `1.25.0`, both through mise:

  ```bash
  mise install swift
  mise install
  ```

- Submodules: `git submodule update --init --recursive` (about 1.5 GB).
- **Strip anaconda from `PATH` before anything Swift.** Anaconda's `cctools`
  linker shadows the real one and the provider fails to link:

  ```bash
  export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v anaconda | paste -sd: -)
  ```

- A CBv2-servable model in the Hugging Face cache. Use
  **`mlx-community/Qwen3.5-0.8B-MLX-4bit`** (`hf download
  mlx-community/Qwen3.5-0.8B-MLX-4bit`, 622 MB) unless the box is quiet — see
  [Pick a model that fits](#pick-a-model-that-fits).
- Either the in-memory store (default, no setup) or a running Postgres for
  `--postgres` — see [`test.md`](test.md) for how the harness starts one.
- A built provider binary, or let the stack build one from source the first
  time (`e2e/testbed/provider.go`, `BuildProvider`) — slower.

### Pick a model that fits

The provider refuses a model at load time on a **memory** verdict, not a
kernel one, and the gate counts only `free + inactive` pages
(`provider-swift/Sources/ProviderCore/Inference/SystemMemory.swift`), so a
48 GB Mac with a busy browser fails for a 3.3 GB model. Required is
`padded weights (disk × 1.2) + 5.5 GiB activation floor + 1 GiB minimum KV`
(`provider-swift/Sources/ProviderCore/Inference/ModelLoadAdmission.swift`).

| Model | Required | Verdict on a 48 GB M4 Pro with Chrome and VS Code open |
|---|---|---|
| `mlx-community/Qwen3.5-0.8B-MLX-4bit` | ≈ 7.1 GiB | loads (usable was 7.3–9.1 GiB) |
| `mlx-community/gemma-4-e2b-it-4bit` | ≈ 10.5 GiB | refused — and it needs ≈ 15.5 GiB of `free + inactive` in practice, because the load gate re-samples *after* the weight hash reads 3.3 GB into the page cache |

Check before you start:

```bash
darkbloom doctor | grep 'model fits in RAM'   # must be [PASS]
vm_stat | awk '/Pages (free|speculative|inactive)/{s+=$NF} END{printf "%.1f GiB free+inactive\n", s*16384/2^30}'
```

`DARKBLOOM_MEM_CAP_FRACTION=1.0` buys about 2.8 GiB (reserve 4.8 → 2.0 GiB, a
hard floor). `DARKBLOOM_ACTIVATION_RESERVE_GB` is **raise-only** and cannot
lower the 5.5 GiB activation floor. Nothing else helps: quit Chrome.

## Steps

### 1. Start the stack

```bash
export DARKBLOOM_PROVIDER_BINARY=$PWD/provider-swift/.build/arm64-apple-macosx/debug/darkbloom
export DARKBLOOM_TESTBED_MODEL=mlx-community/Qwen3.5-0.8B-MLX-4bit
make dev-stack
```

This runs `e2e/cmd/devstack/main.go`, which builds a `testbed.SuiteConfig`
pinned to `127.0.0.1:18080` (`SuiteConfig.ListenAddr`) and drives it through
the same `testbed.NewSuite` / `Suite.Start` lifecycle the e2e integration
tests use — never a hand-rolled coordinator or provider launch. It prints the
base URL, a test API key, and the provider's PID, then blocks until you press
Ctrl-C.

Other flags, via `DEVSTACK_ARGS`: `--provider-binary <path>` (skip the
from-source build), `--model <id>` (default: `DARKBLOOM_TESTBED_MODEL`),
`--postgres` (a real Postgres store instead of memory — see below), and
`--api-key <sk-db-…>` (default: `DARKBLOOM_DEV_KEY`; reuse a key the store
already holds instead of minting one).

`--postgres` has two modes, chosen by the first of these that is set:

| Variable (read only under `--postgres`) | Store | Lifecycle |
|---|---|---|
| neither (default) | an ephemeral instance the harness provisions via Docker or a native `initdb`/`postgres` (`e2e/testbed/deps/postgres.go`) | created at start, **removed** at stop — batches, items and keys are gone |
| `DARKBLOOM_DEVSTACK_DATABASE_URL` | the database it names | the stack neither creates nor drops it; migrations run on connect, and every row survives a stop |
| `EIGENINFERENCE_DATABASE_URL` | same, second choice | same |

Prefer `DARKBLOOM_DEVSTACK_DATABASE_URL`. `EIGENINFERENCE_DATABASE_URL` is
accepted because it is the name a coordinator already reads, but the testbed
*writes* that variable when it provisions an ephemeral instance
(`deps.PostgresLifecycle.SetEnv`), so it is not a reliable statement of intent
inside a process that makes databases of its own. Resolution therefore happens
in `e2e/cmd/devstack/main.go`; `testbed.SuiteConfig.DatabaseURL` has no
environment fallback at all, and the startup log names the variable it used.

Only the persistent modes make restart resilience observable, because all
three of the database, the API key and the batch seal key have to survive
together — see [Restart test](#restart-test).

### 2. Run the smoke script

In a second terminal, using the API key printed in step 1:

```bash
DARKBLOOM_DEV_KEY=<the printed key> \
DARKBLOOM_TESTBED_MODEL=mlx-community/Qwen3.5-0.8B-MLX-4bit \
scripts/dev-smoke-chat.sh
```

The script POSTs a streaming chat completion and exits 0 once an SSE `data:`
line arrives, or 1 after 60 seconds with no content. `DARKBLOOM_DEV_URL`
defaults to `http://127.0.0.1:18080`.

### 3. Stop the stack

Ctrl-C in the `make dev-stack` terminal. Non-interactively, signal the **Go
binary**, not `make`: `make dev-stack` runs it under `/bin/sh -c`, so a SIGINT
to the `make` or `sh` PID never reaches it and leaves the provider and port
18080 held.

```bash
kill -INT "$(lsof -t -nP -iTCP:18080 -sTCP:LISTEN)"
```

Signal the listener, not a name: `pgrep -f 'exe/devstack'` matches only the
**first** `go run` of a session. Afterwards Go reuses its cached executable
and the process is `…/Library/Caches/go-build/…/devstack`, which that pattern
misses — SIGINT then goes nowhere and port 18080 stays held.

A clean stop prints `provider stopped` and `dev stack stopped` and releases
port 18080. The provider's config is written into a per-instance temp state
dir (`provider config written path=/var/folders/…/darkbloom-testbed-state-0-…/provider.toml`),
so nothing under `~/.config/darkbloom/` is created or removed by a dev-stack
run.

## Morning checklist

Four terminals' worth of commands, in order, with what each should print.

```bash
# 0. Environment (every terminal that touches Swift)
cd ~/ionet/oss/d-inference
export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer
export PATH=$(echo "$PATH" | tr ':' '\n' | grep -v anaconda | paste -sd: -)
export DARKBLOOM_TESTBED_MODEL=mlx-community/Qwen3.5-0.8B-MLX-4bit
export DARKBLOOM_PROVIDER_BINARY=$PWD/provider-swift/.build/arm64-apple-macosx/debug/darkbloom

# 1. Terminal A — bring the stack up (batch env vars included, see the note below)
export EIGENINFERENCE_BATCH_DEV_INSECURE_KEY=true
export EIGENINFERENCE_BATCH_BLOB_DIR=$TMPDIR/darkbloom-batch
make dev-stack
```

Expect, within about 15 seconds:

```
level=INFO msg="test coordinator started" port=18080 base_url=http://127.0.0.1:18080
level=INFO msg="provider registered" chip="Apple M4 Pro" memory_gb=48 models=1 backend=mlx-swift
level=INFO msg="providers registered" count=1
Dev stack ready.
  Base URL: http://127.0.0.1:18080
  Provider PID: <pid> (model: mlx-community/Qwen3.5-0.8B-MLX-4bit)
```

```bash
# 2. Terminal B — chat path
export DARKBLOOM_DEV_KEY=<the key terminal A printed>
./scripts/dev-smoke-chat.sh
```

Expect exit 0 and:

```
OK: first SSE data line arrived after 4s (model: mlx-community/Qwen3.5-0.8B-MLX-4bit)
```

```bash
# 3. Terminal B — batch API path
COORD=http://127.0.0.1:18080 API_KEY=$DARKBLOOM_DEV_KEY \
MODEL=$DARKBLOOM_TESTBED_MODEL ./scripts/dev-smoke-batch-api.sh
```

Expect:

```
--- warm the model with one synchronous completion ---
model warm
--- POST /v1/files ---
uploaded file-…
--- POST /v1/batches ---
created batch_…
--- GET /v1/batches/batch_… (poll to completed, up to 180s) ---
  status=in_progress completed=0 failed=0 total=3
  status=in_progress completed=1 failed=0 total=3
  status=completed completed=3 failed=0 total=3
batch batch_… completed with 3 of 3 requests
batch API smoke PASSED
```

The warm-up step is load-bearing. The dispatcher claims items only for a model
some provider slot has headroom for, and a provider that has never served the
model has no slot for it — so a batch created against a freshly started stack
sits at `in_progress` indefinitely with nothing in the log. One ordinary chat
completion loads the model and the next dispatcher tick starts claiming.

**`make dev-stack` wires the batch lane, but only if the two
`EIGENINFERENCE_BATCH_*` variables are exported before you start it.**
`e2e/testbed/suite.go` builds the sealed blob store from
`api.ReadBatchConfig()` and starts the dispatcher
(`startTestbedBatchDispatcher`) exactly the way
`coordinator/cmd/coordinator/batch_lane.go` does for the production binary —
but `api.NewBatchBlobStore` returns `(nil, nil)` when there is neither a
mnemonic nor `EIGENINFERENCE_BATCH_DEV_INSECURE_KEY`, and then every batch
route answers `503 batch_unavailable` and the script aborts at
`POST /v1/files`. The variables are read once at startup, so exporting them
after `make dev-stack` is running has no effect: stop the stack, export, start
again. Startup logs `batch lane enabled blob_dir=…` and
`batch lane dispatcher started` when it worked.

Batch environment (exported before `make dev-stack`, and for any coordinator
you start yourself):

| Variable | Dev value | Why |
|---|---|---|
| `EIGENINFERENCE_BATCH_DEV_INSECURE_KEY` | `true` | No mnemonic locally, so the lane would otherwise refuse to start. Blobs become unreadable at restart; that is fine for a dev loop, but leave it unset for the [Restart test](#restart-test) |
| `EIGENINFERENCE_BATCH_BLOB_DIR` | `$TMPDIR/darkbloom-batch` | The default is `/mnt/disks/userdata/batch`, which does not exist on a Mac |
| `EIGENINFERENCE_BATCH_LANE_ENABLED` | unset (`true`) | Set `false` to keep the API serving while nothing dispatches. Read only by `coordinator/cmd/coordinator/batch_lane.go`; `make dev-stack` starts its dispatcher unconditionally once the blob store exists |

Full descriptions:
[`../reference/configuration.md#batch-lane`](../reference/configuration.md#batch-lane).

## Restart test

The batch lane requeues items a previous process left in flight
(`Dispatcher.requeueAfterRestart`). Reaching that path from this loop needs
three things to outlive the process — the rows, the key that addresses them,
and the key the blobs are sealed with — so the recipe sets all three:

```bash
createdb dinf_devstack        # once
export DARKBLOOM_DEVSTACK_DATABASE_URL='postgres://localhost:5432/dinf_devstack?sslmode=disable'
export EIGENINFERENCE_BATCH_BLOB_DIR=/tmp/dinf-devstack-blobs   # a FIXED dir, not mktemp -d
unset EIGENINFERENCE_BATCH_DEV_INSECURE_KEY                     # would seal with a random key
export DEVSTACK_ARGS=--postgres
make dev-stack
```

The stack logs `persistent postgres store source=DARKBLOOM_DEVSTACK_DATABASE_URL`
and `using persistent postgres store (not provisioned or dropped by the
testbed)` and, because no `MNEMONIC` is set, a WARN that it is sealing with
the fixed dev mnemonic — a publicly known BIP39 test vector, chosen so the seal
key is *derived* (the production path, `sealedblob.DeriveKey`) instead of
random. Set `MNEMONIC` yourself to override it; never run a deployment on the
dev value.

Start a batch, let some items settle, then stop and restart with the key the
first run printed:

```bash
# terminal B: upload a 30-line JSONL, POST /v1/batches, poll until completed > 0
kill -INT "$(lsof -t -nP -iTCP:18080 -sTCP:LISTEN)"

# terminal A: same environment, plus the key from the first run's banner
DARKBLOOM_DEV_KEY=<the key terminal A printed> make dev-stack
```

The second start logs the resume and reuses the key:

```
level=INFO msg="reusing configured API key" account_id=testbed-user-0
level=INFO msg="batch lane: requeued items left inflight by a previous process" batches=1 items=3
```

`GET /v1/batches/{id}` with the **old** key answers 200 and the batch runs to
`completed` with every item, including the ones whose results were sealed by
the previous process. Warm the model again first — a restarted provider holds
no slot for it, so the dispatcher claims nothing until one ordinary completion
loads it.

The key can be reused but not chosen: the store generates its own secrets and
persists only a hash, so `--api-key` adopts a key the store already holds and
otherwise WARNs and mints a fresh one. Take the key from the first run's
banner.

Clean up with `dropdb dinf_devstack` and `rm -rf /tmp/dinf-devstack-blobs`;
with neither URL variable set the stack is back to the ephemeral store that is
thrown away on every stop.

## Measured on this loop

2026-09-05, M4 Pro / 48 GB / macOS 15.7, debug provider build with a
hand-built `mlx.metallib`, model `mlx-community/Qwen3.5-0.8B-MLX-4bit`:

| Metric | Value |
|---|---|
| provider spawn → `providers registered` | 14.0 s |
| weight hash (10 files) | 0.5 s |
| `dev-smoke-chat.sh` first SSE `data:` line | 4 s (cold, includes the lazy model load) |
| first cold chat completion | 3564 ms for 27 completion tokens |
| warm streaming, 81 completion tokens | 1273–1280 ms → 62–64 tok/s |
| warm streaming, 128 completion tokens | 1989 ms → 64.4 tok/s |

The coordinator emits the whole completion as **one** content SSE frame (the
response is hashed and SE-signed before release), so per-token TTFT is not
observable at `127.0.0.1:18080`. Use the totals above, or measure inside the
provider.

## Logs

Both the coordinator and the provider log to the `make dev-stack` terminal's
stderr (`slog` text handler) — there is no separate log file. Provider stdout
and stderr are prefixed `provider:stdout` / `provider:stderr`
(`e2e/testbed/suite.go`, `logWriter`).

`darkbloom start --local` writes no daemon state file, and the provider
sanitises inference failures before logging them, so the real reason never
reaches stdout *or* `log stream` (which shows `<private>` without
`sudo log config --mode private_data:on`). `darkbloom doctor` is the only
operator-facing surface for the model-load verdict.

## Verify

- `make dev-stack` prints `Dev stack ready.` with a base URL, an API key, and
  at least one `Provider PID:` line.
- `scripts/dev-smoke-chat.sh` prints `OK: first SSE data line arrived after
  <N>s` and exits 0.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `rate_limit_exceeded` / `all providers at capacity`, provider at 0% CPU, 1–3 ms round trip | The model-load memory admission gate refused before any weight load. It is a memory verdict, not a kernel or metallib one | `darkbloom doctor \| grep 'model fits in RAM'`; free RAM or use a smaller model. `DARKBLOOM_MEM_CAP_FRACTION=1.0` buys ~2.8 GiB and nothing else does |
| Provider fails to link, `ld` errors from `cctools` | Anaconda's linker is ahead of the real one on `PATH` | Strip anaconda from `PATH` (see Prerequisites) |
| `xcode-select` / Metal build errors, and you have no `sudo` | The toolchain is pointed at the CLI tools, not Xcode | `export DEVELOPER_DIR=/Applications/Xcode.app/Contents/Developer` |
| `dev stack failed to start` | Provider never registered, or the checkpoint is not CBv2-servable | Check `DARKBLOOM_TESTBED_MODEL` is in the HF cache and its family is CBv2-supported (`qwen3_5` is) |
| smoke script prints `no SSE 'data:' line within 60s` | Slow first load, or the wrong model was requested | Re-run; confirm the script's `DARKBLOOM_TESTBED_MODEL` matches what `make dev-stack` served |
| `DARKBLOOM_DEV_KEY not set` | Forgot to copy the key `make dev-stack` printed | Copy it from terminal A's startup output |
| Port 18080 still held after you stopped the stack | SIGINT went to `make`/`sh`, not the Go binary — or to a `pgrep -f 'exe/devstack'` pattern that matched nothing because Go reused its cached executable | `kill -INT "$(lsof -t -nP -iTCP:18080 -sTCP:LISTEN)"` |
| A batch stays at `in_progress` with `completed: 0` and nothing in the log | The model is not resident on any provider, so no slot has batch headroom and the dispatcher claims nothing | Serve one ordinary chat completion for that model first (`dev-smoke-batch-api.sh` now does this itself) |
| Every `/v1/files` or `/v1/batches` call returns `503 batch_unavailable` | The coordinator has no batch blob store: `EIGENINFERENCE_BATCH_DEV_INSECURE_KEY` / `EIGENINFERENCE_BATCH_BLOB_DIR` were not exported *before* the stack started | Stop the stack, export both, start again; confirm `batch lane enabled` in the startup log |

## Related

- [`test.md`](test.md): the full e2e suite, its environment variables, and
  what CI runs.
- [`build.md`](build.md): building each component from a fresh clone.
- [`../architecture/hardware-support.md`](../architecture/hardware-support.md):
  the memory model the load gate implements.
- [`../architecture/batch-lane.md`](../architecture/batch-lane.md): what the
  batch smoke script exercises.
- [`../design/tidal-batch-lane.md`](../design/tidal-batch-lane.md): the design
  this dev loop was built to support.
