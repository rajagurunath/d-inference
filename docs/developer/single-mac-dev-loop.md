# Single-Mac dev loop

> Last updated: 2026-09-04 · commit `da6db27f2`

Run a coordinator and one provider on one Mac for manual testing, without
touching the shared dev environment or writing a throwaway launch script. For
developers working on coordinator or provider changes who want a live
`http://127.0.0.1:18080` endpoint to `curl` or point an SDK at.

## Prerequisites

- The one-time host setup from the [tidal batch lane
  plan](../design/tidal-batch-lane.md): `mise install` (Go `1.25.0`, Swift
  `6.3`), `git submodule update --init --recursive`, and a CBv2-servable model
  in the Hugging Face cache (`mlx-community/gemma-4-e2b-it-4bit`, or the
  fallback `mlx-community/Qwen3.5-0.8B-MLX-4bit`).
- Either the in-memory store (default, no setup) or a running Postgres for
  `--postgres` — see [`test.md`](test.md) for how the harness starts one.
- A built provider binary, or let the stack build one from source the first
  time (`e2e/testbed/provider.go`, `BuildProvider`) — slower, and needs the
  Swift toolchain and `libs/mlx-swift`/`libs/mlx-swift-lm` submodules.

## Steps

### 1. Start the stack

```bash
make dev-stack
```

This runs `e2e/cmd/devstack/main.go`, which builds a `testbed.SuiteConfig`
pinned to `127.0.0.1:18080` (`SuiteConfig.ListenAddr`) and drives it through
the same `testbed.NewSuite` / `Suite.Start` lifecycle the e2e integration
tests use — never a hand-rolled coordinator or provider launch. It prints the
base URL, a test API key, and the provider's PID, then blocks until you press
Ctrl-C.

To use a prebuilt provider binary instead of a from-source build (recommended
— a from-source build can take minutes):

```bash
DEVSTACK_ARGS="--provider-binary /path/to/darkbloom" make dev-stack
```

Other flags: `--model <id>` (default: `DARKBLOOM_TESTBED_MODEL` env, else the
testbed default) and `--postgres` (use a real Postgres store instead of
memory; the harness provisions its own ephemeral instance via Docker or a
native `initdb`/`postgres` — `e2e/testbed/deps/postgres.go` — it does not
attach to an already-running server).

### 2. Run the smoke script

In a second terminal, using the API key printed in step 1:

```bash
DARKBLOOM_DEV_KEY=<the printed key> \
DARKBLOOM_TESTBED_MODEL=mlx-community/gemma-4-e2b-it-4bit \
scripts/dev-smoke-batch.sh
```

The script POSTs a streaming chat completion and exits 0 once an SSE `data:`
line arrives, or 1 after 60 seconds with no content. `DARKBLOOM_DEV_URL`
defaults to `http://127.0.0.1:18080` and only needs overriding if you passed
a different `--postgres`/`ListenAddr` setup.

### 3. Stop the stack

Press Ctrl-C in the `make dev-stack` terminal. The command waits on SIGINT,
then calls the suite's `Stop()`, which shuts the provider process and the
coordinator's HTTP server down cleanly.

## Logs

Both the coordinator and the provider log to the `make dev-stack` terminal's
stderr (`slog` text handler) — there is no separate log file. Provider stdout
and stderr are prefixed `provider:stdout` / `provider:stderr`
(`e2e/testbed/suite.go`, `logWriter`).

## Verify

- `make dev-stack` prints `Dev stack ready.` with a base URL, an API key, and
  at least one `Provider PID:` line.
- `scripts/dev-smoke-batch.sh` prints `OK: first SSE data line arrived after
  <N>s` and exits 0.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `dev stack failed to start` | Provider never registered, or checkpoint not CBv2-servable | Check `DARKBLOOM_TESTBED_MODEL` is in the HF cache and is a `gpt_oss`/`gemma4` family checkpoint |
| smoke script prints `no SSE 'data:' line within 60s` | Provider is slow to load a large checkpoint, or the wrong model was requested | Re-run with more patience on first load, or confirm the smoke script's `DARKBLOOM_TESTBED_MODEL` matches what `make dev-stack` served |
| `DARKBLOOM_DEV_KEY not set` | Forgot to copy the key `make dev-stack` printed | Copy it from the first terminal's startup output |

## Related

- [`test.md`](test.md): the full e2e suite, its environment variables, and
  what CI runs.
- [`build.md`](build.md): building each component from a fresh clone.
- [`../design/tidal-batch-lane.md`](../design/tidal-batch-lane.md): the design
  this dev loop was built to support.
