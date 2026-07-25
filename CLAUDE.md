# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

**recur** — a production-grade, head-to-head comparison of three agentic
control-loop architectures (**ReAct**, **Reflection**, **Plan-Execute**),
built on LangGraph and evaluated against the same multi-hop research/QA
benchmark. Dual purpose: portfolio piece + interview-prep material (ADRs
and writeup must be defensible out loud in a system-design interview).

**Core design principle:** all three loop patterns run through one shared
harness — one `Task` definition, one `Tool` interface, one retry/backoff
policy, one scoring function. The only variable under test is the
control-loop architecture itself. If a loop-specific hack is tempting, it
almost always belongs in the harness instead, made generic. Never let a
pattern silently drift onto its own tools, scoring, or retry logic.

## Resolved project decisions

- **Benchmark domain:** GNN / mechanistic-interpretability papers —
  multi-hop questions require cross-referencing across papers/findings.
- **Search backend:** fixed local corpus (not a live API) — prioritizes
  fully reproducible evals over production realism.
- **Project name:** recur (final).

## Commands

Environment is managed with `uv`, venv at `.venv`.

```bash
# create/refresh the venv
uv venv .venv

# install deps (runtime + dev)
uv pip install --python .venv -r requirements-dev.txt

# run the full test suite
.venv/Scripts/python -m pytest -q

# run a single test file / test
.venv/Scripts/python -m pytest tests/test_contracts.py -q
.venv/Scripts/python -m pytest tests/test_contracts.py::test_call_tool_retries_until_success -q

# lint
.venv/Scripts/python -m ruff check .
```

On Windows the venv's Python is at `.venv/Scripts/python`, not
`.venv/bin/python`.

## Architecture

```
harness/       shared contracts: Task, Tool, AgentLoop, ToolResult, TaskResult, reliability
loops/         one file per pattern: react_loop.py, reflection_loop.py, plan_execute_loop.py, common.py
eval/          QA dataset, corpus, comparison runner, mock LLM policy, CI regression gate
service/       FastAPI service: submit a task, poll for its result
tests/         unit + integration tests, run in CI on every PR
docs/adrs/     one ADR per loop pattern documenting observed trade-offs
```

### `harness/contracts.py` — the seam everything is built against

- `Tool` — abstract base; subclasses implement `_run(**kwargs)`. The
  public `run()` wraps `_run` with timing and exception isolation: **a
  tool call never raises out of a loop** — failures come back as a
  `ToolResult(ok=False, error=...)` so the loop can reason about them as
  an observation rather than crashing.
- `AgentLoop` — abstract base every pattern subclasses. Its `_call_tool`
  method is the single retry/backoff wrapper all loops must route tool
  calls through (`LoopRunConfig.tool_max_retries` /
  `tool_backoff_base_s`, exponential). Do not reimplement retry logic
  inside an individual loop — extend `_call_tool`/`LoopRunConfig` instead.
- `Task` / `TaskResult` — the fixed input/output shape for every loop run.
  `ScoringFn` is a pluggable `(predicted, gold) -> (correct, score)`
  signature; `exact_match_scorer` is the default but partial-credit
  scorers should conform to the same signature.

### `harness/tools.py`

Concrete `Tool` implementations (`WebSearchTool`, `CalculatorTool`,
`ScratchpadTool`). Loops depend only on the `Tool` interface, never on
these concrete classes directly — new tools should live here and be
injected into loops via the `tools: list[Tool]` constructor argument.
`WebSearchTool` takes a `backend` callable; per the resolved decision
above, wire it to a fixed local corpus, not a live search API.

### `harness/tracker.py`

MLflow logging layer, kept separate from loop logic so instrumentation
doesn't leak into control-flow code. `track_run` context-manages a single
run; `log_result` logs a `TaskResult`'s metrics + trace artifact;
`log_comparison_summary` aggregates `TaskResult`s by `loop_name` into one
cross-pattern comparison artifact — call once after a full eval sweep,
not per-task.

### `harness/reliability.py`

Shared reliability primitives every loop inherits unchanged: `new_checkpointer()`
(LangGraph `MemorySaver`), `CircuitBreaker` (per-tool consecutive-failure
counter — `AgentLoop._call_tool` consults it before every call and
short-circuits after `LoopRunConfig.circuit_breaker_threshold` whole-call
failures in a row), `Timeout` (wall-clock deadline checked between graph
steps, not preemptive mid-node), and `run_graph()` — the single helper
every loop's `run()` calls instead of `self._graph.invoke(...)`, so
checkpoint-resume-on-crash and timeout enforcement are identical across
patterns. Extend reliability behavior here, never inside one loop file.

### Loop patterns (`loops/`)

Each pattern is one file, implemented as a LangGraph `StateGraph` with
conditional edges for routing (continue looping / replan / terminate),
compiled with a checkpointer, and run via `harness.reliability.run_graph`.
`loops/common.py` holds the ACTION/FINAL_ANSWER tool-use protocol
(prompt formatting, action parsing) shared by every pattern that lets the
LLM call a tool mid-reasoning — extend it there, not per-pattern, if the
protocol needs to change. Reliability behavior lives in the shared
harness layer, not duplicated per pattern — a loop file should only
contain the reasoning/control-flow logic specific to that pattern.

### `eval/mock_agent.py`

A deterministic, stateless (prompt-in, text-out) mock reasoning policy
shared by all three loops, used because no live LLM API key is configured
in this environment (see `harness/llm.py`'s `LLM`/`MockLLM`/`ScriptedLLM`).
It issues one search and a naive one-sentence extraction as the final
answer — **not a real reasoning policy**. It exists to validate loop
*mechanism* (control flow, tool calls, termination) end-to-end, not
accuracy — `eval/check_regression.py`'s CI gate checks for crash-free
completion, not an accuracy floor, for the same reason. Wiring a real
provider is one new `LLM` subclass in `harness/llm.py`; only then does a
real accuracy comparison (and a real accuracy-floor CI gate) become
meaningful.

### `service/` — FastAPI service

`service/app.py` exposes `POST /tasks` (submit a question + loop name,
returns a `job_id` immediately) and `GET /tasks/{job_id}` (poll status/
result). `service/jobs.py`'s `JobStore` runs each loop in a background
thread so a multi-iteration run never blocks the submitting request.
Dockerfile builds and serves it on port 8000 (`docker build -t recur .`
then `docker run -p 8000:8000 recur`).

### Reference test suite (`tests/test_contracts.py`)

Any reimplementation or extension of the harness contract must keep this
file green. It covers: tool success/failure wrapping, retry-until-success
on a flaky tool, giving up after max retries, unknown-tool handling
(returns an error `ToolResult`, never raises), exact-match scorer
(case-insensitive, whitespace-trimmed), and `Task` field defaults.

## Delivery plan status

All 8 sprints are done: shared harness + reliability layer, the
18-question multi-hop benchmark, all three loop patterns, the FastAPI
service, and the comparison writeup + per-pattern ADRs (`docs/writeup.md`,
`docs/adrs/`) — see `tests/` (66+ passing tests) and
`.github/workflows/ci.yml` (runs the test suite plus
`eval/check_regression.py` on every push/PR to `main`). The main tracked
follow-up is wiring a real LLM into `harness/llm.py` so accuracy (not just
structural cost) becomes a meaningful cross-pattern signal — see
`docs/writeup.md`'s caveat. Each sprint had an explicit Definition of
Done; keep that discipline for any future work here rather than letting
scope silently expand.
