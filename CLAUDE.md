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
harness/       shared contracts: Task, Tool, AgentLoop, ToolResult, TaskResult
loops/         one file per pattern: react_loop.py, reflection_loop.py, plan_execute_loop.py
eval/          QA dataset + comparison runner
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

### Loop patterns (`loops/`)

Each pattern is one file, implemented as a LangGraph `StateGraph` with
conditional edges for routing (continue looping / replan / terminate).
Reliability behavior (retries, and later checkpointing/circuit breakers)
lives in the shared harness layer, not duplicated per pattern — a loop
file should only contain the reasoning/control-flow logic specific to
that pattern.

### Reference test suite (`tests/test_contracts.py`)

Any reimplementation or extension of the harness contract must keep this
file green. It covers: tool success/failure wrapping, retry-until-success
on a flaky tool, giving up after max retries, unknown-tool handling
(returns an error `ToolResult`, never raises), exact-match scorer
(case-insensitive, whitespace-trimmed), and `Task` field defaults.

## Delivery plan status

Sprint 0 (harness & contract) is done — see `tests/test_contracts.py`
(10 passing tests). Sprints 1-8 (benchmark, ReAct/Reflection/Plan-Execute
loops, reliability layer, service layer, comparison writeup, polish) are
tracked but not yet started. Each sprint has an explicit Definition of
Done; scope that doesn't fit rolls into the next sprint rather than
silently expanding the current one.
