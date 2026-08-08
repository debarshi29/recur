# ADR 0003: Plan-Execute loop

## Context

Plan-Execute plans the full step sequence up front, executes each step,
and replans -- discarding the rest of the stale plan rather than
restarting the whole run -- when a step is judged to have failed. It's
the pattern expected to handle genuinely multi-hop tasks best (the plan
gives it lookahead ReAct doesn't have) at the highest structural cost of
the three, with a distinct failure mode: a planner can hallucinate
infeasible steps that no amount of execution effort will complete.

## Decision

Implemented as a five-node LangGraph `StateGraph`: `plan` (emits
`PLAN: step1 | step2 | ...` or a direct `FINAL_ANSWER`), `execute` (per
current step, same tool-capable ACTION/FINAL_ANSWER/STEP_DONE/REPLAN
protocol), `observe` (shared tool execution), `replan` (triggered by an
explicit `REPLAN: <reason>` from `execute`; produces a new plan that
replaces the remainder of the stale one, rather than restarting from
`plan`), and `synthesize` (produces a final answer once the plan is
exhausted without one). Replanning is model-driven, not automatic on
every tool failure -- the model sees the failed observation and decides
whether the step (not just the tool call) needs to change, which is
closer to how a planner would actually recover in production.

## Observed trade-offs (structural, mock-LLM run)

| metric | value | vs. ReAct |
|---|---|---|
| avg iterations/task | 5.00 | +3.00 (2.5x) |
| avg tool calls/task | 2.00 | +1.00 (2x) |
| avg tokens/task | 921.4 | +578.0 (2.7x) |
| avg wall-clock/task | 11.7ms | +4.6ms |

This is the most expensive pattern on every axis under the mock policy's
fixed two-step plan (search, then answer) -- each step gets its own
act/observe exchange, so cost compounds with plan length in a way ReAct's
single-shot loop and Reflection's fixed one-extra-call don't. The mock
policy never emits `REPLAN` itself (see `eval/mock_agent.py`), so this
run validates the plan/execute/synthesize mechanism and its baseline
cost, not the replan-on-failure recovery path -- that path is covered
directly in `tests/test_plan_execute_loop.py::test_plan_execute_replans_after_a_failed_step`
via a scripted LLM, but not exercised in the benchmark sweep.

## Real-LLM run (Groq `llama-3.3-70b-versatile`)

The theoretical case for Plan-Execute -- that an upfront plan avoids
ReAct's step-by-step wandering on genuinely multi-hop tasks -- has now
been measured directly, not just structurally. See `docs/writeup.md`'s
real-LLM section for the full run:

| metric | value |
|---|---|
| accuracy | 72.2% (13/18) |
| avg iterations/task | 4.39 |
| avg tool calls/task | 2.22 |
| avg tokens/task | 1642.7 |
| avg wall-clock/task | 10.5s |

That theoretical case isn't borne out on this benchmark: Plan-Execute is
the least accurate of the three patterns (72.2% vs. ReAct's 88.9% and
Reflection's 83.3%) despite paying the highest tool-call cost. This is
now measured under `paraphrase_scorer` (not the original overly-strict
`exact_match_scorer` that depressed the first pass's numbers uniformly),
so the gap is a real signal to investigate rather than a scoring
confound waiting to be ruled out -- see `docs/writeup.md`'s "What the
first real-LLM pass found" for that history. As with `0001`/`0002`, 18
tasks is still a small sample and the replan-on-failure path still isn't
exercised by either sweep, so this doesn't rule out Plan-Execute winning
on a benchmark with deeper multi-hop dependencies or noisier tool
failures than this corpus has.

## Consequences

Plan-Execute's cost scales with plan length, so it's the wrong choice for
shallow, single-hop questions (ReAct answers them for a fraction of the
cost) and the right choice for tasks that genuinely need multiple
dependent steps where an upfront plan avoids ReAct's step-by-step
wandering. Its structural risk, carried into Sprint 7's synthesis: an
infeasible or wrong initial plan is a single point of failure unless the
model recognizes a step has failed (not just that a tool call errored)
and actually invokes `REPLAN` -- silent execution of a bad plan to
completion, with `synthesize` papering over it, is the failure mode a
real-LLM eval needs to specifically probe for.
