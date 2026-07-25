# ADR 0001: ReAct loop

## Context

ReAct interleaves reasoning and action: at each step the model sees the
question and the trace so far, and emits either a tool call or a final
answer. It reacts to each observation before deciding the next step,
with no separate planning or self-critique phase. It's the cheapest,
simplest control loop in this project and the baseline the other two
patterns are compared against.

## Decision

Implemented as a two-node LangGraph `StateGraph`: `reason_act` (one LLM
call producing `ACTION: <tool> | <json>` or `FINAL_ANSWER: <answer>`) and
`observe` (executes the tool through `AgentLoop._call_tool`, the shared
retry/circuit-breaker wrapper). The graph loops `reason_act -> observe ->
reason_act -> ...` until a final answer is emitted or `max_iterations` is
hit. No planning step, no critique step -- the entire control-flow
surface is these two nodes and one conditional edge.

## Observed trade-offs (structural, mock-LLM run)

Against the mock reasoning policy (`eval/mock_agent.py`) over all 18
benchmark tasks:

| metric | value |
|---|---|
| avg iterations/task | 2.00 |
| avg tool calls/task | 1.00 |
| avg tokens/task | 343.4 |
| avg wall-clock/task | 7.1ms |

This is the cheapest of the three patterns on every structural axis, as
expected -- one reasoning call, one action, one answer, nothing else.
That's the real, measured cost profile. It is **not** an accuracy
comparison: the mock policy answers identically regardless of pattern, so
accuracy is 0% uniformly across all three loops here (see
`eval/mock_agent.py`'s docstring). Only a real LLM can validate ReAct's
expected weakness -- wandering or failing to backtrack on multi-hop
questions that need combining facts across more than one search.

## Consequences

ReAct is the right default when task cost/latency matters more than
robustness to multi-hop reasoning failures, and when tasks are shallow
enough that one-step-at-a-time reactivity doesn't need lookahead. Its
lack of any self-correction mechanism is also its main structural risk
carried into Sprint 7's synthesis: a single bad tool call or a
misinterpreted observation has nothing downstream to catch it, unlike
Reflection or Plan-Execute's replan branch.
