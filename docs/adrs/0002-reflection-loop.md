# ADR 0002: Reflection loop

## Context

Reflection generates a draft answer, critiques it, and revises if the
critique flags a problem -- trading extra LLM calls for a chance to catch
a subtly-wrong first-pass answer before committing to it. Draft
generation has the same tool-use capability as ReAct (same
ACTION/FINAL_ANSWER protocol, `loops/common.py`), so any difference
against ReAct should come from the critique/revise mechanism, not from a
capability gap.

## Decision

Implemented as a three-node LangGraph `StateGraph`: `draft` (tool-capable,
same protocol as ReAct's `reason_act`), `observe` (tool execution, shared
with ReAct), and `critique` (one LLM call judging the draft: `GOOD` or
`REVISE: <feedback>`). On `REVISE`, the draft is discarded (not just
flagged) and control returns to `draft` with the feedback folded into the
next prompt, so a stale accepted draft can never leak into the next
round. Bounded by the same `max_iterations` budget as every other
pattern, shared across both draft and critique calls.

## Observed trade-offs (structural, mock-LLM run)

| metric | value | vs. ReAct |
|---|---|---|
| avg iterations/task | 3.00 | +1.00 (the critique call) |
| avg tool calls/task | 1.00 | unchanged |
| avg tokens/task | 403.3 | +59.9 (+17%) |
| avg wall-clock/task | 7.6ms | +0.4ms |

The mock critique step always returns `GOOD` (documented in
`eval/mock_agent.py`: a mock critic can't judge quality any better than
the mock drafter produced it, so simulating revision would be theater).
That means this run validates the *mechanism's fixed cost* -- one extra
LLM call every round, ~17% more tokens, for a pattern that in this run
never actually exercises its revision path. A real LLM is needed to
measure the thing Reflection is actually for: how often that extra call
turns a wrong draft into a right one, and whether that revision rate
justifies the added cost.

## Real-LLM run (Groq `llama-3.3-70b-versatile`)

That measurement has now happened, and it surfaced a real architectural
gap the mock run couldn't show -- see `docs/writeup.md`'s real-LLM
section for the full trace evidence:

| metric | value |
|---|---|
| accuracy | 83.3% (15/18) |
| avg iterations/task | 5.61 |
| avg tool calls/task | 1.22 |
| avg tokens/task | 1656.8 |
| avg wall-clock/task | 13.4s |

The critic **never once converges to `GOOD`** across all 18 tasks --
it kept finding new, increasingly pedantic objections to substantively
correct drafts ("lacks... additional context," "unclear... without
external verification"). Unbounded, that ran every task to
`max_iterations` with an empty final answer (0% accuracy on the first
pass). Fixed by `max_critique_rounds` (`loops/reflection_loop.py`,
default 3): once hit, the current draft is force-accepted instead of
revised again. The 83.3% above is the loop returning *an* answer instead
of giving up -- the critic's behavior itself didn't change, and its
revision path (the reason this pattern exists) still isn't shown to be
worth its cost on this benchmark: Reflection pays ~3x ReAct's tokens and
~10x its wall-clock for *worse* accuracy (83.3% vs. 88.9%).

## Consequences

Reflection's cost is paid on every task, not just the ones where the
critique catches something -- so it's a poor fit when the draft is
usually already correct (the overhead buys nothing) and a better fit
when first-pass answers are often subtly wrong in ways a critique
prompt can catch (e.g. arithmetic slips, unsupported claims) but a fresh
attempt from scratch could not reliably fix. Its structural weakness,
carried into Sprint 7's synthesis: nothing stops a bad critique from
revising a correct answer into a wrong one -- there's no ground truth in
the loop, only the critique's own (fallible) judgment.
