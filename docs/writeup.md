# recur: ReAct vs. Reflection vs. Plan-Execute

A head-to-head comparison of three agentic control-loop architectures —
ReAct, Reflection, and Plan-Execute — built on one shared harness (one
`Task`/`Tool` contract, one retry/circuit-breaker/timeout layer, one
scoring function) so that any measured difference between patterns is
architecture, not implementation drift. See `docs/adrs/000{1,2,3}-*.md`
for the per-pattern design decisions and trade-offs this writeup
synthesizes.

## Read this caveat before the numbers

Every number below comes from a **real, measured run** of all three loops
against the 18-question multi-hop benchmark (`eval/qa_dataset.py`) — no
number here is estimated or invented. But the run uses a deterministic
**mock reasoning policy** (`eval/mock_agent.py`), not a real LLM, because
no live provider API key is configured in this environment. The mock
policy issues one fixed search-then-answer strategy regardless of loop
pattern, so:

- **Structural metrics (iterations, tool calls, tokens, wall-clock) are
  real and pattern-differentiated** — they reflect each pattern's actual
  control-flow shape and its real cost multiplier over ReAct.
- **Accuracy is not a meaningful signal in this run** — all three loops
  score 0% because the mock policy's naive one-sentence extraction almost
  never exact-matches the gold answers, identically across patterns. This
  says nothing about which architecture reasons better.

Wiring a real provider is one new `LLM` subclass in `harness/llm.py`; the
rest of the harness, benchmark, and comparison runner need no changes to
produce a real accuracy comparison. That is the natural next step before
treating this as a finished benchmark result rather than an architecture
and infrastructure validation.

## Comparison table (real, measured, mock-policy run)

| pattern | avg iterations/task | avg tool calls/task | avg tokens/task | avg wall-clock/task | cost vs. ReAct |
|---|---|---|---|---|---|
| **ReAct** | 2.00 | 1.00 | 343.4 | 7.1ms | 1.0x (baseline) |
| **Reflection** | 3.00 | 1.00 | 403.3 | 7.6ms | ~1.2x tokens |
| **Plan-Execute** | 5.00 | 2.00 | 921.4 | 11.7ms | ~2.7x tokens |

Reproduce with `python -m eval.run_comparison` (writes to
`sqlite:///mlflow.db`; inspect via `mlflow ui --backend-store-uri
sqlite:///mlflow.db`).

## What this run validates

1. **The shared-harness principle holds.** All three loops route tool
   calls through the same retry/circuit-breaker wrapper, use the same 18
   tasks, and are scored identically. The cost differences above are
   attributable to control-flow shape alone.
2. **Each pattern's cost scales the way its design predicts.** Reflection
   adds one fixed critique call per round (+17% tokens). Plan-Execute's
   cost scales with plan length — two steps means every step gets its own
   act/observe exchange (2.7x tokens). ReAct pays for exactly what it
   uses, once.
3. **Reliability is centralized, not per-pattern** (Sprint 5): all three
   loops inherit the same checkpoint-resume-on-crash and circuit-breaker
   behavior from `harness/reliability.py`, verified directly in
   `tests/test_reliability.py` rather than reimplemented per loop.
4. **The replan-on-failure path is implemented and tested**
   (`tests/test_plan_execute_loop.py`), but not exercised by this
   particular benchmark sweep, since the mock policy never triggers a
   tool failure that would prompt a `REPLAN`.

## Recommendation, by task shape (architectural reasoning, pending real-LLM validation)

- **Shallow, single-hop lookups:** ReAct. Paying for a critique step or an
  upfront plan buys nothing when one search answers the question, and
  ReAct's cost profile is the cheapest of the three on every axis
  measured here.
- **Tasks where a first-pass answer is often subtly wrong in a
  catchable way** (arithmetic slips, unsupported claims, missing a
  qualifier) but a plan wouldn't help avoid the mistake in the first
  place: Reflection. The fixed one-extra-call cost is worth it exactly
  when critique quality is good enough to catch more errors than it
  introduces — a claim this run cannot verify without a real LLM, since
  the mock critique never revises.
- **Genuinely multi-hop tasks needing dependent steps** (the kind
  `eval/qa_dataset.py` tags `expected_hops >= 2`, 12 of the 18 questions):
  Plan-Execute, on the hypothesis that an upfront plan avoids the
  step-by-step wandering ReAct is prone to on tasks needing lookahead.
  This is the pattern's strongest theoretical case and also its highest
  structural cost (2.7x tokens) — worth it only if its accuracy on
  exactly these multi-hop-tagged questions is measurably better than
  ReAct's, which requires a real LLM to test.

**Bottom line:** this project's infrastructure — harness, benchmark,
reliability layer, and comparison tooling — is complete and validated.
The interview-defensible claim today is architectural and cost-based
("Plan-Execute costs 2.7x ReAct in tokens because its cost scales with
plan length, not because it's inherently 'better'"), not yet an accuracy
claim. Wiring a real LLM into `harness/llm.py` and re-running
`eval/run_comparison.py` is the next step to make the accuracy half of
this comparison as defensible as the cost half already is.
