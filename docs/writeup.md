# recur: ReAct vs. Reflection vs. Plan-Execute

A head-to-head comparison of three agentic control-loop architectures —
ReAct, Reflection, and Plan-Execute — built on one shared harness (one
`Task`/`Tool` contract, one retry/circuit-breaker/timeout layer, one
scoring function) so that any measured difference between patterns is
architecture, not implementation drift. See `docs/adrs/000{1,2,3}-*.md`
for the per-pattern design decisions and trade-offs this writeup
synthesizes.

## Read this caveat before the numbers

This writeup now has two measured runs: an original **mock-policy run**
(deterministic, no LLM, validates control-flow mechanism only) and a
**real-LLM run** against Groq (`llama-3.3-70b-versatile`, via
`harness/llm.py`'s `GroqLLM`), which is the first run where accuracy is a
real signal at all. Both are kept below rather than overwriting the mock
numbers, since the mock run remains the reference for "does the control
flow work," while the real run is the reference for "how do the patterns
actually perform."

The real-LLM run below is the *second* one taken against Groq. The first
pass (accuracy 27.8% / 0.0% / 11.1% for ReAct / Reflection / Plan-Execute)
surfaced two problems that made those numbers untrustworthy on their own
terms rather than a real accuracy comparison -- see "What the first
real-LLM pass found" below for the evidence. Both are now fixed
(`paraphrase_scorer` in `harness/contracts.py`; `max_critique_rounds` in
`loops/reflection_loop.py`), and the table below is the re-run with both
fixes in place.

### Mock-policy run (mechanism validation, not accuracy)

The mock policy issues one fixed search-then-answer strategy regardless
of loop pattern, so:

- **Structural metrics (iterations, tool calls, tokens, wall-clock) are
  real and pattern-differentiated** — they reflect each pattern's actual
  control-flow shape and its real cost multiplier over ReAct.
- **Accuracy is not a meaningful signal in this run** — all three loops
  score 0% because the mock policy's naive one-sentence extraction almost
  never exact-matches the gold answers, identically across patterns. This
  says nothing about which architecture reasons better.

| pattern | avg iterations/task | avg tool calls/task | avg tokens/task | avg wall-clock/task | cost vs. ReAct |
|---|---|---|---|---|---|
| **ReAct** | 2.00 | 1.00 | 343.4 | 7.1ms | 1.0x (baseline) |
| **Reflection** | 3.00 | 1.00 | 403.3 | 7.6ms | ~1.2x tokens |
| **Plan-Execute** | 5.00 | 2.00 | 921.4 | 11.7ms | ~2.7x tokens |

Reproduce with `python -m eval.run_comparison --llm mock` (writes to
`sqlite:///mlflow.db`; inspect via `mlflow ui --backend-store-uri
sqlite:///mlflow.db`).

### Real-LLM run (Groq `llama-3.3-70b-versatile`)

Reproduce with `python -m eval.run_comparison --llm groq` (requires
`GROQ_API_KEY`, see `.env.example`; add `--scorer exact` to reproduce the
original strict numbers below instead).

| pattern | accuracy | avg iterations/task | avg tool calls/task | avg tokens/task | avg wall-clock/task |
|---|---|---|---|---|---|
| **ReAct** | 88.9% (16/18) | 2.00 | 1.00 | 546.2 | 1.3s |
| **Reflection** | 83.3% (15/18) | 5.61 | 1.22 | 1656.8 | 13.4s |
| **Plan-Execute** | 72.2% (13/18) | 4.39 | 2.22 | 1642.7 | 10.5s |

ReAct remains the cheapest pattern on every axis and now also the most
accurate; Reflection and Plan-Execute both trail it, in line with what
"pay for a critique/plan step that a single-hop question doesn't need"
predicts.

### What the first real-LLM pass found

The numbers above are a *re-run*. The first pass against Groq scored
27.8% / 0.0% / 11.1% (ReAct / Reflection / Plan-Execute) under
`exact_match_scorer`, and reading the traces directly surfaced two
problems with trusting that table at face value — not "Reflection is
bad" and "ReAct wins," but two confounds that needed fixing first:

1. **`exact_match_scorer` was too strict for free-text LLM answers, and
   it depressed every pattern's accuracy roughly equally.** Traces showed
   answers marked `correct=False` that were substantively right — task
   `q14`'s gold answer is `when features are sparse`; ReAct's prediction
   was `when features are sparse.` (one trailing period). Task `q17`'s
   gold is `Elhage et al., 2021`; ReAct's prediction was `Elhage et al. in
   2021.` — correct content, different surface form, scored wrong.
   **Fixed** by `paraphrase_scorer` (`harness/contracts.py`): normalizes
   punctuation/case, then falls back to token-containment (every gold
   word present in the prediction) before failing. Now the default for
   `eval.run_comparison`.
2. **Reflection's 0% was a real, reproducible control-flow gap, not a
   scoring artifact.** Tracing `reflection`'s runs showed the critique
   step repeatedly rejecting substantively correct drafts with
   increasingly pedantic feedback ("lacks... additional context,"
   "unclear... without external verification") rather than ever emitting
   `GOOD`, so the loop exhausted `max_iterations` (6) and returned an
   empty final answer regardless of scorer strictness, since there was no
   answer to score. Only 1 of 18 tasks converged to `GOOD` before the cap.
   **Fixed** by `max_critique_rounds` (`loops/reflection_loop.py`,
   default 3, separate from and tighter than `max_iterations`): once hit,
   the current draft is force-accepted instead of revised again, with a
   `best_draft` fallback if `max_iterations` is hit first.

**The critic still never converges to `GOOD` in the re-run** — traces for
`q16`/`q17`/`q18` below show `max_critique_rounds` forcing acceptance on
round 3 every time, not the critic changing its mind. That's the fix
working as designed, not a surprise: it bounds the cost of an
overly-critical LLM judge rather than teaching it to be less critical.

```
TASK q17 (correct=True, score=0.9, iterations=7)
  draft     -> ACTION: web_search | {"query": "mathematical framework..."}
  draft     -> FINAL_ANSWER: Elhage et al. in 2021.
  critique  -> REVISE: seems mostly correct but lacks confirmation of the...
  draft     -> FINAL_ANSWER: Chris Elhage et al. in 2021, as described in...
  critique  -> REVISE: seems plausible, but without verifying the actual...
  draft     -> FINAL_ANSWER: Elhage et al. in 2021.
  critique  -> REVISE: incomplete as it does not fully address the "group"...
                        (3rd REVISE == max_critique_rounds -> force-accepted)
```

Reflection's avg tokens (1656.8) barely moved from the first pass
(1661.0) — it still pays for the same number of rounds, since the critic
still never says `GOOD` and `max_critique_rounds` was set to be tighter
than `max_iterations`, not the other way around. What changed is that the
loop now returns *an* answer instead of giving up, which is the entire
point of the bound.

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
   (`tests/test_plan_execute_loop.py`), but not exercised by either sweep,
   since the corpus search backend rarely fails and neither policy
   triggers a `REPLAN`.
5. **A real LLM validates the cost-scaling predictions from the mock run
   and surfaced one the mock run couldn't show: self-critique needs a
   convergence bound.** Reflection's real avg iterations (5.61, near the
   6-iteration cap) and 0% convergence-to-`GOOD` rate — not visible under
   a mock critic that always says `GOOD` immediately — is a genuine
   architectural finding, and `max_critique_rounds` (see above) is now the
   harness-level answer to it, verified by the re-run's accuracy recovery
   (0.0% → 83.3%) without the critic's underlying behavior changing at
   all.

## Recommendation, by task shape

- **Shallow, single-hop lookups:** ReAct. Paying for a critique step or an
  upfront plan buys nothing when one search answers the question, and
  ReAct's cost profile is the cheapest of the three on every axis
  measured here — and it's also the highest-accuracy pattern measured
  (88.9%, vs. 83.3% and 72.2%).
- **Reflection, now that it has a convergence bound, is viable but still
  the most expensive pattern for the accuracy it buys.** Its critic still
  never converges to `GOOD` in this benchmark (0/18 tasks) — the fix
  doesn't change that, only bounds its cost — so it pays ~3x ReAct's
  tokens and ~10x its wall-clock for slightly *worse* accuracy (83.3% vs.
  88.9%). Worth keeping only where a genuine second look at a draft is
  valuable and the cost is acceptable; not recommended as a default over
  ReAct for tasks ReAct already handles well.
- **Genuinely multi-hop tasks needing dependent steps** (the kind
  `eval/qa_dataset.py` tags `expected_hops >= 2`, 12 of the 18 questions):
  Plan-Execute's theoretical case — an upfront plan avoids ReAct's
  step-by-step wandering — still isn't borne out here (72.2% vs. ReAct's
  88.9%), now under a scorer that isn't depressing the comparison. 18
  tasks is still a small sample, but the gap is now measured under
  matched conditions rather than a confound waiting to be ruled out.

**Bottom line:** this project's infrastructure — harness, benchmark,
reliability layer, comparison tooling, real LLM provider, paraphrase-
tolerant scorer, and Reflection's convergence bound — is complete and
validated. The two follow-ups this real run originally surfaced are both
resolved and re-measured, not just implemented: fixing them took
Reflection's accuracy from 0.0% to 83.3% and Plan-Execute's from 11.1% to
72.2%, without touching the loops' reasoning logic — the entire gap was
scoring strictness and an unbounded critique loop, not the underlying
control-flow architectures.
