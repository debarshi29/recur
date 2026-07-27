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
(deterministic, no LLM, validates control-flow mechanism only) and a new
**real-LLM run** against Groq (`llama-3.3-70b-versatile`, via
`harness/llm.py`'s `GroqLLM`), which is the first run where accuracy is a
real signal at all. Both are kept below rather than overwriting the mock
numbers, since the mock run remains the reference for "does the control
flow work," while the real run is the reference for "how do the patterns
actually perform."

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
`GROQ_API_KEY`, see `.env.example`).

| pattern | accuracy | avg iterations/task | avg tool calls/task | avg tokens/task | avg wall-clock/task |
|---|---|---|---|---|---|
| **ReAct** | 27.8% (5/18) | 2.06 | 1.06 | 560.4 | 4.3s |
| **Reflection** | 0.0% (0/18) | 5.22 | 1.44 | 1661.0 | 12.9s |
| **Plan-Execute** | 11.1% (2/18) | 4.39 | 2.33 | 1771.3 | 10.8s |

**Two things in this table need explanation before they're read as
"Reflection is bad" and "ReAct wins":**

1. **`exact_match_scorer` is too strict for free-text LLM answers, and it
   is depressing every pattern's accuracy roughly equally.** Inspecting
   traces directly shows answers marked `correct=False` that are
   substantively right — e.g. task `q14`'s gold answer is `when features
   are sparse`; ReAct's prediction was `when features are sparse.` (one
   trailing period). Task `q17`'s gold is `Elhage et al., 2021`; ReAct's
   prediction was `Elhage et al. in 2021.` — correct content, different
   surface form, scored wrong. The mock-run's scoring caveat ("accuracy
   isn't meaningful") has now been replaced by a narrower one: **accuracy
   here is a lower bound**, and a real comparison needs a scorer that
   tolerates paraphrase (substring/containment check, or an LLM-judge)
   before the gap between patterns can be trusted. This is now the
   project's next concrete follow-up, not "wire a real LLM" (done).
2. **Reflection's 0% is a real, reproducible behavior, not a scoring
   artifact** — and it's the most interesting finding in this run.
   Tracing `reflection`'s runs shows the critique step repeatedly
   rejecting substantively correct drafts with increasingly pedantic
   feedback ("lacks... additional context," "unclear... without external
   verification") rather than ever emitting `GOOD`, so the loop exhausts
   `max_iterations` (6) and returns an empty final answer — scored wrong
   regardless of scorer strictness, since there's no answer to score. Only
   1 of 18 tasks converged to `GOOD` before the cap. This is a known
   failure mode of naive self-critique loops: without an explicit
   "good enough" bar or a critique budget separate from the draft budget,
   a sufficiently capable critic can nitpick a correct-but-imperfect
   answer forever. Plan-Execute doesn't hit this because it never
   critiques — it only replans on tool *failure*, which this benchmark's
   corpus search rarely produces.

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
   and adds one the mock run couldn't show: self-critique needs a
   convergence bound.** Reflection's real avg iterations (5.22, near the
   6-iteration cap) and 0% convergence-to-`GOOD` rate is a genuine
   architectural finding — see the caveat above — not visible under a
   mock critic that always says `GOOD` immediately.

## Recommendation, by task shape

- **Shallow, single-hop lookups:** ReAct. Paying for a critique step or an
  upfront plan buys nothing when one search answers the question, and
  ReAct's cost profile is the cheapest of the three on every axis
  measured here — and it's now also the highest-accuracy pattern measured
  (27.8%, vs. 11.1% and 0%), even accounting for `exact_match_scorer`
  depressing all three roughly equally.
- **Reflection, as implemented here, is not recommended without a
  convergence bound.** The real run shows its core assumption — a fixed
  one-extra-call cost buys error-catching — doesn't hold once the critic
  is a real, capable LLM: it nitpicks rather than converging, so the cost
  is closer to `max_iterations`-worth of calls (in this run, ~2.5x
  ReAct's tokens) for a worse, not better, accuracy outcome. Fixing this
  is a scope-bounded follow-up: cap critique rounds separately from draft
  rounds, or accept the best draft seen if no round reaches `GOOD`, rather
  than returning empty on cap-out.
- **Genuinely multi-hop tasks needing dependent steps** (the kind
  `eval/qa_dataset.py` tags `expected_hops >= 2`, 12 of the 18 questions):
  Plan-Execute's theoretical case — an upfront plan avoids ReAct's
  step-by-step wandering — isn't borne out in this run (11.1% vs. ReAct's
  27.8%), but 18 tasks and one scoring pass is too small a sample and too
  blunt a scorer to treat that as disproof. Worth re-testing once a
  paraphrase-tolerant scorer is in place.

**Bottom line:** this project's infrastructure — harness, benchmark,
reliability layer, comparison tooling, and now a real LLM provider — is
complete and validated, and the accuracy comparison this writeup called
for is no longer hypothetical. The two concrete follow-ups this real run
surfaced are: (1) replace `exact_match_scorer` with something paraphrase-
tolerant so the accuracy gap between patterns can be trusted rather than
read as a floor, and (2) give Reflection's critique loop a convergence
bound so it stops trading Plan-Execute's cost for worse accuracy than
ReAct.
