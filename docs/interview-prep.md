# recur — defense prep

Hard questions for each design decision in ReAct, Reflection, and
Plan-Execute, grounded in the actual ADRs, code, and the real-Groq run —
not generic system-design trivia. Rehearse the answer out loud before
reading the angle. Companion to `docs/adrs/000{1,2,3}-*.md` and
`docs/writeup.md`, which this draws on directly.

Questions marked **honest gap** are ones where the right answer is
admitting a real limitation, not defending the design — those are
usually the ones worth over-preparing.

## ADR 0001 — ReAct

Two nodes, no self-correction, cheapest pattern measured — and the
highest-accuracy one in the real run.

**A bad tool call happens on iteration 1. What actually catches it?**
(`docs/adrs/0001-react-loop.md`, Consequences)
- Nothing does, by design — the ADR states this plainly. The
  retry/backoff wrapper and circuit breaker in `AgentLoop._call_tool`
  handle *transient tool failure* (a search backend erroring), not a
  *semantic* misread of a correct observation.
- That's the accepted structural risk of the baseline, not an oversight
  — it's exactly what Reflection's critique step and Plan-Execute's
  replan branch exist to catch, at a measured cost premium.

**ReAct won the real run at 88.9%. Doesn't that mean the benchmark
doesn't need multi-hop reasoning at all?**
(`eval/qa_dataset.py`, `expected_hops >= 2` on 12/18 tasks)
- Careful here — it means ReAct isn't *losing* on this benchmark, not
  that multi-hop reasoning is unnecessary. A capable model with tool
  access can often synthesize what looks like a multi-hop answer from
  one well-targeted search over a small fixed corpus.
- 18 tasks and one run is a thin basis for "ReAct wins categorically" —
  the honest claim is narrower: on *this* corpus, at *this* scale, the
  extra machinery isn't earning its cost yet.

**Why split reasoning and tool execution into two nodes instead of one
that does both?** (`loops/react_loop.py`, `reason_act` → `observe`)
- So the tool call can be wrapped by `_call_tool`'s retry/circuit-breaker
  independent of the LLM call — and so `observe` is the literal same
  node logic Reflection reuses for its own draft step.
- Collapsing them would mean re-deriving retry semantics per pattern —
  precisely what the shared-harness principle forbids.

## ADR 0002 — Reflection

Draft → critique → revise. The critic never once said `GOOD` against a
real model — the interesting part is what that forced you to build.

**Your critic converged to `GOOD` on 1/18 tasks first pass, then 0/18
after the fix. Is the critique step actually working?**
(`docs/writeup.md`, "Only 1 of 18 tasks converged to GOOD" / "0/18 in
the re-run")
- It's working exactly as a genuinely critical judge behaves —
  nitpicking. The bug wasn't non-convergence, it was that
  non-convergence had *no bound*, so the loop paid for it by returning
  nothing.
- Draw the line clearly: a loop-mechanism bug (control flow doesn't
  terminate sensibly) is this project's problem to fix; a critic being
  too harsh is a prompting/calibration problem, out of scope for a
  harness comparison.

**Why `max_critique_rounds = 3` specifically? What breaks at 1, or at
10?** (`loops/reflection_loop.py`, `ReflectionLoop.__init__`) — **honest gap**
- No principled derivation — it's an untuned bound picked to demonstrate
  the mechanism. 1 defeats the point of critique entirely (no real
  revision cycle); values approaching `max_iterations` (6 in the eval
  config) just reintroduce the original problem.
- The honest next step is sweeping this against a validation set rather
  than asserting 3 is correct.

**Walk me through why a LangGraph conditional-edge function's state
writes silently don't persist.** (`loops/reflection_loop.py`, commit
`9c91793`)
- A router is consumed synchronously to pick the next node — it isn't a
  graph step in its own right, so nothing re-emits a "values" event
  capturing its mutation once `END` is next and no further node runs to
  carry it forward.
- Generalizes past LangGraph: in any framework with a node/edge
  distinction, treat routers as pure predicates only — never rely on
  them for side effects.

**You discard the draft entirely on `REVISE`. Doesn't that throw away a
usable fallback answer?** (`loops/reflection_loop.py`, `_critique` /
`best_draft`)
- That's exactly why `best_draft` exists as a separate field from
  `draft`: `draft` means "in-flight, possibly stale, must be null between
  rounds"; `best_draft` means "last known-good candidate, kept solely for
  cap-out fallback."
- Conflating the two was the original design — keeping them apart is
  what let the fallback logic live in `run()` without touching the
  revise/discard invariant.

**Eventually you'll want an LLM-judge scorer. Isn't grading Groq's
answer with Groq itself circular?** (`docs/writeup.md`, Recommendation
section)
- Yes — that's a real self-grading bias risk, which is exactly why
  `paraphrase_scorer` stays a cheap, deterministic heuristic rather than
  reaching for the model that's sitting right there.
- An LLM-judge is flagged as a real future upgrade, not because it's a
  bad idea, but because it needs its own prompt design and its own
  reliability wrapper — a separately-scoped piece of work, not a
  two-line change.

## ADR 0003 — Plan-Execute

Plans up front, replans on model-judged step failure. Most expensive
pattern measured — and the lowest-accuracy one in the real run.

**This was supposed to win on multi-hop tasks. It scored lowest (72.2%).
What's your actual hypothesis?** (`docs/writeup.md`, Real-LLM run table)
- Cost compounds with plan length — every step is its own act/observe
  exchange (2.7x ReAct's tokens in the mock run), which is also 2.7x the
  chances for one hallucinated or misread step to poison the final
  synthesis.
- The replan path — the mechanism meant to catch exactly that — was
  never triggered in either sweep, so it isn't earning its keep here;
  more calls without more correction is a worse trade than ReAct's fewer
  calls with no correction at all.

**`REPLAN` is model-driven, not automatic on tool failure. What's the
risk in that choice?** (`docs/adrs/0003-plan-execute-loop.md`,
Consequences)
- A step can be observably wrong — an empty or off-target search result
  — without the model recognizing that the *step*, not just the tool
  call, needs to change. Synthesize then papers over it with a confident
  wrong answer.
- This is the ADR's own named failure mode: "silent execution of a bad
  plan to completion." Neither benchmark run confirms or rules it out,
  because REPLAN was never exercised in the sweep.

**Replan-on-failure is only tested with a scripted LLM in unit tests,
never in the actual benchmark. Isn't that an evaluation gap, not just an
implementation detail?** (`tests/test_plan_execute_loop.py`,
`test_plan_execute_replans_after_a_failed_step`) — **honest gap**
- Yes, and it's named as such directly in the writeup rather than glossed
  over. The fixed local corpus (chosen for reproducibility, not realism)
  rarely fails, so the path that matters most under real-world flakiness
  is the one this benchmark structurally can't stress.

## Harness & CI

The seam every pattern is built against — and where the discipline of
"extend the harness, not the pattern" actually got tested.

**Give a concrete case where you were tempted to break the shared-harness
principle, and what stopped you.** (`CLAUDE.md`, Core design principle)
- `max_critique_rounds` is Reflection-only state. It would've been easy
  to bolt it onto the shared `LoopRunConfig` since that's where
  `max_iterations` already lives — instead it's a constructor argument on
  `ReflectionLoop` itself, because it's pattern-specific control flow,
  not shared reliability behavior.
- Same logic kept the cap-out fallback fix inside `ReflectionLoop.run()`
  rather than special-casing `run_graph` — every pattern still goes
  through the identical checkpoint/timeout wrapper.

**Why is retry/circuit-breaking a wrapper method on `AgentLoop` instead
of built into `Tool` itself?** (`harness/contracts.py`,
`AgentLoop._call_tool`)
- `Tool.run()` already isolates exceptions per call — that's the tool's
  own contract. Retry count and backoff are a *policy* choice that can
  differ per loop instance (`LoopRunConfig` is per-`AgentLoop`), and the
  same `WebSearchTool` instance is shared across all three patterns.
- Baking retry into `Tool` would fuse "what a tool does" with "how
  forgiving a loop is about it failing" — two orthogonal concerns.

**CI only ever runs the mock policy. How do you know a loop-logic change
doesn't quietly regress real accuracy?** (`eval/check_regression.py`,
module docstring) — **honest gap**
- You don't, automatically — that's a documented, deliberate tradeoff
  (determinism, cost, no live-provider flakiness in CI). CI only
  guarantees crash-free completion, not accuracy.
- The actual safety net right now is manual: rerun
  `eval.run_comparison --llm groq` and read the traces — literally what
  happened to validate both fixes this session. A next step worth
  naming: a small, human-curated "golden trace" set replayed through
  `ScriptedLLM`, deterministic and free, that catches control-flow
  regressions without needing a live model.

**Why MLflow instead of just structured logs or a dataframe?**
(`harness/tracker.py`)
- Needed comparable historical runs across sweeps (not just one
  session's), a queryable backend, and artifact storage for full
  per-task traces. Flat files could do this too — MLflow's
  run/experiment model gives it off the shelf, and it's the same tool
  used broadly in ML work, so it's a defensible reach rather than a
  reinvented tracker.

## Methodology — the real-LLM run

The part most worth over-preparing: how solid is the 0%→83% headline,
really?

**Accuracy moved from 0% to 83% for Reflection and 27.8% to 88.9% for
ReAct — code for ReAct didn't change at all between runs. How much of
that is really the scorer, versus Groq just sampling better the second
time?** (`harness/llm.py`, `GroqLLM.generate` — no `temperature` pinned)
— **honest gap**
- This is the sharpest edge in the whole writeup, and worth owning
  directly: `GroqLLM` never pins `temperature`, so the two passes are two
  independent live samples, not the same predictions re-scored. Some of
  the delta is provably the scorer (the q14/q17 trace examples show the
  exact same wording scored differently) — but a fully clean isolation
  would re-score the *first pass's saved trace text* under both scorers
  rather than re-running live.
- The honest answer: qualitative trace evidence supports the scorer as
  the dominant effect; a rigorous version of this claim needs that
  re-scoring step, which wasn't done. Say exactly that if asked — it's a
  better answer than overselling the number.

**`paraphrase_scorer` falls back to token-containment. What's the
failure mode of that heuristic?** (`harness/contracts.py`,
`paraphrase_scorer`)
- It can only ever raise a wrong-looking answer to 0.9, never claim full
  exact-match confidence — but it can still false-positive: a prediction
  containing all of gold's words embedded in an unrelated sentence
  (trivial case: gold is a bare year, "2021") would score correct despite
  being off-topic.
- Known, named tradeoff — the real fix for that class of error is an
  LLM-judge, not a better regex.

**Only 18 tasks. What breaks first if this scaled to 180?**
(`eval/qa_dataset.py`, `TASKS`)
- Not the harness — cost is linear in tasks × patterns, nothing
  quadratic. What actually strains first is wall-clock (Reflection/
  Plan-Execute already run ~10–13s/task against Groq) and corpus quality:
  enough genuinely distinct multi-hop chains to avoid accidentally-easy
  overlapping answers.

**Convince me single-scorer accuracy is even the right top-line metric
for open-ended QA.** (`docs/writeup.md`, "Recommendation, by task shape")
- It isn't "the" right metric — it's a pragmatic one for a project that
  needs to run unattended, cheaply, and reproducibly. A production eval
  would likely blend this with human or pairwise LLM-judge preference;
  the scorer choice here is itself a modeling decision that shapes which
  pattern looks like it "won."

## Curveballs

Off-script generalization questions — the ones that test whether you
understand the design, not just remember it.

**Add a fourth pattern tomorrow — Tree-of-Thought, multi-agent debate,
whatever. What in the harness do you not have to touch?**
- `Tool`, `AgentLoop._call_tool`'s retry/circuit-breaker, `run_graph`'s
  checkpoint/timeout wrapper, `ScoringFn`, `Task`/`TaskResult`, and
  `tracker`'s logging — all untouched. New work is one `loops/x_loop.py`
  file plus one factory entry in `LOOP_FACTORIES` /
  `REAL_LLM_LOOP_FACTORIES`.

**What's the single most likely thing here to bite you in production
that this benchmark structurally can't show you?**
- The circuit breaker and retry policy are tuned against a fixed local
  corpus that basically never fails — a live search API's real failure
  modes (rate limits, partial results, schema drift) are entirely
  untested. Separately, `Timeout` is only checked between graph steps, so
  one slow/hanging LLM call inside a node isn't interrupted — only
  noticed after it returns.

**If you had one more week and had to spend it on exactly one thing,
what would it be and why not the other candidates?**
- Re-scoring the first pass's saved traces under both scorers to
  properly isolate scorer-effect from Groq sampling variance (see
  Methodology) — it's the one open question that undermines trusting the
  headline number, versus an LLM-judge scorer or a wider benchmark, both
  of which are real but don't currently threaten the validity of what's
  already been measured.
