"""
CI regression gate for the benchmark sweep.

This gate always runs against `LOOP_FACTORIES` (the mock-LLM registry),
never `REAL_LLM_LOOP_FACTORIES` -- CI needs to be deterministic and
requires no API key. Under the mock reasoning policy (see
eval/mock_agent.py's docstring), per-task accuracy against
exact_match_scorer is not a meaningful signal -- it measures a fixed
heuristic's luck against 18 questions, not reasoning quality. What *is*
meaningful, and worth gating CI on, is that every registered loop pattern
completes the full benchmark without raising: a regression here means a
real break (a parsing bug, an infinite loop, a crash) introduced by a
change, not just a wrong answer.

A real per-loop accuracy floor against `REAL_LLM_LOOP_FACTORIES` would
need to tolerate real-provider flakiness (rate limits, transient
failures) and cost real API calls on every CI run -- both reasons to keep
that as a manually-run comparison (`python -m eval.run_comparison
--llm groq`) rather than a CI gate.
"""
from __future__ import annotations

from eval.qa_dataset import TASKS
from eval.run_comparison import LOOP_FACTORIES, run_all
from harness import tracker


def main() -> int:
    tracker.configure()
    results = run_all(loop_factories=LOOP_FACTORIES, tasks=TASKS)

    expected = len(LOOP_FACTORIES) * len(TASKS)
    if len(results) != expected:
        print(f"FAIL: expected {expected} task runs, got {len(results)}")
        return 1

    print(
        f"OK: {len(results)} task runs completed across "
        f"{len(LOOP_FACTORIES)} loop pattern(s) with no crashes"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
