"""
CI regression gate for the benchmark sweep.

Under the current mock reasoning policy (see eval/mock_agent.py's
docstring), per-task accuracy against exact_match_scorer is not a
meaningful signal -- it measures a fixed heuristic's luck against 18
questions, not reasoning quality. What *is* meaningful, and worth gating
CI on, is that every registered loop pattern completes the full benchmark
without raising: a regression here means a real break (a parsing bug, an
infinite loop, a crash) introduced by a change, not just a wrong answer.

Once a real LLM is wired into harness/llm.py (tracked as follow-up work),
extend this with a real per-loop accuracy floor -- the plumbing
(LOOP_FACTORIES, run_all, TaskResult.correct) is already in place for
that; only the threshold check itself needs to be added below.
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
