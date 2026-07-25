"""
Comparison runner: dataset -> loop.run() -> log via tracker.

This module is deliberately just plumbing: dataset loading, wiring tools
into each registered loop, running every loop over every task, and
logging results plus a cross-pattern comparison summary through
harness.tracker. It carries no loop-specific or scoring logic of its own,
so it stays valid unmodified as ReAct, Reflection, and Plan-Execute land
in later sprints -- each just needs to register a factory in
`LOOP_FACTORIES`.
"""
from __future__ import annotations

from typing import Callable

from eval import corpus
from eval.qa_dataset import TASKS
from harness.contracts import AgentLoop, ScoringFn, TaskResult, exact_match_scorer
from harness.tools import CalculatorTool, ScratchpadTool, WebSearchTool
from harness import tracker

# Populated as each loop pattern lands (Sprints 2-4): name -> factory
# producing a fresh AgentLoop instance (fresh per run, so loops with
# internal state like ScratchpadTool don't leak across tasks).
LOOP_FACTORIES: dict[str, Callable[[], AgentLoop]] = {}


def default_tools() -> list:
    return [WebSearchTool(backend=corpus.search), CalculatorTool(), ScratchpadTool()]


def run_all(
    loop_factories: dict[str, Callable[[], AgentLoop]] | None = None,
    tasks=TASKS,
    scorer: ScoringFn = exact_match_scorer,
) -> list[TaskResult]:
    """Run every registered loop over every task, logging each result and
    a final comparison summary. Returns the flat list of TaskResults."""
    loop_factories = loop_factories if loop_factories is not None else LOOP_FACTORIES
    if not loop_factories:
        raise RuntimeError(
            "No loops registered in LOOP_FACTORIES -- register a factory "
            "for each pattern (react, reflection, plan_execute) before running."
        )

    results: list[TaskResult] = []
    for loop_name, make_loop in loop_factories.items():
        for task in tasks:
            loop = make_loop()
            with tracker.track_run(loop_name, task.id, vars(loop.config)):
                result = loop.run(task, scorer=scorer)
                tracker.log_result(result)
                results.append(result)

    tracker.log_comparison_summary(results)
    return results


def main() -> None:
    tracker.configure()
    run_all()


if __name__ == "__main__":
    main()
