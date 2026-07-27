"""
Comparison runner: dataset -> loop.run() -> log via tracker.

This module is deliberately just plumbing: dataset loading, wiring tools
into each registered loop, running every loop over every task, and
logging results plus a cross-pattern comparison summary through
harness.tracker. It carries no loop-specific or scoring logic of its own,
so it stays valid unmodified as ReAct, Reflection, and Plan-Execute land
in later sprints -- each just needs to register a factory in
`LOOP_FACTORIES`.

Two factory registries are provided: `LOOP_FACTORIES` (mock LLM -- what
`eval/check_regression.py`'s CI gate runs, since it needs to be
deterministic and require no API key) and `REAL_LLM_LOOP_FACTORIES` (Groq
-- what `main()` runs by default here, now that a real provider is wired
into harness/llm.py, to produce a real accuracy comparison). Select
between them with `--llm mock|groq`.
"""
from __future__ import annotations

import argparse
from collections.abc import Callable

from eval import corpus
from eval.mock_agent import make_mock_llm
from eval.qa_dataset import TASKS
from harness import tracker
from harness.contracts import AgentLoop, LoopRunConfig, ScoringFn, TaskResult, exact_match_scorer
from harness.llm import GroqLLM
from harness.tools import CalculatorTool, ScratchpadTool, WebSearchTool
from loops.plan_execute_loop import PlanExecuteLoop
from loops.react_loop import ReActLoop
from loops.reflection_loop import ReflectionLoop


def default_tools() -> list:
    return [WebSearchTool(backend=corpus.search), CalculatorTool(), ScratchpadTool()]


def _make_react_loop() -> AgentLoop:
    return ReActLoop(
        default_tools(),
        make_mock_llm(),
        config=LoopRunConfig(max_iterations=6, tool_backoff_base_s=0.0),
    )


def _make_reflection_loop() -> AgentLoop:
    return ReflectionLoop(
        default_tools(),
        make_mock_llm(),
        config=LoopRunConfig(max_iterations=6, tool_backoff_base_s=0.0),
    )


def _make_plan_execute_loop() -> AgentLoop:
    return PlanExecuteLoop(
        default_tools(),
        make_mock_llm(),
        config=LoopRunConfig(max_iterations=8, tool_backoff_base_s=0.0),
    )


# name -> factory producing a fresh AgentLoop instance (fresh per run, so
# loops with internal state like ScratchpadTool don't leak across tasks).
LOOP_FACTORIES: dict[str, Callable[[], AgentLoop]] = {
    "react": _make_react_loop,
    "reflection": _make_reflection_loop,
    "plan_execute": _make_plan_execute_loop,
}


def _make_react_loop_real() -> AgentLoop:
    return ReActLoop(
        default_tools(),
        GroqLLM(),
        config=LoopRunConfig(max_iterations=6, tool_backoff_base_s=0.0),
    )


def _make_reflection_loop_real() -> AgentLoop:
    return ReflectionLoop(
        default_tools(),
        GroqLLM(),
        config=LoopRunConfig(max_iterations=6, tool_backoff_base_s=0.0),
    )


def _make_plan_execute_loop_real() -> AgentLoop:
    return PlanExecuteLoop(
        default_tools(),
        GroqLLM(),
        config=LoopRunConfig(max_iterations=8, tool_backoff_base_s=0.0),
    )


# Same shape as LOOP_FACTORIES, backed by the real Groq provider instead of
# the mock policy -- requires GROQ_API_KEY (see .env.example).
REAL_LLM_LOOP_FACTORIES: dict[str, Callable[[], AgentLoop]] = {
    "react": _make_react_loop_real,
    "reflection": _make_reflection_loop_real,
    "plan_execute": _make_plan_execute_loop_real,
}


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
    parser = argparse.ArgumentParser(description="Run the loop-pattern comparison sweep.")
    parser.add_argument(
        "--llm",
        choices=["mock", "groq"],
        default="mock",
        help="mock: deterministic, no API key needed (default). groq: real provider, requires GROQ_API_KEY.",
    )
    args = parser.parse_args()

    tracker.configure()
    factories = REAL_LLM_LOOP_FACTORIES if args.llm == "groq" else LOOP_FACTORIES
    run_all(loop_factories=factories)


if __name__ == "__main__":
    main()
