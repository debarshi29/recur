from __future__ import annotations

from harness.contracts import LoopRunConfig, Task
from harness.llm import MockLLM, ScriptedLLM
from harness.tools import CalculatorTool
from loops.plan_execute_loop import (
    PlanExecuteLoop,
    build_execute_prompt,
    build_plan_prompt,
    build_replan_prompt,
    build_synthesize_prompt,
)

# ---------------------------------------------------------------------------
# Node-level unit tests
# ---------------------------------------------------------------------------

def test_build_plan_prompt_includes_question_and_tools():
    task = Task(id="t1", question="What is 2+2?", gold_answer="4")
    tool = CalculatorTool()
    prompt = build_plan_prompt(task, {tool.name: tool}, trace=[])

    assert "What is 2+2?" in prompt
    assert "calculator" in prompt
    assert "PLAN: step1 | step2" in prompt


def test_build_execute_prompt_marks_final_step():
    task = Task(id="t1", question="q", gold_answer="a")
    prompt = build_execute_prompt(task, {}, step="do the thing", step_idx=1, plan_len=2, trace=[])
    assert "(final step)" in prompt
    assert "do the thing" in prompt


def test_build_execute_prompt_does_not_mark_non_final_step():
    task = Task(id="t1", question="q", gold_answer="a")
    prompt = build_execute_prompt(task, {}, step="do the thing", step_idx=0, plan_len=2, trace=[])
    assert "(final step)" not in prompt


def test_build_replan_prompt_includes_failure_reason():
    task = Task(id="t1", question="q", gold_answer="a")
    prompt = build_replan_prompt(task, {}, failed_step="bad step", reason="tool unavailable", trace=[])
    assert "bad step" in prompt
    assert "tool unavailable" in prompt


def test_build_synthesize_prompt_includes_trace():
    task = Task(id="t1", question="q", gold_answer="a")
    trace = [{"stage": "execute", "thought": "STEP_DONE: found it"}]
    prompt = build_synthesize_prompt(task, trace)
    assert "STEP_DONE: found it" in prompt


# ---------------------------------------------------------------------------
# Full-run integration tests
# ---------------------------------------------------------------------------

def test_plan_execute_runs_a_two_step_plan_to_completion():
    llm = ScriptedLLM(
        [
            "PLAN: use calculator | give final answer",
            'ACTION: calculator | {"expression": "2+2"}',
            "STEP_DONE: got 4",
            "FINAL_ANSWER: 4",
        ]
    )
    loop = PlanExecuteLoop([CalculatorTool()], llm)
    task = Task(id="t1", question="What is 2+2?", gold_answer="4")

    result = loop.run(task)

    assert result.correct is True
    assert result.predicted_answer == "4"
    assert result.iterations == 4
    assert result.tool_calls == 1
    assert [s["stage"] for s in result.trace] == ["plan", "execute", "execute", "execute"]


def test_plan_execute_replans_after_a_failed_step():
    llm = ScriptedLLM(
        [
            "PLAN: use bogus tool | fallback",
            "ACTION: does_not_exist | {}",
            "REPLAN: does_not_exist tool unavailable",
            "PLAN: use calculator | give final answer",
            'ACTION: calculator | {"expression": "3+3"}',
            "STEP_DONE: ok",
            "FINAL_ANSWER: 6",
        ]
    )
    loop = PlanExecuteLoop([CalculatorTool()], llm)
    task = Task(id="t1", question="What is 3+3?", gold_answer="6")

    result = loop.run(task)

    assert result.correct is True
    assert result.predicted_answer == "6"
    assert result.tool_calls == 2  # one failed unknown-tool attempt, one successful call
    stages = [s["stage"] for s in result.trace]
    assert "replan" in stages


def test_plan_execute_synthesizes_when_plan_exhausts_without_final_answer():
    llm = ScriptedLLM(
        [
            "PLAN: just note it",
            "STEP_DONE: noted",
            "FINAL_ANSWER: 4",  # from the synthesize node
        ]
    )
    loop = PlanExecuteLoop([CalculatorTool()], llm)
    task = Task(id="t1", question="What is 2+2?", gold_answer="4")

    result = loop.run(task)

    assert result.correct is True
    assert [s["stage"] for s in result.trace] == ["plan", "execute", "synthesize"]


def test_plan_execute_stops_at_max_iterations_without_crashing():
    llm = MockLLM(lambda prompt: "hmm not sure")
    config = LoopRunConfig(max_iterations=3, tool_backoff_base_s=0.0)
    loop = PlanExecuteLoop([CalculatorTool()], llm, config=config)
    task = Task(id="t1", question="never resolves", gold_answer="anything")

    result = loop.run(task)

    assert result.iterations == 3
    assert result.predicted_answer == ""
    assert result.correct is False
