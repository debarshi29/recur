from __future__ import annotations

from harness.contracts import LoopRunConfig, Task
from harness.llm import MockLLM, ScriptedLLM
from harness.tools import CalculatorTool
from loops.reflection_loop import ReflectionLoop, build_critique_prompt, build_draft_prompt

# ---------------------------------------------------------------------------
# Node-level unit tests
# ---------------------------------------------------------------------------

def test_build_draft_prompt_includes_question_tools_and_feedback():
    task = Task(id="t1", question="What is 2+2?", gold_answer="4")
    tool = CalculatorTool()
    prompt = build_draft_prompt(task, {tool.name: tool}, trace=[], feedback="be more precise")

    assert "What is 2+2?" in prompt
    assert "calculator" in prompt
    assert "be more precise" in prompt


def test_build_draft_prompt_omits_feedback_block_when_none():
    task = Task(id="t1", question="q", gold_answer="a")
    prompt = build_draft_prompt(task, {}, trace=[], feedback=None)
    assert "Critique feedback" not in prompt


def test_build_critique_prompt_includes_question_and_draft():
    task = Task(id="t1", question="What is 2+2?", gold_answer="4")
    prompt = build_critique_prompt(task, "4")
    assert "What is 2+2?" in prompt
    assert "Candidate answer: 4" in prompt


# ---------------------------------------------------------------------------
# Full-run integration tests
# ---------------------------------------------------------------------------

def test_reflection_loop_accepts_good_draft_immediately():
    llm = ScriptedLLM(
        [
            "FINAL_ANSWER: 4",
            "GOOD",
        ]
    )
    loop = ReflectionLoop([CalculatorTool()], llm)
    task = Task(id="t1", question="What is 2+2?", gold_answer="4")

    result = loop.run(task)

    assert result.correct is True
    assert result.predicted_answer == "4"
    assert result.iterations == 2
    assert [s["stage"] for s in result.trace] == ["draft", "critique"]


def test_reflection_loop_revises_after_critique_feedback():
    llm = ScriptedLLM(
        [
            "FINAL_ANSWER: 5",
            "REVISE: arithmetic is wrong",
            "FINAL_ANSWER: 4",
            "GOOD",
        ]
    )
    loop = ReflectionLoop([CalculatorTool()], llm)
    task = Task(id="t1", question="What is 2+2?", gold_answer="4")

    result = loop.run(task)

    assert result.correct is True
    assert result.predicted_answer == "4"
    assert result.iterations == 4
    assert [s["stage"] for s in result.trace] == ["draft", "critique", "draft", "critique"]


def test_reflection_loop_calls_tool_during_draft():
    llm = ScriptedLLM(
        [
            'ACTION: calculator | {"expression": "2+2"}',
            "FINAL_ANSWER: 4",
            "GOOD",
        ]
    )
    loop = ReflectionLoop([CalculatorTool()], llm)
    task = Task(id="t1", question="What is 2+2?", gold_answer="4")

    result = loop.run(task)

    assert result.correct is True
    assert result.tool_calls == 1


def test_reflection_loop_stops_at_max_iterations_without_crashing():
    llm = MockLLM(lambda prompt: "REVISE: never satisfied" if "Candidate answer" in prompt else "FINAL_ANSWER: 4")
    config = LoopRunConfig(max_iterations=4, tool_backoff_base_s=0.0)
    loop = ReflectionLoop([CalculatorTool()], llm, config=config)
    task = Task(id="t1", question="never accepted", gold_answer="anything")

    result = loop.run(task)

    assert result.iterations == 4
    assert result.correct is False


def test_reflection_loop_force_accepts_draft_after_max_critique_rounds():
    """A critic that never says GOOD shouldn't burn the whole
    max_iterations budget or return an empty answer -- max_critique_rounds
    should force-accept the current draft first."""
    llm = MockLLM(lambda prompt: "REVISE: never satisfied" if "Candidate answer" in prompt else "FINAL_ANSWER: 4")
    config = LoopRunConfig(max_iterations=20, tool_backoff_base_s=0.0)
    loop = ReflectionLoop([CalculatorTool()], llm, config=config, max_critique_rounds=2)
    task = Task(id="t1", question="What is 2+2?", gold_answer="4")

    result = loop.run(task)

    assert result.predicted_answer == "4"
    assert result.correct is True
    # 2 draft/critique rounds, forced accept on the 2nd critique -- well
    # under the max_iterations=20 budget.
    assert result.iterations == 4


def test_reflection_loop_falls_back_to_best_draft_on_iteration_cap():
    """If max_iterations is hit while still mid-revision (e.g.
    max_critique_rounds set high relative to max_iterations), the loop
    should return the last real draft rather than an empty answer."""
    llm = MockLLM(lambda prompt: "REVISE: never satisfied" if "Candidate answer" in prompt else "FINAL_ANSWER: 4")
    config = LoopRunConfig(max_iterations=4, tool_backoff_base_s=0.0)
    loop = ReflectionLoop([CalculatorTool()], llm, config=config, max_critique_rounds=100)
    task = Task(id="t1", question="never accepted", gold_answer="anything")

    result = loop.run(task)

    assert result.iterations == 4
    assert result.predicted_answer == "4"
    assert result.correct is False
