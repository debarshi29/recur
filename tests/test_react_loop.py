from __future__ import annotations

from harness.contracts import LoopRunConfig, Task
from harness.llm import MockLLM, ScriptedLLM
from harness.tools import CalculatorTool
from loops.react_loop import ReActLoop, build_prompt, parse_action

# ---------------------------------------------------------------------------
# Node-level unit tests
# ---------------------------------------------------------------------------

def test_build_prompt_includes_question_tools_and_history():
    task = Task(id="t1", question="What is 2+2?", gold_answer="4")
    tool = CalculatorTool()
    prompt = build_prompt(task, {tool.name: tool}, trace=[])

    assert "What is 2+2?" in prompt
    assert "calculator" in prompt
    assert "(none yet)" in prompt


def test_build_prompt_renders_prior_trace_steps():
    task = Task(id="t1", question="q", gold_answer="a")
    trace = [{"thought": "ACTION: calculator | {}", "observation": "4"}]
    prompt = build_prompt(task, {}, trace)

    assert "ACTION: calculator | {}" in prompt
    assert "Observation: 4" in prompt


def test_parse_action_extracts_tool_name_and_kwargs():
    name, kwargs = parse_action('ACTION: calculator | {"expression": "2+2"}')
    assert name == "calculator"
    assert kwargs == {"expression": "2+2"}


def test_parse_action_handles_no_kwargs():
    name, kwargs = parse_action("ACTION: scratchpad |")
    assert name == "scratchpad"
    assert kwargs == {}


# ---------------------------------------------------------------------------
# Full-run integration tests
# ---------------------------------------------------------------------------

def test_react_loop_calls_tool_then_answers_correctly():
    llm = ScriptedLLM(
        [
            'ACTION: calculator | {"expression": "2+2"}',
            "FINAL_ANSWER: 4",
        ]
    )
    loop = ReActLoop([CalculatorTool()], llm)
    task = Task(id="t1", question="What is 2+2?", gold_answer="4")

    result = loop.run(task)

    assert result.correct is True
    assert result.predicted_answer == "4"
    assert result.iterations == 2
    assert result.tool_calls == 1
    assert len(result.trace) == 2


def test_react_loop_recovers_from_unknown_tool_without_crashing():
    llm = ScriptedLLM(
        [
            "ACTION: does_not_exist | {}",
            "FINAL_ANSWER: 4",
        ]
    )
    loop = ReActLoop([CalculatorTool()], llm)
    task = Task(id="t1", question="What is 2+2?", gold_answer="4")

    result = loop.run(task)

    assert result.correct is True
    assert "Unknown tool" in result.trace[0]["observation"]


def test_react_loop_stops_at_max_iterations_without_crashing():
    llm = MockLLM(lambda prompt: 'ACTION: calculator | {"expression": "1+1"}')
    config = LoopRunConfig(max_iterations=3, tool_backoff_base_s=0.0)
    loop = ReActLoop([CalculatorTool()], llm, config=config)
    task = Task(id="t1", question="never resolves", gold_answer="anything")

    result = loop.run(task)

    assert result.iterations == 3
    assert result.predicted_answer == ""
    assert result.correct is False
