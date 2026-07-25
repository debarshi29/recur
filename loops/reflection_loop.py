"""
Reflection loop: generate an initial answer, critique it, revise if the
critique flags a problem, repeat until the critique accepts the draft or
max_iterations is hit.

Draft generation can call tools via the same ACTION/FINAL_ANSWER protocol
ReAct uses (loops/common.py), so Reflection has the same underlying
capability as ReAct -- the only variable under test is that a draft gets
critiqued and possibly revised before being accepted as final, at the
cost of extra LLM calls per round.
"""
from __future__ import annotations

import time
from typing import Any, TypedDict

from langgraph.graph import END, StateGraph

from harness.contracts import (
    AgentLoop,
    LoopRunConfig,
    ScoringFn,
    Task,
    TaskResult,
    Tool,
    exact_match_scorer,
)
from harness.llm import LLM
from harness.reliability import new_checkpointer, run_graph
from loops.common import ACTION_PREFIX, FINAL_PREFIX, parse_action, tool_lines

GOOD_PREFIX = "GOOD"
REVISE_PREFIX = "REVISE:"


class ReflectionState(TypedDict):
    task: Task
    trace: list[dict[str, Any]]
    iteration: int
    tool_calls: int
    total_tokens: int
    draft: str | None
    feedback: str | None
    final_answer: str | None


def build_draft_prompt(
    task: Task,
    tools: dict[str, Tool],
    trace: list[dict[str, Any]],
    feedback: str | None,
) -> str:
    history_lines: list[str] = []
    for step in trace:
        if step.get("stage") != "draft":
            continue
        history_lines.append(f"Thought/Action: {step['thought']}")
        if "observation" in step:
            history_lines.append(f"Observation: {step['observation']}")
    history = "\n".join(history_lines) if history_lines else "(none yet)"
    feedback_block = f"\nCritique feedback to address: {feedback}\n" if feedback else ""
    return (
        f"Question: {task.question}\n\n"
        f"Available tools:\n{tool_lines(tools)}\n\n"
        f"History so far:\n{history}\n"
        f"{feedback_block}\n"
        f"Respond with exactly one line: either '{FINAL_PREFIX} <answer>' or "
        f"'{ACTION_PREFIX} <tool_name> | <json kwargs>'."
    )


def build_critique_prompt(task: Task, draft: str) -> str:
    return (
        f"Question: {task.question}\n"
        f"Candidate answer: {draft}\n\n"
        f"Critique the candidate answer against the question. Respond with "
        f"exactly one line: either '{GOOD_PREFIX}' if it is correct and "
        f"complete, or '{REVISE_PREFIX} <specific feedback>' if it needs "
        f"revision."
    )


class ReflectionLoop(AgentLoop):
    name = "reflection"

    def __init__(self, tools: list[Tool], llm: LLM, config: LoopRunConfig | None = None):
        super().__init__(tools, config)
        self.llm = llm
        self.checkpointer = new_checkpointer()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ReflectionState)
        graph.add_node("draft", self._draft)
        graph.add_node("observe", self._observe)
        graph.add_node("critique", self._critique)
        graph.set_entry_point("draft")
        graph.add_conditional_edges(
            "draft",
            self._route_after_draft,
            {"observe": "observe", "critique": "critique", "end": END},
        )
        graph.add_edge("observe", "draft")
        graph.add_conditional_edges(
            "critique",
            self._route_after_critique,
            {"revise": "draft", "end": END},
        )
        return graph.compile(checkpointer=self.checkpointer)

    def _draft(self, state: ReflectionState) -> ReflectionState:
        prompt = build_draft_prompt(state["task"], self.tools, state["trace"], state["feedback"])
        response = self.llm.generate(prompt)
        state["total_tokens"] += response.total_tokens
        state["iteration"] += 1
        text = response.text.strip()
        step: dict[str, Any] = {"stage": "draft", "thought": text}
        if text.startswith(FINAL_PREFIX):
            state["draft"] = text[len(FINAL_PREFIX):].strip()
        state["trace"].append(step)
        return state

    def _observe(self, state: ReflectionState) -> ReflectionState:
        step = state["trace"][-1]
        try:
            tool_name, kwargs = parse_action(step["thought"])
        except Exception as e:  # noqa: BLE001 — a malformed action is an observation, not a crash
            step["observation"] = f"error parsing action: {e}"
            return state
        result = self._call_tool(tool_name, **kwargs)
        state["tool_calls"] += 1
        step["observation"] = result.output if result.ok else f"ERROR: {result.error}"
        return state

    def _critique(self, state: ReflectionState) -> ReflectionState:
        prompt = build_critique_prompt(state["task"], state["draft"] or "")
        response = self.llm.generate(prompt)
        state["total_tokens"] += response.total_tokens
        state["iteration"] += 1
        text = response.text.strip()
        state["trace"].append({"stage": "critique", "thought": text})
        if text.startswith(REVISE_PREFIX):
            state["feedback"] = text[len(REVISE_PREFIX):].strip()
            state["draft"] = None  # discard the unrevised draft so a stale
            # value can't be mistaken for a fresh one on the next round
        else:
            state["final_answer"] = state["draft"]
        return state

    def _route_after_draft(self, state: ReflectionState) -> str:
        if state["draft"] is not None:
            return "critique"
        if state["iteration"] >= self.config.max_iterations:
            state["final_answer"] = ""
            return "end"
        return "observe"

    def _route_after_critique(self, state: ReflectionState) -> str:
        if state["final_answer"] is not None:
            return "end"
        if state["iteration"] >= self.config.max_iterations:
            state["final_answer"] = ""
            return "end"
        return "revise"

    def run(self, task: Task, scorer: ScoringFn = exact_match_scorer) -> TaskResult:
        start = time.perf_counter()
        initial_state: ReflectionState = {
            "task": task,
            "trace": [],
            "iteration": 0,
            "tool_calls": 0,
            "total_tokens": 0,
            "draft": None,
            "feedback": None,
            "final_answer": None,
        }
        final_state = run_graph(
            self._graph,
            initial_state,
            thread_id=task.id,
            timeout_s=self.config.timeout_s,
            recursion_limit=self.config.max_iterations * 6 + 10,
        )
        predicted = final_state.get("final_answer") or ""
        correct, score = scorer(predicted, task.gold_answer)
        return TaskResult(
            task_id=task.id,
            loop_name=self.name,
            predicted_answer=predicted,
            correct=correct,
            score=score,
            iterations=final_state["iteration"],
            tool_calls=final_state["tool_calls"],
            total_tokens=final_state["total_tokens"],
            wall_clock_ms=(time.perf_counter() - start) * 1000,
            trace=final_state["trace"],
        )
