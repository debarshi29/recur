"""
ReAct loop: interleave reasoning and action, one step at a time, reacting
to each observation before deciding the next step.

Built as a LangGraph StateGraph with two nodes -- reason_act (ask the LLM
for either a tool action or a final answer) and observe (execute that
action through the shared harness's retry-wrapped _call_tool) -- looping
between them until the LLM emits a final answer or max_iterations is hit.
Reliability (tool retry/backoff) and scoring are inherited unchanged from
AgentLoop; only this control-flow shape is pattern-specific.
"""
from __future__ import annotations

import json
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

FINAL_PREFIX = "FINAL_ANSWER:"
ACTION_PREFIX = "ACTION:"


class ReActState(TypedDict):
    task: Task
    trace: list[dict[str, Any]]
    iteration: int
    tool_calls: int
    total_tokens: int
    final_answer: str | None


def build_prompt(task: Task, tools: dict[str, Tool], trace: list[dict[str, Any]]) -> str:
    tool_lines = "\n".join(f"- {name}: {tool.description}" for name, tool in tools.items())
    history_lines: list[str] = []
    for step in trace:
        history_lines.append(f"Thought/Action: {step['thought']}")
        if "observation" in step:
            history_lines.append(f"Observation: {step['observation']}")
    history = "\n".join(history_lines) if history_lines else "(none yet)"
    return (
        f"Question: {task.question}\n\n"
        f"Available tools:\n{tool_lines}\n\n"
        f"History so far:\n{history}\n\n"
        f"Respond with exactly one line: either '{FINAL_PREFIX} <answer>' or "
        f"'{ACTION_PREFIX} <tool_name> | <json kwargs>'."
    )


def parse_action(text: str) -> tuple[str, dict[str, Any]]:
    body = text[len(ACTION_PREFIX):].strip()
    name_part, _, kwargs_part = body.partition("|")
    name = name_part.strip()
    kwargs_part = kwargs_part.strip()
    kwargs = json.loads(kwargs_part) if kwargs_part else {}
    return name, kwargs


class ReActLoop(AgentLoop):
    name = "react"

    def __init__(self, tools: list[Tool], llm: LLM, config: LoopRunConfig | None = None):
        super().__init__(tools, config)
        self.llm = llm
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(ReActState)
        graph.add_node("reason_act", self._reason_act)
        graph.add_node("observe", self._observe)
        graph.set_entry_point("reason_act")
        graph.add_conditional_edges(
            "reason_act",
            self._route_after_reason,
            {"observe": "observe", "end": END},
        )
        graph.add_edge("observe", "reason_act")
        return graph.compile()

    def _reason_act(self, state: ReActState) -> ReActState:
        prompt = build_prompt(state["task"], self.tools, state["trace"])
        response = self.llm.generate(prompt)
        state["total_tokens"] += response.total_tokens
        state["iteration"] += 1
        text = response.text.strip()
        step: dict[str, Any] = {"thought": text}
        if text.startswith(FINAL_PREFIX):
            state["final_answer"] = text[len(FINAL_PREFIX):].strip()
        state["trace"].append(step)
        return state

    def _observe(self, state: ReActState) -> ReActState:
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

    def _route_after_reason(self, state: ReActState) -> str:
        if state["final_answer"] is not None:
            return "end"
        if state["iteration"] >= self.config.max_iterations:
            return "end"
        return "observe"

    def run(self, task: Task, scorer: ScoringFn = exact_match_scorer) -> TaskResult:
        start = time.perf_counter()
        initial_state: ReActState = {
            "task": task,
            "trace": [],
            "iteration": 0,
            "tool_calls": 0,
            "total_tokens": 0,
            "final_answer": None,
        }
        final_state = self._graph.invoke(
            initial_state,
            config={"recursion_limit": self.config.max_iterations * 4 + 10},
        )
        predicted = final_state["final_answer"] or ""
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
