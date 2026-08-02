"""
Reflection loop: generate an initial answer, critique it, revise if the
critique flags a problem, repeat until the critique accepts the draft,
max_critique_rounds is hit (the draft is force-accepted), or
max_iterations is hit (the best draft seen is used as a fallback).

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
    critique_rounds: int
    best_draft: str | None


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
    """
    Convergence bound: a real, capable critic can nitpick a
    correct-but-imperfect draft indefinitely rather than ever emitting
    GOOD (see docs/writeup.md's real-LLM section -- Reflection's critique
    step trended toward `max_iterations` with a 0% GOOD-convergence rate).
    `max_critique_rounds` caps how many REVISE rounds are honored,
    separate from and typically tighter than `max_iterations`; once hit,
    the current draft is force-accepted rather than discarded. On any
    cap-out (including the pre-existing `max_iterations` bound), the loop
    falls back to `best_draft` -- the most recent non-None draft seen --
    instead of returning an empty final answer.
    """

    name = "reflection"

    def __init__(
        self,
        tools: list[Tool],
        llm: LLM,
        config: LoopRunConfig | None = None,
        max_critique_rounds: int = 3,
    ):
        super().__init__(tools, config)
        self.llm = llm
        self.max_critique_rounds = max_critique_rounds
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
            state["best_draft"] = state["draft"]
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
        state["critique_rounds"] += 1
        text = response.text.strip()
        state["trace"].append({"stage": "critique", "thought": text})
        if text.startswith(REVISE_PREFIX) and state["critique_rounds"] < self.max_critique_rounds:
            state["feedback"] = text[len(REVISE_PREFIX):].strip()
            state["draft"] = None  # discard the unrevised draft so a stale
            # value can't be mistaken for a fresh one on the next round
        else:
            # Either the critic said GOOD, or it hit max_critique_rounds --
            # force-accept the current draft rather than revise forever.
            state["final_answer"] = state["draft"]
        return state

    def _route_after_draft(self, state: ReflectionState) -> str:
        # Routing functions only decide the next node -- they must not
        # mutate state, since a router's writes aren't persisted the way
        # a node's return value is (no node runs afterward to capture
        # them). Cap-out fallback to best_draft is handled in run().
        if state["draft"] is not None:
            return "critique"
        if state["iteration"] >= self.config.max_iterations:
            return "end"
        return "observe"

    def _route_after_critique(self, state: ReflectionState) -> str:
        if state["final_answer"] is not None:
            return "end"
        if state["iteration"] >= self.config.max_iterations:
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
            "critique_rounds": 0,
            "best_draft": None,
        }
        final_state = run_graph(
            self._graph,
            initial_state,
            thread_id=task.id,
            timeout_s=self.config.timeout_s,
            recursion_limit=self.config.max_iterations * 6 + 10,
        )
        # final_answer stays None when max_iterations is hit mid-revision
        # (no node runs afterward to set it) -- fall back to the last
        # real draft seen rather than an empty answer.
        predicted = final_state.get("final_answer")
        if predicted is None:
            predicted = final_state.get("best_draft") or ""
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
