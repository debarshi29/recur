"""
Plan-Execute loop: plan the full step sequence up front, execute each
step (each step can call a tool via the shared ACTION/FINAL_ANSWER
protocol), and replan -- discarding the rest of the stale plan rather
than restarting from scratch -- whenever the model judges a step to have
failed.

Graph shape:

    plan -> execute -> observe -> execute -> ... -> synthesize -> END
              (or)-> replan -> execute (new plan) -> ...

`plan` and `replan` both emit `PLAN: step1 | step2 | ...`; `execute`
emits `ACTION: ...`, `STEP_DONE: ...`, `REPLAN: <reason>`, or
`FINAL_ANSWER: ...` for the current step. Reaching the end of the plan
without a final answer routes to `synthesize`, which asks the model to
produce one from the accumulated trace.
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

PLAN_PREFIX = "PLAN:"
STEP_DONE_PREFIX = "STEP_DONE:"
REPLAN_PREFIX = "REPLAN:"


class PlanExecuteState(TypedDict):
    task: Task
    trace: list[dict[str, Any]]
    iteration: int
    tool_calls: int
    total_tokens: int
    plan: list[str] | None
    step_idx: int
    replans: int
    final_answer: str | None


def build_plan_prompt(task: Task, tools: dict[str, Tool], trace: list[dict[str, Any]]) -> str:
    return (
        f"Question: {task.question}\n\n"
        f"Available tools:\n{tool_lines(tools)}\n\n"
        f"Produce a short plan as 'PLAN: step1 | step2 | ...' (2-4 concise "
        f"steps), or '{FINAL_PREFIX} <answer>' directly if the question "
        f"needs no plan."
    )


def build_execute_prompt(
    task: Task,
    tools: dict[str, Tool],
    step: str,
    step_idx: int,
    plan_len: int,
    trace: list[dict[str, Any]],
) -> str:
    history_lines: list[str] = []
    for record in trace:
        if record.get("stage") == "execute" and record.get("step") == step:
            history_lines.append(f"Thought/Action: {record['thought']}")
            if "observation" in record:
                history_lines.append(f"Observation: {record['observation']}")
    history = "\n".join(history_lines) if history_lines else "(none yet)"
    final_marker = " (final step)" if step_idx == plan_len - 1 else ""
    return (
        f"Question: {task.question}\n\n"
        f"Available tools:\n{tool_lines(tools)}\n\n"
        f"Current step{final_marker}: {step}\n\n"
        f"History for this step:\n{history}\n\n"
        f"Respond with exactly one line: '{ACTION_PREFIX} <tool_name> | "
        f"<json kwargs>' to use a tool, '{STEP_DONE_PREFIX} <note>' once "
        f"this step is complete, '{FINAL_PREFIX} <answer>' if you can "
        f"answer the whole question now, or '{REPLAN_PREFIX} <reason>' if "
        f"this step has failed and the plan needs to change."
    )


def build_replan_prompt(
    task: Task, tools: dict[str, Tool], failed_step: str, reason: str, trace: list[dict[str, Any]]
) -> str:
    return (
        f"Question: {task.question}\n\n"
        f"Available tools:\n{tool_lines(tools)}\n\n"
        f"The current step failed: {failed_step}\n"
        f"Reason given: {reason}\n\n"
        f"Produce a new plan as 'PLAN: step1 | step2 | ...' to recover, or "
        f"'{FINAL_PREFIX} <answer>' if you can answer despite the failure."
    )


def build_synthesize_prompt(task: Task, trace: list[dict[str, Any]]) -> str:
    lines = [f"{r.get('stage')}: {r['thought']}" for r in trace]
    return (
        f"Question: {task.question}\n\n"
        f"All planned steps are complete. Full trace:\n" + "\n".join(lines) + "\n\n"
        f"Respond with exactly one line: '{FINAL_PREFIX} <answer>'."
    )


class PlanExecuteLoop(AgentLoop):
    name = "plan_execute"

    def __init__(self, tools: list[Tool], llm: LLM, config: LoopRunConfig | None = None):
        super().__init__(tools, config)
        self.llm = llm
        self.checkpointer = new_checkpointer()
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(PlanExecuteState)
        graph.add_node("plan", self._plan)
        graph.add_node("execute", self._execute)
        graph.add_node("observe", self._observe)
        graph.add_node("replan", self._replan)
        graph.add_node("synthesize", self._synthesize)
        graph.set_entry_point("plan")
        graph.add_conditional_edges(
            "plan",
            self._route_after_plan,
            {"execute": "execute", "plan": "plan", "end": END},
        )
        graph.add_conditional_edges(
            "execute",
            self._route_after_execute,
            {
                "observe": "observe",
                "replan": "replan",
                "synthesize": "synthesize",
                "execute": "execute",
                "end": END,
            },
        )
        graph.add_edge("observe", "execute")
        graph.add_conditional_edges(
            "replan",
            self._route_after_replan,
            {"execute": "execute", "replan": "replan", "end": END},
        )
        graph.add_edge("synthesize", END)
        return graph.compile(checkpointer=self.checkpointer)

    # -- nodes --------------------------------------------------------

    def _plan(self, state: PlanExecuteState) -> PlanExecuteState:
        prompt = build_plan_prompt(state["task"], self.tools, state["trace"])
        response = self.llm.generate(prompt)
        state["total_tokens"] += response.total_tokens
        state["iteration"] += 1
        text = response.text.strip()
        state["trace"].append({"stage": "plan", "thought": text})
        if text.startswith(FINAL_PREFIX):
            state["final_answer"] = text[len(FINAL_PREFIX):].strip()
        elif text.startswith(PLAN_PREFIX):
            steps = [s.strip() for s in text[len(PLAN_PREFIX):].split("|") if s.strip()]
            state["plan"] = steps
            state["step_idx"] = 0
        return state

    def _execute(self, state: PlanExecuteState) -> PlanExecuteState:
        plan = state["plan"] or []
        step = plan[state["step_idx"]]
        prompt = build_execute_prompt(
            state["task"], self.tools, step, state["step_idx"], len(plan), state["trace"]
        )
        response = self.llm.generate(prompt)
        state["total_tokens"] += response.total_tokens
        state["iteration"] += 1
        text = response.text.strip()
        state["trace"].append({"stage": "execute", "step": step, "thought": text})
        if text.startswith(FINAL_PREFIX):
            state["final_answer"] = text[len(FINAL_PREFIX):].strip()
        elif text.startswith(STEP_DONE_PREFIX):
            state["step_idx"] += 1
        return state

    def _observe(self, state: PlanExecuteState) -> PlanExecuteState:
        step_record = state["trace"][-1]
        try:
            tool_name, kwargs = parse_action(step_record["thought"])
        except Exception as e:  # noqa: BLE001 — a malformed action is an observation, not a crash
            step_record["observation"] = f"error parsing action: {e}"
            return state
        result = self._call_tool(tool_name, **kwargs)
        state["tool_calls"] += 1
        step_record["observation"] = result.output if result.ok else f"ERROR: {result.error}"
        return state

    def _replan(self, state: PlanExecuteState) -> PlanExecuteState:
        plan = state["plan"] or []
        failed_step = plan[state["step_idx"]] if state["step_idx"] < len(plan) else "(unknown step)"
        reason = state["trace"][-1]["thought"][len(REPLAN_PREFIX):].strip()
        prompt = build_replan_prompt(state["task"], self.tools, failed_step, reason, state["trace"])
        response = self.llm.generate(prompt)
        state["total_tokens"] += response.total_tokens
        state["iteration"] += 1
        text = response.text.strip()
        state["trace"].append({"stage": "replan", "thought": text})
        if text.startswith(FINAL_PREFIX):
            state["final_answer"] = text[len(FINAL_PREFIX):].strip()
        elif text.startswith(PLAN_PREFIX):
            steps = [s.strip() for s in text[len(PLAN_PREFIX):].split("|") if s.strip()]
            state["plan"] = steps
            state["step_idx"] = 0
        state["replans"] += 1
        return state

    def _synthesize(self, state: PlanExecuteState) -> PlanExecuteState:
        prompt = build_synthesize_prompt(state["task"], state["trace"])
        response = self.llm.generate(prompt)
        state["total_tokens"] += response.total_tokens
        state["iteration"] += 1
        text = response.text.strip()
        if text.startswith(FINAL_PREFIX):
            text = text[len(FINAL_PREFIX):].strip()
        state["final_answer"] = text
        state["trace"].append({"stage": "synthesize", "thought": text})
        return state

    # -- routing --------------------------------------------------------

    def _route_after_plan(self, state: PlanExecuteState) -> str:
        if state["final_answer"] is not None:
            return "end"
        if state["plan"]:
            return "execute"
        if state["iteration"] >= self.config.max_iterations:
            state["final_answer"] = ""
            return "end"
        return "plan"

    def _route_after_execute(self, state: PlanExecuteState) -> str:
        if state["final_answer"] is not None:
            return "end"
        last_thought = state["trace"][-1]["thought"]
        if last_thought.startswith(ACTION_PREFIX):
            return "observe"
        if last_thought.startswith(REPLAN_PREFIX):
            return "replan"
        plan = state["plan"] or []
        if state["step_idx"] >= len(plan):
            return "synthesize"
        if state["iteration"] >= self.config.max_iterations:
            state["final_answer"] = ""
            return "end"
        return "execute"

    def _route_after_replan(self, state: PlanExecuteState) -> str:
        if state["final_answer"] is not None:
            return "end"
        if state["plan"]:
            return "execute"
        if state["iteration"] >= self.config.max_iterations:
            state["final_answer"] = ""
            return "end"
        return "replan"

    def run(self, task: Task, scorer: ScoringFn = exact_match_scorer) -> TaskResult:
        start = time.perf_counter()
        initial_state: PlanExecuteState = {
            "task": task,
            "trace": [],
            "iteration": 0,
            "tool_calls": 0,
            "total_tokens": 0,
            "plan": None,
            "step_idx": 0,
            "replans": 0,
            "final_answer": None,
        }
        final_state = run_graph(
            self._graph,
            initial_state,
            thread_id=task.id,
            timeout_s=self.config.timeout_s,
            recursion_limit=self.config.max_iterations * 6 + 20,
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
