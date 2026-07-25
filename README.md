# recur

A production-grade, head-to-head comparison of three agentic control-loop
architectures — **ReAct**, **Reflection**, and **Plan-Execute** — built on
LangGraph and evaluated against the same multi-hop research/QA benchmark.

All three patterns share one harness: one `Task` definition, one `Tool`
interface, one retry/backoff policy, one scoring function. The only
variable under test is the control-loop architecture itself.

## Status

Sprint 0 (harness & contract) complete. See `docs/` for sprint plan and ADRs
as they land.

## Layout

```
harness/       shared contracts: Task, Tool, AgentLoop, ToolResult, TaskResult
loops/         one file per pattern: react_loop.py, reflection_loop.py, plan_execute_loop.py
eval/          QA dataset + comparison runner
tests/         unit + integration tests, run in CI on every PR
docs/adrs/     one ADR per loop pattern documenting observed trade-offs
```

## Development

```bash
pip install -r requirements-dev.txt
pytest
```
