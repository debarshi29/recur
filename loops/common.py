"""
Shared ACTION/FINAL_ANSWER tool-use protocol used by every loop pattern
that lets the LLM call a tool mid-reasoning (ReAct's act/observe step;
Reflection's and Plan-Execute's draft/execute steps). Keeping this parsing
in one place means a prompt-format change happens once, not once per
pattern.
"""
from __future__ import annotations

import json
from typing import Any

from harness.contracts import Tool

FINAL_PREFIX = "FINAL_ANSWER:"
ACTION_PREFIX = "ACTION:"


def tool_lines(tools: dict[str, Tool]) -> str:
    return "\n".join(f"- {name}: {tool.description}" for name, tool in tools.items())


def parse_action(text: str) -> tuple[str, dict[str, Any]]:
    body = text[len(ACTION_PREFIX):].strip()
    name_part, _, kwargs_part = body.partition("|")
    name = name_part.strip()
    kwargs_part = kwargs_part.strip()
    kwargs = json.loads(kwargs_part) if kwargs_part else {}
    return name, kwargs
