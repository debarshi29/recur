"""Pydantic request/response models for the FastAPI service."""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

LoopName = Literal["react", "reflection", "plan_execute"]


class TaskSubmitRequest(BaseModel):
    question: str
    gold_answer: str = ""
    loop: LoopName = "react"
    expected_hops: int = 1


class JobCreatedResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: dict[str, Any] | None = Field(default=None)
    error: str | None = Field(default=None)
