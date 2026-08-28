from __future__ import annotations

import os


BETA_DEFAULT_MODEL = "gpt-5.6-terra"
BETA_DEFAULT_REASONING_EFFORT = "medium"


def beta_model_name() -> str:
    return str(os.getenv("BETA_MODEL") or BETA_DEFAULT_MODEL).strip() or BETA_DEFAULT_MODEL


def beta_reasoning_effort() -> str:
    value = str(os.getenv("BETA_REASONING_EFFORT") or BETA_DEFAULT_REASONING_EFFORT).strip().lower()
    return value if value in {"none", "low", "medium", "high", "xhigh", "max"} else BETA_DEFAULT_REASONING_EFFORT


__all__ = [
    "BETA_DEFAULT_MODEL",
    "BETA_DEFAULT_REASONING_EFFORT",
    "beta_model_name",
    "beta_reasoning_effort",
]
