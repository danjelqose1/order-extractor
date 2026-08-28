from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Dict, Type

from pydantic import BaseModel


BETA_DEFAULT_MODEL = "gpt-5.6-terra"
BETA_DEFAULT_REASONING_EFFORT = "medium"


def beta_model_name() -> str:
    return str(os.getenv("BETA_MODEL") or BETA_DEFAULT_MODEL).strip() or BETA_DEFAULT_MODEL


def beta_reasoning_effort() -> str:
    value = str(os.getenv("BETA_REASONING_EFFORT") or BETA_DEFAULT_REASONING_EFFORT).strip().lower()
    return value if value in {"none", "low", "medium", "high", "xhigh", "max"} else BETA_DEFAULT_REASONING_EFFORT


def strict_json_schema(model: Type[BaseModel]) -> Dict[str, Any]:
    """Return an OpenAI strict-output schema with every object key required."""
    schema = deepcopy(model.model_json_schema())

    def require_all_properties(node: Any) -> None:
        if isinstance(node, dict):
            properties = node.get("properties")
            if isinstance(properties, dict):
                node["required"] = list(properties.keys())
            for value in node.values():
                require_all_properties(value)
        elif isinstance(node, list):
            for value in node:
                require_all_properties(value)

    require_all_properties(schema)
    return schema


__all__ = [
    "BETA_DEFAULT_MODEL",
    "BETA_DEFAULT_REASONING_EFFORT",
    "beta_model_name",
    "beta_reasoning_effort",
    "strict_json_schema",
]
