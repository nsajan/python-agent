"""Example tool that evaluates simple arithmetic expressions."""

from __future__ import annotations

import math
from typing import Any, Dict

from toolkit import ToolSpec


PARAMETERS_SCHEMA = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "description": "Mathematical expression to evaluate using Python's math module.",
        }
    },
    "required": ["expression"],
}


def evaluate_math(arguments: Dict[str, Any]) -> str:
    """Evaluate a mathematical expression with access to the math module."""

    expression = arguments.get("expression")
    if not isinstance(expression, str):
        raise ValueError("The 'expression' argument must be a string.")

    # Create a restricted evaluation environment exposing only safe math functions.
    safe_globals = {name: getattr(math, name) for name in dir(math) if not name.startswith("_")}
    safe_globals["__builtins__"] = {}
    try:
        result = eval(expression, safe_globals, {})  # noqa: S307 - controlled globals
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Failed to evaluate expression '{expression}': {exc}") from exc

    return str(result)


TOOL = ToolSpec(
    name="evaluate_math",
    description=(
        "Safely evaluate a mathematical expression. Accepts an 'expression' string "
        "that can use functions from Python's math module."
    ),
    parameters=PARAMETERS_SCHEMA,
    run=evaluate_math,
    author="Sample Tool",
)
