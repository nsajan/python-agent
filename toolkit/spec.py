"""Data structures shared between the chat app and custom tools."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional


@dataclass
class ToolSpec:
    """Descriptor for a tool that can be called by the language model."""

    name: str
    description: str
    parameters: Dict[str, Any]
    run: Callable[[Dict[str, Any]], Any]
    author: Optional[str] = None
