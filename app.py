"""Streamlit chat app with OpenAI API integration and dynamic tool execution."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

# Configure Streamlit runtime for hosting platforms like Railway where the
# application must bind to the provided ``PORT`` on all interfaces. The
# variables only take effect if Streamlit is launched via ``streamlit run``,
# which is how the project is deployed. Setting sensible defaults here avoids
# having to duplicate the configuration in multiple deploy targets.
os.environ.setdefault("STREAMLIT_SERVER_ADDRESS", "0.0.0.0")
if "PORT" in os.environ:
    os.environ.setdefault("STREAMLIT_SERVER_PORT", os.environ["PORT"])
os.environ.setdefault("STREAMLIT_SERVER_HEADLESS", "true")

import streamlit as st

from toolkit import ToolSpec


OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_MODEL = "gpt-4o-mini"
TOOLS_PATH = Path(__file__).parent / "tools"


class ToolLoadingError(RuntimeError):
    """Raised when a tool module cannot be loaded correctly."""


def to_openai_tool(tool: ToolSpec) -> Dict[str, Any]:
    """Return the OpenAI tool schema representation."""

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
        },
    }


@st.cache_resource(show_spinner=False)
def load_tools(tool_dir: Path = TOOLS_PATH) -> Dict[str, ToolSpec]:
    """Discover and load tool specs from Python modules in ``tool_dir``."""

    tools: Dict[str, ToolSpec] = {}
    if not tool_dir.exists():
        return tools

    for path in sorted(tool_dir.glob("*.py")):
        if path.name == "__init__.py":
            continue
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            raise ToolLoadingError(f"Unable to create spec for {path.name}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore[assignment]
        tool: Optional[ToolSpec] = getattr(module, "TOOL", None)
        if tool is None:
            raise ToolLoadingError(
                f"Tool module '{path.name}' must define a global TOOL variable."
            )
        if tool.name in tools:
            raise ToolLoadingError(
                f"Duplicate tool name '{tool.name}' detected in {path.name}."
            )
        tools[tool.name] = tool
    return tools


def get_api_key(user_supplied_key: str | None = None) -> Optional[str]:
    """Return the API key from user input or environment."""

    key = (user_supplied_key or "").strip()
    if key:
        return key
    return os.getenv("OPENAI_API_KEY")


def call_openai(
    api_key: str,
    messages: List[Dict[str, Any]],
    tools: Dict[str, ToolSpec],
    model: str,
    temperature: float,
) -> Dict[str, Any]:
    """Send the conversation to OpenAI's API and return the raw response."""

    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
    }

    if tools:
        payload["tools"] = [to_openai_tool(tool) for tool in tools.values()]
        payload["tool_choice"] = "auto"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    response = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def execute_tool(tool: ToolSpec, arguments_json: str) -> str:
    """Execute a tool and return its stringified output."""

    try:
        arguments = json.loads(arguments_json or "{}")
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Tool '{tool.name}' received invalid JSON arguments: {arguments_json}"
        ) from exc

    result = tool.run(arguments)
    if not isinstance(result, str):
        result = json.dumps(result, ensure_ascii=False)
    return result


def stream_openai_response(
    api_key: str,
    base_messages: List[Dict[str, Any]],
    tools: Dict[str, ToolSpec],
    model: str,
    temperature: float,
) -> str:
    """Handle the full tool-calling loop and return the assistant's reply."""

    messages = list(base_messages)

    while True:
        raw_response = call_openai(
            api_key=api_key,
            messages=messages,
            tools=tools,
            model=model,
            temperature=temperature,
        )
        choice = raw_response["choices"][0]
        message = choice["message"]

        if message.get("tool_calls"):
            messages.append(message)
            for tool_call in message["tool_calls"]:
                call_id = tool_call.get("id")
                function_data = tool_call.get("function", {})
                tool_name = function_data.get("name")
                if tool_name not in tools:
                    raise ToolLoadingError(f"Tool '{tool_name}' not found.")
                tool_spec = tools[tool_name]
                tool_output = execute_tool(tool_spec, function_data.get("arguments", "{}"))
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "name": tool_name,
                        "content": tool_output,
                    }
                )
            # Continue loop to get final assistant message after tool outputs.
            continue

        messages.append(message)
        return message.get("content", "")


def render_sidebar(tool_specs: Dict[str, ToolSpec]) -> Dict[str, Any]:
    """Render sidebar controls and return the current configuration."""

    with st.sidebar:
        st.header("Configuration")
        api_key_input = st.text_input(
            "OpenAI API Key",
            type="password",
            help="Key is used only for this session. Leave blank to use OPENAI_API_KEY env var.",
        )
        model = st.text_input("Model", value=DEFAULT_MODEL)
        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)

        st.markdown("---")
        st.subheader("Available Tools")
        if not tool_specs:
            st.caption("No custom tools found in the tools directory.")
        else:
            for tool in tool_specs.values():
                with st.expander(tool.name, expanded=False):
                    st.markdown(f"**Description:** {tool.description}")
                    if tool.author:
                        st.markdown(f"**Author:** {tool.author}")
                    st.code(json.dumps(tool.parameters, indent=2))

    return {
        "api_key_input": api_key_input,
        "model": model,
        "temperature": temperature,
    }


def init_session_state() -> None:
    """Initialize Streamlit session state variables used by the app."""

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "tool_specs" not in st.session_state:
        st.session_state.tool_specs = load_tools()


def main() -> None:
    st.set_page_config(page_title="OpenAI Tool Chat", page_icon="🤖", layout="wide")
    st.title("🔧 OpenAI Tool-Enabled Chat")
    st.caption(
        "Chat with OpenAI models, optionally enhanced with Python tools defined in the `tools/` directory."
    )

    init_session_state()
    config = render_sidebar(st.session_state.tool_specs)

    api_key = get_api_key(config["api_key_input"])
    if not api_key:
        st.warning("Please provide an OpenAI API key to start chatting.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input("Send a message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        if not api_key:
            error_msg = "No API key configured."
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
            with st.chat_message("assistant"):
                st.error(error_msg)
            return

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Generating response…")
            try:
                reply = stream_openai_response(
                    api_key=api_key,
                    base_messages=st.session_state.messages,
                    tools=st.session_state.tool_specs,
                    model=config["model"],
                    temperature=config["temperature"],
                )
            except requests.HTTPError as exc:
                reply = f"OpenAI API error: {exc}"
            except Exception as exc:  # noqa: BLE001
                reply = f"Unexpected error: {exc}"
            placeholder.markdown(reply)

        st.session_state.messages.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
