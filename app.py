"""Streamlit chat app with OpenAI API integration and dynamic tool execution."""
from __future__ import annotations

import importlib.util
import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
DEFAULT_SYSTEM_PROMPT = (
    "You are an AI assistant that can run Python helper functions. "
    "Use the knowledge base provided by the user to ground your replies."
)
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
        st.subheader("Knowledge Base")
        system_prompt = st.text_area(
            "System Prompt / Knowledge Base",
            value=st.session_state.system_prompt,
            height=200,
            help="Provide instructions or reference knowledge that every response should follow.",
        )
        if system_prompt != st.session_state.system_prompt:
            st.session_state.system_prompt = system_prompt

        st.markdown("---")
        st.subheader("Available Tools")
        if not tool_specs:
            st.caption("No custom tools found in the tools directory.")
        else:
            runtime_names = set(st.session_state.runtime_tools.keys())
            for name in sorted(tool_specs):
                tool = tool_specs[name]
                with st.expander(tool.name, expanded=False):
                    st.markdown(f"**Description:** {tool.description}")
                    if tool.author:
                        st.markdown(f"**Author:** {tool.author}")
                    origin = "Live session" if tool.name in runtime_names else "tools directory"
                    st.caption(f"Origin: {origin}")
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
    if "file_tools" not in st.session_state:
        st.session_state.file_tools = load_tools()
    if "runtime_tools" not in st.session_state:
        st.session_state.runtime_tools = {}
    st.session_state.tool_specs = {**st.session_state.file_tools, **st.session_state.runtime_tools}
    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT
    if "builder_name" not in st.session_state:
        st.session_state.builder_name = "runtime_tool"
    if "builder_description" not in st.session_state:
        st.session_state.builder_description = "Describe what your tool does."
    if "builder_parameters" not in st.session_state:
        st.session_state.builder_parameters = json.dumps(
            {
                "type": "object",
                "properties": {},
                "required": [],
            },
            indent=2,
        )
    if "builder_code" not in st.session_state:
        st.session_state.builder_code = (
            "def run_tool(args: dict) -> str:\n"
            "    \"\"\"Return a friendly greeting using the provided arguments.\"\"\"\n"
            "    name = args.get('name', 'there')\n"
            "    return f'Hello, {name}! This response came from a runtime tool.'\n"
        )
    if "builder_test_args" not in st.session_state:
        st.session_state.builder_test_args = json.dumps({"name": "Streamlit"}, indent=2)
    if "builder_validation_state" not in st.session_state:
        st.session_state.builder_validation_state = None
    if "builder_validation_message" not in st.session_state:
        st.session_state.builder_validation_message = None
    if "builder_validation_preview" not in st.session_state:
        st.session_state.builder_validation_preview = None
    if "builder_validated_payload" not in st.session_state:
        st.session_state.builder_validated_payload = None
    if "builder_validated_snapshot" not in st.session_state:
        st.session_state.builder_validated_snapshot = None


def validate_runtime_tool(
    name: str,
    description: str,
    parameters_text: str,
    code: str,
    test_args_text: str,
) -> Tuple[ToolSpec, str]:
    """Validate a runtime tool definition and return its spec and test output."""

    if not name.strip():
        raise ValueError("Tool name is required.")

    try:
        parameters = json.loads(parameters_text) if parameters_text.strip() else {}
    except json.JSONDecodeError as exc:
        raise ValueError("Parameters must be valid JSON.") from exc
    if not isinstance(parameters, dict):
        raise ValueError("Parameters JSON must describe an object schema.")

    namespace: Dict[str, Any] = {}
    try:
        exec(code, {}, namespace)  # noqa: S102 - exec is required for the live tool editor
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Tool code failed to execute: {exc}") from exc

    run_callable = namespace.get("run_tool")
    if run_callable is None or not callable(run_callable):
        raise ValueError("Define a callable `run_tool(args: dict)` function in the snippet.")

    try:
        test_args = json.loads(test_args_text) if test_args_text.strip() else {}
    except json.JSONDecodeError as exc:
        raise ValueError("Test arguments must be valid JSON.") from exc
    if not isinstance(test_args, dict):
        raise ValueError("Test arguments must be a JSON object.")

    try:
        preview_result = run_callable(test_args)
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Tool execution raised an error: {exc}") from exc

    if not isinstance(preview_result, str):
        preview_render = json.dumps(preview_result, ensure_ascii=False, indent=2)
    else:
        preview_render = preview_result

    tool_spec = ToolSpec(
        name=name.strip(),
        description=description.strip() or "Runtime tool",
        parameters=parameters,
        run=run_callable,
    )

    return tool_spec, preview_render


def render_tool_workshop() -> None:
    """Render the live tool editor and handle validation/addition workflow."""

    st.subheader("🛠️ Tool Workshop")
    st.caption(
        "Author Python helpers at runtime. Define a `run_tool(args: dict)` function, "
        "validate it with sample input, then add it to this session's toolset."
    )

    name = st.text_input("Tool Name", key="builder_name")
    description = st.text_input("Description", key="builder_description")
    parameters_text = st.text_area(
        "JSON Schema Parameters",
        key="builder_parameters",
        height=160,
        help="Provide the JSON schema describing the tool arguments (as used by the OpenAI API).",
    )
    test_args_text = st.text_area(
        "Test Arguments (JSON)",
        key="builder_test_args",
        height=120,
        help="Sample payload that will be sent to `run_tool` during validation.",
    )
    code = st.text_area(
        "Tool Code",
        key="builder_code",
        height=240,
        help="Write a Python snippet that defines a `run_tool(args: dict)` function.",
    )

    current_snapshot = (name, description, parameters_text, code)
    stored_snapshot = st.session_state.builder_validated_snapshot
    if stored_snapshot and stored_snapshot != current_snapshot:
        st.session_state.builder_validated_payload = None
        st.session_state.builder_validation_state = None
        st.session_state.builder_validation_message = None
        st.session_state.builder_validation_preview = None
        st.session_state.builder_validated_snapshot = None

    validate_clicked = st.button("Validate Tool", type="primary")
    add_clicked = st.button(
        "Add Tool",
        disabled=st.session_state.builder_validated_payload is None,
    )

    if validate_clicked:
        try:
            _tool_spec, preview = validate_runtime_tool(
                name=name,
                description=description,
                parameters_text=parameters_text,
                code=code,
                test_args_text=test_args_text,
            )
        except Exception as exc:  # noqa: BLE001
            st.session_state.builder_validated_payload = None
            st.session_state.builder_validation_state = "error"
            st.session_state.builder_validation_message = str(exc)
            st.session_state.builder_validation_preview = traceback.format_exc()
            st.session_state.builder_validated_snapshot = None
        else:
            st.session_state.builder_validated_payload = {
                "name": name,
                "description": description,
                "parameters_text": parameters_text,
                "code": code,
                "test_args_text": test_args_text,
            }
            st.session_state.builder_validation_state = "success"
            st.session_state.builder_validation_message = (
                "Tool validated successfully. Review the preview output below."
            )
            st.session_state.builder_validation_preview = preview
            st.session_state.builder_validated_snapshot = current_snapshot

    payload = st.session_state.builder_validated_payload
    if add_clicked and payload is not None:
        try:
            tool, _ = validate_runtime_tool(**payload)
        except Exception as exc:  # noqa: BLE001
            st.session_state.builder_validation_state = "error"
            st.session_state.builder_validation_message = str(exc)
            st.session_state.builder_validation_preview = traceback.format_exc()
            st.session_state.builder_validated_payload = None
            st.session_state.builder_validated_snapshot = None
        else:
            if tool.name in st.session_state.tool_specs:
                st.session_state.builder_validation_state = "error"
                st.session_state.builder_validation_message = (
                    f"A tool named '{tool.name}' already exists. Choose another name."
                )
                st.session_state.builder_validation_preview = None
            else:
                st.session_state.runtime_tools[tool.name] = tool
                st.session_state.tool_specs = {
                    **st.session_state.file_tools,
                    **st.session_state.runtime_tools,
                }
                st.session_state.builder_validation_state = "success"
                st.session_state.builder_validation_message = (
                    f"Tool '{tool.name}' added. It is now available for the assistant."
                )
                st.session_state.builder_validation_preview = None
                st.session_state.builder_validated_payload = None
                st.session_state.builder_validated_snapshot = None

    status = st.session_state.builder_validation_state
    message = st.session_state.builder_validation_message
    preview = st.session_state.builder_validation_preview
    if status == "success" and message:
        st.success(message)
        if preview:
            st.code(preview, language="text")
    elif status == "error" and message:
        st.error(message)
        if preview:
            st.code(preview, language="python")


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
                    base_messages=
                        [{"role": "system", "content": st.session_state.system_prompt}]
                        + st.session_state.messages,
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

    st.divider()
    render_tool_workshop()


if __name__ == "__main__":
    main()
