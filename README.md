# Streamlit OpenAI Tool Chat

A Streamlit-based chat interface for OpenAI models that supports tool calling through
custom Python scripts. Place scripts in the `tools/` directory to expose their
functionality to the assistant.

## Features

- Chat UI built with Streamlit's conversational components.
- Configurable OpenAI model, temperature, and API key input via the sidebar.
- Automatic discovery of Python tools in the `tools/` folder.
- Example math evaluation tool demonstrating tool registration.
- Robust tool-calling loop that executes tools requested by the model.

## Getting Started

1. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

2. **Set your OpenAI API key**

   Set the `OPENAI_API_KEY` environment variable or enter the key in the app's sidebar.

3. **Run the Streamlit app**

   ```bash
   streamlit run app.py
   ```

4. **Add custom tools**

   Create Python files in the `tools/` directory. Each file must expose a `TOOL`
   variable that is an instance of `toolkit.ToolSpec` with a callable `run`
   function. The function receives a dictionary of arguments and should return a
   string or JSON-serializable object.

## Example Tool

The repository includes `tools/sample_math.py`, which safely evaluates
mathematical expressions using Python's `math` module. Use it as a template for
creating your own automation scripts.

## Configuration Notes

- The app calls the [Chat Completions API](https://platform.openai.com/docs/api-reference/chat)
  with the selected model and any discovered tools.
- Tool execution occurs on the server running Streamlit, so ensure your tools are
  safe and trusted before deploying.
