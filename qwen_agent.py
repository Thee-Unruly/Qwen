#!/usr/bin/env python3
"""
qwen_agent.py — a local AI agent powered by Qwen3 + Ollama, with real tool use.

This is the "impressive" demo: instead of just answering a prompt, the model
decides WHICH tool it needs, calls it, reads the result, and keeps going until
it can give a final answer. That decision-making loop — not the chat itself —
is what's worth showing a technical audience.

Runs fully offline once the model is downloaded. No API key needed.

TOOLS AVAILABLE TO THE AGENT
    calculate(expression)   - safe arithmetic, no eval() footguns
    list_notes()             - lists the sample notes files
    search_notes(query)      - keyword search across sample_notes/*.md
    read_file(path)          - reads a specific file

SETUP
    1. Install Ollama:      https://ollama.com/download
    2. Pull a model:        ollama pull qwen3:4b   (needs tool-calling support)
    3. Install the client:  pip install ollama

USAGE
    python qwen_agent.py --check
    python qwen_agent.py "What's our Q3 engineering budget split?"
    python qwen_agent.py "Is the payments migration still at risk? Check notes and meeting notes."
    python qwen_agent.py "What's 18 out of 50 as a percentage, and what does that tell you about the beta pilot?"
"""

import argparse
import ast
import operator
import os
import sys
import glob

MODEL_DEFAULT = "qwen3:4b"
NOTES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_notes")
MAX_TOOL_ITERATIONS = 5
MAX_FILE_CHARS = 6000

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
GREEN = "\033[32m"
MAGENTA = "\033[35m"


def color(text, code):
    if not sys.stdout.isatty():
        return text
    return f"{code}{text}{RESET}"


def die(message, hint=None):
    print(color(f"\n✗ {message}", RED))
    if hint:
        print(color(f"  → {hint}", DIM))
    sys.exit(1)


# ---------------------------------------------------------------------------
# Tools — plain Python functions. Each returns a string the model will read.
# ---------------------------------------------------------------------------

_SAFE_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub,
    ast.Mult: operator.mul, ast.Div: operator.truediv,
    ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _safe_eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _SAFE_OPS:
        return _SAFE_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("Unsupported expression")


def calculate(expression: str) -> str:
    """Safely evaluate arithmetic — no eval(), no code execution risk."""
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception:
        return f"Error: could not evaluate '{expression}' as arithmetic."


def list_notes() -> str:
    files = sorted(glob.glob(os.path.join(NOTES_DIR, "*.md")))
    if not files:
        return "No notes found."
    return "\n".join(os.path.basename(f) for f in files)


def search_notes(query: str) -> str:
    files = sorted(glob.glob(os.path.join(NOTES_DIR, "*.md")))
    if not files:
        return "No notes available to search."

    query_lower = query.lower()
    hits = []
    for path in files:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        for i, line in enumerate(lines):
            if query_lower in line.lower():
                snippet = "".join(lines[max(0, i - 1):i + 2]).strip()
                hits.append(f"[{os.path.basename(path)}]\n{snippet}")

    if not hits:
        return f"No matches for '{query}' in notes."
    return "\n\n".join(hits[:5])


def read_file(path: str) -> str:
    full_path = path if os.path.isabs(path) else os.path.join(NOTES_DIR, path)
    try:
        with open(full_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except IsADirectoryError:
        return f"Error: '{path}' is a directory."
    if len(content) > MAX_FILE_CHARS:
        content = content[:MAX_FILE_CHARS] + "\n...[truncated]"
    return content


TOOL_FUNCTIONS = {
    "calculate": calculate,
    "list_notes": list_notes,
    "search_notes": search_notes,
    "read_file": read_file,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate an arithmetic expression, e.g. '18/50*100'.",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "A math expression"}},
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_notes",
            "description": "List the available note files.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_notes",
            "description": "Search all note files for a keyword and return matching snippets.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string", "description": "Keyword to search for"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the full contents of a specific note file by name, e.g. 'roadmap.md'.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string", "description": "Filename to read"}},
                "required": ["path"],
            },
        },
    },
]

SYSTEM_PROMPT = (
    "You are a helpful local assistant with access to tools for arithmetic and "
    "searching a small set of team notes. Use tools when they'd give you real "
    "information instead of guessing. Be concise and concrete in your final answer."
)


# ---------------------------------------------------------------------------
# Setup checks
# ---------------------------------------------------------------------------

def check_ollama_installed():
    try:
        import ollama  # noqa: F401
    except ImportError:
        die("The 'ollama' Python package isn't installed.", "Run: pip install ollama")


def check_ollama_running_and_model(model):
    import ollama
    try:
        models_response = ollama.list()
    except Exception:
        die(
            "Can't reach the Ollama server.",
            "Make sure Ollama is running: run 'ollama serve' or open the Ollama app.",
        )
        return

    available = {m.get("model", m.get("name", "")) for m in models_response.get("models", [])}
    match = any(model in name or name.split(":")[0] == model.split(":")[0] for name in available)
    if not match:
        die(f"Model '{model}' isn't downloaded yet.", f"Run: ollama pull {model}")


def run_check(model):
    print(color("Checking setup...", DIM))
    check_ollama_installed()
    print(color("  ✓ ollama package installed", GREEN))
    check_ollama_running_and_model(model)
    print(color(f"  ✓ Ollama running, '{model}' available", GREEN))
    print(color(f"  ✓ {len(glob.glob(os.path.join(NOTES_DIR, '*.md')))} sample note(s) found", GREEN))
    print(color("\nAll good — ready to demo.\n", BOLD))


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def _get(obj, key, default=None):
    """Read a field from either a dict chunk or an SDK object chunk."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def run_turn(model, messages, think, stream_fn):
    """
    Runs one streaming turn. Prints Qwen's thinking trace (if the model/mode
    produces one) and its content live, token by token. Returns
    (content_text, tool_calls) once the stream ends.
    """
    content_parts = []
    tool_calls = None
    thinking_header_shown = False
    content_header_shown = False

    for chunk in stream_fn(model=model, messages=messages, tools=TOOL_SCHEMAS, think=think, stream=True):
        msg = _get(chunk, "message", {})
        thinking_piece = _get(msg, "thinking")
        content_piece = _get(msg, "content")
        tc = _get(msg, "tool_calls")

        if thinking_piece:
            if not thinking_header_shown:
                print(color("\n  🧠 thinking...", DIM))
                thinking_header_shown = True
            print(color(thinking_piece, DIM), end="", flush=True)

        if content_piece:
            if not content_header_shown:
                if thinking_header_shown:
                    print()  # newline after the thinking block
                content_header_shown = True
            print(content_piece, end="", flush=True)
            content_parts.append(content_piece)

        if tc:
            tool_calls = tc

    if thinking_header_shown or content_header_shown:
        print()  # tidy trailing newline after streaming

    return "".join(content_parts), tool_calls


def run_agent(model, user_question, think=False, stream_fn=None):
    """stream_fn is injectable for testing; defaults to ollama.chat (which
    returns a generator of chunks when stream=True)."""
    if stream_fn is None:
        import ollama
        stream_fn = ollama.chat

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]

    print(f"\n{BOLD}Question:{RESET} {user_question}" if sys.stdout.isatty() else f"\nQuestion: {user_question}")

    for step in range(MAX_TOOL_ITERATIONS):
        try:
            content, tool_calls = run_turn(model, messages, think, stream_fn)
        except Exception as e:
            die("Something went wrong talking to Ollama.", f"Details: {e}")
            return

        if not tool_calls:
            return content

        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})

        for call in tool_calls:
            fn = _get(call, "function", {})
            name = _get(fn, "name")
            args = _get(fn, "arguments")
            if not isinstance(args, dict):
                args = {}

            print(color(f"\n  🔧 calling {name}({args})", MAGENTA))

            tool_fn = TOOL_FUNCTIONS.get(name)
            if tool_fn is None:
                result = f"Error: unknown tool '{name}'"
            else:
                try:
                    result = tool_fn(**args)
                except Exception as e:
                    result = f"Error running {name}: {e}"

            preview = result if len(result) < 200 else result[:200] + "..."
            print(color(f"     → {preview}", DIM))

            messages.append({"role": "tool", "content": str(result), "name": name})

    die("Agent hit the max reasoning steps without a final answer.", "Try a simpler question, or raise MAX_TOOL_ITERATIONS.")


def main():
    parser = argparse.ArgumentParser(description="A local, tool-using AI agent powered by Qwen3.")
    parser.add_argument("question", nargs="?", help="Question for the agent")
    parser.add_argument("--model", default=MODEL_DEFAULT, help=f"Ollama model tag (default: {MODEL_DEFAULT})")
    parser.add_argument("--think", action="store_true", help="Show Qwen3's reasoning trace before its answer")
    parser.add_argument("--check", action="store_true", help="Verify setup, then exit")
    args = parser.parse_args()

    check_ollama_installed()

    if args.check:
        run_check(args.model)
        return

    if not args.question:
        parser.error("question is required unless you pass --check")

    check_ollama_running_and_model(args.model)
    run_agent(args.model, args.question, think=args.think)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(color("\nCancelled.", DIM))
        sys.exit(130)