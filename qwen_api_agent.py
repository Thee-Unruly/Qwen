#!/usr/bin/env python3
"""
qwen_api_agent.py — same agent as qwen_agent.py, but talks to Qwen over the
network via Alibaba's DashScope API instead of running locally through
Ollama. Use this if your laptop can't download Ollama + the model in time,
or if you'd rather demo against the real hosted models.

Uses the official OpenAI-compatible endpoint, so it's the standard
"OpenAI SDK, swap base_url and api_key" pattern Alibaba documents.

SETUP
    1. Get a DashScope API key: https://bailian.console.alibabacloud.com
       (Model Studio) -> API Keys -> Create API key.
       New accounts get a free token quota valid 90 days, no card required,
       on the International (Singapore) endpoint.
    2. Install the client:  pip install openai
    3. Set your key:        export DASHSCOPE_API_KEY="sk-..."
       (or pass --api-key on the command line)

USAGE
    python qwen_api_agent.py --check
    python qwen_api_agent.py "Is the payments migration still at risk?"
    python qwen_api_agent.py --model qwen-flash "What's 18/50 as a percentage?"

NOTES
    - Default model is qwen-plus (good balance of quality/cost/tool-calling).
      qwen-flash is cheaper and faster if you're rate-limited or budget-tight;
      qwen-max is the strongest if you want the flagship for a hero question.
    - Uses the International (Singapore) endpoint by default. If your key was
      issued in the Beijing region, pass --base-url with the China endpoint
      (see --help).
"""

import argparse
import ast
import glob
import json
import operator
import os
import sys

DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
DEFAULT_MODEL = "qwen-plus"
NOTES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_notes")
MAX_TOOL_ITERATIONS = 5
MAX_FILE_CHARS = 6000

RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
MAGENTA = "\033[35m"
RED = "\033[31m"
GREEN = "\033[32m"


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
# Tools — identical behavior to the Ollama version, so both scripts feel
# the same to an audience even though the backend differs.
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
    try:
        tree = ast.parse(expression, mode="eval")
        return str(_safe_eval(tree.body))
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
    "You are a helpful assistant with access to tools for arithmetic and "
    "searching a small set of team notes. Use tools when they'd give you real "
    "information instead of guessing. Be concise and concrete in your final answer."
)


# ---------------------------------------------------------------------------
# Setup checks
# ---------------------------------------------------------------------------

def check_openai_installed():
    try:
        import openai  # noqa: F401
    except ImportError:
        die("The 'openai' Python package isn't installed.", "Run: pip install openai")


def get_api_key(cli_key):
    key = cli_key or os.environ.get("DASHSCOPE_API_KEY")
    if not key:
        die(
            "No DashScope API key found.",
            "Set it with: export DASHSCOPE_API_KEY='sk-...'  or pass --api-key",
        )
    return key


def make_client(api_key, base_url):
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url)


def run_check(model, api_key, base_url):
    print(color("Checking setup...", DIM))
    check_openai_installed()
    print(color("  ✓ openai package installed", GREEN))
    get_api_key(api_key)
    print(color("  ✓ API key found", GREEN))

    client = make_client(get_api_key(api_key), base_url)
    try:
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'ok' and nothing else."}],
            max_tokens=5,
        )
    except Exception as e:
        msg = str(e)
        if "401" in msg or "Unauthorized" in msg or "invalid" in msg.lower():
            die("API key was rejected.", "Double-check it was copied correctly and hasn't expired.")
        elif "Connection" in msg or "timeout" in msg.lower():
            die(
                "Can't reach the DashScope API.",
                "Check your internet connection, or that your network allows HTTPS to aliyuncs.com.",
            )
        else:
            die("The test call to DashScope failed.", f"Details: {msg}")
        return

    print(color(f"  ✓ Reached DashScope, model '{model}' responded", GREEN))
    print(color(f"  ✓ {len(glob.glob(os.path.join(NOTES_DIR, '*.md')))} sample note(s) found", GREEN))
    print(color("\nAll good — ready to demo.\n", BOLD))


# ---------------------------------------------------------------------------
# Streaming turn: reconstructs tool_calls from delta fragments, matching the
# standard OpenAI streaming pattern, and prints content live as it arrives.
# ---------------------------------------------------------------------------

def run_turn(client, model, messages):
    content_parts = []
    tool_calls_acc = {}  # index -> {id, name, arguments}
    content_started = False

    try:
        stream = client.chat.completions.create(
            model=model, messages=messages, tools=TOOL_SCHEMAS, stream=True,
        )
    except Exception as e:
        die("Something went wrong talking to DashScope.", f"Details: {e}")
        return None, None

    try:
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta

            reasoning_piece = getattr(delta, "reasoning_content", None)
            if reasoning_piece:
                print(color(reasoning_piece, DIM), end="", flush=True)

            if delta.content:
                if not content_started and reasoning_piece is None and tool_calls_acc:
                    pass
                content_started = True
                print(delta.content, end="", flush=True)
                content_parts.append(delta.content)

            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc_delta.id:
                        tool_calls_acc[idx]["id"] = tc_delta.id
                    if tc_delta.function:
                        if tc_delta.function.name:
                            tool_calls_acc[idx]["name"] += tc_delta.function.name
                        if tc_delta.function.arguments:
                            tool_calls_acc[idx]["arguments"] += tc_delta.function.arguments
    except KeyboardInterrupt:
        print(color("\n\n(stopped early)", DIM))
        sys.exit(130)
    except Exception as e:
        die("Something went wrong mid-stream.", f"Details: {e}")
        return None, None

    if content_parts or tool_calls_acc:
        print()

    tool_calls = None
    if tool_calls_acc:
        tool_calls = []
        for v in tool_calls_acc.values():
            try:
                args = json.loads(v["arguments"]) if v["arguments"] else {}
            except json.JSONDecodeError:
                args = {}
            tool_calls.append({"id": v["id"], "name": v["name"], "arguments": args})

    return "".join(content_parts), tool_calls


def run_agent(client, model, user_question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_question},
    ]

    print(f"\n{BOLD}Question:{RESET} {user_question}" if sys.stdout.isatty() else f"\nQuestion: {user_question}")

    for step in range(MAX_TOOL_ITERATIONS):
        content, tool_calls = run_turn(client, model, messages)

        if not tool_calls:
            return content

        # Feed the assistant's tool-call request back into the conversation
        # in the exact shape the API expects.
        messages.append({
            "role": "assistant",
            "content": content or "",
            "tool_calls": [
                {
                    "id": call["id"] or f"call_{i}",
                    "type": "function",
                    "function": {"name": call["name"], "arguments": json.dumps(call["arguments"])},
                }
                for i, call in enumerate(tool_calls)
            ],
        })

        for call in tool_calls:
            name = call["name"]
            args = call["arguments"]
            call_id = call["id"] or "call_0"

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

            messages.append({"role": "tool", "tool_call_id": call_id, "content": str(result)})

    die("Agent hit the max reasoning steps without a final answer.", "Try a simpler question, or raise MAX_TOOL_ITERATIONS.")


def main():
    parser = argparse.ArgumentParser(description="A Qwen agent using the DashScope API (network fallback for qwen_agent.py).")
    parser.add_argument("question", nargs="?", help="Question for the agent")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Qwen model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--api-key", default=None, help="DashScope API key (defaults to $DASHSCOPE_API_KEY)")
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL,
        help=f"API base URL (default: International/Singapore — {DEFAULT_BASE_URL}). "
             f"Use https://dashscope.aliyuncs.com/compatible-mode/v1 for the Beijing region.",
    )
    parser.add_argument("--check", action="store_true", help="Verify setup, then exit")
    args = parser.parse_args()

    check_openai_installed()

    if args.check:
        run_check(args.model, args.api_key, args.base_url)
        return

    if not args.question:
        parser.error("question is required unless you pass --check")

    api_key = get_api_key(args.api_key)
    client = make_client(api_key, args.base_url)
    run_agent(client, args.model, args.question)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(color("\nCancelled.", DIM))
        sys.exit(130)