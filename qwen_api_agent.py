#!/usr/bin/env python3
"""
qwen_api_agent.py — same agent as qwen_agent.py, but talks to Qwen over the
network via a hosted API instead of running locally through Ollama. Use this
if your laptop can't download Ollama + the model in time, or your DashScope
account is stuck pending activation.

Defaults to OpenRouter, which needs no purchase/activation step and offers
Qwen3 Coder free with tool-calling support. DashScope is available as an
alternative via --provider dashscope.

SETUP (OpenRouter — default, recommended if DashScope access is delayed)
    1. Sign up free at https://openrouter.ai (no card required)
    2. Create a key: openrouter.ai -> Keys -> Create Key
    3. Install the client:  pip install openai python-dotenv
    4. Put it in a .env file next to this script:
           OPENROUTER_API_KEY=sk-or-v1-...

SETUP (DashScope — alternative)
    1. Get a key: https://bailian.console.alibabacloud.com -> API Keys
       (new accounts get a free 90-day quota; some models need a one-click
       activation step in the Model Square even on the free tier)
    2. Put it in .env:  DASHSCOPE_API_KEY=sk-...

USAGE
    python qwen_api_agent.py --check
    python qwen_api_agent.py "Is the payments migration still at risk?"
    python qwen_api_agent.py --provider dashscope --model qwen-plus "..."
"""

import argparse
import ast
import glob
import json
import operator
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()  # reads .env in the current directory (if present) into os.environ
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False

PROVIDERS = {
    "openrouter": {
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "qwen/qwen3-coder:free",
        "env_key": "OPENROUTER_API_KEY",
        "signup_hint": "Get a free key at https://openrouter.ai (no card required, no activation step).",
    },
    "dashscope": {
        "base_url": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
        "env_key": "DASHSCOPE_API_KEY",
        "signup_hint": "Get a key at https://bailian.console.alibabacloud.com "
                        "(may need per-model activation in the Model Square).",
    },
}
DEFAULT_PROVIDER = "openrouter"
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


def check_dotenv_installed():
    if not _DOTENV_AVAILABLE:
        die(
            "The 'python-dotenv' package isn't installed.",
            "Run: pip install python-dotenv   (needed to read the .env file)",
        )


def get_api_key(cli_key, provider):
    env_key = PROVIDERS[provider]["env_key"]
    key = cli_key or os.environ.get(env_key)
    if not key:
        die(
            f"No {provider} API key found.",
            f"Add {env_key}=... to a .env file next to this script "
            f"(or export it, or pass --api-key). {PROVIDERS[provider]['signup_hint']}",
        )
    return key


def make_client(api_key, base_url, provider):
    from openai import OpenAI
    headers = {}
    if provider == "openrouter":
        # Optional but recommended by OpenRouter for their public leaderboards;
        # harmless either way, just identifies the app making the request.
        headers = {
            "HTTP-Referer": "https://github.com/build-with-qwen-demo",
            "X-Title": "Build with Qwen Workshop Demo",
        }
    return OpenAI(api_key=api_key, base_url=base_url, default_headers=headers)


def run_check(model, api_key, base_url, provider):
    print(color("Checking setup...", DIM))
    check_openai_installed()
    print(color("  ✓ openai package installed", GREEN))

    if _DOTENV_AVAILABLE:
        print(color("  ✓ python-dotenv installed (.env file will be read)", GREEN))
    else:
        print(color("  ⚠ python-dotenv not installed — .env file will be ignored, "
                     "only OS env vars / --api-key will work", "\033[33m"))
        print(color("     → Run: pip install python-dotenv", DIM))

    key = get_api_key(api_key, provider)
    print(color("  ✓ API key found", GREEN))

    client = make_client(key, base_url, provider)
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
        elif "403" in msg or "AccessDenied" in msg:
            die(
                f"Access denied for model '{model}' on {provider}.",
                f"{PROVIDERS[provider]['signup_hint']} Or try --provider "
                f"{'dashscope' if provider == 'openrouter' else 'openrouter'} instead.",
            )
        elif "Connection" in msg or "timeout" in msg.lower():
            die(
                f"Can't reach the {provider} API.",
                "Check your internet connection, or that your network allows the required HTTPS domain.",
            )
        else:
            die(f"The test call to {provider} failed.", f"Details: {msg}")
        return

    print(color(f"  ✓ Reached {provider}, model '{model}' responded", GREEN))
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
    parser = argparse.ArgumentParser(
        description="A Qwen agent using a hosted API (OpenRouter by default, DashScope as an alternative) "
                     "— network fallback for qwen_agent.py."
    )
    parser.add_argument("question", nargs="?", help="Question for the agent")
    parser.add_argument(
        "--provider", choices=list(PROVIDERS.keys()), default=DEFAULT_PROVIDER,
        help=f"Which API to use (default: {DEFAULT_PROVIDER})",
    )
    parser.add_argument("--model", default=None, help="Model name (defaults to the provider's recommended free/cheap model)")
    parser.add_argument("--api-key", default=None, help="API key (defaults to the provider's env var, e.g. $OPENROUTER_API_KEY)")
    parser.add_argument("--base-url", default=None, help="Override the API base URL (defaults to the provider's standard endpoint)")
    parser.add_argument("--check", action="store_true", help="Verify setup, then exit")
    args = parser.parse_args()

    check_openai_installed()

    provider_config = PROVIDERS[args.provider]
    model = args.model or provider_config["default_model"]
    base_url = args.base_url or provider_config["base_url"]

    if args.check:
        run_check(model, args.api_key, base_url, args.provider)
        return

    if not args.question:
        parser.error("question is required unless you pass --check")

    api_key = get_api_key(args.api_key, args.provider)
    client = make_client(api_key, base_url, args.provider)
    run_agent(client, model, args.question)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(color("\nCancelled.", DIM))
        sys.exit(130)