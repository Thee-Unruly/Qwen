# Qwen Local Agent Demo

A local AI agent powered by Qwen3 (via Ollama) that can reason about which
tool it needs, call it, read the result, and give a final answer — live,
streaming, fully offline. No API key required.

Built for the "Build with Qwen" workshop demo.

## Why this exists

Most AI demos are just "type a prompt, get an answer." This one shows the
model actually *deciding* what to do — check a calculator, search some notes,
read a file — before it answers. That decision loop is what people mean by
an "AI agent," and it's the part worth watching, not just the final text.

It also works with zero dependency on API access or venue Wi-Fi, since
everything runs locally through Ollama.

## What's included

| File | What it does |
|---|---|
| `qwen_agent.py` | The local agent — runs via Ollama, no API key needed |
| `qwen_api_agent.py` | Same agent, but calls Qwen over the DashScope API instead — use this if Ollama won't install on someone's laptop |
| `.env.example` | Template for your DashScope API key — copy to `.env` and fill in |
| `sample_notes/roadmap.md` | Sample "company" data for the agent to search |
| `sample_notes/meeting_notes.md` | Sample "company" data for the agent to search |
| `sample_notes/customer_feedback.md` | Sample "company" data for the agent to search |

## Setup — local version (`qwen_agent.py`, works offline)

```bash
# 1. Install Ollama
#    https://ollama.com/download

# 2. Pull the model (a few GB — do this on good Wi-Fi tonight)
ollama pull qwen3:4b

# 3. Install the Python client
pip install ollama

# 4. Verify everything works
python qwen_agent.py --check
```

You should see:
```
Checking setup...
  ✓ ollama package installed
  ✓ Ollama running, 'qwen3:4b' available
  ✓ 3 sample note(s) found

All good — ready to demo.
```

## Setup — API version (`qwen_api_agent.py`, for laptops that can't run Ollama)

Uses **OpenRouter by default** — free, no purchase or activation step, works
immediately after signup. DashScope is available as a fallback provider if
you'd rather use it (`--provider dashscope`), but some models there need a
manual activation step in the Model Square even on the free tier.

```bash
# 1. Install the dependencies
pip install openai python-dotenv

# 2. Get a free OpenRouter API key
#    https://openrouter.ai -> sign up (no card required) -> Keys -> Create Key

# 3. Copy the template and add your key
cp .env.example .env
# then edit .env and set: OPENROUTER_API_KEY=sk-or-v1-your-real-key

# 4. Verify everything works
python qwen_api_agent.py --check
```

You should see:
```
Checking setup...
  ✓ openai package installed
  ✓ python-dotenv installed (.env file will be read)
  ✓ API key found
  ✓ Reached openrouter, model 'qwen/qwen3-coder:free' responded
  ✓ 3 sample note(s) found

All good — ready to demo.
```

**To use DashScope instead:**
```bash
# in .env: DASHSCOPE_API_KEY=sk-your-key
python qwen_api_agent.py --provider dashscope --check
```

**Never commit your real `.env` file** — it holds your live API key. If this
folder ever becomes a git repo, add `.env` to `.gitignore` (keep `.env.example`,
which has no real key, tracked instead).

## Usage

```bash
# Local (Ollama) version
python qwen_agent.py "Is the payments migration still at risk?"
python qwen_agent.py --think "What's 18 out of 50 as a percentage, and what does that mean for the beta pilot?"
python qwen_agent.py --model qwen3:8b "Who's responsible for Swahili localization?"

# API version (OpenRouter by default)
python qwen_api_agent.py "Is the payments migration still at risk?"
python qwen_api_agent.py --provider dashscope --model qwen-plus "What's 18/50 as a percentage?"
```

## Available tools

The agent can call any of these on its own — you don't tell it which one to use:

- **`calculate(expression)`** — safe arithmetic (no `eval()`, rejects anything that isn't math)
- **`list_notes()`** — lists the sample note files
- **`search_notes(query)`** — keyword search across all notes, returns matching snippets
- **`read_file(path)`** — reads one note file in full

## Good demo questions

These are picked so the agent has to actually choose the right tool, not just answer from general knowledge:

- "Is the payments migration still at risk?" → should search notes
- "What's 18 out of 50 as a percentage?" → should calculate
- "Who's handling Swahili localization and when's the next sync?" → should search notes
- "What's the Q3 engineering budget split, and what's 25% of $180,000?" → should search AND calculate

## Troubleshooting

| Problem | Fix |
|---|---|
| `Can't reach the Ollama server` | Run `ollama serve`, or open the Ollama app |
| `Model isn't downloaded yet` | Run `ollama pull qwen3:4b` |
| `ollama package isn't installed` | Run `pip install ollama` |
| Model doesn't call the right tool | Try `qwen3:8b` if available — better tool-calling judgment than the 4B model |
| Agent loops without answering | It hit the 5-step safety limit — ask a simpler question |
| `No DashScope API key found` / `No openrouter API key found` | Copy `.env.example` to `.env` and add your real key |
| `python-dotenv not installed` | Run `pip install python-dotenv` — otherwise `.env` is ignored |
| `API key was rejected` | Check for a typo/extra space, or that the key hasn't expired |
| `Can't reach the {provider} API` | Check internet access, or that the venue network allows the provider's domain |
| `403 AccessDenied.Unpurchased` (DashScope) | The model needs activation in Model Studio's Model Square — or just switch to `--provider openrouter` |

## After the workshop: switching to the hosted API

Once your DashScope API key is active, the same tool-calling pattern works
with the hosted API — same shape, just swap the client:

```python
from openai import OpenAI
client = OpenAI(
    api_key="YOUR_DASHSCOPE_KEY",
    base_url="https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
)
```

## License / credit

Demo built for the AI Collective's "Build with Qwen" workshop, Nairobi.
Qwen3 is developed by Alibaba, released under Apache 2.0.