## Usage

```bash
# Quick tool-use demo
python qwen_agent.py "Is the payments migration still at risk?"

# Show the model's reasoning trace before it answers (slower, more impressive)
python qwen_agent.py --think "What's 18 out of 50 as a percentage, and what does that mean for the beta pilot?"

# Use a bigger local model if you pre-downloaded one
python qwen_agent.py --model qwen3:8b "Who's responsible for Swahili localization?"
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