# outloud v0.2 Design

## Summary

Rename readthis → outloud. Add multi-provider TTS (Gemini, OpenAI, KittenTTS local), cost calculator, improved TUI with provider switching, and open source polish.

## Providers

| Provider | Voices | Style Control | Cost | Dependencies |
|----------|--------|--------------|------|--------------|
| Gemini Flash TTS | 30 | Text prompt | ~$0.01-0.04/1K chars | aiohttp |
| Gemini Pro TTS | 30 | Text prompt | ~$0.04-0.12/1K chars | aiohttp |
| OpenAI gpt-4o-mini-tts | 13 | `instructions` param | ~$0.015/1K chars | aiohttp |
| OpenAI tts-1 | 10 | None | $15/1M chars | aiohttp |
| KittenTTS (local) | 8 | None | Free | onnxruntime, spacy, espeak |

### Provider Interface

```python
class TTSProvider(ABC):
    name: str
    requires_api_key: bool

    async def generate(text, voice, style, speed, output_path, on_progress) -> AudioResult
    def voices() -> list[Voice]
    def styles() -> dict[str, str]
    def estimate_cost(text) -> CostEstimate | None
    def is_available() -> bool
```

### Smart Defaults

1. Check env for GEMINI_API_KEY → Gemini available
2. Check env for OPENAI_API_KEY → OpenAI available
3. Check if kittentts importable → Local available
4. Default to first available; if none → setup hint

### KittenTTS as Optional Dep

Heavy deps (spacy, onnxruntime, espeak) make it opt-in:
- `pipx install outloud-tts` — cloud only
- `pipx install outloud-tts[local]` — includes KittenTTS

## Cost Calculator

Live estimate in TUI as text changes. Shows per-provider comparison.

Pricing data in `cost.py`, easily updatable:
- Gemini Flash: $0.50/1M input tokens, $10/1M output tokens
- Gemini Pro: $1.00/1M input tokens, $20/1M output tokens
- OpenAI gpt-4o-mini-tts: $0.60/1M input tokens, $12/1M output tokens
- OpenAI tts-1: $15/1M characters
- KittenTTS: Free

## TUI Changes

- Provider radio buttons in sidebar (above voice list)
- Voice list updates when provider changes
- Cost estimate below controls, updates live
- Speed slider widget (0.25x - 4.0x)
- Star prompt after 3rd generation (once, dismissable)
- Config persisted to ~/.config/outloud/config.toml

## Package Structure

```
src/outloud/
├── __init__.py
├── cli.py
├── tui.py
├── config.py
├── cost.py
└── providers/
    ├── __init__.py    # Base class + ProviderManager
    ├── gemini.py
    ├── openai.py
    └── kitten.py
```

## Install & Publish

- PyPI: `outloud-tts`
- Command: `outloud`
- Config: `~/.config/outloud/config.toml`
