# outloud

Beautiful TUI for text-to-speech. Gemini, OpenAI, or local. One command.

<!-- TODO: Add GIF demo here -->

## Features

- **Multi-provider** — Google Gemini, OpenAI, or KittenTTS (local, free, no API key)
- **Beautiful TUI** — full terminal UI with voice picker, style presets, progress bar
- **Cost calculator** — see estimated cost before generating
- **30+ voices** — across all providers
- **Style control** — warm, whisper, energetic, audiobook, newsreader, and more
- **Smart defaults** — auto-detects available providers, zero config needed
- **Direct CLI mode** — pipe text, pass files, or use inline for scripting
- **History** — click to replay any past generation

## Install

```bash
# Cloud providers (Gemini + OpenAI) — lightweight
pipx install outloud-tts

# Include local TTS (KittenTTS) — free, no API key needed
pipx install outloud-tts[local]
```

Requires `ffmpeg`:

```bash
brew install ffmpeg    # macOS
sudo apt install ffmpeg  # Linux
```

## Quick Start

```bash
# Launch the TUI
outloud

# Direct generation
outloud --direct "Hello world"
outloud --direct ~/Documents/article.md
pbpaste | outloud --direct

# Pick a provider
outloud --direct -p OpenAI "Hello from OpenAI"
outloud --direct -p Local "Free and local"

# List voices and providers
outloud --voices
outloud --providers
```

## Providers

| Provider | Voices | Style Control | Cost | API Key |
|----------|--------|--------------|------|---------|
| **Gemini** | 30 | Yes | ~$0.01-0.04/1K chars | `GEMINI_API_KEY` |
| **OpenAI** | 13 | Yes (gpt-4o-mini-tts) | ~$0.015/1K chars | `OPENAI_API_KEY` |
| **Local** | 8 | No | Free | None |

## Setup

Set one or more API keys:

```bash
export GEMINI_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

Or skip API keys entirely — install with `[local]` for free offline TTS.

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `Ctrl+G` | Generate |
| `Ctrl+P` | Play last |
| `Ctrl+O` | Open output folder |
| `Ctrl+L` | Clear text |
| `Ctrl+Q` | Quit |

## Support

If outloud saved you time, consider:

- [Star this repo](https://github.com/bacharyehya/outloud) — it helps others find it
- [Buy me a coffee](https://buymeacoffee.com/bash) — building from Lebanon on 29 Mbps

## License

MIT
