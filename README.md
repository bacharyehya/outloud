# readthis

Beautiful TUI for text-to-speech via Google Gemini.

## Install

```bash
pipx install readthis-tts
```

## Usage

```bash
# Launch the TUI
readthis

# Direct CLI generation
readthis --direct "Hello world"
readthis --direct ~/Documents/article.md
pbpaste | readthis --direct

# List voices and styles
readthis --voices
readthis --styles
```

## Setup

Set your Gemini API key:

```bash
export GEMINI_API_KEY="your-key-here"
```

Requires `ffmpeg` for audio processing:

```bash
brew install ffmpeg
```

## License

MIT
