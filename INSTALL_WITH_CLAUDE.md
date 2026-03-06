# Hey Claude! Install outloud for me.

You're helping someone who is NOT a developer. Do everything for them. No jargon. No choices. Just make it work.

## What to do

1. Check if Python 3.11+ is installed. If not, install it:
   - macOS: `brew install python` (install Homebrew first if needed: `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"`)
   - Linux: `sudo apt install python3 python3-pip python3-venv` or equivalent
   - Windows: Download from python.org and install with "Add to PATH" checked

2. Check if pipx is installed. If not: `brew install pipx` (macOS) or `pip install --user pipx` (Linux/Windows), then `pipx ensurepath`

3. Check if ffmpeg is installed. If not: `brew install ffmpeg` (macOS) or `sudo apt install ffmpeg` (Linux)

4. Install outloud: `pipx install outloud-tts`

5. Ask the user: "Do you have a Google Gemini API key or an OpenAI API key?"
   - If yes: help them set it as an environment variable (`export GEMINI_API_KEY="their-key"` or `export OPENAI_API_KEY="their-key"`) and add it to their shell profile so it persists
   - If no: walk them through getting a free Gemini API key at https://aistudio.google.com/apikey (it's free, they just need a Google account)

6. Run `outloud` to launch the app and confirm it works

7. Tell them: "You're all set! Type `outloud` anytime to turn text into speech. Paste your text, pick a voice, hit Generate. Your audio files save to Downloads."

## Important
- Be friendly and patient. Explain what you're doing in simple terms.
- If something fails, fix it. Don't dump errors on them.
- Test that it actually works before saying you're done.
