"""outloud CLI — entry point that routes to TUI or direct generation."""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys

from . import __version__
from .providers import ProviderManager


REPO_URL = "https://github.com/bacharyehya/outloud"
COFFEE_URL = "https://buymeacoffee.com/bash"


def resolve_input(args_text: list[str]) -> str:
    if not args_text:
        if not sys.stdin.isatty():
            return sys.stdin.read().strip()
        return ""

    if len(args_text) == 1:
        candidate = args_text[0].strip().strip("'\"")
        if os.path.isfile(candidate):
            text_exts = {
                ".txt", ".md", ".markdown", ".rst", ".org", ".html", ".htm",
                ".csv", ".json", ".xml", ".yaml", ".yml", ".log", ".py",
                ".js", ".ts", ".swift", ".rs", ".go", ".rb", ".sh", ".zsh",
                ".bash", ".css", ".scss", ".sql", ".toml", ".ini", ".cfg", "",
            }
            ext = os.path.splitext(candidate)[1].lower()
            if ext in text_exts:
                with open(candidate, "r", errors="replace") as f:
                    return f.read().strip()

    return " ".join(args_text).strip()


def main():
    parser = argparse.ArgumentParser(
        description="outloud — Text-to-speech in your terminal. Local or cloud.",
        epilog="Run without arguments to launch the TUI.\n\n"
               "Examples:\n"
               "  outloud                              # launch TUI\n"
               "  outloud --direct hello world          # generate directly\n"
               "  outloud --direct ~/notes.txt          # read file\n"
               "  pbpaste | outloud --direct            # pipe from clipboard\n"
               "  outloud --voices                      # list all voices\n"
               "  outloud --providers                   # list available providers\n\n"
               f"GitHub: {REPO_URL}\n"
               f"Support: {COFFEE_URL}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("text", nargs="*", help="Text to speak, or path to a text file (direct mode)")
    parser.add_argument("-d", "--direct", action="store_true", help="Skip TUI, generate directly")
    parser.add_argument("-p", "--provider", default=None, help="Provider: Gemini, OpenAI, or Local")
    parser.add_argument("-v", "--voice", default=None, help="Voice name")
    parser.add_argument("-i", "--instructions", default=None, help="Style instructions")
    parser.add_argument("-s", "--speed", type=float, default=1.0, help="Playback speed 0.25-4.0")
    parser.add_argument("--play", action="store_true", help="Play after generating (direct mode)")
    parser.add_argument("--voices", action="store_true", help="List available voices")
    parser.add_argument("--providers", action="store_true", help="List available providers")
    parser.add_argument("--support", action="store_true", help="Open support/donation page")
    parser.add_argument("--version", action="version", version=f"outloud {__version__}")
    args = parser.parse_args()

    if args.support:
        print(f"Thanks for considering supporting outloud!")
        print(f"Buy me a coffee: {COFFEE_URL}")
        print(f"Star on GitHub: {REPO_URL}")
        subprocess.Popen(["open", COFFEE_URL])
        sys.exit(0)

    manager = ProviderManager()

    if args.providers:
        print("Available providers:")
        for name, p in manager.providers.items():
            status = "ready" if p.is_available() else "no API key" if p.requires_api_key else "not installed"
            print(f"  {name}: {status}")
        if not manager.available:
            print("\nNo providers available. Set GEMINI_API_KEY or OPENAI_API_KEY.")
            print("For free local TTS: pip install outloud-tts[local]")
        sys.exit(0)

    if args.voices:
        provider_name = args.provider
        providers = [manager.get(provider_name)] if provider_name and manager.get(provider_name) else list(manager.providers.values())
        for p in providers:
            print(f"\n{p.name} voices:")
            for v in p.voices():
                default = " (default)" if v.id == p.default_voice() else ""
                gender = f" [{v.gender}]" if v.gender else ""
                print(f"  {v.name}{gender}{default}")
        sys.exit(0)

    # Direct mode
    if args.direct or not sys.stdin.isatty() or args.text:
        text = resolve_input(args.text)
        if not text:
            if sys.stdin.isatty() and not args.text:
                from .tui import run_tui
                run_tui()
                return
            parser.print_help()
            sys.exit(1)

        provider = None
        if args.provider:
            provider = manager.get(args.provider)
            if not provider or not provider.is_available():
                print(f"Provider '{args.provider}' not available.", file=sys.stderr)
                sys.exit(1)
        else:
            provider = manager.default()
            if not provider:
                print("No providers available. Set GEMINI_API_KEY or OPENAI_API_KEY.", file=sys.stderr)
                sys.exit(1)

        voice = args.voice or provider.default_voice()
        style = args.instructions or ""
        est = provider.estimate_cost(text)

        print(f"{len(text):,} chars -> {est.chunks} chunk{'s' if est.chunks != 1 else ''} via {provider.name} — {est.display}")

        def on_progress(idx, total, success):
            status = "done" if success else "FAILED"
            print(f"  Chunk {idx+1}/{total} {status}")

        result = asyncio.run(provider.generate(
            text=text, voice=voice, style=style, speed=args.speed, on_progress=on_progress,
        ))

        if not result:
            print("Generation failed.", file=sys.stderr)
            sys.exit(1)

        mins, secs = divmod(result.duration_s, 60)
        print(f"Saved: {result.path} ({result.size_kb:,}KB, {mins}m{secs:02d}s)")

        if args.play:
            subprocess.run(["afplay", result.path])
        return

    # Default: TUI mode
    from .tui import run_tui
    initial = resolve_input(args.text)
    run_tui(initial_text=initial)
