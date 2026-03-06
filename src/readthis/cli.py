"""readthis CLI — entry point that routes to TUI or direct generation."""

import argparse
import asyncio
import os
import subprocess
import sys

from . import engine


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
        description="readthis — Text-to-speech via Google Gemini TTS",
        epilog="Run without arguments to launch the TUI.\n"
               "Pass text or a file path for direct CLI generation.\n\n"
               "Examples:\n"
               "  readthis                          # launch TUI\n"
               "  readthis --direct hello world      # generate directly\n"
               "  readthis --direct ~/notes.txt      # read file\n"
               "  pbpaste | readthis --direct        # pipe from clipboard\n"
               "  readthis --voices                  # list voices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("text", nargs="*", help="Text to speak, or path to a text file (direct mode)")
    parser.add_argument("-d", "--direct", action="store_true", help="Skip TUI, generate directly")
    parser.add_argument("-v", "--voice", default=engine.DEFAULT_VOICE, help=f"Voice (default: {engine.DEFAULT_VOICE})")
    parser.add_argument("-m", "--model", default=engine.DEFAULT_MODEL, help=f"Model (default: {engine.DEFAULT_MODEL})")
    parser.add_argument("-i", "--instructions", default=None, help="Style instructions")
    parser.add_argument("-s", "--speed", type=float, default=None, help="Playback speed 0.25-4.0")
    parser.add_argument("-n", "--no-play", action="store_true", help="Save only, don't play")
    parser.add_argument("-p", "--play", action="store_true", help="Play after generating (direct mode)")
    parser.add_argument("--voices", action="store_true", help="List available voices")
    parser.add_argument("--styles", action="store_true", help="List style presets")
    args = parser.parse_args()

    if args.voices:
        print("Available voices:")
        for v in engine.VOICES:
            marker = " (default)" if v == engine.DEFAULT_VOICE else ""
            print(f"  {v}{marker}")
        sys.exit(0)

    if args.styles:
        print("Style presets:")
        for name, prompt in engine.STYLE_PRESETS.items():
            print(f"  {name}: {prompt or '(no instructions)'}")
        sys.exit(0)

    # Direct mode: CLI generation
    if args.direct or not sys.stdin.isatty() or args.text:
        text = resolve_input(args.text)
        if not text:
            # If nothing piped and no text args, launch TUI
            if sys.stdin.isatty() and not args.text:
                from .tui import run_tui
                run_tui()
                return
            parser.print_help()
            sys.exit(1)

        style = args.instructions or engine.STYLE_PRESETS.get(engine.DEFAULT_STYLE, "")

        chars = len(text)
        chunks = engine.chunk_text(text)
        print(f"{chars:,} chars -> {len(chunks)} chunk{'s' if len(chunks) != 1 else ''} (voice: {args.voice})")

        def on_progress(idx, total, success):
            status = "done" if success else "FAILED"
            print(f"  Chunk {idx+1}/{total} {status}")

        result = asyncio.run(engine.generate(
            text=text,
            voice=args.voice,
            style=style,
            model=args.model,
            speed=args.speed,
            on_progress=on_progress,
        ))

        if not result:
            print("Generation failed.", file=sys.stderr)
            sys.exit(1)

        mins, secs = divmod(result.duration_s, 60)
        print(f"Saved: {result.path} ({result.size_kb:,}KB, {mins}m{secs:02d}s)")

        if args.play and not args.no_play:
            subprocess.run(["afplay", result.path])
        return

    # Default: TUI mode
    from .tui import run_tui
    initial = resolve_input(args.text)
    run_tui(initial_text=initial)
