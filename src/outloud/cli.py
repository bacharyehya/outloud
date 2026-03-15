"""outloud CLI — entry point that routes to TUI or direct generation, plus STT."""

from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time

from . import __version__
from .stt import is_audio_file, TranscriptResult


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


# -- STT format helpers --

def _format_timestamp_srt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_timestamp_vtt(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds % 1) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def _print_srt(result: TranscriptResult) -> str:
    lines = []
    for i, seg in enumerate(result.segments, 1):
        lines.append(str(i))
        lines.append(f"{_format_timestamp_srt(seg.start)} --> {_format_timestamp_srt(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def _print_vtt(result: TranscriptResult) -> str:
    lines = ["WEBVTT", ""]
    for seg in result.segments:
        lines.append(f"{_format_timestamp_vtt(seg.start)} --> {_format_timestamp_vtt(seg.end)}")
        lines.append(seg.text)
        lines.append("")
    return "\n".join(lines)


def _print_json(result: TranscriptResult) -> str:
    import json
    return json.dumps({
        "text": result.text,
        "language": result.language,
        "duration_s": result.duration_s,
        "provider": result.provider,
        "source": result.source,
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in result.segments
        ],
    }, indent=2)


def _copy_to_clipboard(text: str):
    try:
        subprocess.run(["pbcopy"], input=text.encode(), check=True)
    except Exception:
        pass


def _output_transcript(result: TranscriptResult, fmt: str, copy: bool):
    """Output transcript — metadata to stderr (TTY), clean text to stdout."""
    is_tty = sys.stdout.isatty()

    if fmt == "srt":
        print(_print_srt(result))
    elif fmt == "vtt":
        print(_print_vtt(result))
    elif fmt == "json":
        print(_print_json(result))
    else:
        # text format
        if is_tty:
            mins, secs = divmod(result.duration_s, 60)
            print(f"[{result.provider}] {result.language} | {mins:.0f}m{secs:02.0f}s | {result.source}", file=sys.stderr)
            print(file=sys.stderr)
        print(result.text)

    if copy and is_tty:
        text_to_copy = result.text if fmt == "text" else (
            _print_srt(result) if fmt == "srt" else
            _print_vtt(result) if fmt == "vtt" else
            _print_json(result)
        )
        _copy_to_clipboard(text_to_copy)
        print("Copied to clipboard.", file=sys.stderr)


def _run_stt(args, path: str):
    """Transcribe an audio file."""
    from .stt import transcribe_mlx, transcribe_openai
    from .config import Config

    config = Config.load()
    provider = getattr(args, "stt_provider", None) or config.stt_provider
    model = getattr(args, "model", None) or config.whisper_model
    language = getattr(args, "language", None) or config.stt_language or None
    task = "translate" if getattr(args, "translate", False) else "transcribe"
    fmt = getattr(args, "format", "text") or "text"
    copy = getattr(args, "copy", None)
    if copy is None:
        copy = config.stt_copy_clipboard

    if provider == "openai":
        result = asyncio.run(transcribe_openai(path, language=language, task=task))
    else:
        result = transcribe_mlx(path, model=model, language=language, task=task)

    _output_transcript(result, fmt, copy)


def _run_stt_record(args):
    """Record from mic, then transcribe."""
    from .stt import record_from_mic

    duration = getattr(args, "duration", None)
    wav_path = record_from_mic(duration=duration)
    try:
        _run_stt(args, wav_path)
    finally:
        os.unlink(wav_path)


def main():
    parser = argparse.ArgumentParser(
        description="outloud — Speak and listen. TTS + STT in your terminal.",
        epilog="Run without arguments to launch the TUI.\n\n"
               "Examples:\n"
               "  outloud                              # launch TUI\n"
               "  outloud --direct hello world          # TTS: generate speech\n"
               "  outloud meeting.m4a                   # STT: transcribe audio\n"
               "  outloud --record                      # STT: record + transcribe\n"
               "  outloud video.mp4 --format srt        # STT: extract subtitles\n"
               "  outloud --record | outloud --direct   # round-trip: voice → text → speech\n"
               "  outloud --voices                      # list all voices\n\n"
               f"GitHub: {REPO_URL}\n"
               f"Support: {COFFEE_URL}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("text", nargs="*", help="Text to speak, or audio file to transcribe")
    parser.add_argument("-d", "--direct", action="store_true", help="Skip TUI, generate TTS directly")
    parser.add_argument("-p", "--provider", default=None, help="TTS provider: Gemini, OpenAI, or Local")
    parser.add_argument("-v", "--voice", default=None, help="Voice name")
    parser.add_argument("-i", "--instructions", default=None, help="Style instructions")
    parser.add_argument("-s", "--speed", type=float, default=1.0, help="Playback speed 0.25-4.0")
    parser.add_argument("--play", action="store_true", help="Play after generating (direct mode)")
    parser.add_argument("--voices", action="store_true", help="List available voices")
    parser.add_argument("--providers", action="store_true", help="List available providers")
    parser.add_argument("--support", action="store_true", help="Open support/donation page")
    parser.add_argument("--version", action="version", version=f"outloud {__version__}")

    # STT arguments
    stt_group = parser.add_argument_group("speech-to-text")
    stt_group.add_argument("-r", "--record", action="store_true", help="Record from mic, then transcribe")
    stt_group.add_argument("--duration", type=float, default=None, help="Recording length in seconds (default: until Ctrl+C)")
    stt_group.add_argument("--language", default=None, help="Language hint (en, ar, fr...) — auto-detects if omitted")
    stt_group.add_argument("--translate", action="store_true", help="Translate to English instead of transcribing")
    stt_group.add_argument("--format", default="text", choices=["text", "srt", "vtt", "json"], help="Transcript output format (default: text)")
    stt_group.add_argument("--copy", action="store_true", default=None, help="Copy transcript to clipboard")
    stt_group.add_argument("--no-copy", action="store_true", help="Don't copy transcript to clipboard")
    stt_group.add_argument("--stt-provider", default=None, choices=["mlx", "openai"], help="STT provider (default: mlx)")
    stt_group.add_argument("--model", default=None, help="Whisper model override")

    args = parser.parse_args()

    # Handle --no-copy
    if args.no_copy:
        args.copy = False

    # STT: record mode
    if args.record:
        _run_stt_record(args)
        return

    # STT: audio file detection
    if args.text and len(args.text) == 1:
        candidate = args.text[0].strip().strip("'\"")
        expanded = os.path.expanduser(candidate)
        if os.path.isfile(expanded) and is_audio_file(expanded):
            _run_stt(args, expanded)
            return

    if args.support:
        import webbrowser
        print(f"Thanks for considering supporting outloud!")
        print(f"Buy me a coffee: {COFFEE_URL}")
        print(f"Star on GitHub: {REPO_URL}")
        webbrowser.open(COFFEE_URL)
        sys.exit(0)

    from .providers import ProviderManager
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

        t0 = time.monotonic()
        failures = 0

        def on_progress(idx, total, success, error=""):
            nonlocal failures
            elapsed = time.monotonic() - t0
            if success:
                print(f"  Chunk {idx+1}/{total} done ({elapsed:.0f}s)")
            else:
                failures += 1
                reason = f": {error}" if error else ""
                print(f"  Chunk {idx+1}/{total} FAILED{reason} ({elapsed:.0f}s)")

        result = asyncio.run(provider.generate(
            text=text, voice=voice, style=style, speed=args.speed, on_progress=on_progress,
        ))

        if not result:
            print("Generation failed — all chunks failed. Check your API key and network.", file=sys.stderr)
            sys.exit(1)

        mins, secs = divmod(result.duration_s, 60)
        fail_str = f", {result.chunks_failed} failed" if result.chunks_failed else ""
        cost_str = f", actual cost: ${result.actual_cost:.4f}" if result.actual_cost is not None else ""
        print(f"Saved: {result.path} ({result.size_kb:,}KB, {mins}m{secs:02d}s, {result.elapsed_s}s elapsed{fail_str}{cost_str})")

        if args.play:
            from .audio import play_audio
            proc = play_audio(result.path)
            if proc:
                proc.wait()
        return

    # Default: TUI mode
    from .tui import run_tui
    initial = resolve_input(args.text)
    run_tui(initial_text=initial)
