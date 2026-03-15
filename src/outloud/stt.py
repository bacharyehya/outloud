"""outloud STT — Speech-to-text via MLX Whisper (local) or OpenAI Whisper API."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".mp4", ".mov", ".flac", ".ogg", ".opus",
    ".aac", ".wma", ".aiff", ".aif", ".webm", ".oga", ".spx", ".caf",
}

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
WHISPER_PYTHON = os.path.expanduser("~/whisper-env/bin/python")
WHISPER_CLI = os.path.expanduser("~/whisper-env/bin/mlx_whisper")


def is_audio_file(path: str) -> bool:
    """Check if a path points to an audio/video file by extension."""
    return Path(path).suffix.lower() in AUDIO_EXTENSIONS


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


@dataclass
class TranscriptResult:
    text: str
    language: str = ""
    duration_s: float = 0.0
    segments: list[TranscriptSegment] = field(default_factory=list)
    provider: str = "mlx"
    source: str = ""


def _convert_to_wav(path: str) -> str:
    """Convert any audio/video to 16kHz mono WAV for Whisper. Returns path to WAV."""
    p = Path(path)
    if p.suffix.lower() == ".wav":
        return path

    tmp = tempfile.mktemp(suffix=".wav", prefix="outloud_")
    subprocess.run(
        ["ffmpeg", "-y", "-i", path, "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le", tmp],
        capture_output=True, check=True,
    )
    return tmp


def record_from_mic_managed(output_path: str | None = None) -> subprocess.Popen:
    """Start recording from mic, return immediately. Caller calls proc.terminate() to stop.

    Uses SIGTERM instead of SIGINT to avoid conflicts with Textual's signal handling.
    """
    path = output_path or tempfile.mktemp(suffix=".wav", prefix="outloud_rec_")

    cmd = [
        "ffmpeg", "-y",
        "-f", "avfoundation", "-i", ":default",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
        path,
    ]
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
    proc._output_path = path  # stash for caller convenience
    return proc


def validate_recording(path: str) -> bool:
    """Check recording file exists and has meaningful audio data (>4KB ~= 0.25s)."""
    p = Path(path)
    return p.exists() and p.stat().st_size > 4096


def record_from_mic(duration: float | None = None) -> str:
    """Record from default mic via ffmpeg avfoundation. Returns path to WAV.

    Records until Ctrl+C if no duration specified.
    """
    tmp = tempfile.mktemp(suffix=".wav", prefix="outloud_rec_")

    cmd = [
        "ffmpeg", "-y",
        "-f", "avfoundation", "-i", ":default",
        "-ar", "16000", "-ac", "1", "-c:a", "pcm_s16le",
    ]
    if duration:
        cmd += ["-t", str(duration)]
    cmd.append(tmp)

    print("Recording... (Ctrl+C to stop)", file=sys.stderr)
    proc = subprocess.Popen(cmd, stderr=subprocess.PIPE)

    try:
        proc.wait()
    except KeyboardInterrupt:
        proc.send_signal(signal.SIGINT)
        proc.wait()
        print(file=sys.stderr)

    if not Path(tmp).exists() or Path(tmp).stat().st_size < 100:
        raise RuntimeError("Recording failed — no audio captured")

    return tmp


def transcribe_mlx(
    path: str,
    model: str = DEFAULT_MODEL,
    language: str | None = None,
    task: str = "transcribe",
    output_format: str = "json",
) -> TranscriptResult:
    """Transcribe audio using mlx-whisper CLI subprocess."""
    wav_path = _convert_to_wav(path)
    cleanup_wav = wav_path != path

    try:
        with tempfile.TemporaryDirectory(prefix="outloud_stt_") as tmpdir:
            cmd = [
                WHISPER_CLI,
                wav_path,
                "--model", model,
                "--output-format", "json",
                "--output-dir", tmpdir,
                "--task", task,
            ]
            if language:
                cmd += ["--language", language]

            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"mlx_whisper failed: {proc.stderr[:500]}")

            json_files = list(Path(tmpdir).glob("*.json"))
            if not json_files:
                raise RuntimeError("mlx_whisper produced no output")

            data = json.loads(json_files[0].read_text())

            segments = []
            total_duration = 0.0
            for seg in data.get("segments", []):
                segments.append(TranscriptSegment(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"].strip(),
                ))
                total_duration = max(total_duration, seg["end"])

            return TranscriptResult(
                text=data.get("text", "").strip(),
                language=data.get("language", ""),
                duration_s=total_duration,
                segments=segments,
                provider="mlx",
                source=path,
            )
    finally:
        if cleanup_wav and Path(wav_path).exists():
            Path(wav_path).unlink()


async def transcribe_openai(
    path: str,
    language: str | None = None,
    task: str = "transcribe",
) -> TranscriptResult:
    """Transcribe audio using OpenAI Whisper API (cloud fallback)."""
    import aiohttp

    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        try:
            api_key = subprocess.run(
                ["security", "find-generic-password", "-s", "OPENAI_API_KEY", "-a", "mcp", "-w"],
                capture_output=True, text=True,
            ).stdout.strip()
        except FileNotFoundError:
            pass  # not macOS

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")

    wav_path = _convert_to_wav(path)
    cleanup_wav = wav_path != path

    try:
        endpoint = "https://api.openai.com/v1/audio/translations" if task == "translate" else "https://api.openai.com/v1/audio/transcriptions"

        data = aiohttp.FormData()
        data.add_field("file", open(wav_path, "rb"), filename="audio.wav", content_type="audio/wav")
        data.add_field("model", "whisper-1")
        data.add_field("response_format", "verbose_json")
        if language and task != "translate":
            data.add_field("language", language)

        async with aiohttp.ClientSession() as session:
            async with session.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                data=data,
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise RuntimeError(f"OpenAI API error {resp.status}: {body[:300]}")
                result = await resp.json()

        segments = []
        for seg in result.get("segments", []):
            segments.append(TranscriptSegment(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
            ))

        return TranscriptResult(
            text=result.get("text", "").strip(),
            language=result.get("language", ""),
            duration_s=result.get("duration", 0.0),
            segments=segments,
            provider="openai",
            source=path,
        )
    finally:
        if cleanup_wav and Path(wav_path).exists():
            Path(wav_path).unlink()
