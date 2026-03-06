"""Gemini TTS engine — async chunked text-to-speech."""

import asyncio
import base64
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass

import aiohttp

CHAR_LIMIT = 4096
MAX_CONCURRENT = 8
API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"

VOICES = [
    "Zephyr", "Puck", "Charon", "Kore", "Fenrir", "Leda", "Orus",
    "Aoede", "Callirrhoe", "Autonoe", "Enceladus", "Iapetus",
    "Umbriel", "Algieba", "Despina", "Erinome", "Algenib",
    "Rasalgethi", "Laomedeia", "Achernar", "Alnilam", "Schedar",
    "Gacrux", "Pulcherrima", "Achird", "Zubenelgenubi",
    "Vindemiatrix", "Sadachbia", "Sadaltager", "Sulafat",
]

STYLE_PRESETS = {
    "Default": "",
    "Warm British": "Speak with a refined British accent. Deep, warm, philosophical tone. Thoughtful pacing, like a late-night conversation between old friends.",
    "Whisper": "Whisper softly, intimate and gentle.",
    "Energetic": "Fast-paced, energetic, like a tech keynote presenter.",
    "Newsreader": "Calm, neutral newsreader. Clear and professional.",
    "Excited": "Excited and enthusiastic, like announcing something amazing.",
    "Audiobook": "Narrate like a premium audiobook. Rich, immersive, measured pacing.",
    "Storyteller": "Speak like a master storyteller. Vivid, dramatic, with natural pauses for effect.",
    "Calm": "Speak in a calm, soothing, meditative tone. Slow and relaxed.",
}

DEFAULT_VOICE = "Fenrir"
DEFAULT_STYLE = "Warm British"


@dataclass
class TTSResult:
    path: str
    size_kb: int
    duration_s: int
    chunks: int


def chunk_text(text: str, limit: int = CHAR_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break

        cut = limit
        for sep in [". ", "! ", "? ", "\n\n", "\n", ", ", " "]:
            idx = remaining[:limit].rfind(sep)
            if idx > 0:
                cut = idx + len(sep)
                break

        chunks.append(remaining[:cut])
        remaining = remaining[cut:]

    return chunks


def pcm_to_wav(pcm_path: str, wav_path: str):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1",
         "-i", pcm_path, wav_path],
        capture_output=True,
    )


async def tts_chunk(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    api_key: str,
    text: str,
    out_path: str,
    voice: str,
    model: str,
    max_retries: int = 3,
) -> bool:
    url = f"{API_BASE}/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {
                "voiceConfig": {
                    "prebuiltVoiceConfig": {"voiceName": voice}
                }
            }
        }
    }

    for attempt in range(1, max_retries + 1):
        async with semaphore:
            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 429 or resp.status >= 500:
                        pass  # retry
                    elif resp.status != 200:
                        return False
                    else:
                        data = await resp.json()
                        b64 = (data.get("candidates", [{}])[0]
                               .get("content", {})
                               .get("parts", [{}])[0]
                               .get("inlineData", {})
                               .get("data"))
                        if not b64:
                            return False

                        pcm_bytes = base64.b64decode(b64)
                        pcm_path = out_path + ".pcm"
                        with open(pcm_path, "wb") as f:
                            f.write(pcm_bytes)
                        pcm_to_wav(pcm_path, out_path)
                        os.remove(pcm_path)
                        return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            except Exception:
                return False

        wait = 2 ** attempt
        await asyncio.sleep(wait)

    return False


async def generate(
    text: str,
    voice: str = DEFAULT_VOICE,
    style: str = "",
    model: str = DEFAULT_MODEL,
    speed: float | None = None,
    output_dir: str | None = None,
    on_progress: callable = None,
) -> TTSResult | None:
    """Generate TTS audio. on_progress(chunk_idx, total, success) is called per chunk."""
    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY not set (also accepts GOOGLE_API_KEY)")

    if style:
        text = f"[{style}]\n\n{text}"

    chunks = chunk_text(text)
    total = len(chunks)

    if output_dir is None:
        output_dir = os.path.expanduser("~/Downloads")
    os.makedirs(output_dir, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"readthis_{ts}.wav")
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    if total == 1:
        async with aiohttp.ClientSession() as session:
            ok = await tts_chunk(session, semaphore, api_key, chunks[0], out_file, voice, model)
            if on_progress:
                on_progress(0, 1, ok)
            if not ok:
                return None
    else:
        tmpdir = tempfile.mkdtemp()
        chunk_files = [os.path.join(tmpdir, f"chunk_{i:03d}.wav") for i in range(total)]

        async with aiohttp.ClientSession() as session:
            tasks = []
            for i, (chunk, path) in enumerate(zip(chunks, chunk_files)):
                tasks.append(_tts_with_progress(
                    session, semaphore, api_key, chunk, path, voice, model, i, total, on_progress
                ))
            results = await asyncio.gather(*tasks)

        ok_files = [f for f, ok in zip(chunk_files, results) if ok and os.path.isfile(f)]
        if not ok_files:
            _cleanup_tmpdir(tmpdir, chunk_files)
            return None

        list_file = os.path.join(tmpdir, "list.txt")
        with open(list_file, "w") as f:
            for cf in ok_files:
                f.write(f"file '{cf}'\n")

        subprocess.run(
            ["ffmpeg", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", out_file, "-y"],
            capture_output=True,
        )
        _cleanup_tmpdir(tmpdir, chunk_files, list_file)

    if speed and speed != 1.0:
        sped_file = out_file.replace(".wav", "_speed.wav")
        subprocess.run(
            ["ffmpeg", "-i", out_file, "-filter:a", f"atempo={speed}", sped_file, "-y"],
            capture_output=True,
        )
        os.replace(sped_file, out_file)

    size_kb = os.path.getsize(out_file) // 1024
    duration_s = _get_duration(out_file)

    return TTSResult(path=out_file, size_kb=size_kb, duration_s=duration_s, chunks=total)


async def _tts_with_progress(session, semaphore, api_key, chunk, path, voice, model, idx, total, on_progress):
    ok = await tts_chunk(session, semaphore, api_key, chunk, path, voice, model)
    if on_progress:
        on_progress(idx, total, ok)
    return ok


def _cleanup_tmpdir(tmpdir, chunk_files, list_file=None):
    for cf in chunk_files:
        if os.path.isfile(cf):
            os.remove(cf)
    if list_file and os.path.isfile(list_file):
        os.remove(list_file)
    if os.path.isdir(tmpdir):
        os.rmdir(tmpdir)


def _get_duration(path: str) -> int:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True,
        )
        return int(float(result.stdout.strip() or "0"))
    except Exception:
        return 0


def play_audio(path: str):
    subprocess.Popen(["afplay", path])
