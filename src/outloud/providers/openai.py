"""OpenAI TTS provider — gpt-4o-mini-tts and tts-1."""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import time

import aiohttp

from . import AudioResult, CostEstimate, TTSProvider, Voice

CHAR_LIMIT = 4096
MAX_CONCURRENT = 4
API_URL = "https://api.openai.com/v1/audio/speech"

MODELS = {
    "gpt-4o-mini-tts": {"label": "Mini TTS", "input_per_m_tokens": 0.60, "output_per_m_tokens": 12.00, "supports_instructions": True},
    "tts-1": {"label": "TTS-1", "per_m_chars": 15.00, "supports_instructions": False},
    "tts-1-hd": {"label": "TTS-1 HD", "per_m_chars": 30.00, "supports_instructions": False},
}
DEFAULT_MODEL = "gpt-4o-mini-tts"

VOICES = [
    Voice("alloy", "Alloy"), Voice("ash", "Ash"), Voice("ballad", "Ballad"),
    Voice("coral", "Coral"), Voice("echo", "Echo"), Voice("fable", "Fable"),
    Voice("nova", "Nova"), Voice("onyx", "Onyx"), Voice("sage", "Sage"),
    Voice("shimmer", "Shimmer"), Voice("verse", "Verse"),
    Voice("marin", "Marin"), Voice("cedar", "Cedar"),
]

# Only these work with tts-1 / tts-1-hd
TTS1_VOICES = {"alloy", "ash", "coral", "echo", "fable", "nova", "onyx", "sage", "shimmer"}

STYLE_PRESETS = {
    "Default": "",
    "Warm": "Speak in a warm, friendly, conversational tone.",
    "Whisper": "Whisper softly, intimate and gentle.",
    "Energetic": "Fast-paced, energetic, like a tech keynote presenter.",
    "Newsreader": "Calm, neutral newsreader. Clear and professional.",
    "Storyteller": "Speak like a master storyteller. Vivid, dramatic, with natural pauses.",
    "Calm": "Speak in a calm, soothing, meditative tone.",
}


def _chunk_text(text: str, limit: int = CHAR_LIMIT) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks, remaining = [], text
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


async def _tts_chunk(session, semaphore, api_key, text, out_path, voice, model, style="", speed=1.0, max_retries=3):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "input": text,
        "voice": voice,
        "response_format": "wav",
    }
    model_info = MODELS.get(model, MODELS[DEFAULT_MODEL])
    if model_info.get("supports_instructions") and style:
        payload["instructions"] = style
    if not model_info.get("supports_instructions") and speed != 1.0:
        payload["speed"] = max(0.25, min(4.0, speed))

    for attempt in range(1, max_retries + 1):
        async with semaphore:
            try:
                async with session.post(API_URL, headers=headers, json=payload) as resp:
                    if resp.status == 429 or resp.status >= 500:
                        pass
                    elif resp.status != 200:
                        return False
                    else:
                        data = await resp.read()
                        with open(out_path, "wb") as f:
                            f.write(data)
                        return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            except Exception:
                return False
        await asyncio.sleep(2 ** attempt)
    return False


class OpenAIProvider(TTSProvider):
    name = "OpenAI"
    requires_api_key = True

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def _api_key(self) -> str | None:
        return os.environ.get("OPENAI_API_KEY")

    def is_available(self) -> bool:
        return self._api_key() is not None

    def voices(self) -> list[Voice]:
        if self.model in ("tts-1", "tts-1-hd"):
            return [v for v in VOICES if v.id in TTS1_VOICES]
        return VOICES

    def styles(self) -> dict[str, str]:
        if MODELS.get(self.model, {}).get("supports_instructions"):
            return STYLE_PRESETS
        return {"Default": ""}

    def default_voice(self) -> str:
        return "coral"

    def estimate_cost(self, text: str) -> CostEstimate:
        chunks = _chunk_text(text)
        model_info = MODELS.get(self.model, MODELS[DEFAULT_MODEL])
        if "per_m_chars" in model_info:
            cost = len(text) / 1_000_000 * model_info["per_m_chars"]
        else:
            input_tokens = len(text) / 4
            output_tokens = len(chunks) * 400
            cost = (input_tokens / 1_000_000 * model_info["input_per_m_tokens"] +
                    output_tokens / 1_000_000 * model_info["output_per_m_tokens"])
        return CostEstimate(provider=self.name, chars=len(text), chunks=len(chunks), estimated_usd=cost)

    async def generate(self, text, voice="coral", style="", speed=1.0, output_dir=None, on_progress=None):
        api_key = self._api_key()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        chunks = _chunk_text(text)
        total = len(chunks)
        output_dir = output_dir or os.path.expanduser("~/Downloads")
        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(output_dir, f"outloud_{ts}.wav")
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        if total == 1:
            async with aiohttp.ClientSession() as session:
                ok = await _tts_chunk(session, semaphore, api_key, chunks[0], out_file, voice, self.model, style, speed)
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
                    tasks.append(self._gen_with_progress(session, semaphore, api_key, chunk, path, voice, style, speed, i, total, on_progress))
                results = await asyncio.gather(*tasks)

            ok_files = [f for f, ok in zip(chunk_files, results) if ok and os.path.isfile(f)]
            if not ok_files:
                self._cleanup(tmpdir, chunk_files)
                return None

            list_file = os.path.join(tmpdir, "list.txt")
            with open(list_file, "w") as f:
                for cf in ok_files:
                    f.write(f"file '{cf}'\n")
            subprocess.run(["ffmpeg", "-f", "concat", "-safe", "0", "-i", list_file, "-c", "copy", out_file, "-y"], capture_output=True)
            self._cleanup(tmpdir, chunk_files, list_file)

        return AudioResult(
            path=out_file, size_kb=os.path.getsize(out_file) // 1024,
            duration_s=self._get_duration(out_file), chunks=total,
            provider=self.name, voice=voice,
        )

    async def _gen_with_progress(self, session, semaphore, api_key, chunk, path, voice, style, speed, idx, total, on_progress):
        ok = await _tts_chunk(session, semaphore, api_key, chunk, path, voice, self.model, style, speed)
        if on_progress:
            on_progress(idx, total, ok)
        return ok

    def _cleanup(self, tmpdir, chunk_files, list_file=None):
        for cf in chunk_files:
            if os.path.isfile(cf):
                os.remove(cf)
        if list_file and os.path.isfile(list_file):
            os.remove(list_file)
        if os.path.isdir(tmpdir):
            os.rmdir(tmpdir)

    def _get_duration(self, path: str) -> int:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                capture_output=True, text=True)
            return int(float(r.stdout.strip() or "0"))
        except Exception:
            return 0
