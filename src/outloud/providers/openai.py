"""OpenAI TTS provider — gpt-4o-mini-tts and tts-1."""

from __future__ import annotations

import asyncio
import os
import subprocess
import tempfile
import time

import aiohttp

from . import AudioResult, CostEstimate, TTSProvider, Voice, chunk_text, get_duration, concat_chunks, cleanup_tmpdir

CHAR_LIMIT = 4096
MAX_CONCURRENT = 8
REQUEST_TIMEOUT = 60
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


async def _tts_chunk(session, semaphore, api_key, text, out_path, voice, model, style="", speed=1.0, max_retries=3):
    """Generate a single TTS chunk. Returns (success: bool, error: str)."""
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

    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    last_error = ""

    for attempt in range(1, max_retries + 1):
        async with semaphore:
            try:
                async with session.post(API_URL, headers=headers, json=payload, timeout=timeout) as resp:
                    if resp.status == 429:
                        last_error = "rate limited"
                    elif resp.status >= 500:
                        last_error = f"server error {resp.status}"
                    elif resp.status != 200:
                        body = await resp.text()
                        return False, f"HTTP {resp.status}: {body[:150]}"
                    else:
                        data = await resp.read()
                        if len(data) < 100:
                            return False, "empty audio response"
                        with open(out_path, "wb") as f:
                            f.write(data)
                        return True, ""
            except asyncio.TimeoutError:
                last_error = f"timeout ({REQUEST_TIMEOUT}s)"
            except aiohttp.ClientError as e:
                last_error = f"connection error: {e}"
            except Exception as e:
                return False, f"unexpected: {e}"
        if attempt < max_retries:
            await asyncio.sleep(min(1.0 * attempt, 3.0))
    return False, f"failed after {max_retries} retries: {last_error}"


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
        chunks = chunk_text(text, CHAR_LIMIT)
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

        t0 = time.monotonic()
        chunks = chunk_text(text, CHAR_LIMIT)
        total = len(chunks)
        output_dir = output_dir or os.path.expanduser("~/Downloads")
        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(output_dir, f"outloud_{ts}.wav")
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        failed_count = 0

        if total == 1:
            async with aiohttp.ClientSession() as session:
                ok, err = await _tts_chunk(session, semaphore, api_key, chunks[0], out_file, voice, self.model, style, speed)
                if on_progress:
                    on_progress(0, 1, ok, err)
                if not ok:
                    return None
        else:
            tmpdir = tempfile.mkdtemp()
            chunk_files = [os.path.join(tmpdir, f"chunk_{i:03d}.wav") for i in range(total)]
            try:
                async with aiohttp.ClientSession() as session:
                    tasks = []
                    for i, (chunk, path) in enumerate(zip(chunks, chunk_files)):
                        tasks.append(self._gen_with_progress(session, semaphore, api_key, chunk, path, voice, style, speed, i, total, on_progress))
                    results = await asyncio.gather(*tasks)

                ok_files = [f for f, (ok, _) in zip(chunk_files, results) if ok and os.path.isfile(f)]
                failed_count = sum(1 for _, (ok, _) in zip(chunk_files, results) if not ok)

                if not ok_files:
                    return None

                concat_chunks(ok_files, out_file, tmpdir)
            finally:
                cleanup_tmpdir(tmpdir, chunk_files)

        elapsed = time.monotonic() - t0
        if not os.path.isfile(out_file):
            return None

        return AudioResult(
            path=out_file, size_kb=os.path.getsize(out_file) // 1024,
            duration_s=get_duration(out_file), chunks=total,
            chunks_failed=failed_count,
            provider=self.name, voice=voice, elapsed_s=round(elapsed, 1),
        )

    async def _gen_with_progress(self, session, semaphore, api_key, chunk, path, voice, style, speed, idx, total, on_progress):
        ok, err = await _tts_chunk(session, semaphore, api_key, chunk, path, voice, self.model, style, speed)
        if on_progress:
            on_progress(idx, total, ok, err)
        return ok, err
