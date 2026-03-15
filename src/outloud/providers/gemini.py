"""Gemini TTS provider — Flash and Pro models."""

from __future__ import annotations

import asyncio
import base64
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass

import aiohttp

from . import AudioResult, CostEstimate, TTSProvider, Voice, chunk_text, get_duration, concat_chunks, cleanup_tmpdir

MAX_CONCURRENT = 4
REQUEST_TIMEOUT = 60  # seconds per chunk
API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"

MODELS = {
    "gemini-2.5-flash-preview-tts": {"label": "Flash", "input_per_m": 0.50, "output_per_m": 10.00},
    "gemini-2.5-pro-preview-tts": {"label": "Pro", "input_per_m": 1.00, "output_per_m": 20.00},
}
DEFAULT_MODEL = "gemini-2.5-flash-preview-tts"

VOICES = [
    Voice("Zephyr", "Zephyr"), Voice("Puck", "Puck"), Voice("Charon", "Charon"),
    Voice("Kore", "Kore"), Voice("Fenrir", "Fenrir"), Voice("Leda", "Leda"),
    Voice("Orus", "Orus"), Voice("Aoede", "Aoede"), Voice("Callirrhoe", "Callirrhoe"),
    Voice("Autonoe", "Autonoe"), Voice("Enceladus", "Enceladus"), Voice("Iapetus", "Iapetus"),
    Voice("Umbriel", "Umbriel"), Voice("Algieba", "Algieba"), Voice("Despina", "Despina"),
    Voice("Erinome", "Erinome"), Voice("Algenib", "Algenib"), Voice("Rasalgethi", "Rasalgethi"),
    Voice("Laomedeia", "Laomedeia"), Voice("Achernar", "Achernar"), Voice("Alnilam", "Alnilam"),
    Voice("Schedar", "Schedar"), Voice("Gacrux", "Gacrux"), Voice("Pulcherrima", "Pulcherrima"),
    Voice("Achird", "Achird"), Voice("Zubenelgenubi", "Zubenelgenubi"),
    Voice("Vindemiatrix", "Vindemiatrix"), Voice("Sadachbia", "Sadachbia"),
    Voice("Sadaltager", "Sadaltager"), Voice("Sulafat", "Sulafat"),
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


def _pcm_to_wav(pcm_path: str, wav_path: str):
    result = subprocess.run(
        ["ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", pcm_path, wav_path],
        capture_output=True, timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"PCM→WAV failed: {result.stderr.decode()[-200:]}")


@dataclass
class _ChunkResult:
    ok: bool
    error: str = ""
    input_tokens: int = 0
    output_tokens: int = 0


async def _tts_chunk(session, semaphore, api_key, text, out_path, voice, model, style="", max_retries=6):
    """Generate a single TTS chunk. Returns _ChunkResult with token usage."""
    import random

    if style:
        styled_text = f"[{style}]\n\n{text}"
    else:
        styled_text = text
    url = f"{API_BASE}/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": styled_text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}
        }
    }
    timeout = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
    last_error = ""

    for attempt in range(1, max_retries + 1):
        async with semaphore:
            try:
                async with session.post(url, json=payload, timeout=timeout) as resp:
                    if resp.status == 429:
                        last_error = "rate limited"
                    elif resp.status >= 500:
                        last_error = f"server error {resp.status}"
                    elif resp.status != 200:
                        body = await resp.text()
                        return _ChunkResult(ok=False, error=f"HTTP {resp.status}: {body[:150]}")
                    else:
                        data = await resp.json()
                        b64 = (data.get("candidates", [{}])[0]
                               .get("content", {}).get("parts", [{}])[0]
                               .get("inlineData", {}).get("data"))
                        if not b64:
                            return _ChunkResult(ok=False, error="empty audio data in response")

                        # Extract real token usage
                        usage = data.get("usageMetadata", {})
                        in_tok = usage.get("promptTokenCount", 0)
                        out_tok = usage.get("candidatesTokenCount", 0) or usage.get("totalTokenCount", 0) - in_tok

                        pcm_path = out_path + ".pcm"
                        with open(pcm_path, "wb") as f:
                            f.write(base64.b64decode(b64))
                        try:
                            _pcm_to_wav(pcm_path, out_path)
                        finally:
                            try:
                                os.remove(pcm_path)
                            except FileNotFoundError:
                                pass
                        return _ChunkResult(ok=True, input_tokens=in_tok, output_tokens=out_tok)
            except asyncio.TimeoutError:
                last_error = f"timeout ({REQUEST_TIMEOUT}s)"
            except aiohttp.ClientError as e:
                last_error = f"connection error: {e}"
            except Exception as e:
                return _ChunkResult(ok=False, error=f"unexpected: {e}")
        if attempt < max_retries:
            base_delay = min(2.0 ** attempt, 15.0)
            jitter = random.uniform(0, base_delay * 0.5)
            await asyncio.sleep(base_delay + jitter)
    return _ChunkResult(ok=False, error=f"failed after {max_retries} retries: {last_error}")


class GeminiProvider(TTSProvider):
    name = "Gemini"
    requires_api_key = True

    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model

    def _api_key(self) -> str | None:
        return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

    def is_available(self) -> bool:
        return self._api_key() is not None

    def voices(self) -> list[Voice]:
        return VOICES

    def styles(self) -> dict[str, str]:
        return STYLE_PRESETS

    def default_voice(self) -> str:
        return "Fenrir"

    def estimate_cost(self, text: str) -> CostEstimate:
        chunks = chunk_text(text)
        input_tokens = len(text) / 4
        # Estimate audio duration: ~1400 chars/min TTS speech rate (calibrated from real usage)
        # Audio output tokens: ~30 tokens/second (conservative estimate)
        estimated_seconds = len(text) / 1400 * 60
        output_tokens = estimated_seconds * 30
        pricing = MODELS.get(self.model, MODELS[DEFAULT_MODEL])
        cost = (input_tokens / 1_000_000 * pricing["input_per_m"] +
                output_tokens / 1_000_000 * pricing["output_per_m"])
        return CostEstimate(provider=self.name, chars=len(text), chunks=len(chunks), estimated_usd=cost)

    def _calc_actual_cost(self, total_input: int, total_output: int) -> float | None:
        if total_input == 0 and total_output == 0:
            return None
        pricing = MODELS.get(self.model, MODELS[DEFAULT_MODEL])
        return (total_input / 1_000_000 * pricing["input_per_m"] +
                total_output / 1_000_000 * pricing["output_per_m"])

    async def generate(self, text, voice="Fenrir", style="", speed=None, output_dir=None, on_progress=None):
        api_key = self._api_key()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

        t0 = time.monotonic()
        chunks = chunk_text(text)
        total = len(chunks)
        output_dir = output_dir or os.path.expanduser("~/Downloads")
        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(output_dir, f"outloud_{ts}.wav")
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        failed_count = 0
        total_input_tokens = 0
        total_output_tokens = 0

        if total == 1:
            async with aiohttp.ClientSession() as session:
                cr = await _tts_chunk(session, semaphore, api_key, chunks[0], out_file, voice, self.model, style=style)
                total_input_tokens += cr.input_tokens
                total_output_tokens += cr.output_tokens
                if on_progress:
                    on_progress(0, 1, cr.ok, cr.error)
                if not cr.ok:
                    return None
        else:
            tmpdir = tempfile.mkdtemp()
            chunk_files = [os.path.join(tmpdir, f"chunk_{i:03d}.wav") for i in range(total)]
            try:
                async with aiohttp.ClientSession() as session:
                    tasks = []
                    for i, (chunk, path) in enumerate(zip(chunks, chunk_files)):
                        tasks.append(self._gen_with_progress(session, semaphore, api_key, chunk, path, voice, i, total, on_progress, style=style, delay=i * 0.15))
                    results = await asyncio.gather(*tasks)

                ok_files = []
                for f, cr in zip(chunk_files, results):
                    total_input_tokens += cr.input_tokens
                    total_output_tokens += cr.output_tokens
                    if cr.ok and os.path.isfile(f):
                        ok_files.append(f)
                    elif not cr.ok:
                        failed_count += 1

                if not ok_files:
                    return None

                concat_chunks(ok_files, out_file, tmpdir)
            finally:
                cleanup_tmpdir(tmpdir, chunk_files)

        if speed and speed != 1.0:
            sped = out_file.replace(".wav", "_speed.wav")
            r = subprocess.run(["ffmpeg", "-i", out_file, "-filter:a", f"atempo={speed}", sped, "-y"], capture_output=True)
            if r.returncode == 0:
                os.replace(sped, out_file)
            else:
                try:
                    os.remove(sped)
                except FileNotFoundError:
                    pass

        elapsed = time.monotonic() - t0
        if not os.path.isfile(out_file):
            return None

        return AudioResult(
            path=out_file, size_kb=os.path.getsize(out_file) // 1024,
            duration_s=get_duration(out_file), chunks=total,
            chunks_failed=failed_count,
            provider=self.name, voice=voice, elapsed_s=round(elapsed, 1),
            actual_cost=self._calc_actual_cost(total_input_tokens, total_output_tokens),
        )

    async def _gen_with_progress(self, session, semaphore, api_key, chunk, path, voice, idx, total, on_progress, style="", delay=0):
        if delay > 0:
            await asyncio.sleep(delay)
        cr = await _tts_chunk(session, semaphore, api_key, chunk, path, voice, self.model, style=style)
        if on_progress:
            on_progress(idx, total, cr.ok, cr.error)
        return cr
