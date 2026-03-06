"""Gemini TTS provider — Flash and Pro models."""

from __future__ import annotations

import asyncio
import base64
import os
import subprocess
import tempfile
import time

import aiohttp

from . import AudioResult, CostEstimate, TTSProvider, Voice

CHAR_LIMIT = 4096
MAX_CONCURRENT = 8
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


def _pcm_to_wav(pcm_path: str, wav_path: str):
    subprocess.run(
        ["ffmpeg", "-y", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", pcm_path, wav_path],
        capture_output=True,
    )


async def _tts_chunk(session, semaphore, api_key, text, out_path, voice, model, max_retries=3):
    url = f"{API_BASE}/{model}:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": text}]}],
        "generationConfig": {
            "responseModalities": ["AUDIO"],
            "speechConfig": {"voiceConfig": {"prebuiltVoiceConfig": {"voiceName": voice}}}
        }
    }
    for attempt in range(1, max_retries + 1):
        async with semaphore:
            try:
                async with session.post(url, json=payload) as resp:
                    if resp.status == 429 or resp.status >= 500:
                        pass
                    elif resp.status != 200:
                        return False
                    else:
                        data = await resp.json()
                        b64 = (data.get("candidates", [{}])[0]
                               .get("content", {}).get("parts", [{}])[0]
                               .get("inlineData", {}).get("data"))
                        if not b64:
                            return False
                        pcm_path = out_path + ".pcm"
                        with open(pcm_path, "wb") as f:
                            f.write(base64.b64decode(b64))
                        _pcm_to_wav(pcm_path, out_path)
                        os.remove(pcm_path)
                        return True
            except (aiohttp.ClientError, asyncio.TimeoutError):
                pass
            except Exception:
                return False
        await asyncio.sleep(2 ** attempt)
    return False


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
        chunks = _chunk_text(text)
        # Rough estimate: ~4 chars per token for input, ~32 tokens per second of audio
        input_tokens = len(text) / 4
        # ~150 words/min speech, ~0.75 tokens per word, ~50 output tokens per chunk
        output_tokens = len(chunks) * 400
        pricing = MODELS.get(self.model, MODELS[DEFAULT_MODEL])
        cost = (input_tokens / 1_000_000 * pricing["input_per_m"] +
                output_tokens / 1_000_000 * pricing["output_per_m"])
        return CostEstimate(provider=self.name, chars=len(text), chunks=len(chunks), estimated_usd=cost)

    async def generate(self, text, voice="Fenrir", style="", speed=None, output_dir=None, on_progress=None):
        api_key = self._api_key()
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")

        if style:
            text = f"[{style}]\n\n{text}"

        chunks = _chunk_text(text)
        total = len(chunks)
        output_dir = output_dir or os.path.expanduser("~/Downloads")
        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(output_dir, f"outloud_{ts}.wav")
        semaphore = asyncio.Semaphore(MAX_CONCURRENT)

        if total == 1:
            async with aiohttp.ClientSession() as session:
                ok = await _tts_chunk(session, semaphore, api_key, chunks[0], out_file, voice, self.model)
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
                    tasks.append(self._gen_with_progress(session, semaphore, api_key, chunk, path, voice, i, total, on_progress))
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

        if speed and speed != 1.0:
            sped = out_file.replace(".wav", "_speed.wav")
            subprocess.run(["ffmpeg", "-i", out_file, "-filter:a", f"atempo={speed}", sped, "-y"], capture_output=True)
            os.replace(sped, out_file)

        return AudioResult(
            path=out_file, size_kb=os.path.getsize(out_file) // 1024,
            duration_s=self._get_duration(out_file), chunks=total,
            provider=self.name, voice=voice,
        )

    async def _gen_with_progress(self, session, semaphore, api_key, chunk, path, voice, idx, total, on_progress):
        ok = await _tts_chunk(session, semaphore, api_key, chunk, path, voice, self.model)
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
