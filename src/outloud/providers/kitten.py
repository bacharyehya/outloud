"""KittenTTS provider — local, free, no API key needed."""

from __future__ import annotations

import asyncio
import os
import time

from . import AudioResult, CostEstimate, TTSProvider, Voice, get_duration

VOICES = [
    Voice("Bella", "Bella", "female"), Voice("Jasper", "Jasper", "male"),
    Voice("Luna", "Luna", "female"), Voice("Bruno", "Bruno", "male"),
    Voice("Rosie", "Rosie", "female"), Voice("Hugo", "Hugo", "male"),
    Voice("Kiki", "Kiki", "female"), Voice("Leo", "Leo", "male"),
]

MODELS = {
    "mini": "KittenML/kitten-tts-mini-0.8",
    "micro": "KittenML/kitten-tts-micro-0.8",
    "nano": "KittenML/kitten-tts-nano-0.8-int8",
}


class KittenProvider(TTSProvider):
    name = "Local"
    requires_api_key = False

    def __init__(self, model: str = "mini"):
        import kittentts  # noqa: F401
        import soundfile  # noqa: F401
        self._model_id = MODELS.get(model, MODELS["mini"])
        self._tts = None

    def _get_tts(self):
        if self._tts is None:
            from kittentts import KittenTTS
            self._tts = KittenTTS(self._model_id)
        return self._tts

    def is_available(self) -> bool:
        try:
            import kittentts  # noqa: F401
            return True
        except ImportError:
            return False

    def voices(self) -> list[Voice]:
        return VOICES

    def styles(self) -> dict[str, str]:
        return {"Default": ""}

    def default_voice(self) -> str:
        return "Bella"

    def estimate_cost(self, text: str) -> CostEstimate:
        chunks = max(1, (len(text) + 399) // 400)
        return CostEstimate(provider=self.name, chars=len(text), chunks=chunks, estimated_usd=0.0)

    async def generate(self, text, voice="Bella", style="", speed=1.0, output_dir=None, on_progress=None):
        output_dir = output_dir or os.path.expanduser("~/Downloads")
        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_file = os.path.join(output_dir, f"outloud_{ts}.wav")

        t0 = time.monotonic()
        tts = self._get_tts()
        chunks = max(1, (len(text) + 399) // 400)
        error = ""

        def _generate():
            import soundfile as sf
            audio = tts.generate(text, voice=voice)
            sf.write(out_file, audio, 24000)

        try:
            await asyncio.to_thread(_generate)
        except Exception as e:
            error = str(e)
            if on_progress:
                on_progress(0, chunks, False, error)
            return None

        if on_progress:
            on_progress(chunks - 1, chunks, True, "")

        elapsed = time.monotonic() - t0
        if not os.path.isfile(out_file):
            return None

        return AudioResult(
            path=out_file, size_kb=os.path.getsize(out_file) // 1024,
            duration_s=get_duration(out_file), chunks=chunks,
            chunks_failed=0,
            provider=self.name, voice=voice, elapsed_s=round(elapsed, 1),
        )
