"""Provider architecture for outloud TTS."""

from __future__ import annotations

import os
import subprocess
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Voice:
    id: str
    name: str
    gender: str | None = None


@dataclass
class CostEstimate:
    provider: str
    chars: int
    chunks: int
    estimated_usd: float

    @property
    def display(self) -> str:
        if self.estimated_usd == 0:
            return f"{self.provider}: Free (local)"
        return f"{self.provider}: ~${self.estimated_usd:.4f} ({self.chunks} chunk{'s' if self.chunks != 1 else ''})"


@dataclass
class AudioResult:
    path: str
    size_kb: int
    duration_s: int
    chunks: int
    chunks_failed: int
    provider: str
    voice: str
    elapsed_s: float


class TTSProvider(ABC):
    """Base class for all TTS providers."""

    name: str = "base"
    requires_api_key: bool = True

    @abstractmethod
    async def generate(
        self,
        text: str,
        voice: str,
        style: str = "",
        speed: float = 1.0,
        output_dir: str | None = None,
        on_progress: callable = None,
    ) -> AudioResult | None:
        ...

    @abstractmethod
    def voices(self) -> list[Voice]:
        ...

    def styles(self) -> dict[str, str]:
        return {}

    @abstractmethod
    def estimate_cost(self, text: str) -> CostEstimate:
        ...

    @abstractmethod
    def is_available(self) -> bool:
        ...

    def default_voice(self) -> str:
        v = self.voices()
        return v[0].id if v else ""


# -- Shared utilities --

def chunk_text(text: str, limit: int = 2000) -> list[str]:
    """Split text into chunks at sentence/paragraph boundaries. Filters empty chunks."""
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
    return [c for c in chunks if c.strip()]


def get_duration(path: str) -> int:
    """Get audio duration in seconds via ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True, text=True, timeout=10)
        return int(float(r.stdout.strip() or "0"))
    except Exception:
        return 0


def concat_chunks(ok_files: list[str], out_file: str, tmpdir: str):
    """Concatenate WAV chunks with audio normalization."""
    list_file = os.path.join(tmpdir, "list.txt")
    with open(list_file, "w") as f:
        for cf in ok_files:
            f.write(f"file '{cf}'\n")
    result = subprocess.run([
        "ffmpeg", "-f", "concat", "-safe", "0", "-i", list_file,
        "-af", "dynaudnorm=p=0.9:s=5",
        "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le",
        out_file, "-y",
    ], capture_output=True, timeout=120)
    os.remove(list_file)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg concat failed: {result.stderr.decode()[-200:]}")


def cleanup_tmpdir(tmpdir: str, chunk_files: list[str]):
    """Remove temporary chunk files and directory."""
    for cf in chunk_files:
        try:
            os.remove(cf)
        except FileNotFoundError:
            pass
    try:
        os.rmdir(tmpdir)
    except OSError:
        pass


class ProviderManager:
    """Discovers and manages available TTS providers."""

    def __init__(self):
        self._providers: dict[str, TTSProvider] = {}
        self._discover()

    def _discover(self):
        # Gemini
        from .gemini import GeminiProvider
        g = GeminiProvider()
        self._providers[g.name] = g

        # OpenAI
        from .openai import OpenAIProvider
        o = OpenAIProvider()
        self._providers[o.name] = o

        # KittenTTS (optional)
        try:
            from .kitten import KittenProvider
            k = KittenProvider()
            self._providers[k.name] = k
        except ImportError:
            pass

    @property
    def providers(self) -> dict[str, TTSProvider]:
        return self._providers

    @property
    def available(self) -> dict[str, TTSProvider]:
        return {k: v for k, v in self._providers.items() if v.is_available()}

    def get(self, name: str) -> TTSProvider | None:
        return self._providers.get(name)

    def default(self) -> TTSProvider | None:
        """Return best available provider: Gemini > OpenAI > Local."""
        for name in ["Gemini", "OpenAI", "Local"]:
            p = self._providers.get(name)
            if p and p.is_available():
                return p
        return None
