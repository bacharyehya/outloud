"""Provider architecture for outloud TTS."""

from __future__ import annotations

import os
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
    provider: str
    voice: str


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
