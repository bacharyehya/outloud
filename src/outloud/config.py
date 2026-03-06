"""Config persistence for outloud."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

CONFIG_DIR = Path(os.environ.get("XDG_CONFIG_HOME", "~/.config")).expanduser() / "outloud"
CONFIG_FILE = CONFIG_DIR / "config.toml"


@dataclass
class Config:
    default_provider: str = ""
    default_voice: str = ""
    default_style: str = "Default"
    output_dir: str = "~/Downloads"
    speed: float = 1.0
    generation_count: int = 0
    star_dismissed: bool = False

    def save(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        lines = [
            f'default_provider = "{self.default_provider}"',
            f'default_voice = "{self.default_voice}"',
            f'default_style = "{self.default_style}"',
            f'output_dir = "{self.output_dir}"',
            f'speed = {self.speed}',
            f'generation_count = {self.generation_count}',
            f'star_dismissed = {"true" if self.star_dismissed else "false"}',
        ]
        CONFIG_FILE.write_text("\n".join(lines) + "\n")

    @classmethod
    def load(cls) -> "Config":
        c = cls()
        if not CONFIG_FILE.exists():
            return c
        try:
            for line in CONFIG_FILE.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key, val = key.strip(), val.strip().strip('"')
                if key == "default_provider":
                    c.default_provider = val
                elif key == "default_voice":
                    c.default_voice = val
                elif key == "default_style":
                    c.default_style = val
                elif key == "output_dir":
                    c.output_dir = val
                elif key == "speed":
                    c.speed = float(val)
                elif key == "generation_count":
                    c.generation_count = int(val)
                elif key == "star_dismissed":
                    c.star_dismissed = val == "true"
        except Exception:
            pass
        return c
