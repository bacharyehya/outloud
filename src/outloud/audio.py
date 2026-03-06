"""Cross-platform audio playback."""

import shutil
import subprocess
import sys


def play_audio(path: str) -> subprocess.Popen | None:
    """Play a WAV file using the best available system player. Returns the process."""
    plat = sys.platform

    if plat == "darwin":
        return subprocess.Popen(["afplay", path])

    if plat == "win32":
        # PowerShell can play audio without extra deps
        cmd = f'(New-Object Media.SoundPlayer "{path}").PlaySync()'
        return subprocess.Popen(["powershell", "-c", cmd])

    # Linux / FreeBSD / other Unix
    for player in ["aplay", "paplay", "mpv", "ffplay", "cvlc"]:
        if shutil.which(player):
            args = [player]
            if player == "ffplay":
                args += ["-nodisp", "-autoexit"]
            elif player == "mpv":
                args += ["--no-video"]
            elif player == "cvlc":
                args += ["--play-and-exit"]
            args.append(path)
            return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    return None
