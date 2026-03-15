"""outloud TUI — Beautiful terminal interface for multi-provider TTS."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import (
    Button,
    Footer,
    Header,
    Label,
    OptionList,
    ProgressBar,
    RadioButton,
    RadioSet,
    Select,
    Static,
    TextArea,
)
from textual.widgets.option_list import Option

from .audio import play_audio
from .config import Config
from .providers import ProviderManager, TTSProvider
from .stt import record_from_mic_managed, validate_recording, transcribe_mlx, transcribe_openai


REPO_URL = "https://github.com/bacharyehya/outloud"
COFFEE_URL = "https://buymeacoffee.com/bash"


class Sidebar(VerticalScroll):
    DEFAULT_CSS = """
    Sidebar {
        width: 30;
        border: solid $primary-background;
        padding: 0 1;
    }
    Sidebar > Label {
        text-style: bold;
        margin-bottom: 1;
        color: $text;
    }
    Sidebar RadioSet {
        height: auto;
        margin-bottom: 1;
    }
    Sidebar OptionList {
        height: 1fr;
        min-height: 8;
    }
    #history-label {
        margin-top: 1;
    }
    #history-list {
        height: auto;
        max-height: 10;
    }
    """

    def __init__(self, manager: ProviderManager, config: Config):
        super().__init__()
        self._manager = manager
        self._config = config

    def compose(self) -> ComposeResult:
        yield Label("Provider")
        with RadioSet(id="provider-radio"):
            for name, provider in self._manager.providers.items():
                avail = provider.is_available()
                label = f"{name}" if avail else f"{name} (no key)"
                rb = RadioButton(label, value=avail, name=name, id=f"provider-{name}")
                if not avail:
                    rb.disabled = True
                yield rb
        yield Label("Voice")
        yield OptionList(id="voice-list")
        yield Label("History", id="history-label")
        yield OptionList(id="history-list")

    def on_mount(self) -> None:
        self._select_default_provider()
        self._refresh_voices()

    def _select_default_provider(self):
        radio_set = self.query_one("#provider-radio", RadioSet)
        default = self._manager.default()
        if not default:
            return
        target = self._config.default_provider or default.name
        for rb in radio_set.query(RadioButton):
            if rb.name == target and not rb.disabled:
                rb.value = True
                break

    def _refresh_voices(self):
        ol = self.query_one("#voice-list", OptionList)
        ol.clear_options()
        provider = self.selected_provider
        if not provider:
            return
        for v in provider.voices():
            ol.add_option(Option(v.name, id=v.id))
        default = self._config.default_voice or provider.default_voice()
        for i, v in enumerate(provider.voices()):
            if v.id == default:
                ol.highlighted = i
                break

    @property
    def selected_provider(self) -> TTSProvider | None:
        radio_set = self.query_one("#provider-radio", RadioSet)
        for rb in radio_set.query(RadioButton):
            if rb.value and not rb.disabled:
                return self._manager.get(rb.name)
        return self._manager.default()

    @property
    def selected_voice(self) -> str:
        provider = self.selected_provider
        if not provider:
            return ""
        ol = self.query_one("#voice-list", OptionList)
        voices = provider.voices()
        if ol.highlighted is not None and ol.highlighted < len(voices):
            return voices[ol.highlighted].id
        return provider.default_voice()

    def on_radio_set_changed(self, event: RadioSet.Changed) -> None:
        self._refresh_voices()
        self.app._update_styles_and_cost()


class OutloudApp(App):
    CSS = """
    Screen {
        layout: horizontal;
    }

    #main-panel {
        width: 1fr;
        padding: 1 2;
    }

    #text-input {
        height: 1fr;
        margin-bottom: 1;
    }

    #controls-row-1 {
        height: auto;
        margin-bottom: 1;
    }

    #controls-row-2 {
        height: auto;
        margin-bottom: 1;
    }

    #style-select {
        width: 30;
        margin-right: 2;
    }

    #cost-label {
        margin-left: 2;
        color: $text-muted;
        content-align-vertical: middle;
    }

    #status-bar {
        height: auto;
        margin-top: 1;
    }

    #status-text {
        margin-bottom: 1;
        color: $text-muted;
    }

    #star-banner {
        display: none;
        height: auto;
        margin: 1 0;
        padding: 1 2;
        background: $primary-background;
        color: $text;
    }

    #star-banner.visible {
        display: block;
    }

    #progress {
        display: none;
    }

    #progress.visible {
        display: block;
    }

    Button {
        margin-right: 1;
    }

    #btn-record.recording {
        background: $error;
        color: $text;
    }

    #recording-timer {
        display: none;
        margin-left: 1;
        color: $error;
        content-align-vertical: middle;
    }

    #recording-timer.visible {
        display: block;
    }

    #stt-provider-select {
        width: 20;
        margin-left: 2;
    }
    """

    TITLE = "outloud"
    SUB_TITLE = "Text-to-Speech"

    BINDINGS = [
        Binding("ctrl+g", "generate", "Generate", show=True),
        Binding("ctrl+r", "toggle_record", "Record", show=True),
        Binding("ctrl+p", "play_last", "Play Last", show=True),
        Binding("ctrl+o", "open_downloads", "Open Folder", show=True),
        Binding("ctrl+l", "clear_text", "Clear", show=True),
        Binding("ctrl+q", "quit", "Quit", show=True),
    ]

    def __init__(self, initial_text: str = ""):
        super().__init__()
        self._initial_text = initial_text
        self._manager = ProviderManager()
        self._config = Config.load()
        self._history: list = []
        self._generating = False
        self._chunks_done = 0
        self._chunks_total = 0
        # STT state
        self._recording = False
        self._recording_proc: subprocess.Popen | None = None
        self._recording_path: str = ""
        self._recording_start: float = 0.0
        self._recording_timer = None
        self._transcribing = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Sidebar(self._manager, self._config)
        with Vertical(id="main-panel"):
            yield TextArea(self._initial_text, id="text-input", language=None, show_line_numbers=False)
            with Horizontal(id="controls-row-1"):
                yield Select([("Default", "Default")], value="Default", id="style-select", allow_blank=False)
                yield Static("", id="cost-label")
            with Horizontal(id="controls-row-2"):
                yield Button("Generate", variant="success", id="btn-generate")
                yield Button("Record", variant="default", id="btn-record")
                yield Button("Play Last", variant="primary", id="btn-play")
                yield Button("Open Folder", variant="warning", id="btn-open")
                yield Button("Clear", variant="error", id="btn-clear")
                yield Static("", id="recording-timer")
                stt_options = [("MLX (local)", "mlx"), ("OpenAI", "openai")]
                stt_default = self._config.stt_provider if self._config.stt_provider in ("mlx", "openai") else "mlx"
                yield Select(stt_options, value=stt_default, id="stt-provider-select", allow_blank=False)
            yield Static("", id="star-banner")
            with Vertical(id="status-bar"):
                yield Static("Ready. Paste text above and hit Generate (Ctrl+G).", id="status-text")
                yield ProgressBar(total=100, show_eta=False, id="progress")
        yield Footer()

    def on_mount(self) -> None:
        self._update_styles_and_cost()
        ta = self.query_one("#text-input", TextArea)
        ta.focus()
        if not self._manager.available:
            self._set_status("No providers available. Set GEMINI_API_KEY or OPENAI_API_KEY, or install outloud-tts[local].")

    def _update_styles_and_cost(self) -> None:
        sidebar = self.query_one(Sidebar)
        provider = sidebar.selected_provider
        style_select = self.query_one("#style-select", Select)
        if provider:
            styles = provider.styles()
            options = [(name, name) for name in styles]
            style_select.set_options(options)
            if options:
                default = self._config.default_style if self._config.default_style in styles else options[0][1]
                style_select.value = default
            sub = provider.name
            if hasattr(provider, 'model'):
                sub += f" ({provider.model.split('/')[-1]})"
            self.sub_title = sub
        self._update_cost()

    def _update_cost(self) -> None:
        sidebar = self.query_one(Sidebar)
        provider = sidebar.selected_provider
        cost_label = self.query_one("#cost-label", Static)
        if not provider:
            cost_label.update("")
            return
        ta = self.query_one("#text-input", TextArea)
        text = ta.text.strip()
        if not text:
            cost_label.update(f"{len(self.query_one('#text-input', TextArea).text)} chars")
            return
        est = provider.estimate_cost(text)
        cost_label.update(est.display)

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._update_cost()

    def _set_status(self, msg: str) -> None:
        self.query_one("#status-text", Static).update(msg)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        actions = {
            "btn-generate": self.action_generate,
            "btn-record": self.action_toggle_record,
            "btn-play": self.action_play_last,
            "btn-open": self.action_open_downloads,
            "btn-clear": self.action_clear_text,
        }
        action = actions.get(event.button.id)
        if action:
            action()

    def action_generate(self) -> None:
        if self._generating:
            self._set_status("Already generating...")
            return
        ta = self.query_one("#text-input", TextArea)
        text = ta.text.strip()
        if not text:
            self._set_status("Nothing to generate. Paste some text first.")
            return
        sidebar = self.query_one(Sidebar)
        provider = sidebar.selected_provider
        if not provider:
            self._set_status("No provider available.")
            return
        self._do_generate(text, provider, sidebar.selected_voice)

    @work(thread=True)
    def _do_generate(self, text: str, provider: TTSProvider, voice: str) -> None:
        import asyncio

        self._generating = True
        style_name = self.app.query_one("#style-select", Select).value
        style = provider.styles().get(style_name, "") if style_name else ""

        est = provider.estimate_cost(text)
        self._chunks_total = est.chunks
        self._chunks_done = 0

        self.call_from_thread(self._show_progress, len(text), est.chunks, provider.name, est.display)

        def on_progress(idx, total, success):
            self._chunks_done += 1
            pct = int(self._chunks_done / max(self._chunks_total, 1) * 100)
            status = "ok" if success else "FAILED"
            self.call_from_thread(self._update_progress, pct, idx + 1, total, status)

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                provider.generate(text=text, voice=voice, style=style, speed=self._config.speed, on_progress=on_progress)
            )
        except Exception as e:
            self.call_from_thread(self._generation_error, str(e))
            return
        finally:
            loop.close()
            self._generating = False

        if result:
            self._history.append(result)
            self._config.generation_count += 1
            self._config.save()
            self.call_from_thread(self._generation_done, result)
        else:
            self.call_from_thread(self._generation_error, "Generation failed")

    def _show_progress(self, chars: int, chunks: int, provider: str, cost: str) -> None:
        pb = self.query_one("#progress", ProgressBar)
        pb.add_class("visible")
        pb.update(progress=0, total=100)
        self._set_status(f"Generating via {provider}... {chars:,} chars, {chunks} chunk{'s' if chunks != 1 else ''} — {cost}")
        self.query_one("#btn-generate", Button).disabled = True

    def _update_progress(self, pct: int, chunk_idx: int, total: int, status: str) -> None:
        pb = self.query_one("#progress", ProgressBar)
        pb.update(progress=pct, total=100)
        self._set_status(f"Chunk {chunk_idx}/{total} {status} ({pct}%)")

    def _generation_done(self, result) -> None:
        pb = self.query_one("#progress", ProgressBar)
        pb.remove_class("visible")
        self.query_one("#btn-generate", Button).disabled = False

        mins, secs = divmod(result.duration_s, 60)
        self._set_status(f"Done! {result.size_kb:,}KB, {mins}m{secs:02d}s via {result.provider} — {result.path}")

        history = self.query_one("#history-list", OptionList)
        fname = Path(result.path).name
        history.add_option(Option(f"{fname} ({result.size_kb:,}KB)", id=result.path))

        # Star prompt after 3rd generation
        if self._config.generation_count >= 3 and not self._config.star_dismissed:
            banner = self.query_one("#star-banner", Static)
            banner.update(f"Enjoying outloud? Star us on GitHub: {REPO_URL}")
            banner.add_class("visible")
            self._config.star_dismissed = True
            self._config.save()

    def _generation_error(self, error: str) -> None:
        pb = self.query_one("#progress", ProgressBar)
        pb.remove_class("visible")
        self.query_one("#btn-generate", Button).disabled = False
        self._set_status(f"Error: {error}")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "history-list":
            path = event.option.id
            if path and os.path.isfile(path):
                play_audio(path)
                self._set_status(f"Playing: {Path(path).name}")

    # --- STT Recording ---

    def action_toggle_record(self) -> None:
        if self._transcribing:
            self._set_status("Transcription in progress, please wait...")
            return
        if self._recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        if self._generating:
            self._set_status("Stop generation before recording.")
            return
        self._recording_path = tempfile.mktemp(suffix=".wav", prefix="outloud_rec_")
        try:
            self._recording_proc = record_from_mic_managed(self._recording_path)
        except Exception as e:
            self._set_status(f"Recording failed: {e}")
            return

        self._recording = True
        self._recording_start = time.monotonic()

        btn = self.query_one("#btn-record", Button)
        btn.label = "Stop"
        btn.add_class("recording")
        self.query_one("#btn-generate", Button).disabled = True

        timer_label = self.query_one("#recording-timer", Static)
        timer_label.update("● 00:00")
        timer_label.add_class("visible")

        self._recording_timer = self.set_interval(0.5, self._update_recording_timer)
        self.sub_title = "Recording..."
        self._set_status("Recording... Press Ctrl+R or Stop to finish.")

    def _update_recording_timer(self) -> None:
        elapsed = time.monotonic() - self._recording_start
        mins, secs = divmod(int(elapsed), 60)
        # Pulsing dot: alternate between ● and ○ every 0.5s
        dot = "●" if int(elapsed * 2) % 2 == 0 else "○"
        self.query_one("#recording-timer", Static).update(f"{dot} {mins:02d}:{secs:02d}")

    def _stop_recording(self) -> None:
        if self._recording_timer:
            self._recording_timer.stop()
            self._recording_timer = None

        if self._recording_proc:
            self._recording_proc.terminate()
            try:
                self._recording_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._recording_proc.kill()
                self._recording_proc.wait()
            self._recording_proc = None

        self._recording = False

        btn = self.query_one("#btn-record", Button)
        btn.label = "Record"
        btn.remove_class("recording")

        timer_label = self.query_one("#recording-timer", Static)
        timer_label.remove_class("visible")

        if not validate_recording(self._recording_path):
            self._set_status("Recording too short or empty. Try again.")
            self.query_one("#btn-generate", Button).disabled = False
            self.sub_title = "Text-to-Speech"
            return

        elapsed = time.monotonic() - self._recording_start
        self._set_status(f"Recorded {int(elapsed)}s. Transcribing...")
        self.sub_title = "Transcribing..."
        self._do_transcribe()

    @work(thread=True)
    def _do_transcribe(self) -> None:
        import asyncio

        self._transcribing = True
        self.call_from_thread(self._show_transcribe_progress)

        stt_provider = self.app.query_one("#stt-provider-select", Select).value
        path = self._recording_path
        language = self._config.stt_language or None

        try:
            if stt_provider == "openai":
                loop = asyncio.new_event_loop()
                try:
                    result = loop.run_until_complete(transcribe_openai(path, language=language))
                finally:
                    loop.close()
            else:
                result = transcribe_mlx(
                    path,
                    model=self._config.whisper_model,
                    language=language,
                )
            self.call_from_thread(self._transcription_done, result)
        except Exception as e:
            self.call_from_thread(self._transcription_error, str(e))
        finally:
            self._transcribing = False
            # Clean up temp recording
            try:
                Path(path).unlink(missing_ok=True)
            except Exception:
                pass

    def _show_transcribe_progress(self) -> None:
        pb = self.query_one("#progress", ProgressBar)
        pb.add_class("visible")
        pb.update(progress=50, total=100)

    def _transcription_done(self, result) -> None:
        pb = self.query_one("#progress", ProgressBar)
        pb.remove_class("visible")
        self.query_one("#btn-generate", Button).disabled = False

        ta = self.query_one("#text-input", TextArea)
        existing = ta.text.strip()
        if existing:
            ta.load_text(existing + "\n\n" + result.text)
        else:
            ta.load_text(result.text)

        mins, secs = divmod(result.duration_s, 60)
        self._set_status(
            f"Transcribed {int(mins)}m{int(secs):02d}s audio via {result.provider} — "
            f"{len(result.text)} chars, {len(result.segments)} segments"
        )
        self.sub_title = "Text-to-Speech"
        self._update_cost()

    def _transcription_error(self, error: str) -> None:
        pb = self.query_one("#progress", ProgressBar)
        pb.remove_class("visible")
        self.query_one("#btn-generate", Button).disabled = False
        self._set_status(f"Transcription error: {error}")
        self.sub_title = "Text-to-Speech"

    def action_play_last(self) -> None:
        if self._history:
            last = self._history[-1]
            if os.path.isfile(last.path):
                play_audio(last.path)
                self._set_status(f"Playing: {Path(last.path).name}")
        else:
            self._set_status("No audio generated yet.")

    def action_open_downloads(self) -> None:
        downloads = os.path.expanduser(self._config.output_dir)
        if sys.platform == "darwin":
            subprocess.Popen(["open", downloads])
        elif sys.platform == "win32":
            subprocess.Popen(["explorer", downloads])
        else:
            subprocess.Popen(["xdg-open", downloads])

    def action_clear_text(self) -> None:
        ta = self.query_one("#text-input", TextArea)
        ta.clear()
        ta.focus()
        self._set_status("Cleared.")
        self._update_cost()


def run_tui(initial_text: str = ""):
    app = OutloudApp(initial_text=initial_text)
    app.run()
