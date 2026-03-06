"""outloud TUI — Beautiful terminal interface for multi-provider TTS."""

from __future__ import annotations

import os
import subprocess
import sys
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
    """

    TITLE = "outloud"
    SUB_TITLE = "Text-to-Speech"

    BINDINGS = [
        Binding("ctrl+g", "generate", "Generate", show=True),
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
                yield Button("Play Last", variant="primary", id="btn-play")
                yield Button("Open Folder", variant="warning", id="btn-open")
                yield Button("Clear", variant="error", id="btn-clear")
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
