"""readthis TUI — Beautiful terminal interface for Gemini TTS."""

import os
import subprocess
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
    Select,
    Static,
    TextArea,
)
from textual.widgets.option_list import Option

from . import engine


class VoicePicker(VerticalScroll):
    DEFAULT_CSS = """
    VoicePicker {
        width: 28;
        border: solid $primary-background;
        padding: 0 1;
    }
    VoicePicker > Label {
        text-style: bold;
        margin-bottom: 1;
        color: $text;
    }
    VoicePicker OptionList {
        height: 1fr;
    }
    """

    def compose(self) -> ComposeResult:
        yield Label("Voice")
        ol = OptionList(
            *[Option(v, id=v) for v in engine.VOICES],
            id="voice-list",
        )
        yield ol

    def on_mount(self) -> None:
        ol = self.query_one("#voice-list", OptionList)
        try:
            idx = engine.VOICES.index(engine.DEFAULT_VOICE)
            ol.highlighted = idx
        except ValueError:
            pass

    @property
    def selected_voice(self) -> str:
        ol = self.query_one("#voice-list", OptionList)
        if ol.highlighted is not None:
            return engine.VOICES[ol.highlighted]
        return engine.DEFAULT_VOICE


class ReadThisApp(App):
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

    #controls {
        height: auto;
        margin-bottom: 1;
        align-horizontal: left;
    }

    #controls Button {
        margin-right: 1;
    }

    #style-select {
        width: 40;
        margin-right: 1;
    }

    #status-bar {
        height: auto;
        margin-top: 1;
    }

    #status-text {
        margin-bottom: 1;
        color: $text-muted;
    }

    #progress {
        display: none;
    }

    #progress.visible {
        display: block;
    }

    #output-label {
        text-style: bold;
        margin-top: 1;
        color: $text;
    }

    #history {
        height: auto;
        max-height: 12;
        margin-top: 1;
    }

    #history OptionList {
        height: auto;
        max-height: 10;
    }

    .btn-generate {
        background: $success;
        color: $text;
    }

    .btn-play {
        background: $primary;
        color: $text;
    }

    .btn-open {
        background: $warning;
        color: $text;
    }

    .btn-clear {
        background: $error;
        color: $text;
    }
    """

    TITLE = "readthis"
    SUB_TITLE = "Gemini TTS"

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
        self._history: list[engine.TTSResult] = []
        self._generating = False
        self._chunks_done = 0
        self._chunks_total = 0

    def compose(self) -> ComposeResult:
        yield Header()
        yield VoicePicker()
        with Vertical(id="main-panel"):
            yield TextArea(
                self._initial_text,
                id="text-input",
                language=None,
                show_line_numbers=False,
            )
            with Horizontal(id="controls"):
                yield Select(
                    [(name, name) for name in engine.STYLE_PRESETS],
                    value=engine.DEFAULT_STYLE,
                    id="style-select",
                    allow_blank=False,
                )
                yield Button("Generate", variant="success", classes="btn-generate", id="btn-generate")
                yield Button("Play Last", variant="primary", classes="btn-play", id="btn-play")
                yield Button("Open Folder", variant="warning", classes="btn-open", id="btn-open")
                yield Button("Clear", variant="error", classes="btn-clear", id="btn-clear")
            with Vertical(id="status-bar"):
                yield Static("Ready. Paste text above and hit Generate (Ctrl+G).", id="status-text")
                yield ProgressBar(total=100, show_eta=False, id="progress")
            yield Label("History", id="output-label")
            yield OptionList(id="history")
        yield Footer()

    def on_mount(self) -> None:
        ta = self.query_one("#text-input", TextArea)
        ta.focus()

    def _set_status(self, msg: str) -> None:
        self.query_one("#status-text", Static).update(msg)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-generate":
            self.action_generate()
        elif event.button.id == "btn-play":
            self.action_play_last()
        elif event.button.id == "btn-open":
            self.action_open_downloads()
        elif event.button.id == "btn-clear":
            self.action_clear_text()

    def action_generate(self) -> None:
        if self._generating:
            self._set_status("Already generating...")
            return
        ta = self.query_one("#text-input", TextArea)
        text = ta.text.strip()
        if not text:
            self._set_status("Nothing to generate. Paste some text first.")
            return
        self._do_generate(text)

    @work(thread=True)
    def _do_generate(self, text: str) -> None:
        import asyncio

        self._generating = True
        voice = self.app.query_one(VoicePicker).selected_voice
        style_name = self.app.query_one("#style-select", Select).value
        style = engine.STYLE_PRESETS.get(style_name, "")

        chars = len(text)
        chunks = engine.chunk_text(text if not style else f"[{style}]\n\n{text}")
        self._chunks_total = len(chunks)
        self._chunks_done = 0

        self.call_from_thread(self._show_progress, chars, self._chunks_total)

        def on_progress(idx, total, success):
            self._chunks_done += 1
            pct = int(self._chunks_done / self._chunks_total * 100)
            status = "ok" if success else "FAILED"
            self.call_from_thread(self._update_progress, pct, idx + 1, total, status)

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                engine.generate(
                    text=text,
                    voice=voice,
                    style=style,
                    on_progress=on_progress,
                )
            )
        except Exception as e:
            self.call_from_thread(self._generation_error, str(e))
            return
        finally:
            loop.close()
            self._generating = False

        if result:
            self._history.append(result)
            self.call_from_thread(self._generation_done, result)
        else:
            self.call_from_thread(self._generation_error, "All chunks failed")

    def _show_progress(self, chars: int, chunks: int) -> None:
        pb = self.query_one("#progress", ProgressBar)
        pb.add_class("visible")
        pb.update(progress=0, total=100)
        self._set_status(f"Generating... {chars:,} chars, {chunks} chunk{'s' if chunks != 1 else ''}")
        self.query_one("#btn-generate", Button).disabled = True

    def _update_progress(self, pct: int, chunk_idx: int, total: int, status: str) -> None:
        pb = self.query_one("#progress", ProgressBar)
        pb.update(progress=pct, total=100)
        self._set_status(f"Chunk {chunk_idx}/{total} {status} ({pct}%)")

    def _generation_done(self, result: engine.TTSResult) -> None:
        pb = self.query_one("#progress", ProgressBar)
        pb.remove_class("visible")
        self.query_one("#btn-generate", Button).disabled = False

        mins, secs = divmod(result.duration_s, 60)
        self._set_status(f"Done! {result.size_kb:,}KB, {mins}m{secs:02d}s — {result.path}")

        history = self.query_one("#history", OptionList)
        fname = Path(result.path).name
        history.add_option(Option(f"{fname}  ({result.size_kb:,}KB, {mins}m{secs:02d}s)", id=result.path))

    def _generation_error(self, error: str) -> None:
        pb = self.query_one("#progress", ProgressBar)
        pb.remove_class("visible")
        self.query_one("#btn-generate", Button).disabled = False
        self._set_status(f"Error: {error}")

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "history":
            path = event.option.id
            if path and os.path.isfile(path):
                engine.play_audio(path)
                self._set_status(f"Playing: {Path(path).name}")

    def action_play_last(self) -> None:
        if self._history:
            last = self._history[-1]
            if os.path.isfile(last.path):
                engine.play_audio(last.path)
                self._set_status(f"Playing: {Path(last.path).name}")
        else:
            self._set_status("No audio generated yet.")

    def action_open_downloads(self) -> None:
        downloads = os.path.expanduser("~/Downloads")
        subprocess.Popen(["open", downloads])

    def action_clear_text(self) -> None:
        ta = self.query_one("#text-input", TextArea)
        ta.clear()
        ta.focus()
        self._set_status("Cleared.")


def run_tui(initial_text: str = ""):
    app = ReadThisApp(initial_text=initial_text)
    app.run()
