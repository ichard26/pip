from __future__ import annotations

import contextlib
import itertools
import logging
import sys
import time
from collections.abc import Generator
from threading import Event, Thread
from typing import Final, Protocol

from pip._vendor.rich.console import Console
from pip._vendor.rich.live import Live
from pip._vendor.rich.text import Text

from pip._internal.utils.logging import get_console, get_indentation

logger = logging.getLogger(__name__)

SPINNER_CHARS: Final = r"-\|/"
SPINS_PER_SECOND: Final = 8
NONINTERACTIVE_SPINNER_INTERVAL: Final = 60


class SpinnerInterface(Protocol):
    def start(self) -> None: ...
    def finish(self, label: str) -> None: ...


class RateLimiter:
    def __init__(self, min_update_interval_seconds: float) -> None:
        self._min_update_interval_seconds = min_update_interval_seconds
        self._last_update: float = 0

    def ready(self) -> bool:
        now = time.time()
        delta = now - self._last_update
        return delta >= self._min_update_interval_seconds

    def reset(self) -> None:
        self._last_update = time.time()


class _NoopSpinner(SpinnerInterface):
    def start(self) -> None:
        pass

    def finish(self, label: str) -> None:
        pass


class _RichSpinner(SpinnerInterface):
    """
    Custom rich spinner that matches the style of the legacy spinners.

    (*) Updates will be handled in a background thread by a rich live panel
        which will call render() automatically at the appropriate time.
    """

    def __init__(self, label: str, console: Console) -> None:
        self.label = label
        self._console = console
        self._spin_cycle = itertools.cycle(SPINNER_CHARS)
        self._spinner_text = ""
        self._finished = False
        self._indent = get_indentation() * " "
        self._live: Live | None = None

    def __rich__(self) -> Text:
        if not self._finished:
            self._spinner_text = next(self._spin_cycle)

        return Text.assemble(self._indent, self.label, " ... ", self._spinner_text)

    def start(self) -> None:
        self._live = Live(
            self, refresh_per_second=SPINS_PER_SECOND, console=self._console
        )
        self._live.start(refresh=True)

    def finish(self, status: str) -> None:
        """Stop spinning and set a final status message."""
        if not self._finished:
            self._finished = True
            if self._live is not None:
                self._spinner_text = status
                self._live.stop()
            else:
                final_line = Text.assemble(self._indent, self.label, " ... ", status)
                self._console.print(final_line)


class _NonInteractiveSpinner(SpinnerInterface):
    """
    Used for dumb terminals, non-interactive installs (no tty), etc.
    We still print updates occasionally (once every 60 seconds by default) to
    act as a keep-alive for systems like Travis-CI that take lack-of-output as
    an indication that a task has frozen.
    """

    def __init__(self, label: str, console: Console) -> None:
        self._label = label
        self._console = console
        self._indent = get_indentation() * " "
        self._thread: Thread | None = None
        self._finish_event = Event()
        self._print_line("started")

    def _print_line(self, message: str) -> None:
        line = Text(f"{self._indent}{self._label}: {message}")
        self._console.print(line)

    def _report_progress(self) -> None:
        while not self._finish_event.wait(NONINTERACTIVE_SPINNER_INTERVAL):
            self._print_line("still running ...")

    def start(self) -> None:
        self._thread = Thread(target=self._report_progress)
        self._thread.start()

    def finish(self, status: str) -> None:
        if not self._finish_event.is_set():
            self._finish_event.set()
            if self._thread is not None:
                self._thread.join()
            self._print_line(f"finished with status '{status}'")


@contextlib.contextmanager
def open_spinner(
    label: str, console: Console | None = None, *, autostart: bool = True
) -> Generator[SpinnerInterface]:
    if not logger.isEnabledFor(logging.INFO):
        # Don't show spinner if --quiet is given.
        yield _NoopSpinner()
        return

    console = console or get_console()
    if sys.stdout.isatty():
        spinner: SpinnerInterface = _RichSpinner(label, console)
    else:
        spinner = _NonInteractiveSpinner(label, console)
    if autostart:
        spinner.start()
    try:
        yield spinner
    except KeyboardInterrupt:
        spinner.finish("canceled")
        raise
    except Exception:
        spinner.finish("error")
        raise
    finally:
        spinner.finish("done")
