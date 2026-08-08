from __future__ import annotations

import contextlib
import itertools
import logging
import sys
import time
from collections.abc import Generator
from threading import Event, Thread
from typing import IO, Any, Final, Protocol, Self

from pip._vendor.rich.console import (
    Console,
    ConsoleOptions,
    RenderableType,
    RenderResult,
)
from pip._vendor.rich.live import Live
from pip._vendor.rich.measure import Measurement
from pip._vendor.rich.text import Text

from pip._internal.utils.compat import WINDOWS
from pip._internal.utils.logging import get_console, get_indentation

logger = logging.getLogger(__name__)

SPINNER_CHARS: Final = r"-\|/"
SPINS_PER_SECOND: Final = 8
NONINTERACTIVE_SPINNER_INTERVAL: Final = 60


class SpinnerInterface:
    def spin(self) -> None:
        raise NotImplementedError()

    def finish(self, final_status: str) -> None:
        raise NotImplementedError()


class InteractiveSpinner(SpinnerInterface):
    def __init__(
        self,
        message: str,
        file: IO[str] | None = None,
        spin_chars: str = SPINNER_CHARS,
        # Empirically, 8 updates/second looks nice
        min_update_interval_seconds: float = 1 / SPINS_PER_SECOND,
    ):
        self._message = message
        if file is None:
            file = sys.stdout
        self._file = file
        self._rate_limiter = RateLimiter(min_update_interval_seconds)
        self._finished = False

        self._spin_cycle = itertools.cycle(spin_chars)

        self._file.write(" " * get_indentation() + self._message + " ... ")
        self._width = 0

    def _write(self, status: str) -> None:
        assert not self._finished
        # Erase what we wrote before by backspacing to the beginning, writing
        # spaces to overwrite the old text, and then backspacing again
        backup = "\b" * self._width
        self._file.write(backup + " " * self._width + backup)
        # Now we have a blank slate to add our status
        self._file.write(status)
        self._width = len(status)
        self._file.flush()
        self._rate_limiter.reset()

    def spin(self) -> None:
        if self._finished:
            return
        if not self._rate_limiter.ready():
            return
        self._write(next(self._spin_cycle))

    def finish(self, final_status: str) -> None:
        if self._finished:
            return
        self._write(final_status)
        self._file.write("\n")
        self._file.flush()
        self._finished = True


# Used for dumb terminals, non-interactive installs (no tty), etc.
# We still print updates occasionally (once every 60 seconds by default) to
# act as a keep-alive for systems like Travis-CI that take lack-of-output as
# an indication that a task has frozen.
class NonInteractiveSpinner(SpinnerInterface):
    def __init__(self, message: str, min_update_interval_seconds: float = 60.0) -> None:
        self._message = message
        self._finished = False
        self._rate_limiter = RateLimiter(min_update_interval_seconds)
        self._update("started")

    def _update(self, status: str) -> None:
        assert not self._finished
        self._rate_limiter.reset()
        logger.info("%s: %s", self._message, status)

    def spin(self) -> None:
        if self._finished:
            return
        if not self._rate_limiter.ready():
            return
        self._update("still running...")

    def finish(self, final_status: str) -> None:
        if self._finished:
            return
        self._update(f"finished with status '{final_status}'")
        self._finished = True


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


@contextlib.contextmanager
def open_spinner(message: str) -> Generator[SpinnerInterface, None, None]:
    # Interactive spinner goes directly to sys.stdout rather than being routed
    # through the logging system, but it acts like it has level INFO,
    # i.e. it's only displayed if we're at level INFO or better.
    # Non-interactive spinner goes through the logging system, so it is always
    # in sync with logging configuration.
    if sys.stdout.isatty() and logger.getEffectiveLevel() <= logging.INFO:
        spinner: SpinnerInterface = InteractiveSpinner(message)
    else:
        spinner = NonInteractiveSpinner(message)
    try:
        with hidden_cursor(sys.stdout):
            yield spinner
    except KeyboardInterrupt:
        spinner.finish("canceled")
        raise
    except Exception:
        spinner.finish("error")
        raise
    else:
        spinner.finish("done")


class _PipSpinnerInterface(Protocol):
    def __init__(self, label: str, console: Console) -> None: ...
    def __enter__(self) -> Self: ...
    def __exit__(self, *args: Any, **kwargs: Any) -> None: ...
    def finish(self, label: str) -> None: ...


class _PipRichSpinner(_PipSpinnerInterface):
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

    def __enter__(self) -> Self:
        self._live = Live(
            self, refresh_per_second=SPINS_PER_SECOND, console=self._console
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        self._live.__exit__(*args, **kwargs)

    def __rich_console__(
        self, console: Console, options: ConsoleOptions
    ) -> RenderResult:
        yield self.render()

    def __rich_measure__(
        self, console: Console, options: ConsoleOptions
    ) -> Measurement:
        text = self.render()
        return Measurement.get(console, options, text)

    def render(self) -> RenderableType:
        if not self._finished:
            self._spinner_text = next(self._spin_cycle)

        return Text.assemble(self._indent, self.label, " ... ", self._spinner_text)

    def finish(self, status: str) -> None:
        """Stop spinning and set a final status message."""
        self._spinner_text = status
        self._finished = True


class _PipNonInteractiveRichSpinner(_PipSpinnerInterface):
    """
    Used for dumb terminals, non-interactive installs (no tty), etc.
    We still print updates occasionally (once every 60 seconds by default) to
    act as a keep-alive for systems like Travis-CI that take lack-of-output as
    an indication that a task has frozen.
    """

    def __init__(self, label: str, console: Console) -> None:
        self._label = label
        self._console = console
        self._final_status = "unknown"
        self._indent = get_indentation() * " "
        self._finish_event = Event()

    def __enter__(self) -> Self:
        self._print_line("started")
        self._thread = Thread(target=self._report_progress)
        self._thread.start()
        return self

    def __exit__(self, *args: Any, **kwargs: Any) -> None:
        # Normally this finish() is called before leaving the spinner context,
        # but an unhandled exception may break this, so defensively tell the
        # thread to stop (to avoid a possible hang).
        self._finish_event.set()
        self._thread.join()

    def _print_line(self, message: str) -> None:
        line = Text(f"{self._indent}{self._label}: {message}")
        self._console.print(line)

    def _report_progress(self) -> None:
        while not self._finish_event.wait(NONINTERACTIVE_SPINNER_INTERVAL):
            self._print_line("still running ...")

        # Finish event was set, we're done here.
        self._print_line(f"finished with status '{self._final_status}'")

    def finish(self, status: str) -> None:
        self._final_status = status
        self._finish_event.set()


@contextlib.contextmanager
def open_rich_spinner(label: str, console: Console | None = None) -> Generator[None]:
    if not logger.isEnabledFor(logging.INFO):
        # Don't show spinner if --quiet is given.
        yield
        return

    console = console or get_console()
    if sys.stdout.isatty():
        spinner: _PipSpinnerInterface = _PipRichSpinner(label, console)
    else:
        spinner = _PipNonInteractiveRichSpinner(label, console)
    with spinner:
        try:
            yield
        except KeyboardInterrupt:
            spinner.finish("canceled")
            raise
        except Exception:
            spinner.finish("error")
            raise
        else:
            spinner.finish("done")


HIDE_CURSOR = "\x1b[?25l"
SHOW_CURSOR = "\x1b[?25h"


@contextlib.contextmanager
def hidden_cursor(file: IO[str]) -> Generator[None, None, None]:
    # The Windows terminal does not support the hide/show cursor ANSI codes,
    # even via colorama. So don't even try.
    if WINDOWS:
        yield
    # We don't want to clutter the output with control characters if we're
    # writing to a file, or if the user is running with --quiet.
    # See https://github.com/pypa/pip/issues/3418
    elif not file.isatty() or logger.getEffectiveLevel() > logging.INFO:
        yield
    else:
        file.write(HIDE_CURSOR)
        try:
            yield
        finally:
            file.write(SHOW_CURSOR)
