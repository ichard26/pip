# -------------------------------------------------------------------------- #
# NOTE: Importing from pip's internals or vendored modules should be AVOIDED
#       so this module remains fast to import, minimizing the overhead of
#       spawning a new bytecode compiler subprocess.
# -------------------------------------------------------------------------- #

import compileall
import importlib
import multiprocessing
import os
import sys
import warnings
from contextlib import redirect_stdout
from io import StringIO
from typing import TYPE_CHECKING, Iterable, NamedTuple, Protocol, TextIO

if TYPE_CHECKING:
    from pip._vendor.typing_extensions import Self

MAX_WORKERS = 4


# TODO: use StringIO directly once Python 3.8 is dropped
class StreamWrapper(StringIO):
    orig_stream: TextIO

    @classmethod
    def from_stream(cls, orig_stream: TextIO) -> "StreamWrapper":
        ret = cls()
        ret.orig_stream = orig_stream
        return ret

    # compileall.compile_dir() needs stdout.encoding to print to stdout
    # type ignore is because TextIOBase.encoding is writeable
    @property
    def encoding(self) -> str:  # type: ignore
        return self.orig_stream.encoding


class CompileResult(NamedTuple):
    source_path: str
    pyc_path: str
    is_success: bool
    log: str


def _compile_single(path: str) -> CompileResult:
    stdout = StreamWrapper.from_stream(sys.stdout)
    with warnings.catch_warnings(), redirect_stdout(stdout):
        warnings.filterwarnings("ignore")
        is_success = compileall.compile_file(path, force=True, quiet=True)
    pyc_path = importlib.util.cache_from_source(path)
    # XXX: compile_file() should return a bool (typeshed bug?)
    return CompileResult(path, pyc_path, bool(is_success), stdout.getvalue())


class BytecodeCompiler(Protocol):
    """Abstraction for compiling Python modules into bytecode in bulk."""

    def __call__(self, paths: Iterable[str]) -> Iterable[CompileResult]: ...

    def __enter__(self) -> "Self":
        return self

    def __exit__(self, *args: object) -> None:
        return


class SerialCompiler(BytecodeCompiler):
    """Compile a set of Python modules one by one in-process."""

    def __call__(self, paths: Iterable[str]) -> Iterable[CompileResult]:
        for p in paths:
            yield _compile_single(p)


class ParallelCompiler(BytecodeCompiler):
    """Compile a set of Python modules using a pool of subprocesses."""

    def __init__(self, workers: int = 0) -> None:
        if workers == 0:
            try:
                # New in Python 3.13.
                cpus = os.process_cpu_count()  # type: ignore[attr-defined]
            except AttributeError:
                cpus = os.cpu_count()
            workers = min(MAX_WORKERS, cpus or 1)
        # HACK: multiprocessing imports the main module while initializing subprocesses
        # so the global state is retained in the subprocesses. Unfortunately, when pip
        # is run from a console script wrapper, the wrapper unconditionally imports
        # pip._internal.cli.main and everything else it requires. This is *slow*.
        #
        # This module is wholly independent from the rest of the codebase, so we can
        # avoid the costly re-import of pip by replacing sys.modules["__main__"] with
        # any random module that does functionally nothing (e.g., pip.__init__).
        original_main = sys.modules["__main__"]
        sys.modules["__main__"] = sys.modules["pip"]
        try:
            # ctx = multiprocessing.get_context("spawn")
            # self.pool = ctx.Pool(workers)
            self.pool = multiprocessing.Pool(workers)
        finally:
            sys.modules["__main__"] = original_main

    def __call__(self, paths: Iterable[str]) -> Iterable[CompileResult]:
        yield from self.pool.map(_compile_single, paths)

    def __exit__(self, *args: object) -> None:
        self.pool.close()
