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
        self.pool = multiprocessing.Pool(workers)

    def __call__(self, paths: Iterable[str]) -> Iterable[CompileResult]:
        yield from self.pool.map(_compile_single, paths)

    def __exit__(self, *args: object) -> None:
        self.pool.close()
        self.pool.join()
