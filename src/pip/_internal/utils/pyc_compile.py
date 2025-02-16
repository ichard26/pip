# -------------------------------------------------------------------------- #
# NOTE: Importing from pip's internals or vendored modules should be AVOIDED
#       so this module remains fast to import, minimizing the overhead of
#       spawning a new bytecode compiler subprocess.
# -------------------------------------------------------------------------- #

import compileall
import importlib
import os
import sys
import warnings
from contextlib import redirect_stdout
from io import StringIO
from typing import (
    TYPE_CHECKING,
    Iterable,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    TextIO,
    Union,
)

if TYPE_CHECKING:
    from pip._vendor.typing_extensions import Self

StartMethod = Literal["spawn", "forkserver", "fork"]
WorkerSetting = Union[int, Literal["auto"], Literal["none"]]

DEFAULT_START_METHOD: StartMethod = None
WORKER_LIMIT = 8


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
    py_path: str
    pyc_path: str
    is_success: bool
    compile_output: str


def _compile_single(py_path: str) -> CompileResult:
    stdout = StreamWrapper.from_stream(sys.stdout)
    with warnings.catch_warnings(), redirect_stdout(stdout):
        warnings.filterwarnings("ignore")
        is_success = compileall.compile_file(py_path, force=True, quiet=True)
    pyc_path = importlib.util.cache_from_source(py_path)
    # XXX: compile_file() should return a bool (typeshed bug?)
    return CompileResult(py_path, pyc_path, bool(is_success), stdout.getvalue())


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

    def __init__(
        self, workers: int, start_method: Optional[StartMethod] = None
    ) -> None:
        import multiprocessing

        self.workers = workers
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
            ctx = multiprocessing.get_context(start_method or DEFAULT_START_METHOD)
            self.pool = ctx.Pool(workers)
        finally:
            sys.modules["__main__"] = original_main

    def __call__(self, paths: Iterable[str]) -> Iterable[CompileResult]:
        yield from self.pool.map(_compile_single, paths)

    def __exit__(self, *args: object) -> None:
        self.pool.close()


def create_bytecode_compiler(
    preferred_workers: WorkerSetting = "auto",
) -> BytecodeCompiler:
    """
    TODO: explain this logic
    """
    import logging

    from pip._internal.utils.misc import strtobool

    if strtobool(os.getenv("_PIP_SERIAL", "0")):
        preferred_workers = "none"

    logger = logging.getLogger(__name__)

    try:
        # New in Python 3.13.
        cpus: Optional[int] = os.process_cpu_count()  # type: ignore
    except AttributeError:
        # Poor man's fallback. We won't respect PYTHON_CPU_COUNT, but the envvar
        # was only added in Python 3.13 anyway.
        try:
            cpus = len(os.sched_getaffinity(0))  # exists on unix (usually)
        except AttributeError:
            cpus = os.cpu_count()

    logger.debug("Detected CPU count: %s", cpus)
    _log_note = {
        "auto": f"(will use up to {WORKER_LIMIT})",
        "none": "(parallelization is disabled)",
    }.get(
        preferred_workers, ""  # type: ignore
    )
    logger.debug("Configured worker count: %s %s", preferred_workers, _log_note)

    # Case 1: Only one worker would be used, parallelization is thus pointless.
    if preferred_workers == "none" or (cpus == 1 or cpus is None):
        logger.debug("Bytecode will be compiled serially")
        return SerialCompiler()

    # Case 2: Attempt to initialize a parallelized compiler.
    if preferred_workers == "auto":
        workers = min(cpus, WORKER_LIMIT)
    else:
        workers = preferred_workers
    try:
        compiler = ParallelCompiler(workers)
        logger.debug("Bytecode will be compiled using %s workers", workers)
        return compiler
    except (ImportError, NotImplementedError, OSError) as e:
        # Case 3: multiprocessing is broken, fall back to serial compilation.
        logger.debug(
            "Bytecode will be compiled serially (multiprocessing is unavailable)",
            exc_info=e,
        )
        return SerialCompiler()
