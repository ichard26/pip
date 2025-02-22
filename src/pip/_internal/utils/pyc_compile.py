# -------------------------------------------------------------------------- #
# NOTE: Importing from pip's internals or vendored modules should be AVOIDED
#       so this module remains fast to import, minimizing the overhead of
#       spawning a new bytecode compiler worker.
# -------------------------------------------------------------------------- #

import compileall
import importlib
import os
import sys
import warnings
from contextlib import contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Iterable,
    Iterator,
    Literal,
    NamedTuple,
    Optional,
    Protocol,
    TextIO,
    Union,
)

if TYPE_CHECKING:
    from pip._vendor.typing_extensions import Self

WorkerSetting = Union[int, Literal["auto"]]

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


@contextmanager
def _patch_main_module_hack() -> Iterator[None]:
    """Temporarily replace __main__ to reduce the subprocess startup overhead.

    multiprocessing imports the main module while initializing subprocesses
    so the global state is retained in the subprocesses. Unfortunately, when pip
    is run from a console script wrapper, the wrapper unconditionally imports
    pip._internal.cli.main and everything else it requires. This is *slow*.

    This module is wholly independent(*) from the rest of the codebase, so we can
    avoid the costly re-import of pip by replacing sys.modules["__main__"] with
    any random module that does functionally nothing (e.g., pip.__init__).

    (*) This module's entrypoint does import from pip. This is fine as it's only
        called in the main process where the imports have already executed.
    """
    original_main = sys.modules["__main__"]
    sys.modules["__main__"] = sys.modules["pip"]
    try:
        yield
    finally:
        sys.modules["__main__"] = original_main


class CompileResult(NamedTuple):
    py_path: str
    pyc_path: str
    is_success: bool
    compile_output: str


def _compile_single(py_path: Union[str, Path]) -> CompileResult:
    # For some reason, compile_file() will silently do nothing and return True
    # even if the source file does not exist.
    if not os.path.exists(py_path):
        raise FileNotFoundError(f"Python file {py_path!s} does not exist!")

    stdout = StreamWrapper.from_stream(sys.stdout)
    with warnings.catch_warnings(), redirect_stdout(stdout):
        warnings.filterwarnings("ignore")
        success = compileall.compile_file(py_path, force=True, quiet=True)
    pyc_path = importlib.util.cache_from_source(py_path)  # type: ignore[arg-type]
    return CompileResult(
        str(py_path), pyc_path, success, stdout.getvalue()  # type: ignore[arg-type]
    )


class BytecodeCompiler(Protocol):
    """Abstraction for compiling Python modules into bytecode in bulk."""

    def __call__(self, paths: Iterable[str]) -> Iterable[CompileResult]: ...

    def __enter__(self) -> "Self":
        return self

    def __exit__(self, *args: object) -> None:
        return


class SerialCompiler(BytecodeCompiler):
    """Compile a set of Python modules one by one in-process."""

    def __call__(self, paths: Iterable[Union[str, Path]]) -> Iterable[CompileResult]:
        for p in paths:
            yield _compile_single(p)


class ParallelCompiler(BytecodeCompiler):
    """Compile a set of Python modules using a pool of subprocesses."""

    def __init__(self, workers: int) -> None:
        from concurrent import futures

        # The concurrent executors are smart and will only spin up new workers
        # if there are no more idle workers available for a task. Thus we don't
        # need to adjust the worker count for tiny bulk compile jobs that
        # wouldn't be able to fully utilize all of the workers. (If the fork
        # multiprocessing method is used, this is NOT true, but fork is
        # sufficiently fast that it doesn't really matter).
        if sys.version_info >= (3, 14):
            self.pool = futures.InterpreterPoolExecutor(workers)
        else:
            self.pool = futures.ProcessPoolExecutor(workers)
        self.workers = workers

    def __call__(self, paths: Iterable[Union[str, Path]]) -> Iterable[CompileResult]:
        # The concurrent executors add new workers on-demand, thus this patching
        # needs to be active until all paths have been processed.
        with _patch_main_module_hack():
            yield from self.pool.map(_compile_single, paths)

    def __exit__(self, *args: object) -> None:
        # It's pointless to block on pool finalization, let it occur in background.
        self.pool.shutdown(wait=False)


def create_bytecode_compiler(max_workers: WorkerSetting = "auto") -> BytecodeCompiler:
    """Return a bytecode compiler appropriate for the workload and platform.

    Parallelization will only be used if:
      - There is "enough" code to be compiled to offset the worker startup overhead
      - There are 2 or more CPUs available
      - The maximum # of workers permitted is at least 2

    A maximum worker count of "auto" will use the number of CPUs available to the
    process or system, up to a hard-coded limit (to avoid resource exhaustion).
    """
    import logging

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

    logger = logging.getLogger(__name__)
    logger.debug("Detected CPU count: %s", cpus)
    logger.debug("Configured worker count: %s", max_workers)

    # Case 1: Parallelization is disabled or pointless (there's only one CPU).
    if max_workers == 1 or cpus == 1 or cpus is None:
        logger.debug("Bytecode will be compiled serially")
        return SerialCompiler()

    # Case 2: Attempt to initialize a parallelized compiler.
    workers = min(cpus, WORKER_LIMIT) if max_workers == "auto" else max_workers
    try:
        compiler = ParallelCompiler(workers)
        logger.debug("Bytecode will be compiled using at most %s workers", workers)
        return compiler
    except (ImportError, NotImplementedError, OSError) as e:
        # Case 3: multiprocessing is broken, fall back to serial compilation.
        logger.debug("Err! Falling back to serial bytecode compilation", exc_info=e)
        return SerialCompiler()
