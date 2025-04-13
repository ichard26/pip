# -------------------------------------------------------------------------- #
# NOTE: Importing from pip's internals or vendored modules should be AVOIDED
#       to minimize the overhead of spawning a new bytecode compiler worker.
# -------------------------------------------------------------------------- #

import compileall
import importlib
import os
import sys
import warnings
from collections.abc import Callable, Iterable, Iterator
from contextlib import AbstractContextManager, contextmanager, redirect_stdout
from io import StringIO
from pathlib import Path
from typing import Literal, NamedTuple, Optional, Protocol, Union

WorkerSetting = Union[int, Literal["auto"]]

CODE_SIZE_THRESHOLD = 1000 * 1000  # 1 MB of .py code
WORKER_LIMIT = 8


@contextmanager
def _patch_main_module_hack() -> Iterator[None]:
    """Temporarily replace __main__ to reduce worker startup overhead.

    concurrent.futures imports the __main__ module while initializing new
    workers (so global state is persisted). Unfortunately, when pip is
    invoked via a console script, the wrapper unconditionally imports
    pip._internal.cli.main and its dependencies. This is *slow*.

    The compilation code does not depend on this, so avoid the costly
    re-import of pip by replacing __main__ with a lightweight module.
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
    # compile_file() returns True silently even if the source file is nonexistent.
    if not os.path.exists(py_path):
        raise FileNotFoundError(f"Python file '{py_path!s}' does not exist")

    with warnings.catch_warnings(), redirect_stdout(StringIO()) as stdout:
        warnings.filterwarnings("ignore")
        success = compileall.compile_file(py_path, force=True, quiet=True)
    pyc_path = importlib.util.cache_from_source(py_path)
    return CompileResult(str(py_path), pyc_path, success, stdout.getvalue())


class BytecodeCompiler(Protocol, AbstractContextManager):  # type: ignore[type-arg]
    """Abstraction for compiling Python modules into bytecode in bulk."""

    def __call__(self, paths: Iterable[str]) -> Iterable[CompileResult]: ...
    def __exit__(self, *exe: object) -> None:
        return None


class SerialCompiler(BytecodeCompiler):
    """Compile a set of Python modules one by one in-process."""

    def __call__(self, paths: Iterable[Union[str, Path]]) -> Iterable[CompileResult]:
        for p in paths:
            yield _compile_single(p)


class ParallelCompiler(BytecodeCompiler):
    """Compile a set of Python modules using a pool of workers."""

    def __init__(self, workers: int) -> None:
        # The concurrent.futures pools cannot be used as they start workers
        # on-demand.
        import multiprocessing

        with _patch_main_module_hack():
            self.pool = multiprocessing.Pool(workers)
        self.workers = workers

    def __call__(self, paths: Iterable[Union[str, Path]]) -> Iterable[CompileResult]:
        yield from self.pool.map(_compile_single, paths)

    def __exit__(self, *args: object) -> None:
        self.pool.close()


def create_bytecode_compiler(
    max_workers: WorkerSetting = "auto",
    code_size_check: Optional[Callable[[int], bool]] = None,
) -> BytecodeCompiler:
    """Return a bytecode compiler appropriate for the workload and platform.

    A maximum worker count of "auto" will use the number of CPUs available to the
    process or system, up to a hard-coded limit (to avoid resource exhaustion).

    code_size_check is a callable that receives the code size threshold (in
    bytes) for parallelization and returns whether it will be surpassed or not.
    """
    import logging

    try:
        # New in Python 3.13.
        cpus: Optional[int] = os.process_cpu_count()  # type: ignore
    except AttributeError:
        try:
            # Unix-only alternative for process_cpu_count()
            cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            cpus = os.cpu_count()

    logger = logging.getLogger(__name__)
    logger.debug("Detected CPU count: %s", cpus)
    logger.debug("Configured worker count: %s", max_workers)

    # Case 1: Parallelization is disabled or pointless (there's only one CPU).
    if max_workers == 1 or cpus == 1 or cpus is None:
        logger.debug("Bytecode will be compiled serially")
        return SerialCompiler()

    # Case 2: There isn't enough code for parallelization to be worth it.
    if code_size_check is not None and not code_size_check(CODE_SIZE_THRESHOLD):
        logger.debug("Bytecode will be compiled serially (not enough .py code)")
        return SerialCompiler()

    # Case 3: Attempt to initialize a parallelized compiler.
    workers = min(cpus, WORKER_LIMIT) if max_workers == "auto" else max_workers
    try:
        logger.debug("Bytecode will be compiled using at most %s workers", workers)
        return ParallelCompiler(workers)
    except (ImportError, NotImplementedError, OSError) as e:
        # Case 4: multiprocessing is broken, fall back to serial compilation.
        logger.debug("Err! Falling back to serial bytecode compilation", exc_info=e)
        return SerialCompiler()
