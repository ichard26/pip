import contextlib
import compileall
import importlib
import multiprocessing
import sys
import warnings
from io import TextIOBase
from pathlib import Path
from typing import Callable, Protocol, Sequence, Tuple


class BytecodeCompiler(Protocol):
    def __call__(
        self, paths: Sequence[str], *, stdout: TextIOBase = sys.stdout
    ) -> Sequence[str]: ...

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return


class SerialCompiler(BytecodeCompiler):
    def __call__(
        self, paths: Sequence[str], *, stdout: TextIOBase = sys.stdout
    ) -> Sequence[str]:
        pyc_paths = []

        with warnings.catch_warnings(), contextlib.redirect_stdout(stdout):
            warnings.filterwarnings("ignore")
            for path in paths:
                success = compileall.compile_file(path, force=True, quiet=True)
                if success:
                    pyc_paths.append(importlib.util.cache_from_source(path))

        return pyc_paths


class ParallelCompiler(BytecodeCompiler):
    def __init__(self, *, workers: int) -> None:
        self.pool = multiprocessing.Pool(workers)

    def __call__(
        self, paths: Sequence[str], *, stdout: TextIOBase = sys.stdout
    ) -> Sequence[str]:
        pyc_paths = []
        jobs = [(p,) for p in paths]

        for (path, is_success) in self.pool.starmap(self._compile_single, jobs):
            is_success = compileall.compile_file(path, force=True, quiet=True)
            if is_success:
                pyc_paths.append(importlib.util.cache_from_source(path))

        return pyc_paths

    def __exit__(self, *args):
        return self.pool.__exit__(*args)

    @staticmethod
    def _compile_single(path: str) -> Tuple[str, bool]:
        is_success = compileall.compile_file(path, force=True, quiet=True)
        return (path, is_success)
