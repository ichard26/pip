import collections
import logging
import os
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Generator, List, Optional, Sequence, Tuple

from pip._internal.cli.progress_bars import get_install_progress_renderer
from pip._internal.utils.compile import ParallelCompiler, SerialCompiler
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import strtobool

from .req_file import parse_requirements
from .req_install import InstallRequirement
from .req_set import RequirementSet

__all__ = [
    "RequirementSet",
    "InstallRequirement",
    "parse_requirements",
    "install_given_reqs",
]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstallationResult:
    name: str


def _validate_requirements(
    requirements: List[InstallRequirement],
) -> Generator[Tuple[str, InstallRequirement], None, None]:
    for req in requirements:
        assert req.name, f"invalid to-be-installed requirement: {req}"
        yield req.name, req


def install_given_reqs(
    requirements: List[InstallRequirement],
    global_options: Sequence[str],
    root: Optional[str],
    home: Optional[str],
    prefix: Optional[str],
    warn_script_location: bool,
    use_user_site: bool,
    pycompile: bool,
    progress_bar: str,
) -> List[InstallationResult]:
    """
    Install everything in the given list.

    (to be called after having downloaded and unpacked the packages)
    """
    import time

    t0 = time.perf_counter()
    compiler = None
    if pycompile:
        if strtobool(os.getenv("PIP_SERIAL", "0")):
            compiler = SerialCompiler()
        else:
            workers = os.getenv("PIP_WORKERS") or min(4, os.cpu_count())
            compiler = ParallelCompiler(workers=int(workers))

    to_install = collections.OrderedDict(_validate_requirements(requirements))

    if to_install:
        logger.info(
            "Installing collected packages: %s",
            ", ".join(to_install.keys()),
        )

    installed = []

    show_progress = logger.isEnabledFor(logging.INFO) and len(to_install) > 1

    items = iter(to_install.values())
    if show_progress:
        renderer = get_install_progress_renderer(
            bar_type=progress_bar, total=len(to_install)
        )
        items = renderer(items)

    with indent_log(), compiler or nullcontext():
        for requirement in items:
            req_name = requirement.name
            assert req_name is not None
            if requirement.should_reinstall:
                logger.info("Attempting uninstall: %s", req_name)
                with indent_log():
                    uninstalled_pathset = requirement.uninstall(auto_confirm=True)
            else:
                uninstalled_pathset = None

            try:
                t1 = time.perf_counter()
                requirement.install(
                    global_options,
                    root=root,
                    home=home,
                    prefix=prefix,
                    warn_script_location=warn_script_location,
                    use_user_site=use_user_site,
                    compiler=compiler,
                )
            except Exception:
                # if install did not succeed, rollback previous uninstall
                if uninstalled_pathset and not requirement.install_succeeded:
                    uninstalled_pathset.rollback()
                raise
            else:
                if uninstalled_pathset and requirement.install_succeeded:
                    uninstalled_pathset.commit()
                elapsed = time.perf_counter() - t1
                # logger.info(rf"{req_name}: {elapsed:.3f}s")

            installed.append(InstallationResult(req_name))

    elapsed = time.perf_counter() - t0
    logger.info(f"[cyan bold]install total: {elapsed:.3f}s", extra={"markup": True})
    return installed
