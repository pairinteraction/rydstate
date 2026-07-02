from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time
from pathlib import Path

from rydstate.generate_database.generate_database import create_tables_for_misc, create_tables_for_one_species

logger = logging.getLogger(__name__)


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    """Entry point for the generate_database script."""
    parser = argparse.ArgumentParser(
        description="Generate a database, containing energies and matrix elements, for a given species.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=("Example:\n  generate_database Rb --log-level INFO\n"),
    )
    parser.add_argument("species", help="The species name to generate the database for.")
    parser.add_argument(
        "--n-min",
        default=None,
        type=int,
        help="The minimal principal quantum number n for the states to be included in the database. "
        "Default 1 will start with the ground state configuration of the specific species (e.g. n=5 for Rb).",
    )
    parser.add_argument(
        "--n-max",
        default=None,
        type=int,
        help="The maximum principal quantum number n for the states to be included in the database.",
    )
    parser.add_argument(
        "--nu-min",
        default=None,
        type=int,
        help="The minimal effective principal quantum number nu for the states to be included in the database. "
        "Default 0 will include all low lying states.",
    )
    parser.add_argument(
        "--nu-max",
        default=None,
        type=int,
        help="The maximum effective principal quantum number nu for the states to be included in the database.",
    )
    parser.add_argument(
        "--max-delta-nu",
        default=float("inf"),
        type=float,
        help="The maximum difference in effective principal quantum number nu for matrix elements to be calculated.",
    )
    parser.add_argument(
        "--all-nu-up-to",
        default=float("inf"),
        type=float,
        help="Calculate all matrix elements where at least one state has effective principal quantum number nu "
        "smaller than or equal to this value.",
    )
    parser.add_argument(
        "--f-max",
        default=None,
        type=float,
        help="The maximum angular momentum quantum number f for misc database tables.",
    )
    parser.add_argument(
        "--directory",
        default=None,
        type=str,
        help="The directory where the database will be saved.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--warnings-as-exceptions",
        action="store_true",
        help="Treat warnings in rydstate as exceptions.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the species folder if it exists and create a new one.",
    )

    args = parser.parse_args()
    if args.species == "misc":
        if (
            args.n_min is not None
            or args.n_max is not None
            or args.nu_min is not None
            or args.nu_max is not None
            or args.max_delta_nu != float("inf")
            or args.all_nu_up_to != float("inf")
        ):
            parser.error(
                "--n-min, --n-max, --nu-min, --nu-max, --max-delta-nu, and --all-nu-up-to are only valid "
                "when generating a species database."
            )
    elif args.f_max is not None:
        parser.error("--f-max is only valid when generating the misc database.")

    directory = Path(args.directory) if args.directory is not None else Path("database") / args.species
    directory = directory.resolve()
    if directory.exists():
        if args.overwrite:
            shutil.rmtree(directory)
        else:
            raise FileExistsError(f"The folder {directory} already exists. Use --overwrite to overwrite it.")
    directory.mkdir(parents=True)
    os.chdir(directory)

    configure_logging(args.log_level, directory, args.warnings_as_exceptions)

    time_start = time.perf_counter()
    if args.species == "misc":
        if args.f_max is None:
            parser.error("--f-max is required when generating the misc database.")
        create_tables_for_misc(f_max=args.f_max, kappa_max=3)
    else:
        if args.n_max is None and args.nu_max is None:
            parser.error("At least one of --n-max or --nu-max must be provided.")

        if args.n_min is None and args.n_max is None:
            n = None
        else:
            n_min = args.n_min if args.n_min is not None else 1
            n_max = args.n_max if args.n_max is not None else int(args.nu_max) + 10
            n = (n_min, n_max)

        if args.nu_min is None and args.nu_max is None:
            nu = None
        else:
            nu_min = args.nu_min if args.nu_min is not None else 0
            nu_max = args.nu_max if args.nu_max is not None else args.n_max
            nu = (nu_min, nu_max)
        create_tables_for_one_species(
            args.species, n=n, nu=nu, max_delta_nu=args.max_delta_nu, all_nu_up_to=args.all_nu_up_to
        )
    logger.info("Time taken: %.2f seconds", time.perf_counter() - time_start)
    log_memory_usage()


def log_memory_usage() -> None:
    """Log the peak and current memory usage (resident set size) of this process."""
    try:
        import resource  # noqa: PLC0415
    except ImportError:
        # the resource module is not available on some platforms (e.g. Windows), so we skip the peak memory logging
        pass
    else:
        ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss  # kilobytes on Linux, bytes on macOS
        peak_memory = ru_maxrss * (1e-6 if sys.platform == "darwin" else 1e-3)
        logger.info("Peak memory usage: %.2f megabytes", peak_memory)

    statm = Path("/proc/self/statm")
    if statm.exists():
        resident_pages = int(statm.read_text().split()[1])
        current_memory = resident_pages * os.sysconf("SC_PAGE_SIZE") * 1e-6
        logger.info("Memory usage at end of run: %.2f megabytes", current_memory)


def configure_logging(
    log_level: str,
    log_directory: Path | None = None,
    warnings_as_exceptions: bool = False,
) -> None:
    """Initialize the logger."""
    root_logger = logging.getLogger()
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    root_logger.setLevel(getattr(logging, log_level.upper()))

    stream_formatter = logging.Formatter("%(levelname)s %(asctime)s: %(message)s", datefmt="%H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    root_logger.addHandler(stream_handler)

    if log_directory is not None:
        file_formatter = logging.Formatter("%(levelname)s: %(message)s")
        file_handler = logging.FileHandler(log_directory / "log", delay=True)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    if warnings_as_exceptions:
        root_logger.addHandler(WarningsAsExceptionsHandler())


class WarningsAsExceptionsHandler(logging.Handler):
    """Custom logging handler to raise exceptions for log warnings and above."""

    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.WARNING:
            raise RuntimeError(record.getMessage())


if __name__ == "__main__":
    main()
