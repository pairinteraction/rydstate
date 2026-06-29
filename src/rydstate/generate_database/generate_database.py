from __future__ import annotations

import logging
import sqlite3
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rydstate import __version__
from rydstate.angular.wigner_symbols import calc_wigner_3j
from rydstate.basis.basis_mqdt import BasisMQDT
from rydstate.basis.basis_sqdt import BasisSQDT
from rydstate.generate_database.generate_matrix_elements_table import generate_matrix_elements_tables
from rydstate.generate_database.generate_misc_table import generate_wigner_table
from rydstate.generate_database.generate_states_table import generate_states_table

logger = logging.getLogger(__name__)


DATABASE_SQL_FILE = files("rydstate.generate_database").joinpath("database.sql")


def create_tables_for_one_species(
    species_specifier: str,
    n: tuple[int, int] | None = None,
    nu: tuple[float, float] | None = None,
    max_delta_nu: float = np.inf,
    all_nu_up_to: float = np.inf,
) -> None:
    """Create the database tables for a given species in the current directory."""
    logger.info("Start creating database for %s", species_specifier)
    logger.info("n-range=%s", n)
    logger.info("nu-range=%s", nu)
    logger.info("max_delta_nu=%s, all_nu_up_to=%s", max_delta_nu, all_nu_up_to)
    logger.info("rydstate.__version__=%s", __version__)

    # create the database and populate the states and matrix elements tables
    db_file = Path("database.db")
    with sqlite3.connect(db_file) as conn:
        conn.executescript(DATABASE_SQL_FILE.read_text(encoding="utf-8"))
        basis: BasisSQDT[Any] | BasisMQDT
        species = species_specifier.removesuffix("_mqdt").removesuffix("_sqdt")
        if species_specifier.endswith("_mqdt"):
            if nu is None:
                raise ValueError("nu must be provided for MQDT basis")
            basis = BasisMQDT(species, nu=nu)
            if n is not None:
                basis.filter_states("n", n)
        else:
            if n is None:
                raise ValueError("n must be provided for SQDT basis")
            basis = BasisSQDT(species, n=n, coupling_scheme="LS")
            if nu is not None:
                basis.filter_states("nu", nu)
        generate_states_table(basis, conn)
        generate_matrix_elements_tables(basis, conn, max_delta_nu, all_nu_up_to, free_memory=True)
    logger.info("Size of %s: %.6f megabytes", db_file, db_file.stat().st_size * 1e-6)

    # convert the tables to parquet files
    with sqlite3.connect(db_file) as conn:
        tables: pd.DataFrame = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
        for tkey in tables.to_numpy().flatten():
            table = pd.read_sql_query(f"SELECT * FROM {tkey}", conn)  # noqa: S608
            if len(table) == 0:
                continue
            if tkey == "states":
                table = table.astype({"is_j_total_momentum": bool, "is_calculated_with_mqdt": bool})

            parquet_file = Path(f"{tkey}.parquet")
            table.to_parquet(parquet_file, index=False, compression="zstd")

            logger.info("Size of %s: %.6f megabytes", parquet_file, parquet_file.stat().st_size * 1e-6)
            if logging.getLogger().isEnabledFor(logging.INFO):
                table.info(verbose=True)
                with Path("log").open("a") as buf:
                    table.info(buf=buf)


def create_tables_for_misc(f_max: float, kappa_max: int = 3) -> None:
    """Create misc databases, i.e. the wigner table in the current directory."""
    logger.info("Start creating misc database")
    logger.info("f_max=%s", f_max)
    logger.info("kappa_max=%d", kappa_max)
    logger.info("rydstate.__version__=%s", __version__)

    # create the database and populate the wigner table
    db_file = Path("database.db")
    with sqlite3.connect(db_file) as conn:
        conn.executescript(DATABASE_SQL_FILE.read_text(encoding="utf-8"))
        generate_wigner_table(f_max, kappa_max, conn)
    logger.info("Size of %s: %.6f megabytes", db_file, db_file.stat().st_size * 1e-6)

    # convert the table to a parquet file
    with sqlite3.connect(db_file) as conn:
        parquet_file = Path("wigner.parquet")
        table = pd.read_sql_query("SELECT * FROM wigner", conn)
        table.to_parquet(parquet_file, index=False, compression="zstd")

        logger.info("Size of %s: %.6f megabytes", parquet_file, parquet_file.stat().st_size * 1e-6)
        if logging.getLogger().isEnabledFor(logging.INFO):
            table.info(verbose=True)
            with Path("log").open("a") as buf:
                table.info(buf=buf)

    logger.info("calc_wigner_3j: %s", calc_wigner_3j.cache_info())  # type: ignore [attr-defined]
