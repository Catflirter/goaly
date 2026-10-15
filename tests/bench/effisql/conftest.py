"""Integration fixtures for the Effi-SQL PostgreSQL environment.

``pg_env`` is a whole environment for integration tests. By default it is built for the test
session: a tiny golden build (TPC-H SF0.02 as ``tpch``, SF0.01 as ``tpch_alt``) and five
instances with their own names and ports, removed afterwards, so tests never touch the real
environment. With ``GOALY_EFFISQL_ITEST_REAL=1`` it is the real, running environment instead;
tests that write or restart instances skip then, and GOALY_EFFISQL_ITEST_INSTANCES (such as
``solve-1``) limits them to the named instances. The tiny instances run on CPUs 0, 1, 12 and 13,
never on the real instances' cores.
"""

import os
import secrets
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import psycopg
import pytest
from pydantic import SecretStr

from goaly.bench.effisql import pg_containers
from goaly.bench.effisql.pg_build import Dataset, TableFile, build_golden
from goaly.bench.effisql.pg_instances import INSTANCES, PYTHON_CPUS, Instance, start_instances
from goaly.bench.effisql.pg_settings import (
    ADMIN_ROLE,
    AGENT_ROLE,
    GRADER_ROLE,
    PgCredentials,
    PgEnvConfig,
    load_credentials,
)


def _dataset(db: str, scale: str, files: dict[str, tuple[str, int]]) -> Dataset:
    return Dataset(db, scale, {name: TableFile(sha256=s, rows=r) for name, (s, r) in files.items()})


_REGION = ("5a7c2fe9718db00ff5e5bc82a9ebfa8abc492cc75260d3c0ffb411974f235ab0", 5)
_NATION = ("7d47bc9397da331054fa92b8fb92e4c074004bad72dcbb893012093218dccf6c", 25)
# dbgen below SF0.01 repeats partsupp keys (too few suppliers), so the tiny pair is 0.02 and 0.01.
TINY_TPCH = _dataset(
    "tpch",
    "0.02",
    {
        "region": _REGION,
        "nation": _NATION,
        "supplier": ("7a9d418665ae3cc60dd2963f024c1637af37013353c759efcbaed3227bcb9ae7", 200),
        "customer": ("af62101403dea087d60e7a5cab3f2a2caa91acb18fa5be26a855e63a72898c05", 3000),
        "part": ("e84cdb3d89a51bcf13466d0134ff40e040ff14d0a4d5df3e776125cea4a14f63", 4000),
        "partsupp": ("a5571e78d54dfdf19a664aa1ef6e6182f9a3c9037dd0e03a53767545ad54d892", 16000),
        "orders": ("cfa1d861694ba2fca8c1c34e040b01d08c45d414becb52b56a63ecafe4be8016", 30000),
        "lineitem": ("187773548b644829bcdb13580423a58b08aac8c8886fc2b4ba22fb041c6c33ee", 120515),
    },
)
TINY_ALT = _dataset(
    "tpch_alt",
    "0.01",
    {
        "region": _REGION,
        "nation": _NATION,
        "supplier": ("b199bef3350840676cfe4be096091851bc47b6a57cd8a71b3f559d42b7d9dacd", 100),
        "customer": ("22156f2770387f5adadbc72774f2282d353aedd5092bd8fc54871b3bf5e74cba", 1500),
        "part": ("604232ee10e95dca875c196f759c07babf34293dfac8c99b9f1f0c2708f0561b", 2000),
        "partsupp": ("906f58419af6ad5d62489a7e2105257654bb6d8458a7fd8fe19b76655aac50c7", 8000),
        "orders": ("a444603dfba6c47e902e24b517608a5eb3b117127e99dff16a40f4eaa47b812c", 15000),
        "lineitem": ("517b566190fbeadc638602554d109a463631e19788936ccb97196ebd407b51f1", 60175),
    },
)
# The tiny instances listen on 25432-25436: below Linux's ephemeral range, so no client socket
# can hold their ports when they start.
_PORT_OFFSET = -30_000


@dataclass(frozen=True)
class PgEnv:
    config: PgEnvConfig
    credentials: PgCredentials
    instances: tuple[Instance, ...]
    real: bool

    def instance(self, kind: str) -> Instance:
        """The first instance of ``kind``; the test skips if the run left none of that kind."""
        for instance in self.instances:
            if instance.kind == kind:
                return instance
        pytest.skip(f"no {kind} instance in GOALY_EFFISQL_ITEST_INSTANCES")

    def connect(
        self, instance: Instance, role: str, db: str, **kwargs: Any
    ) -> psycopg.Connection[Any]:
        passwords = {
            ADMIN_ROLE: self.credentials.admin_password,
            AGENT_ROLE: self.credentials.agent_password,
            GRADER_ROLE: self.credentials.grader_password,
        }
        return psycopg.connect(
            host=self.config.host,
            port=instance.port,
            user=role,
            password=passwords[role].get_secret_value(),
            dbname=db,
            connect_timeout=10,
            **kwargs,
        )


@pytest.fixture(scope="session")
def pg_env(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PgEnv]:
    if os.environ.get("GOALY_EFFISQL_ITEST_REAL") == "1":
        # GOALY_EFFISQL_ITEST_INSTANCES=solve-1 keeps the tests off instances others are using.
        chosen = os.environ.get("GOALY_EFFISQL_ITEST_INSTANCES", "")
        names = {name.strip() for name in chosen.split(",") if name.strip()}
        instances = tuple(i for i in INSTANCES if not names or i.name in names)
        yield PgEnv(PgEnvConfig(), load_credentials(), instances, real=True)
        return
    root = tmp_path_factory.mktemp("effisql-env")
    real_dbgen = PgEnvConfig().dbgen_dir
    if real_dbgen.is_dir():
        (root / "dbgen").symlink_to(real_dbgen)
    test_cpus = ",".join(str(cpu) for cpu in PYTHON_CPUS)  # Never the real instances' cores.
    config = PgEnvConfig(
        data_root=root,
        build_port=25451,
        build_container="goaly-effisql-itest-env-build",
        build_cpuset=test_cpus,
    )
    credentials = PgCredentials(
        admin_password=SecretStr(secrets.token_urlsafe(16)),
        agent_password=SecretStr(secrets.token_urlsafe(16)),
        grader_password=SecretStr(secrets.token_urlsafe(16)),
        _env_file=None,  # pydantic-settings: read no env file.
    )
    instances = tuple(
        Instance(f"itest-{i.name}", i.kind, i.port + _PORT_OFFSET, PYTHON_CPUS) for i in INSTANCES
    )
    build_golden(config, credentials.admin_password, datasets=[TINY_TPCH, TINY_ALT])
    try:
        start_instances(config, credentials, instances)
        yield PgEnv(config, credentials, instances, real=False)
    finally:
        for instance in instances:
            pg_containers.remove(instance.container, force=True)


@pytest.fixture
def writable_env(pg_env: PgEnv) -> PgEnv:
    """The environment, for a test that writes to it or restarts it: never the real one."""
    if pg_env.real:
        pytest.skip("writes to or restarts instances: runs only on the test environment")
    return pg_env
