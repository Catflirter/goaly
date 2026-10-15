import stat
from pathlib import Path

import pytest
from pydantic import SecretStr

from goaly.bench.effisql.pg_settings import (
    AgentCredentials,
    PgEnvConfig,
    PgSettingsError,
    connection_kwargs,
    default_credentials_file,
    ensure_credentials,
    load_agent_credentials,
    load_credentials,
    write_agent_credentials,
)

VARIABLES = ("GOALY_PG_ADMIN_PASSWORD", "GOALY_PG_AGENT_PASSWORD", "GOALY_PG_GRADER_PASSWORD")


def _values(path: Path) -> dict[str, str]:
    return dict(line.split("=", 1) for line in path.read_text().splitlines())


def test_the_credentials_file_lives_in_the_users_config_directory() -> None:
    assert default_credentials_file() == Path.home() / ".config" / "goaly" / "pg.env"


def test_ensure_credentials_writes_a_private_file_with_three_random_passwords() -> None:
    ensure_credentials()

    path = default_credentials_file()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    values = _values(path)
    assert set(values) == set(VARIABLES)
    assert len(set(values.values())) == 3
    assert all(len(value) >= 40 for value in values.values())


def test_ensure_credentials_keeps_existing_passwords_and_adds_the_missing_ones(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pg.env"
    path.write_text("GOALY_PG_ADMIN_PASSWORD=kept-admin\n")
    path.chmod(0o600)

    ensure_credentials(path)
    first = _values(path)
    ensure_credentials(path)

    assert first["GOALY_PG_ADMIN_PASSWORD"] == "kept-admin"  # noqa: S105  # A test value.
    assert set(first) == set(VARIABLES)
    assert _values(path) == first  # A complete file is left alone.


def test_a_credentials_file_other_users_can_read_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "pg.env"
    path.write_text("".join(f"{name}=x\n" for name in VARIABLES))
    path.chmod(0o640)

    with pytest.raises(PgSettingsError, match="chmod 600"):
        ensure_credentials(path)
    with pytest.raises(PgSettingsError, match="chmod 600"):
        load_credentials(path)


def test_load_credentials_reads_the_file_and_hides_the_values(tmp_path: Path) -> None:
    path = tmp_path / "pg.env"
    ensure_credentials(path)

    credentials = load_credentials(path)

    values = _values(path)
    assert credentials.agent_password.get_secret_value() == values["GOALY_PG_AGENT_PASSWORD"]
    assert values["GOALY_PG_AGENT_PASSWORD"] not in repr(credentials)


def test_environment_variables_are_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "pg.env"
    ensure_credentials(path)
    monkeypatch.setenv("GOALY_PG_GRADER_PASSWORD", "from-the-environment")

    password = load_credentials(path).grader_password.get_secret_value()
    assert password == _values(path)["GOALY_PG_GRADER_PASSWORD"]


def test_a_lost_admin_password_is_not_replaced_once_a_golden_build_exists(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pg.env"
    pgdata = tmp_path / "golden" / "pgdata"
    pgdata.mkdir(parents=True)

    with pytest.raises(PgSettingsError, match="lost the admin password"):
        ensure_credentials(path, golden_pgdata=pgdata)
    assert not path.exists()

    path.write_text("GOALY_PG_ADMIN_PASSWORD=kept\n")
    path.chmod(0o600)
    ensure_credentials(path, golden_pgdata=pgdata)  # Only the other two are new.
    assert _values(path)["GOALY_PG_ADMIN_PASSWORD"] == "kept"  # noqa: S105  # A test value.


def test_the_file_is_read_with_the_dotenv_parser(tmp_path: Path) -> None:
    path = tmp_path / "pg.env"
    path.write_text(
        "# a comment\nexport GOALY_PG_ADMIN_PASSWORD='quoted value'\n"
        "GOALY_PG_AGENT_PASSWORD=a\nGOALY_PG_GRADER_PASSWORD=g\n"
    )
    path.chmod(0o600)

    ensure_credentials(path)  # Complete as dotenv reads it: nothing is rewritten.

    assert load_credentials(path).admin_password.get_secret_value() == "quoted value"
    assert path.read_text().startswith("# a comment")


def test_a_missing_password_is_named_without_echoing_any_value(tmp_path: Path) -> None:
    path = tmp_path / "pg.env"
    path.write_text("GOALY_PG_ADMIN_PASSWORD=do-not-echo\nGOALY_PG_AGENT_PASSWORD=\n")
    path.chmod(0o600)

    with pytest.raises(PgSettingsError) as caught:
        load_credentials(path)

    message = str(caught.value)
    assert "GOALY_PG_AGENT_PASSWORD" in message
    assert "GOALY_PG_GRADER_PASSWORD" in message
    assert "do-not-echo" not in message
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None


def test_load_credentials_without_a_file_names_the_missing_variables(tmp_path: Path) -> None:
    with pytest.raises(PgSettingsError, match="GOALY_PG_ADMIN_PASSWORD is missing"):
        load_credentials(tmp_path / "absent.env")


def test_an_unreadable_file_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "pg.env"
    path.write_bytes(b"GOALY_PG_ADMIN_PASSWORD=\xff\xfe\n")
    path.chmod(0o600)

    with pytest.raises(PgSettingsError, match="Cannot read"):
        load_credentials(path)


def test_the_config_derives_every_path_from_the_data_root(tmp_path: Path) -> None:
    config = PgEnvConfig(data_root=tmp_path)

    assert PgEnvConfig().data_root == Path.home() / "data" / "effisql"
    assert config.dbgen_dir == tmp_path / "dbgen"
    assert config.raw_dir("0.3") == tmp_path / "raw" / "sf0.3"
    assert config.golden_pgdata == tmp_path / "golden" / "pgdata"
    assert config.golden_record == tmp_path / "golden" / "golden.json"
    assert config.snapshot_path("tpch") == tmp_path / "snapshots" / "tpch" / "schema_snapshot.json"
    assert config.plans_path == tmp_path / "plans.json"
    assert config.image.startswith("postgres@sha256:")


def test_a_stale_temporary_file_cannot_widen_the_credentials_files_mode(tmp_path: Path) -> None:
    stale = tmp_path / ".pg.env.tmp"
    stale.write_text("old")
    stale.chmod(0o644)

    ensure_credentials(tmp_path / "pg.env")

    assert stat.S_IMODE((tmp_path / "pg.env").stat().st_mode) == 0o600
    leftovers = sorted(path.name for path in tmp_path.iterdir() if "pg.env" in path.name)
    assert leftovers == [".pg.env.tmp", "pg.env"]  # No temporary file of ours remains.


def test_the_solving_side_loads_the_agent_password_and_no_other(tmp_path: Path) -> None:
    path = tmp_path / "pg.env"
    ensure_credentials(path)  # The host's file, with all three passwords.

    agent = load_agent_credentials(path)

    values = _values(path)
    assert agent.agent_password.get_secret_value() == values["GOALY_PG_AGENT_PASSWORD"]
    assert set(type(agent).model_fields) == {"agent_password"}  # Nowhere to hold the others.
    assert not any(value in repr(agent) for value in values.values())
    assert load_credentials(path).agent_only() == agent


def test_an_agent_only_file_is_private_and_holds_one_password(tmp_path: Path) -> None:
    host = tmp_path / "pg.env"
    ensure_credentials(host)
    mounted = tmp_path / "solve" / "pg.env"

    write_agent_credentials(mounted, load_credentials(host))

    assert stat.S_IMODE(mounted.stat().st_mode) == 0o600
    assert _values(mounted) == {"GOALY_PG_AGENT_PASSWORD": _values(host)["GOALY_PG_AGENT_PASSWORD"]}
    assert load_agent_credentials(mounted) == load_agent_credentials(host)
    with pytest.raises(PgSettingsError, match="GOALY_PG_ADMIN_PASSWORD is missing"):
        load_credentials(mounted)  # Nothing in it lets anyone log in as the admin.


def test_an_agent_file_without_the_agent_password_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "pg.env"
    path.write_text("GOALY_PG_ADMIN_PASSWORD=do-not-echo\n")
    path.chmod(0o600)

    with pytest.raises(PgSettingsError, match="GOALY_PG_AGENT_PASSWORD is missing") as caught:
        load_agent_credentials(path)
    assert "do-not-echo" not in str(caught.value)


def test_agent_credentials_ignore_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GOALY_PG_AGENT_PASSWORD", "from-the-environment")

    with pytest.raises(PgSettingsError, match="GOALY_PG_AGENT_PASSWORD is missing"):
        load_agent_credentials(tmp_path / "absent.env")


def test_every_connection_to_the_servers_is_autocommit_with_a_bounded_connect() -> None:
    kwargs = connection_kwargs("/run/pg/solve-1", 55432, "effisql_agent", SecretStr("pw"), "tpch")

    assert kwargs == {
        "host": "/run/pg/solve-1",  # A directory: the server's Unix socket there.
        "port": 55432,
        "user": "effisql_agent",
        "password": "pw",
        "dbname": "tpch",
        "autocommit": True,
        "connect_timeout": 10,
    }
    assert AgentCredentials(agent_password=SecretStr("pw"), _env_file=None).agent_password
