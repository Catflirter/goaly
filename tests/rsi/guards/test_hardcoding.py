"""Tests for goaly.rsi.guards.hardcoding: a candidate embeds no gold answer and no data table.

Every task here is synthetic: no real gold answer is ever used in a test.
"""

from pathlib import Path

import pytest

from goaly.contracts.gen.experiment_pb2 import GUARD_HARD_CODING_SCAN, LANE_EFFISQL
from goaly.rsi.guards.gold_windows import GoldWindows
from goaly.rsi.guards.hardcoding import HardCodingConfig, check_hard_coding
from goaly.rsi.workspace import Workspace
from tests.rsi.gitrepo import git, make_repo, write

PACK = "src/goaly/harness/packs/effisql"
# The standard join every base query of this little benchmark holds.
JOIN = "SELECT l_orderkey FROM lineitem JOIN orders ON o_orderkey = l_orderkey"
BASES = [f"{JOIN} WHERE l_quantity > 10", f"{JOIN} WHERE l_discount < 0.05"]
# A gold whose second half no base query has.
GOLD_ONLY = "WHERE NOT EXISTS (SELECT 1 FROM partsupp WHERE ps_partkey = l_partkey)"
GOLDS = [f"{JOIN} {GOLD_ONLY}"]


@pytest.fixture
def gold() -> GoldWindows:
    return GoldWindows.build(GOLDS, BASES, ["CREATE TABLE lineitem (l_orderkey integer);"])


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    repo = make_repo(tmp_path / "repo")
    write(repo, {f"{PACK}/tables.py": "SIZES = [1, 2, 3]\n" + _literal_list(40, "OLD")})
    git(repo, "add", "--all")
    git(repo, "commit", "--quiet", "--message=v0 with a table it already had")
    git(repo, "tag", "--force", "best/effisql")
    return repo


def _literal_list(count: int, name: str = "TABLE") -> str:
    return f"{name} = [\n" + "".join(f"    {n},\n" for n in range(count)) + "]\n"


def candidate(repo: Path, files: dict[str, str]) -> str:
    write(repo, files)
    git(repo, "add", "--all")
    git(repo, "commit", "--quiet", "--message=candidate")
    return git(repo, "rev-parse", "HEAD").strip()


def scan(
    repo: Path, commit: str, gold: GoldWindows, config: HardCodingConfig | None = None
) -> tuple[bool, str]:
    workspace = Workspace(repo, root=repo.parent / "worktrees")
    with workspace.snapshot(commit) as snapshot:
        result = check_hard_coding(
            repo,
            LANE_EFFISQL,
            parent="best/effisql",
            candidate=commit,
            snapshot=snapshot,
            gold=gold,
            config=config or HardCodingConfig(),
        )
    assert result.guard == GUARD_HARD_CODING_SCAN
    return result.passed, result.detail


def test_a_standard_join_that_base_queries_hold_passes(repo: Path, gold: GoldWindows) -> None:
    prompt = f"Joins such as\n\n    {JOIN}\n\nare fine to keep.\n"
    commit = candidate(repo, {f"{PACK}/prompt.md": prompt})

    assert scan(repo, commit, gold) == (True, "no gold-only fragment, no large literal")


def test_a_gold_only_fragment_fails_without_repeating_it(repo: Path, gold: GoldWindows) -> None:
    commit = candidate(repo, {f"{PACK}/skills/tricks.md": f"# Tricks\n\nTry {GOLD_ONLY}.\n"})

    passed, detail = scan(repo, commit, gold)

    assert not passed
    assert detail == f"{PACK}/skills/tricks.md:3: a fragment of a gold answer"
    assert "partsupp" not in detail


def test_a_gold_only_fragment_split_over_lines_and_case_fails(
    repo: Path, gold: GoldWindows
) -> None:
    lines = GOLD_ONLY.upper().replace(" FROM", "\n        FROM").split("\n")
    commit = candidate(repo, {f"{PACK}/policy.py": "HINT = '''\n" + "\n".join(lines) + "\n'''\n"})

    passed, detail = scan(repo, commit, gold)

    assert not passed
    assert detail.startswith(f"{PACK}/policy.py:2: a fragment of a gold answer")


def test_only_added_lines_are_scanned(repo: Path, gold: GoldWindows) -> None:
    # A fragment already in the best version was scanned when it was added.
    write(repo, {f"{PACK}/prompt.md": f"Old text: {GOLD_ONLY}\n"})
    git(repo, "add", "--all")
    git(repo, "commit", "--quiet", "--message=as if v0 held it")
    git(repo, "tag", "--force", "best/effisql")
    commit = candidate(repo, {f"{PACK}/prompt.md": f"Old text: {GOLD_ONLY}\nNew line.\n"})

    assert scan(repo, commit, gold)[0]


@pytest.mark.parametrize(
    ("source", "line"),
    [
        (_literal_list(33), 1),
        ("ANSWERS = {" + ", ".join(f"'q{n}': {n}" for n in range(33)) + "}\n", 1),
        ("PAIRS = [" + ", ".join(f"({n}, 'x')" for n in range(33)) + "]\n", 1),
        ("SEEN = {" + ", ".join(str(n) for n in range(33)) + "}\n", 1),
        ("ROWS = (" + ", ".join(f"-{n}.5" for n in range(33)) + ",)\n", 1),
        ("CASES = [" + ", ".join(f"{{'id': {n}}}" for n in range(33)) + "]\n", 1),
        ("MIXED = [x, y, " + ", ".join(str(n) for n in range(33)) + "]\n", 1),
    ],
)
def test_a_large_literal_collection_fails(
    repo: Path, gold: GoldWindows, source: str, line: int
) -> None:
    commit = candidate(repo, {f"{PACK}/data.py": "X = 1\n" + source})

    passed, detail = scan(repo, commit, gold)

    assert not passed
    assert detail == f"{PACK}/data.py:{line + 1}: a literal collection of 33 constants"


def test_a_literal_collection_at_the_limit_passes(repo: Path, gold: GoldWindows) -> None:
    commit = candidate(repo, {f"{PACK}/data.py": _literal_list(32)})

    assert scan(repo, commit, gold)[0]


def test_the_limit_comes_from_config(repo: Path, gold: GoldWindows) -> None:
    commit = candidate(repo, {f"{PACK}/data.py": _literal_list(5)})

    passed, detail = scan(repo, commit, gold, HardCodingConfig(max_literal_elements=4))

    assert not passed
    assert detail == f"{PACK}/data.py:1: a literal collection of 5 constants"


def test_an_unchanged_large_literal_in_a_changed_file_passes(repo: Path, gold: GoldWindows) -> None:
    commit = candidate(
        repo, {f"{PACK}/tables.py": "SIZES = [1, 2, 3, 4]\n" + _literal_list(40, "OLD")}
    )

    assert scan(repo, commit, gold)[0]


def test_a_long_string_and_a_computed_collection_pass(repo: Path, gold: GoldWindows) -> None:
    source = "PROMPT = '" + "word " * 500 + "'\nRANGE = [n for n in range(100)]\n"
    commit = candidate(repo, {f"{PACK}/prompts.py": source})

    assert scan(repo, commit, gold)[0]


def test_a_python_file_that_does_not_parse_is_scanned_as_text(
    repo: Path, gold: GoldWindows
) -> None:
    commit = candidate(repo, {f"{PACK}/broken.py": f"def broken(:\n    '{GOLD_ONLY}'\n"})

    passed, detail = scan(repo, commit, gold)

    assert not passed
    assert detail == f"{PACK}/broken.py:2: a fragment of a gold answer"


def test_a_deleted_file_is_not_scanned(repo: Path, gold: GoldWindows) -> None:
    (repo / PACK / "tables.py").unlink()
    commit = candidate(repo, {})

    assert scan(repo, commit, gold)[0]
