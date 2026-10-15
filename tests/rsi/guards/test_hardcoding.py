"""Tests for goaly.rsi.guards.hardcoding: a candidate embeds no gold answer and no data table.

Every task here is synthetic: no real gold answer is ever used in a test.
"""

from pathlib import Path

import pytest

from goaly.contracts.gen.experiment_pb2 import GUARD_HARD_CODING_SCAN
from goaly.rsi.guards import hardcoding
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
        ("PAIRS = [" + ", ".join(f"({n}, 'x')" for n in range(17)) + "]\n", 1),  # 34 leaves.
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
    assert detail.startswith(f"{PACK}/data.py:{line + 1}: a literal collection of 3")


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
    assert detail.splitlines() == [
        f"{PACK}/broken.py:2: a fragment of a gold answer",
        f"{PACK}/broken.py:1: cannot be parsed for literal collections",
    ]


def test_a_deleted_file_is_not_scanned(repo: Path, gold: GoldWindows) -> None:
    (repo / PACK / "tables.py").unlink()
    commit = candidate(repo, {})

    assert scan(repo, commit, gold)[0]


# The review of PR #41.


def test_an_empty_store_is_refused_not_passed(repo: Path) -> None:
    commit = candidate(repo, {f"{PACK}/prompt.md": "Be faster.\n"})

    with pytest.raises(ValueError, match="no gold-only window"):
        scan(repo, commit, GoldWindows(frozenset()))


def test_a_python_file_too_deep_to_parse_fails(repo: Path, gold: GoldWindows) -> None:
    commit = candidate(repo, {f"{PACK}/deep.py": "X = " + "+".join(["1"] * 100_000) + "\n"})

    passed, detail = scan(repo, commit, gold)

    assert not passed
    assert detail == f"{PACK}/deep.py:1: cannot be parsed for literal collections"


@pytest.mark.parametrize(
    "source",
    [
        "TABLE = [" + ", ".join(f"[{n}, {n}]" for n in range(20)) + "]\n",  # Nested: 40.
        "TABLE = [*["
        + ", ".join(map(str, range(20)))
        + "], *["
        + ", ".join(map(str, range(20)))
        + "]]\n",
        "TABLE = ["
        + ", ".join(map(str, range(20)))
        + "] + ["
        + ", ".join(map(str, range(20)))
        + "]\n",
    ],
)
def test_a_table_split_into_pieces_still_counts(repo: Path, gold: GoldWindows, source: str) -> None:
    commit = candidate(repo, {f"{PACK}/rewrites.py": source})

    passed, detail = scan(repo, commit, gold)

    assert not passed
    assert "a literal collection of 40 constants" in detail


def test_a_gold_answer_split_over_adjacent_literals_is_found(repo: Path, gold: GoldWindows) -> None:
    half = len(GOLD_ONLY) // 2
    source = f"HINT = ('{GOLD_ONLY[:half]}'\n        '{GOLD_ONLY[half:]}')\n"
    commit = candidate(repo, {f"{PACK}/hint.py": source})

    passed, detail = scan(repo, commit, gold)

    assert not passed
    assert f"{PACK}/hint.py:1: a fragment of a gold answer" in detail.splitlines()


def test_a_gold_answer_with_zero_width_characters_is_found(repo: Path, gold: GoldWindows) -> None:
    hidden = "\u200b".join(GOLD_ONLY)
    commit = candidate(repo, {f"{PACK}/skills/hidden.md": f"Try {hidden}\n"})

    assert not scan(repo, commit, gold)[0]


def test_a_hit_names_its_exact_line_far_into_the_file(repo: Path, gold: GoldWindows) -> None:
    lines = [f"line {n}\n" for n in range(6)] + [f"{GOLD_ONLY}\n"]
    commit = candidate(repo, {f"{PACK}/skills/late.md": "".join(lines)})

    assert scan(repo, commit, gold) == (
        False,
        f"{PACK}/skills/late.md:7: a fragment of a gold answer",
    )


def test_a_gold_answer_split_by_a_blank_line_is_found(repo: Path, gold: GoldWindows) -> None:
    words = GOLD_ONLY.split(" ")
    first, second = " ".join(words[:5]), " ".join(words[5:])
    commit = candidate(repo, {f"{PACK}/skills/split.md": f"{first}\n\n{second}\n"})

    assert not scan(repo, commit, gold)[0]


def test_an_added_line_that_starts_with_plus_is_scanned(repo: Path, gold: GoldWindows) -> None:
    commit = candidate(repo, {f"{PACK}/skills/plus.md": f"+ {GOLD_ONLY}\n"})

    assert not scan(repo, commit, gold)[0]


def test_a_small_list_grown_past_the_limit_from_inside_fails(repo: Path, gold: GoldWindows) -> None:
    write(repo, {f"{PACK}/small.py": "SMALL = [\n    0,\n    1,\n]\n"})
    git(repo, "add", "--all")
    git(repo, "commit", "--quiet", "--message=a small list")
    git(repo, "tag", "--force", "best/effisql")
    grown = "SMALL = [\n    0,\n" + "".join(f"    {n},\n" for n in range(2, 40)) + "    1,\n]\n"
    commit = candidate(repo, {f"{PACK}/small.py": grown})

    passed, detail = scan(repo, commit, gold)

    assert not passed
    assert detail == f"{PACK}/small.py:1: a literal collection of 40 constants"


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("rewrites.json", "[" + ", ".join(map(str, range(40))) + "]"),
        ("rewrites.json", '{"a": ' + "[" + ", ".join(map(str, range(40))) + "]}"),
        ("rows.jsonl", "".join(f'{{"id": {n}}}\n' for n in range(40))),
        ("config.yaml", "rewrites:\n" + "".join(f"  - q{n}\n" for n in range(40))),
        ("config.yaml", "rewrites:\n" + "".join(f"  q{n}: x\n" for n in range(40))),
        ("config.yml", "ids: [" + ", ".join(map(str, range(40))) + "]\n"),
        ("config.toml", "ids = [" + ", ".join(map(str, range(40))) + "]\n"),
        ("table.csv", "".join(f"{n},x\n" for n in range(40))),
        ("skills/table.md", "| q | sql |\n" + "".join(f"| {n} | x |\n" for n in range(40))),
        ("notes.txt", "".join(f"- item {n}\n" for n in range(40))),
    ],
)
def test_a_large_literal_in_a_file_of_any_type_fails(
    repo: Path, gold: GoldWindows, name: str, text: str
) -> None:
    commit = candidate(repo, {f"{PACK}/{name}": text})

    passed, detail = scan(repo, commit, gold)

    assert not passed, detail
    assert "a literal collection of" in detail


@pytest.mark.parametrize(
    ("name", "text"),
    [
        ("config.json", '{"steps": 30, "tools": ["sql", "explain"]}'),
        ("config.yaml", "steps: 30\ntools:\n  - sql\n  - explain\n"),
        ("config.toml", "steps = 30\n"),
        ("prompt.md", "# Plan\n\n- read\n- rewrite\n- check\n"),
    ],
)
def test_small_data_and_text_files_pass(
    repo: Path, gold: GoldWindows, name: str, text: str
) -> None:
    commit = candidate(repo, {f"{PACK}/{name}": text})

    assert scan(repo, commit, gold)[0]


@pytest.mark.parametrize(("name", "text"), [("bad.json", "{"), ("bad.toml", "x = ")])
def test_a_data_file_that_does_not_parse_fails(
    repo: Path, gold: GoldWindows, name: str, text: str
) -> None:
    commit = candidate(repo, {f"{PACK}/{name}": text})

    assert scan(repo, commit, gold) == (
        False,
        f"{PACK}/{name}:1: cannot be parsed for literal collections",
    )


def test_a_table_the_parent_already_had_in_a_data_file_passes(
    repo: Path, gold: GoldWindows
) -> None:
    write(repo, {f"{PACK}/ids.json": "[" + ", ".join(map(str, range(40))) + "]"})
    git(repo, "add", "--all")
    git(repo, "commit", "--quiet", "--message=a table v0 had")
    git(repo, "tag", "--force", "best/effisql")
    commit = candidate(repo, {f"{PACK}/ids.json": "[" + ", ".join(map(str, range(39))) + "]"})

    assert scan(repo, commit, gold)[0]


def test_a_gold_line_and_its_string_are_reported_once(repo: Path, gold: GoldWindows) -> None:
    commit = candidate(repo, {f"{PACK}/hint.py": f"HINT = '{GOLD_ONLY}'\n"})

    assert scan(repo, commit, gold) == (False, f"{PACK}/hint.py:1: a fragment of a gold answer")


def test_a_sql_file_is_scanned_for_gold_alone(repo: Path, gold: GoldWindows) -> None:
    commit = candidate(repo, {f"{PACK}/skills/example.sql": "SELECT 1;\n" * 50})

    assert scan(repo, commit, gold)[0]


def test_a_python_file_too_deep_to_count_fails(
    repo: Path, gold: GoldWindows, monkeypatch: pytest.MonkeyPatch
) -> None:
    def too_deep(node: object) -> int:
        raise RecursionError

    monkeypatch.setattr(hardcoding, "_constants", too_deep)
    commit = candidate(repo, {f"{PACK}/data.py": "X = [1]\n"})

    assert scan(repo, commit, gold) == (
        False,
        f"{PACK}/data.py:1: cannot be checked for literal collections: it nests too deeply",
    )


def test_a_yaml_run_ends_at_a_line_that_is_no_item(repo: Path, gold: GoldWindows) -> None:
    items = "".join(f"  - q{n}\n" for n in range(20))
    commit = candidate(repo, {f"{PACK}/config.yaml": f"a:\n{items}  plain\n{items}"})

    assert scan(repo, commit, gold)[0]
