"""Effi-SQL's official prediction format (spec §3.3): how a Prediction reaches the official runner.

The official prediction record is the task record plus a ``prediction`` field. The official
runner reads that field in one of two modes, and which one the organizers use is unknown, so the
field holds an answer for each, as two fenced blocks and nothing else:

1. A ``sql`` block for ``end2end`` mode, which skips diff blocks and takes a query that begins
   with ``SELECT`` or ``WITH``.
2. A ``diff`` block for the default patch mode, which applies the first diff-like block to
   ``base_sql`` and skips the sql block, which does not look like a diff. Its applier drops one
   base line per ``-`` line without comparing their text, so the block has exactly one ``-`` line
   per base line, then one ``+`` line per line of the query.

Both blocks carry the same query, ended with ``;``: the submitted one in its *official form*, which
:func:`official_query` derives. Comments are dropped, since a comment such as ``-- final query``
matches the runner's final-answer marker and adds a competing ``end2end`` candidate, and a line
the runner's extractors would misread, such as one that begins with ``-``, is joined onto the line
before it. Comments and line breaks outside literals carry no meaning, so the query means the
same.

A query the format cannot carry faithfully, such as two statements or one that mentions the
final-answer marker in a literal, has no official form: :func:`official_answer` says why, and the
record's ``prediction`` is empty, which fails in both modes. The grade phase grades the official
form, so a grade is always what the official runner would see.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass

from goaly.bench.effisql.data import EffiSqlInstance
from goaly.contracts.gen.task_pb2 import Prediction
from goaly.contracts.interfaces import JsonValue

__all__ = [
    "PREDICTION_FIELD",
    "OfficialAnswer",
    "QueryFormatError",
    "encode_prediction",
    "official_answer",
    "official_query",
    "to_official",
]

# The field of the official record that holds the answer.
PREDICTION_FIELD = "prediction"

# The runner's final-answer marker: text it reads as the start of the final query in end2end mode.
# Our record never contains it, so the sql block is the only candidate.
_FINAL_MARKER = re.compile(
    r"final\s+(?:answer|sql|query)|optimized\s+(?:sql|query)|optimization\s+sql"
    r"|rewritten\s+(?:sql|query)|改写后|优化后|最终\s*(?:sql|查询)",
    re.IGNORECASE,
)
# Reasoning tags: patch mode deletes <think> tags from the field, and end2end may drop a tagged
# section.
_REASONING_TAG = re.compile(r"</?(?:think|reasoning)\b", re.IGNORECASE)
# Text that ends a fenced block, or that the patch-mode cleanup reads as a diff header.
_DIFF_HAZARDS = ("```", "---", "+++")
# Line prefixes that end2end's cleanup drops after stripping a line (compared in lower case).
_DROPPED_PREFIXES = ("```", "###", "diagnosis", "analysis")
# What end2end accepts: one statement that begins with SELECT or WITH and a blank.
_QUERY_START = re.compile(r"(?:select|with)\s", re.IGNORECASE)
# The line separators that str.splitlines, and so the runner, splits at, besides a line feed.
_OTHER_LINE_BREAKS = frozenset(
    "\r\x0b\x0c\x1c\x1d\x1e\x85\N{LINE SEPARATOR}\N{PARAGRAPH SEPARATOR}"
)
_DOLLAR_TAG = re.compile(r"\$(?:[^\W\d]\w*)?\$")  # $$ or $tag$; a tag is an identifier.


class QueryFormatError(ValueError):
    """A query has no official form; the message says why, for the grade's feedback."""


@dataclass(frozen=True, slots=True)
class OfficialAnswer:
    """A prediction as the official record carries it.

    ``query`` is the official form of the answer, the query both blocks carry and the grade phase
    grades, or None if there is none: ``problem`` then says why, and ``prediction`` is empty.
    """

    query: str | None
    prediction: str
    problem: str | None = None


def official_answer(inst: EffiSqlInstance, prediction: Prediction) -> OfficialAnswer:
    """Return ``prediction``'s official answer for ``inst``.

    A prediction with no benchmark branch, as after a crash, answers ``base_sql`` unchanged, as
    the v0 fallback does: an identity diff plus ``base_sql``. A blank query, a non-Effi-SQL answer
    and a query with no official form get an empty ``prediction`` and a ``problem``.
    """
    branch = prediction.WhichOneof("benchmark")
    if branch is None:
        sql = inst.base_sql
    elif branch == "effisql":
        sql = prediction.effisql.sql
    else:
        return OfficialAnswer(None, "", f"not an Effi-SQL answer: it has a {branch} answer")
    try:
        query = official_query(sql)
        return OfficialAnswer(query, encode_prediction(inst.base_sql, query))
    except QueryFormatError as error:
        return OfficialAnswer(None, "", str(error))


def to_official(inst: EffiSqlInstance, prediction: Prediction) -> dict[str, JsonValue]:
    """Return the official prediction record: ``inst``'s task record plus its ``prediction``."""
    return {**inst.record, PREDICTION_FIELD: official_answer(inst, prediction).prediction}


def encode_prediction(base_sql: str, query: str) -> str:
    """Return the ``prediction`` text that replaces ``base_sql`` with ``query``, an official form.

    The text is a ``sql`` block, then a ``diff`` block, each holding ``query`` and a final ``;``,
    so both of the runner's modes extract exactly ``query + ";"``. The sql block comes first, and
    the ``;`` ends it as a statement, so even end2end's last resort, which reads the field's first
    statement when no block passes its plausibility test (a CTE written ``AS MATERIALIZED (``, as
    12 base queries are, fails it), finds the query. The diff's ``-`` lines show the base lines
    they drop, except a line that would itself look like a diff header or a marker, which becomes
    a bare ``-``: the applier ignores their text.
    """
    statement = f"{query};"
    base_lines = base_sql.replace("\r\n", "\n").splitlines()  # How the applier counts them.
    query_lines = statement.split("\n")
    diff = [f"@@ -1,{len(base_lines)} +1,{len(query_lines)} @@"]
    diff.extend("-" + line if _safe_removed_line(line) else "-" for line in base_lines)
    diff.extend("+" + line for line in query_lines)
    body = "\n".join(diff)
    return f"```sql\n{statement}\n```\n\n```diff\n{body}\n```"


def _safe_removed_line(line: str) -> bool:
    return not (
        line.startswith("--")  # "-" + "--" is a "---" header.
        or any(hazard in line for hazard in _DIFF_HAZARDS)
        or _FINAL_MARKER.search(line)
    )


def official_query(sql: str) -> str:
    """Return the official form of ``sql``: the query both blocks of the record carry.

    The form is ``sql`` with CRLF line ends made LF, comments dropped, the space at each line's
    end and blank lines removed, a final ``;`` dropped, and every line the runner's extractors
    would misread joined onto the line before it with one space. Raises QueryFormatError if no
    form can carry ``sql`` faithfully in both of the runner's modes.
    """
    text = sql.replace("\r\n", "\n")
    if _OTHER_LINE_BREAKS.intersection(text):
        raise QueryFormatError("the query holds a line break other than a line feed")
    chars, code = _strip_comments(text)
    chars, code = _tidy_lines(chars, code)
    if not chars:
        raise QueryFormatError("the query is empty")
    if any(c == ";" and is_code for c, is_code in zip(chars, code, strict=True)):
        raise QueryFormatError("the answer holds more than one statement")
    query = "".join(chars)
    if not _QUERY_START.match(query):
        raise QueryFormatError("the query does not begin with SELECT or WITH")
    query = _join_misread_lines(chars, code)
    _check_extractable(query)
    return query


def _strip_comments(text: str) -> tuple[list[str], list[bool]]:
    """Drop ``text``'s comments, following PostgreSQL's lexer.

    Returns the remaining characters, each with a flag that is True outside literals and quoted
    identifiers. A line comment goes up to its line feed, which stays; a block comment, which may
    nest, becomes one space. Raises QueryFormatError for an unterminated literal or comment.
    """
    chars: list[str] = []
    code: list[bool] = []
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        pair = text[i : i + 2]
        if pair == "--":
            end = text.find("\n", i)
            i = n if end < 0 else end
        elif pair == "/*":
            i = _block_comment_end(text, i)
            chars.append(" ")
            code.append(True)
        elif c in "'\"" or (c == "$" and _DOLLAR_TAG.match(text, i)) or _escape_string(text, i):
            end = _dollar_quote_end(text, i) if c == "$" else _quoted_end(text, i)
            chars.extend(text[i:end])
            code.extend([False] * (end - i))
            i = end
        else:
            chars.append(c)
            code.append(True)
            i += 1
    return chars, code


def _escape_string(text: str, i: int) -> bool:
    """Whether an E'...' string, with backslash escapes, starts at ``i``."""
    return (
        text[i] in "eE"
        and text[i + 1 : i + 2] == "'"
        and (i == 0 or not (text[i - 1].isalnum() or text[i - 1] in "_$"))
    )


def _block_comment_end(text: str, start: int) -> int:
    depth, i = 0, start
    while i < len(text):
        pair = text[i : i + 2]
        if pair == "/*":
            depth, i = depth + 1, i + 2
        elif pair == "*/":
            depth, i = depth - 1, i + 2
            if depth == 0:
                return i
        else:
            i += 1
    raise QueryFormatError("the query has an unterminated block comment")


def _dollar_quote_end(text: str, start: int) -> int:
    """Return the index just past the dollar-quoted string at ``start``."""
    tag = _DOLLAR_TAG.match(text, start)
    opening = tag.group() if tag else "$$"  # The caller matched it.
    end = text.find(opening, start + len(opening))
    if end < 0:
        raise QueryFormatError("the query has an unterminated dollar-quoted string")
    return end + len(opening)


def _quoted_end(text: str, start: int) -> int:
    """Return the index just past the literal or quoted identifier at ``start``."""
    backslashes = text[start] in "eE"
    i = start + 1 if backslashes else start
    quote = text[i]
    i += 1
    while i < len(text):
        c = text[i]
        if backslashes and c == "\\":
            i += 2
        elif c == quote:
            if text[i + 1 : i + 2] != quote:  # A doubled quote stands for one.
                return i + 1
            i += 2
        else:
            i += 1
    raise QueryFormatError("the query has an unterminated literal or quoted identifier")


def _lines(chars: list[str], code: list[bool]) -> Iterator[tuple[list[str], list[bool], bool]]:
    """Split at line feeds: each line's characters and flags, and whether its break is code."""
    start = 0
    for i, c in enumerate(chars):
        if c == "\n":
            yield chars[start:i], code[start:i], code[i]
            start = i + 1
    yield chars[start:], code[start:], True


def _tidy_lines(chars: list[str], code: list[bool]) -> tuple[list[str], list[bool]]:
    """Remove the space at code line ends, blank code lines, and the final ``;`` and space."""
    out_chars: list[str] = []
    out_code: list[bool] = []
    for line, flags, break_is_code in _lines(chars, code):
        if break_is_code:
            while line and flags[-1] and line[-1].isspace():
                line, flags = line[:-1], flags[:-1]
            if not line and (not out_code or out_code[-1]):
                continue  # A blank line between code lines.
        out_chars.extend([*line, "\n"])
        out_code.extend([*flags, break_is_code])
    while out_chars and out_code[-1] and (out_chars[-1].isspace() or out_chars[-1] == ";"):
        out_chars.pop()
        out_code.pop()
    start = 0
    while start < len(out_chars) and out_code[start] and out_chars[start].isspace():
        start += 1
    return out_chars[start:], out_code[start:]


def _misread(line: str) -> bool:
    """Whether the runner's extractors misread ``line`` as a line of a query block."""
    lowered = line.strip().lower()
    return line.startswith(("+", "-", "@@")) or lowered.startswith(_DROPPED_PREFIXES)


def _join_misread_lines(chars: list[str], code: list[bool]) -> str:
    """Join each misread line onto the line before it, with one space for the line break."""
    lines: list[str] = []
    for line_chars, flags, _ in _lines(chars, code):
        line = "".join(line_chars)
        if lines and _misread(line):
            if flags and not flags[0]:
                raise QueryFormatError(
                    "a line inside a literal begins with text the official runner misreads"
                )
            lines[-1] = f"{lines[-1]} {line.lstrip()}"
        else:
            lines.append(line)
    return "\n".join(lines)


def _check_extractable(query: str) -> None:
    """Raise QueryFormatError unless both of the runner's modes would extract ``query`` intact."""
    if any(hazard in query for hazard in _DIFF_HAZARDS):
        raise QueryFormatError("the query holds ``` or text the official runner reads as a diff")
    if _REASONING_TAG.search(query):
        raise QueryFormatError("the query holds a reasoning tag, which the official runner drops")
    if _FINAL_MARKER.search(query):
        raise QueryFormatError(
            "the query holds a phrase such as 'final query', the official runner's answer marker"
        )
