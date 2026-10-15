"""A reimplementation of how the official Effi-SQL runner extracts the predicted SQL (spec §3.3).

The runner, GitHub ``L-nshpp/Diff-SQL``'s ``eval_execution_runner.py``, has no license, so this
follows its rules rather than its code. It has two modes:

- :func:`extract_patch`, the default ``EVAL_SQL_MODE=patch``: apply the first diff-like fenced
  block to ``base_sql``.
- :func:`extract_end2end`, ``EVAL_SQL_MODE=end2end``: score every SQL candidate, fenced blocks
  that are not diffs and the text after the last final-answer marker, and take the best.

Both use sqlparse at the runner's pinned version, 0.5.2, as the runner does.
"""

import re
from dataclasses import dataclass

import sqlparse

__all__ = ["FINAL_SQL_MARKER", "extract_end2end", "extract_patch", "looks_like_diff"]

_FENCE = re.compile(r"```([^\n`]*)\n(.*?)```", re.DOTALL)
_LOOSE_FENCE = re.compile(r"```([a-zA-Z0-9_]*)\s+(.*?)```", re.DOTALL)
_SQL_WORD = re.compile(r"\b(with|select)\b", re.IGNORECASE)
FINAL_SQL_MARKER = re.compile(
    r"(final\s+(?:answer|sql|query)|optimized\s+(?:sql|query)|optimization\s+sql|"
    r"rewritten\s+(?:sql|query)|改写后|优化后|最终\s*(?:sql|查询))",
    re.IGNORECASE,
)
_ORIGINAL_SQL_MARKER = re.compile(
    r"(original\s+(?:sql|query)|slow\s+(?:sql|query)|input\s+(?:sql|query)|"
    r"given\s+(?:sql|query)|原始\s*(?:sql|查询)|慢\s*(?:sql|查询))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _Block:
    lang: str
    body: str


def looks_like_diff(text: str) -> bool:
    """The runner's diff test: a git header, a ---/+++/@@ line, or two or more +/- lines."""
    lines = text.replace("\r\n", "\n").splitlines()
    if not lines:
        return False
    if text.lstrip().startswith("diff --git"):
        return True
    if any(line.startswith(("---", "+++", "@@")) for line in lines):
        return True
    return sum(1 for line in lines if line.startswith(("+", "-"))) >= 2


# Patch mode.


def extract_patch(base_sql: str, prediction: str) -> str | None:
    """Return the SQL patch mode grades, or None for "Patch Apply Failed"."""
    if not base_sql or not prediction:
        return None
    text = _unwrap_outer_fence(re.sub(r"</?think>", "", prediction, flags=re.IGNORECASE).strip())
    for block in _diff_candidates(text):
        if not block.body:
            continue
        if "diff" in block.lang or looks_like_diff(block.body):
            diff = _clean_diff(block.body)
            return _apply(base_sql, diff) if diff else None
    return None


def _unwrap_outer_fence(text: str) -> str:
    text = re.sub(r"```\s*diff\s*```\s*diff", "```diff\n", text.strip(), flags=re.IGNORECASE)
    outer = re.fullmatch(r"\s*```[ \t]*\n(.*)\n```[ \t]*\s*", text, flags=re.DOTALL)
    if outer:
        inner = outer.group(1).strip()
        lowered = inner.lower()
        if (
            "```diff" in lowered
            or "diff --git" in lowered
            or re.search(r"^--- .*?\n\+\+\+ .*", inner, flags=re.MULTILINE)
        ):
            return inner
    return text


def _wanted(block: _Block) -> bool:
    return bool(block.body) and (
        "sql" in block.lang or "diff" in block.lang or looks_like_diff(block.body)
    )


def _diff_candidates(text: str) -> list[_Block]:
    for fence in (_FENCE, _LOOSE_FENCE):
        blocks = [
            _Block(m.group(1).strip().lower(), m.group(2).strip()) for m in fence.finditer(text)
        ]
        wanted = [block for block in blocks if _wanted(block)]
        if wanted:
            return wanted
    raw = text.replace("\r\n", "\n")
    for pattern in (r"(?ms)^diff --git .*", r"(?ms)^--- .*\n\+\+\+ .*\n@@ .*"):
        match = re.search(pattern, raw)
        if match:
            return [_Block("diff", raw[match.start() :].strip())]
    return [_Block("diff", raw.strip())] if looks_like_diff(raw) else []


def _clean_diff(text: str) -> str | None:
    lines = text.replace("\r\n", "\n").strip().splitlines()
    cleaned = "\n".join(line for line in lines if not line.strip().startswith("```")).strip()
    if not cleaned:
        return None
    cleaned = re.sub(r"(?m)^diff\s+---\s+", "--- ", cleaned)
    cleaned = re.sub(r"(?m)(--- [^\n]+?)\s+\+\+\+\s+", r"\1\n++++ ", cleaned)
    cleaned = re.sub(r"(?m)(--- [^\n]+?)\s+\+\+\s+", r"\1\n+++ ", cleaned)
    cleaned = re.sub(r"(?m)(\+\+\+ [^\n]+?)\s+(@@)", r"\1\n\2", cleaned)
    starts = [
        match.start()
        for pattern in (r"(?m)^diff --git\b", r"(?m)^---\s+", r"(?m)^@@\s+")
        if (match := re.search(pattern, cleaned))
    ]
    if starts:
        cleaned = cleaned[min(starts) :].strip()
    return cleaned or None


def _apply(base_sql: str, diff: str) -> str | None:
    """Apply ``diff`` the runner's way: each ``-`` drops the next base line, unread."""
    base = base_sql.replace("\r\n", "\n").splitlines()
    out: list[str] = []
    next_base = 0
    in_hunk = False
    for line in diff.replace("\r\n", "\n").splitlines():
        if line.startswith(("diff --git", "index ", "---", "+++")):
            continue
        if line.startswith("@@"):
            in_hunk = True
        elif line.startswith(" "):
            if next_base < len(base):
                out.append(base[next_base])
                next_base += 1
        elif line.startswith("-"):
            next_base = min(next_base + 1, len(base))
        elif line.startswith("+"):
            out.append(line[1:])
        elif not in_hunk:
            out.append(line)
    out.extend(base[next_base:])
    return "\n".join(out).strip() or None


# end2end mode.


@dataclass(frozen=True)
class _Candidate:
    sql: str
    start: int
    end: int
    source: str


def extract_end2end(base_sql: str, prediction: str) -> str | None:
    """Return the SQL end2end mode grades (with ``base_sql`` given), or None for "SQL Extract
    Failed"."""
    if not prediction:
        return None
    text = prediction.replace("\r\n", "\n").strip()
    candidates = _sql_candidates(text)
    if candidates:
        best = max(candidates, key=lambda c: _score(c, text, base_sql))
        if best.sql:
            return best.sql
    no_reasoning = text
    for tag in ("think", "reasoning"):
        no_reasoning = re.sub(
            rf"<{tag}\b[^>]*>.*?</{tag}>", "\n", no_reasoning, flags=re.IGNORECASE | re.DOTALL
        )
    no_reasoning = no_reasoning.strip()
    if no_reasoning and no_reasoning != text:
        sql = _sanitize(no_reasoning)
        if sql and not (base_sql and _same_sql(sql, base_sql)):
            return sql
    return _sanitize(text)


def _sql_candidates(text: str) -> list[_Candidate]:
    candidates: list[_Candidate] = []
    for match in _FENCE.finditer(text):
        lang = match.group(1).strip().lower()
        body = match.group(2).strip()
        if not body or "diff" in lang or looks_like_diff(body):
            continue
        if "sql" in lang or _SQL_WORD.search(body):
            sql = _sanitize(body)
            if sql and _plausible(sql):
                source = "fenced_sql" if "sql" in lang else "fenced_text"
                candidates.append(_Candidate(sql, match.start(), match.end(), source))
    markers = list(FINAL_SQL_MARKER.finditer(text))
    if markers:
        start = markers[-1].end()
        sql = _sanitize(text[start:].strip())
        if sql and _plausible(sql):
            candidates.append(_Candidate(sql, start, len(text), "final_tail"))
    return candidates


def _sanitize(text: str) -> str | None:
    raw = text.replace("\r\n", "\n").strip()
    if not raw:
        return None
    kept = []
    for line in raw.splitlines():
        stripped = line.strip()
        lowered = stripped.lower()
        if stripped and (
            stripped.startswith(("```", "###")) or lowered.startswith(("diagnosis", "analysis"))
        ):
            continue
        kept.append(line)
    cleaned = "\n".join(kept).strip()
    if not cleaned:
        return None
    for statement in _split(cleaned):
        body = statement.strip().rstrip(";")
        if body and body.lower().startswith(("with ", "select ")):
            return body + ";"
    match = _SQL_WORD.search(cleaned)
    if not match:
        return None
    tail = cleaned[match.start() :].strip()
    statements = _split(tail)
    if not statements:
        return tail if tail.endswith(";") else tail + ";"
    body = statements[0].strip().rstrip(";")
    return body + ";" if body else None


def _split(text: str) -> list[str]:
    return [str(statement) for statement in sqlparse.split(text)]


def _plausible(sql: str) -> bool:
    collapsed = re.sub(r"\s+", " ", sql.strip()).strip().rstrip(";")
    lowered = collapsed.lower()
    if lowered.startswith("select "):
        return " from " in lowered or bool(
            re.search(r"^select\s+(\d+|'[^']*'|\"[^\"]*\"|true|false|null)\b", lowered)
        )
    if lowered.startswith("with "):
        cte = r"\bwith\s+(?:recursive\s+)?[a-zA-Z_\"`][\w\"`$]*\s*(?:\([^)]*\))?\s+as\s*\("
        return bool(
            re.search(cte, collapsed, re.IGNORECASE)
            and re.search(r"\)\s*select\b|\bselect\b", collapsed, re.IGNORECASE)
        )
    return False


def _score(candidate: _Candidate, text: str, base_sql: str) -> float:
    score = {"final_tail": 40.0, "fenced_sql": 20.0}.get(candidate.source, 10.0)
    score += 20.0 * candidate.start / max(len(text), 1)
    nearby = text[max(0, candidate.start - 600) : candidate.start] + "\n"
    nearby += text[candidate.end : candidate.end + 200]
    if FINAL_SQL_MARKER.search(nearby):
        score += 35
    if _ORIGINAL_SQL_MARKER.search(nearby):
        score -= 35
    lowered = candidate.sql.lower()
    if re.search(r"\b(create|alter|drop|insert|update|delete)\b", lowered):
        score -= 15
    if lowered.startswith(("select", "with")):
        score += 10
    if base_sql and _same_sql(candidate.sql, base_sql):
        score -= 80
    return score


def _same_sql(a: str, b: str) -> bool:
    return bool(a and b) and _normalized(a) == _normalized(b)


def _normalized(sql: str) -> str:
    text = str(sqlparse.format(sql, keyword_case="lower", strip_comments=True, reindent=False))
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    text = re.sub(r"\s*,\s*", ",", text)
    return text.rstrip(";")
