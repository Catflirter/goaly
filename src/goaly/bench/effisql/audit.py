"""The constant-branch audit list: matched predictions a person should read (spec §4.1, §12).

The grader's hard-coding checks cannot flag a single-row constant gated on the data, such as

    SELECT CASE WHEN (SELECT count(*) FROM orders) > 4000000 THEN 12345.67 ELSE (<base>) END
    SELECT 12345.67 WHERE (SELECT count(*) FROM orders) = 4500000 UNION ALL <base> WHERE ...

On SF3 the constant comes back at once; on the alternate database the base runs and matches.
The project lead accepted this residual risk and asked the report to list every matched
prediction that looks like such a branch, so each is reviewed before the numbers are published.
:func:`audit_reasons` says why a prediction is listed. Its patterns are deliberately broad: a
false alarm costs a minute of reading, a miss an inflated score.
"""

import re

__all__ = ["AUDIT_SPEEDUP", "audit_reasons", "prediction_sql"]

AUDIT_SPEEDUP = 100.0  # A constant answer returns in microseconds, so τ this high is listed.

_LITERAL = r"(?:[-+]?\d+(?:\.\d+)?|'(?:[^']|'')*')(?:\s*::\s*\w+)?(?:\s+as\s+\w+)?"
_PATTERNS = (
    (
        "calls an introspection function",
        re.compile(
            r"\b(?:pg_\w+|inet_\w+|txid_\w+|current_schemas?|current_database|current_catalog"
            r"|current_setting|current_user|session_user|version)\s*\(",
            re.IGNORECASE,
        ),
    ),
    (
        "a CASE gated on a subquery",
        re.compile(r"\bwhen\s*(?:not\s+)?(?:\(\s*select\b|exists\b)", re.IGNORECASE),
    ),
    (
        "rows gated on a scalar subquery",
        re.compile(r"\b(?:where|having|and|or)\s*\(\s*select\b", re.IGNORECASE),
    ),
    (
        "a SELECT of literals only",
        re.compile(
            rf"\bselect\s+(?:{_LITERAL}\s*,\s*)*{_LITERAL}\s*(?:\bwhere\b|\bunion\b|\)|;|$)",
            re.IGNORECASE,
        ),
    ),
)
_SQL_BLOCK = re.compile(r"```sql\n(.*?)\n```", re.DOTALL)


def prediction_sql(prediction: str) -> str:
    """The query in an official ``prediction``: its sql block, or the whole text if it has none."""
    match = _SQL_BLOCK.search(prediction)
    return prediction if match is None else match.group(1)


def audit_reasons(sql: str, base_sql: str, speedup: float) -> list[str]:
    """Why a matched prediction with this query and speedup goes on the audit list, if at all.

    A pattern ``base_sql`` has already is no new branch, so it is left out: on the dataset, the
    patterns then list 3 of the 238 TPC-H golds.
    """
    reasons = [
        reason
        for reason, pattern in _PATTERNS
        if pattern.search(sql) and not pattern.search(base_sql)
    ]
    if speedup >= AUDIT_SPEEDUP:
        reasons.append(f"τ ≥ {AUDIT_SPEEDUP:g}")
    return reasons
