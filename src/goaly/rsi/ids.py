"""The IDs goaly.rsi gives experiments, hypotheses and lessons.

An ID matches ID_PATTERN: 1 to 64 letters, digits, ``_`` or ``-``, starting with a letter or a
digit. IDs name branches (``rsi/<lane>/<experiment_id>``), worktree and run directories, commit
trailers and container labels, so nothing in one ever needs quoting.
"""

import re

__all__ = ["ID_PATTERN", "check_id"]

ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}")


def check_id(value: str, field: str) -> None:
    """Raise ValueError, naming ``field``, unless ``value`` matches ID_PATTERN."""
    if not ID_PATTERN.fullmatch(value):
        raise ValueError(
            f"{field} must be 1-64 letters, digits, '_' or '-', starting with a letter or a "
            f"digit; got {value!r}"
        )
