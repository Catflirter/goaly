# The RSI sandbox image (spec §8.8): Python 3.12, git, ripgrep, uv and goaly's locked dependencies,
# the dev group included, so that a pack's tests can run in it.
#
# goaly.rsi.sandbox builds it as goaly-rsi-sandbox:<hash of uv.lock and this file>, from a context
# that holds only pyproject.toml and uv.lock. The worktree is mounted when a container starts,
# never copied in, and goaly itself is not installed: PYTHONPATH points at the mounted worktree's
# src/.
#
# git and ripgrep come from Debian unpinned, on purpose: pinned versions disappear from Debian's
# mirrors and would break the build. So two builds with one tag can differ by Debian's updates to
# those two tools, which run no code of the candidate's.

FROM ghcr.io/astral-sh/uv:0.13.1@sha256:4a1f3666816dafadd20ceb71eac9a7d56d7ef30279f7d64048519ab901ada336 AS uv

FROM python:3.12-slim@sha256:a6e34c598f2467ed0e9a8d349809fcd8b5c603269512df273a0bb1784edc11b1

COPY --from=uv /uv /uvx /usr/local/bin/

RUN apt-get update \
    && apt-get install --yes --no-install-recommends git ripgrep \
    && rm -rf /var/lib/apt/lists/*

ENV UV_PROJECT_ENVIRONMENT=/opt/venv \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1
WORKDIR /opt/goaly
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-install-project --no-cache

# Containers run with a read-only root filesystem, as the host's user, whom /etc/passwd does not
# know: /tmp, a tmpfs, is the home and the only writable place besides the lane's pack.
ENV PATH=/opt/venv/bin:$PATH \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    UV_CACHE_DIR=/tmp/.cache/uv \
    UV_FROZEN=1 \
    UV_NO_SYNC=1 \
    UV_OFFLINE=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    GIT_OPTIONAL_LOCKS=0 \
    LANG=C.UTF-8
WORKDIR /
