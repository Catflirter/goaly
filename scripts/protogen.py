"""Generate src/goaly/contracts/gen/ from the protobuf contracts in proto/ (spec §3.3).

The protoc bundled with the locked grpcio-tools compiles every proto/goaly/v1/*.proto into a
message module, <name>_pb2.py, and through the mypy-protobuf plugin into its type stub,
<name>_pb2.pyi. protoc names our modules after their .proto paths (goaly.v1.task_pb2), but they
live in goaly.contracts.gen, so the generator rewrites those module paths and nothing else.

    python scripts/protogen.py          # make proto: bring the generated files up to date
    python scripts/protogen.py --check  # make proto-check: exit 1 if they are out of date

The generator owns the *_pb2.py and *_pb2.pyi files in the output directory. It writes them,
deletes those that no proto generates any more, and leaves every other file alone.
"""

import argparse
import re
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGE = "goaly.v1"  # protoc names modules by path: goaly/v1/x.proto is goaly.v1.x_pb2.
TARGET_PACKAGE = "goaly.contracts.gen"
OWNED_SUFFIXES = ("_pb2.py", "_pb2.pyi")

# Our module paths, in the two forms protoc and mypy-protobuf write them: "from goaly.v1 import
# task_pb2", and "goaly.v1.task_pb2", the module name the generated classes report. Neither form
# matches what the serialized descriptors hold, such as ".goaly.v1.TaskInput" (a type name) or
# "goaly/v1/task.proto" (a file name), so the descriptors stay exactly as protoc wrote them.
_OWN_MODULE_PATH = re.compile(rf"(?<![\w.]){re.escape(SOURCE_PACKAGE)}(?=(?: import |\.)\w+_pb2\b)")

type Difference = Literal["missing", "extra", "changed"]


class ProtogenError(Exception):
    """The protos cannot be compiled; the message says why."""


def generate(proto_dir: Path) -> dict[str, bytes]:
    """Compile every .proto under ``proto_dir`` and return the generated files by name."""
    protos = _find_protos(proto_dir)
    with tempfile.TemporaryDirectory(prefix="protogen-") as scratch:
        _run_protoc(proto_dir, protos, Path(scratch))
        package_dir = Path(scratch, *SOURCE_PACKAGE.split("."))
        return {
            path.name: _relocate(path.read_bytes().decode("utf-8")).encode("utf-8")
            for path in sorted(package_dir.iterdir())
        }


def compare(out_dir: Path, generated: Mapping[str, bytes]) -> dict[str, Difference]:
    """Say how each file the generator owns in ``out_dir`` differs from ``generated``."""
    owned: set[str] = set()
    if out_dir.is_dir():
        owned = {path.name for path in out_dir.iterdir() if path.name.endswith(OWNED_SUFFIXES)}
    differences: dict[str, Difference] = {}
    for name in sorted(owned.union(generated)):
        if name not in owned:
            differences[name] = "missing"
        elif name not in generated:
            differences[name] = "extra"
        elif (out_dir / name).read_bytes() != generated[name]:
            differences[name] = "changed"
    return differences


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the Python code for proto/goaly/v1/.")
    parser.add_argument(
        "--check", action="store_true", help="change nothing; exit 1 if --out-dir is out of date"
    )
    parser.add_argument(
        "--proto-dir",
        type=Path,
        default=REPO_ROOT / "proto",
        help="the include root that holds goaly/v1/*.proto (default: proto)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "src" / "goaly" / "contracts" / "gen",
        help="where the generated files go (default: src/goaly/contracts/gen)",
    )
    args = parser.parse_args(argv)
    proto_dir: Path = args.proto_dir
    out_dir: Path = args.out_dir

    try:
        generated = generate(proto_dir)
    except ProtogenError as error:
        sys.stderr.write(f"protogen: {error}\n")
        return 1
    differences = compare(out_dir, generated)

    if args.check:
        if not differences:
            sys.stdout.write(f"protogen: {_shown(out_dir)} is up to date.\n")
            return 0
        listed = "".join(f"  {kind + ':':<8} {name}\n" for name, kind in differences.items())
        sys.stderr.write(
            f"protogen: {_shown(out_dir)} is out of date with {_shown(proto_dir)}:\n{listed}"
            "Run `make proto` to regenerate it.\n"
        )
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, kind in differences.items():
        path = out_dir / name
        if kind == "extra":
            path.unlink()
            sys.stdout.write(f"protogen: removed {_shown(path)}\n")
        else:
            path.write_bytes(generated[name])
            sys.stdout.write(f"protogen: wrote {_shown(path)}\n")
    if not differences:
        sys.stdout.write(f"protogen: {_shown(out_dir)} is already up to date.\n")
    return 0


def _find_protos(proto_dir: Path) -> list[str]:
    """Return the paths of the .proto files under ``proto_dir``, relative to it."""
    protos = sorted(path.relative_to(proto_dir) for path in proto_dir.rglob("*.proto"))
    if not protos:
        raise ProtogenError(f"no .proto files under {_shown(proto_dir)}")
    # Only module paths in the source package are rewritten, and all of them go to one package.
    package_dir = Path(*SOURCE_PACKAGE.split("."))
    outside = [proto.as_posix() for proto in protos if proto.parent != package_dir]
    if outside:
        raise ProtogenError(f"every .proto must be in {package_dir}/: {', '.join(outside)}")
    return [proto.as_posix() for proto in protos]


def _run_protoc(proto_dir: Path, protos: Sequence[str], out_dir: Path) -> None:
    """Compile ``protos`` into ``out_dir``. protoc prints its own errors and warnings."""
    command = [
        # Run as a module, grpc_tools.protoc adds the include path of the well-known types.
        sys.executable,
        "-m",
        "grpc_tools.protoc",
        f"--proto_path={proto_dir}",
        f"--plugin=protoc-gen-mypy={Path(sysconfig.get_path('scripts'), 'protoc-gen-mypy')}",
        f"--python_out={out_dir}",
        f"--mypy_out={out_dir}",
        "--mypy_opt=quiet",
        "--fatal_warnings",
        *protos,
    ]
    # The command holds fixed flags and the paths of files found on disk, not untrusted input.
    completed = subprocess.run(command, check=False)  # noqa: S603
    if completed.returncode != 0:
        raise ProtogenError(f"protoc failed with exit status {completed.returncode}")


def _relocate(source: str) -> str:
    """Rewrite our generated modules' paths from goaly.v1 to goaly.contracts.gen."""
    return _OWN_MODULE_PATH.sub(TARGET_PACKAGE, source)


def _shown(path: Path) -> str:
    """Show ``path`` relative to the repository root when it is inside it."""
    resolved = path.resolve()
    return str(resolved.relative_to(REPO_ROOT) if resolved.is_relative_to(REPO_ROOT) else path)


if __name__ == "__main__":
    sys.exit(main())
