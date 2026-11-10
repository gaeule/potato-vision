"""Snapshot and verify pre-existing workspace files."""

import argparse
import sys
from enum import StrEnum
from typing import Final

from public_label_rework.common import OUTPUTS, ROOT, sha256, write_json
from retraining_data.schema import DataError, document, require, text

SNAPSHOT: Final = OUTPUTS / "protected_before.json"
EXCLUDED: Final = frozenset(
    {".omo", ".ruff_cache", "__pycache__", "public_label_rework"}
)


class Command(StrEnum):
    """Protect CLI commands."""

    SNAPSHOT = "snapshot"
    VERIFY = "verify"


def workspace_hashes() -> dict[str, str]:
    """Hash regular files outside excluded workspace subtrees."""
    return {
        str(path.relative_to(ROOT)): sha256(path)
        for path in sorted(ROOT.rglob("*"))
        if path.is_file()
        and not any(part in EXCLUDED for part in path.relative_to(ROOT).parts)
    }


def snapshot() -> None:
    """Write the immutable-before-intake workspace inventory once."""
    require(
        not SNAPSHOT.exists() and not SNAPSHOT.is_symlink(),
        "Snapshot already exists",
    )
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    hashes = workspace_hashes()
    write_json(SNAPSHOT, hashes)
    print(f"PROTECTED {len(hashes)} files")


def verify() -> None:
    """Report changed, missing, or newly added pre-existing workspace files."""
    before = {
        name: text(value) for name, value in document(SNAPSHOT.read_bytes()).items()
    }
    current = workspace_hashes()
    changed = sorted(
        name for name, digest in before.items() if current.get(name) != digest
    )
    changed.extend(sorted(set(current) - set(before)))
    print(f"CHECKED {len(before)} files; CHANGED {changed}")
    require(not changed, "Protected workspace files changed")


def main() -> None:
    """Run snapshot or verification command."""
    parser = argparse.ArgumentParser()
    _ = parser.add_argument("command", choices=("snapshot", "verify"))
    _ = parser.parse_args()
    command = sys.argv[1]
    match command:
        case "snapshot":
            snapshot()
        case "verify":
            verify()
        case other:
            raise DataError(f"Invalid protect command: {other}")


if __name__ == "__main__":
    try:
        main()
    except (DataError, OSError, ValueError) as error:
        print(f"ERROR {error}", file=sys.stderr)
        raise SystemExit(1) from error
