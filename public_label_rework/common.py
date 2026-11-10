"""Paths and safe output helpers for public label rework."""

import hashlib
import json
from pathlib import Path
from typing import Final

from retraining_data.schema import require

ROOT: Final = Path(__file__).resolve().parents[1]
ZIP: Final = ROOT / "final-dataset.zip"
OUTPUTS: Final = ROOT / "public_label_rework/outputs"


def new_output(path: Path) -> Path:
    """Create an unused output path strictly below OUTPUTS."""
    output_root = OUTPUTS.resolve()
    require(OUTPUTS == output_root, "Outputs root must not be a symlink")
    absolute = path.absolute()
    require(absolute == path.resolve(), "Output symlinks are refused")
    require(
        absolute.is_relative_to(output_root) and absolute != output_root,
        f"Output must be below {OUTPUTS}",
    )
    absolute.mkdir(parents=True, exist_ok=False)
    return absolute


def sha256(path: Path) -> str:
    """Hash a file incrementally."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    """Write stable, human-readable UTF-8 JSON."""
    require(path.is_relative_to(OUTPUTS), f"JSON output must be below {OUTPUTS}")
    _ = path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
