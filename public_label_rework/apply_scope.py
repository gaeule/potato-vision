"""Frozen pilot/full authorization and source preflight for decision application."""

import hashlib
import json
from collections.abc import Callable
from datetime import date
from pathlib import Path
from typing import Final

from public_label_rework import decisions as d
from public_label_rework.common import OUTPUTS
from retraining_data.schema import DataError, Json, array, mapping, require, text

SOURCE: Final = OUTPUTS / "source/images/public"
PILOT: Final = OUTPUTS / "lists/pilot_list.json"
REMAINING: Final = OUTPUTS / "lists/remaining_list.json"
CONFIRMATION: Final = OUTPUTS / "pilot/user_confirmation.json"


def canonical(path: Path, expected: Path, detail: str) -> None:
    require(path.absolute() == expected.absolute() == path.resolve(), detail)


def plain_file(path: Path) -> None:
    require(
        path.absolute() == path.resolve() and path.is_file(),
        f"Missing or symlinked scope/source file: {path.name}",
    )


def json_value(data: bytes, *, decoder: Callable[[bytes], Json] = json.loads) -> Json:
    """Decode scope JSON once, using the repository's typed decoder boundary."""
    try:
        return decoder(data)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise DataError("Malformed scope JSON") from error


def members(value: Json) -> tuple[str, ...]:
    images = tuple(d.token(value, "image") for value in array(value))
    require(
        bool(images) and len(set(images)) == len(images),
        "Empty or duplicate image list",
    )
    return images


def frozen(path: Path, count: int) -> tuple[str, ...]:
    digest = path.with_suffix(".sha256")
    plain_file(path)
    plain_file(digest)
    raw = path.read_bytes()
    require(
        hashlib.sha256(raw).hexdigest() == digest.read_text(encoding="utf-8").strip(),
        f"Frozen list SHA-256 mismatch: {path.name}",
    )
    images = members(json_value(raw))
    require(len(images) == count, f"Frozen list must contain exactly {count} images")
    return images


def confirmed() -> None:
    """Only this project's canonical confirmation authorizes the full union."""
    plain_file(CONFIRMATION)
    row = mapping(json_value(CONFIRMATION.read_bytes()))
    require(
        {"date", "basis", "statement_verbatim"} <= row.keys(),
        "Confirmation requires date, basis and statement_verbatim",
    )
    value = text(row["date"])
    try:
        parsed_date = date.fromisoformat(value)
    except ValueError as error:
        raise DataError("Confirmation date must be an ISO calendar date") from error
    require(parsed_date.isoformat() == value, "Confirmation date must use YYYY-MM-DD")
    for key in ("basis", "statement_verbatim"):
        require(bool(text(row[key]).strip()), f"Confirmation {key} must be nonempty")


def source_membership(source: Path, images: tuple[str, ...], full: bool) -> None:
    """Check names and files before opening decision JSONL or any source label."""
    if full:
        expected = set(images)
        require(
            {p.stem for p in source.glob("*.jpg")} == expected
            and {p.stem for p in source.glob("*.json")} == expected,
            "Full source membership mismatch",
        )
    for image in images:
        for suffix in (".jpg", ".json"):
            path = source / (image + suffix)
            require(path.absolute() == path.resolve(), "Source symlink escape")
            require(path.is_file(), f"Missing source file: {path.name}")


def selected_images(inputs: d.ApplyInput) -> tuple[str, ...]:
    """Default to pilot100; full2060 requires both explicit lists and confirmation."""
    if inputs.fixture_gate is not None:
        root = inputs.fixture_gate.root.resolve()
        require(
            root == inputs.fixture_gate.root.absolute()
            and root.parent == OUTPUTS.resolve()
            and root.name.startswith("smoke-decisions-"),
            "Fixture gate requires an isolated temporary root",
        )
        canonical(
            inputs.source, root / "source", "Fixture gate cannot authorize real source"
        )
        canonical(
            inputs.image_list, root / "list.json", "Fixture gate requires its own list"
        )
        require(
            inputs.remaining_list is None, "Fixture gate cannot authorize full scope"
        )
        plain_file(inputs.image_list)
        images = members(json_value(inputs.image_list.read_bytes()))
    else:
        canonical(inputs.source, SOURCE, "Real runs require canonical public source")
        canonical(inputs.image_list, PILOT, "Apply requires the frozen pilot list")
        images = frozen(inputs.image_list, 100)
        if inputs.remaining_list is not None:
            canonical(
                inputs.remaining_list,
                REMAINING,
                "Apply requires canonical remaining list",
            )
            confirmed()
            remaining = frozen(inputs.remaining_list, 1960)
            require(
                set(images).isdisjoint(remaining), "Pilot and remaining lists overlap"
            )
            images += remaining
            require(len(set(images)) == 2060, "Full scope must have 2060 unique images")
    images = tuple(sorted(images))
    source_membership(inputs.source, images, inputs.remaining_list is not None)
    return images


def parse_args(args: list[str]) -> tuple[d.ApplyInput, Path]:
    """Preserve pilot CLI; an optional --remaining-list explicitly requests full."""
    remaining: Path | None = None
    arguments = args
    if "--remaining-list" in args:
        index = args.index("--remaining-list")
        require(
            args.count("--remaining-list") == 1 and index + 1 < len(args),
            "--remaining-list requires exactly one path",
        )
        remaining = Path(args[index + 1])
        arguments = args[:index] + args[index + 2 :]
    require(
        len(arguments) >= 5 and arguments[0] == "--list" and arguments[2] == "--output",
        "Usage: --list PILOT --output DIR [--remaining-list REMAINING] OPS...",
    )
    return (
        d.ApplyInput(
            SOURCE,
            Path(arguments[1]),
            tuple(Path(p) for p in arguments[4:]),
            remaining_list=remaining,
        ),
        Path(arguments[3]),
    )
