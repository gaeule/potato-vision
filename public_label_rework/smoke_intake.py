"""Fixture-driven intake checks using only temporary outputs."""

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path

from public_label_rework.common import OUTPUTS
from public_label_rework.intake import intake_from_zip
from retraining_data.schema import DataError, require


def make_fixture(root: Path, corrupt: bool = False) -> Path:
    """Create a tiny valid public dataset archive."""
    entries: dict[str, bytes] = {
        "manifest.json": json.dumps(
            {
                "records": [
                    {
                        "image": "images/public/public_original01_0001.jpg",
                        "label": "images/public/public_original01_0001.json",
                        "source_prefix": "original01",
                        "image_sha256": hashlib.sha256(
                            b"wrong" if corrupt else b"jpg"
                        ).hexdigest(),
                    }
                ],
                "public_sources": {
                    source: {
                        "url": "https://example.invalid/data",
                        "license": "CC BY 4.0",
                    }
                    for source in (
                        "original01",
                        "original02",
                        "original03",
                        "original06",
                        "original07",
                    )
                },
            }
        ).encode(),
        "images/public/public_original01_0001.jpg": b"jpg",
        "images/public/public_original01_0001.json": b'{"shapes":[{},{}]}',
    }
    for source in (
        "original01",
        "original02",
        "original03",
        "original06",
        "original07",
    ):
        entries[f"sources/{source}/README.roboflow.txt"] = (
            b"URL: https://example.invalid/data\nLicense: CC BY 4.0\n"
        )
        entries[f"sources/{source}/data.yaml"] = b""
    archive_path = root / "fixture.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        for name, content in entries.items():
            archive.writestr(name, content)
    return archive_path


def main() -> None:
    """Prove fixture intake and reject a corrupted image hash."""
    # Given: an isolated output root with one image and paired label.
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="smoke-intake-", dir=OUTPUTS) as directory:
        root = Path(directory)
        archive = make_fixture(root)
        outputs = root / "outputs"
        # When: intake validates a coherent image hash against the manifest.
        receipt = intake_from_zip(
            archive, outputs, protect=False, expected_images=1, expected_boxes=2
        )
        require(receipt["hash_mismatches"] == 0, "Valid fixture hash receipt")
        # Then: a corrupted manifest image hash is rejected at the hash boundary.
        bad_outputs = root / "bad-outputs"
        try:
            _ = intake_from_zip(
                make_fixture(root, corrupt=True),
                bad_outputs,
                protect=False,
                expected_images=1,
                expected_boxes=2,
            )
        except DataError as error:
            require(
                "hash mismatches" in str(error),
                "Corrupt fixture failed for another reason",
            )
        else:
            raise DataError("Corrupted image fixture was accepted")
    print("PASS intake fixture validation and corrupted image rejection")


if __name__ == "__main__":
    main()
