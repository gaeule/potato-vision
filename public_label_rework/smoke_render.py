"""CPU smoke for review renders: python -B -m public_label_rework.smoke_render."""

import io
import random
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Final
from unittest.mock import patch

from PIL import Image, ImageDraw

from public_label_rework.render import (
    TriageItem,
    before_after,
    proposal_sheet,
    triage_sheet,
)
from public_label_rework.render_pages import (
    CAPTION_HEIGHT,
    COLORS,
    PAGE_MAX,
    TEXT,
    Box,
    Font,
    Rect,
    Rgb,
    cell_tile,
    png_bytes,
)
from retraining_data.schema import DataError, array, document, mapping, number

ROOT: Final = Path(__file__).resolve().parent.parent
ARCHIVE: Final = ROOT / "final-dataset.zip"
REAL: Final = (
    "public_original06_0161",
    "public_original02_0857",
    "public_original06_0058",
)


@dataclass(frozen=True, slots=True)
class Fixture:
    name: str
    image: Image.Image
    existing: tuple[Box, ...]
    proposal: tuple[Box, ...]


def _shrunk(index: int, rect: tuple[float, float, float, float]) -> Box:
    x0, y0, x1, y1 = rect
    w, h = (x1 - x0) * 0.05, (y1 - y0) * 0.05
    return Box(index, (x0 + w, y0 + h, x1 - w, y1 - h), "refined", rect)


def _fixture(
    name: str, image: Image.Image, rects: list[tuple[float, float, float, float]]
) -> Fixture:
    existing = tuple(Box(i, r) for i, r in enumerate(rects, 1))
    proposal = [
        _shrunk(i, r) if i % 3 == 0 else Box(i, r) for i, r in enumerate(rects, 1)
    ]
    width, height = image.size
    proposal.append(
        Box(
            len(rects) + 1,
            (width * 0.4, height * 0.4, width * 0.5, height * 0.5),
            "candidate",
        )
    )
    return Fixture(name, image, existing, tuple(proposal))


def _real(archive: zipfile.ZipFile, name: str) -> Fixture:
    image = Image.open(io.BytesIO(archive.read(f"images/public/{name}.jpg")))
    _ = image.load()
    annotation = document(archive.read(f"images/public/{name}.json"))
    rects: list[tuple[float, float, float, float]] = []
    for shape in array(annotation["shapes"]):
        points = [array(p) for p in array(mapping(shape)["points"])]
        xs = [number(p[0]) for p in points]
        ys = [number(p[1]) for p in points]
        rects.append((min(xs), min(ys), max(xs), max(ys)))
    return _fixture(name, image, rects)


def _synthetic() -> Fixture:
    rng = random.Random(7)
    image = Image.new("RGB", (1000, 700), (90, 70, 50))
    draw = ImageDraw.Draw(image)
    rects: list[tuple[float, float, float, float]] = []
    for row in range(4):
        for col in range(6):
            cx, cy = (
                110 + col * 150 + rng.randint(-20, 20),
                100 + row * 165 + rng.randint(-20, 20),
            )
            rx, ry = rng.randint(55, 80), rng.randint(45, 65)
            draw.ellipse(
                (cx - rx, cy - ry, cx + rx, cy + ry),
                fill=(200, 170, 110),
                outline=(120, 90, 50),
            )
            rects.append(
                (float(cx - rx), float(cy - ry), float(cx + rx), float(cy + ry))
            )
    return _fixture("synthetic_24", image, rects)


def _write(output: Path, name: str, data: bytes) -> None:
    with (output / f"{name}.png").open("xb") as handle:
        _ = handle.write(data)


def _expect_error(action: str, call: Callable[[], list[Image.Image]]) -> None:
    try:
        _ = call()
    except DataError:
        return
    raise AssertionError(f"{action} did not raise DataError")


def _assert_bounded(name: str, image: Image.Image) -> None:
    assert max(image.size) <= PAGE_MAX, f"{name} is {image.size}, over {PAGE_MAX}px"


def _assert_edge_chips(fixture: Fixture) -> list[Image.Image]:
    """Observe real glyph bounds at D5 and all four edges without replacing drawing."""
    source = next(b for b in fixture.proposal if b.id == 13)
    cases: list[Rect] = [
        source.xyxy,
        (319, 260, 340, 280),
        (399, 260, 420, 280),
        (350, 239, 370, 260),
        (350, 319, 370, 340),
        (319, 239, 321, 241),
        (399, 239, 401, 241),
        (319, 319, 321, 321),
        (399, 319, 401, 321),
    ]
    panels: list[Image.Image] = []
    original = ImageDraw.ImageDraw.text
    observed: list[tuple[str, Rect, Rgb]] = []

    # The keyword parameters mirror the PIL text boundary being observed.
    def record(
        draw: ImageDraw.ImageDraw,
        xy: tuple[float, float],
        text: str,
        *,
        font: Font,
        fill: Rgb,
    ) -> None:
        observed.append((text, draw.textbbox(xy, text, font=font), fill))
        original(draw, xy, text, font=font, fill=fill)

    for status, color in COLORS.items():
        for rect in cases:
            # Given a real ID on an intersecting box and another ID outside D5.
            observed.clear()
            items = [
                (str(source.id), rect, color, TEXT[status]),
                (str(fixture.proposal[0].id), (0, 0, 10, 10), color, TEXT[status]),
            ]
            # When the real cell renderer draws the original image.
            with patch.object(ImageDraw.ImageDraw, "text", record):
                panel = cell_tile(fixture.image, "D5", items)
            # Then exactly the intersecting ID, its status and full padded glyph fit.
            chips = [
                (label, bounds, fg) for label, bounds, fg in observed if label.isdigit()
            ]
            assert len(chips) == 1, (status, rect, chips)
            label, (left, top, right, bottom), fg = chips[0]
            assert label == str(source.id) and fg == TEXT[status]
            assert 0 <= left - 4 and right + 4 < panel.width, (status, rect, chips)
            assert 0 <= top - 3 and bottom + 3 < panel.height - CAPTION_HEIGHT
            assert panel.getpixel((int(left - 2), int(top + CAPTION_HEIGHT))) == color
            panels.append(panel)
    return panels


def render_check(output: Path) -> None:
    """Check all renderers; persist QA only to a caller-owned new directory."""
    output.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(ARCHIVE) as archive:
        fixtures = [_real(archive, name) for name in REAL]
    fixtures.append(_synthetic())
    _ = _assert_edge_chips(fixtures[0])
    cells = ["A1", "D5", "H8"]
    for fixture in fixtures:
        # Given a fixture image, When each sheet is rendered twice, Then the PNG bytes are identical
        pages = proposal_sheet(fixture.image, fixture.proposal, cells)
        again = proposal_sheet(fixture.image, fixture.proposal, cells)
        assert len(pages) >= 2
        assert [png_bytes(p) for p in pages] == [png_bytes(p) for p in again]
        for index, page in enumerate(pages, 1):
            _assert_bounded(f"proposal_{fixture.name}_p{index}", page)
            _write(output, f"proposal_{fixture.name}_p{index}", png_bytes(page))
        pair = before_after(fixture.image, fixture.existing, fixture.proposal)
        assert png_bytes(pair) == png_bytes(
            before_after(fixture.image, fixture.existing, fixture.proposal)
        )
        _assert_bounded(f"before_after_{fixture.name}", pair)
        _write(output, f"before_after_{fixture.name}", png_bytes(pair))
    items = [
        TriageItem(f.name, f.image, f.existing, flag)
        for f, flag in zip(
            fixtures * 2,
            [
                "dense pile, many unlabelled",
                "rotated envelopes",
                "single box includes sprout",
                "21+ boxes",
                "dup A",
                "dup B",
            ],
            strict=False,
        )
    ]
    sheet = triage_sheet(items)
    assert png_bytes(sheet) == png_bytes(triage_sheet(items))
    _assert_bounded("triage_sheet", sheet)
    _write(output, "triage_sheet", png_bytes(sheet))
    first = fixtures[0]
    clash = (*first.proposal, Box(first.proposal[0].id, (1.0, 1.0, 9.0, 9.0)))
    outside = (Box(1, (0.0, 0.0, 9999.0, 9.0)),)
    _expect_error("id collision", lambda: proposal_sheet(first.image, clash))
    _expect_error("box outside image", lambda: proposal_sheet(first.image, outside))
    _expect_error("empty box list", lambda: proposal_sheet(first.image, ()))
    _expect_error(
        "bad grid cell", lambda: proposal_sheet(first.image, first.proposal, ["I9"])
    )
    print(f"PASS rendered {len(list(output.glob('*.png')))} files to {output}")


def main() -> None:
    """Each smoke owns its output directory and removes it even on failure."""
    with TemporaryDirectory(
        prefix="render-smoke-", dir=ROOT / "public_label_rework" / "outputs"
    ) as temporary:
        output = Path(temporary) / "renders"
        render_check(output)
    print(f"PASS automatic cleanup {temporary}")


if __name__ == "__main__":
    main()
