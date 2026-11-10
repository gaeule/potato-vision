"""Drawing primitives, crop tiles and bounded pagination shared by the review renderers."""

import io
import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final, Literal

from PIL import Image, ImageDraw, ImageFont

from retraining_data.schema import require

type Status = Literal["existing", "refined", "candidate"]
type Rect = tuple[float, float, float, float]
type Rgb = tuple[int, int, int]
type Font = ImageFont.FreeTypeFont | ImageFont.ImageFont
type Item = tuple[str | None, Rect, Rgb, Rgb]

FONT_PATH: Final = Path("/home/sc/.config/Ultralytics/Arial.ttf")
COLORS: Final[dict[Status, Rgb]] = {
    "existing": (230, 40, 40),
    "refined": (40, 200, 70),
    "candidate": (250, 220, 30),
}
TEXT: Final[dict[Status, Rgb]] = {
    "existing": (255, 255, 255),
    "refined": (0, 0, 0),
    "candidate": (0, 0, 0),
}
BACKGROUND: Final[Rgb] = (28, 28, 28)
WHITE: Final[Rgb] = (255, 255, 255)
MAIN_SIDE: Final = 1280
MARGIN: Final = 30
PAGE_WIDTH: Final = MAIN_SIDE + MARGIN
PAGE_MAX: Final = 1600
TITLE_HEIGHT: Final = 34
CAPTION_HEIGHT: Final = 28
CROP_SCALE: Final = 3
CROP_CAP: Final = 360
CELL_CAP: Final = 480
GAP: Final = 8
ROWS: Final = "ABCDEFGH"


@dataclass(frozen=True, slots=True)
class Box:
    """One xyxy box in image pixels; refined boxes may carry the box they replace."""

    id: int
    xyxy: Rect
    status: Status = "existing"
    old_xyxy: Rect | None = None


@cache
def font(size: int) -> Font:
    if FONT_PATH.exists():
        return ImageFont.truetype(str(FONT_PATH), size)
    return ImageFont.load_default(size)


def png_bytes(image: Image.Image) -> bytes:
    """Deterministic PNG encoding: same pixels give the same bytes."""
    buffer = io.BytesIO()
    image.save(buffer, "PNG", compress_level=6)
    return buffer.getvalue()


def check_boxes(image: Image.Image, boxes: Sequence[Box]) -> None:
    width, height = image.size
    seen = set[int]()
    for box in boxes:
        require(box.id not in seen, f"Duplicate box id {box.id}")
        seen.add(box.id)
        for rect in (box.xyxy, box.old_xyxy):
            if rect is None:
                continue
            x0, y0, x1, y1 = rect
            inside = (
                all(math.isfinite(v) for v in rect)
                and 0 <= x0 < x1 <= width
                and 0 <= y0 < y1 <= height
            )
            require(inside, f"Box {box.id} {rect} invalid or outside {width}x{height}")


def _chip(
    view: Image.Image,
    placed: list[Rect],
    item: tuple[str, Rect, Rgb, Rgb],
) -> None:
    label, spot, color, fg = item
    draw = ImageDraw.Draw(view)
    f = font(20)
    left, top, right, bottom = f.getbbox(label)
    cw, ch = right - left + 8, bottom - top + 6
    x0, y0, x1, y1 = spot
    anchors = [
        (x0, y0),
        (x1 - cw, y0),
        (x0, y1 - ch),
        (x1 - cw, y1 - ch),
        ((x0 + x1 - cw) / 2, (y0 + y1 - ch) / 2),
    ]
    anchors = [
        (max(0, min(x, view.width - cw - 1)), max(0, min(y, view.height - ch - 1)))
        for x, y in anchors
    ]
    free = [
        a
        for a in anchors
        if all(
            a[0] + cw <= p[0] or p[2] <= a[0] or a[1] + ch <= p[1] or p[3] <= a[1]
            for p in placed
        )
    ]
    cx, cy = (free or anchors)[0]
    placed.append((cx, cy, cx + cw, cy + ch))
    draw.rectangle((cx, cy, cx + cw, cy + ch), fill=color, outline=BACKGROUND)
    draw.text((cx + 4 - left, cy + 3 - top), label, font=f, fill=fg)


def view_of(
    image: Image.Image,
    region: tuple[int, int, int, int],
    scale: float,
    items: Sequence[Item],
    width: int,
    avoid: Sequence[Rect] = (),
) -> Image.Image:
    x0, y0, x1, y1 = region
    view = image.crop(region).resize(
        (max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale))),
        Image.Resampling.BICUBIC,
    )
    draw = ImageDraw.Draw(view)
    placed: list[Rect] = list(avoid)
    chips: list[tuple[str, Rect, Rgb, Rgb]] = []
    for label, (bx0, by0, bx1, by1), color, fg in items:
        if bx1 <= x0 or bx0 >= x1 or by1 <= y0 or by0 >= y1:
            continue
        spot = (
            (bx0 - x0) * scale,
            (by0 - y0) * scale,
            (bx1 - x0) * scale,
            (by1 - y0) * scale,
        )
        draw.rectangle(spot, outline=color, width=width)
        if label is not None:
            visible = (
                max(0, spot[0]),
                max(0, spot[1]),
                min(view.width - 1, spot[2]),
                min(view.height - 1, spot[3]),
            )
            chips.append((label, visible, color, fg))
    # Crop pixels first, then label visible geometry after all outlines.
    for chip in chips:
        _chip(view, placed, chip)
    return view


def wrap(text: str, f: Font, limit: int, lines: int) -> list[str]:
    out: list[str] = [""]
    for word in text.split():
        trial = f"{out[-1]} {word}".strip()
        if f.getlength(trial) <= limit or not out[-1]:
            out[-1] = trial
        else:
            out.append(word)
    if len(out) > lines:
        out = [*out[: lines - 1], out[lines - 1] + "..."]
    return out


def tile(captions: Sequence[str], panels: Sequence[Image.Image]) -> Image.Image:
    f = font(20)
    result = Image.new(
        "RGB",
        (
            sum(p.width for p in panels) + GAP // 2 * (len(panels) - 1),
            CAPTION_HEIGHT + max(p.height for p in panels),
        ),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(result)
    x = 0
    for caption, panel in zip(captions, panels, strict=True):
        draw.text((x + 2, 3), caption, font=f, fill=WHITE)
        result.paste(panel, (x, CAPTION_HEIGHT))
        x += panel.width + GAP // 2
    return result


def _fit(
    rect: Rect, size: tuple[int, int], cap: int
) -> tuple[tuple[int, int, int, int], float]:
    x0, y0, x1, y1 = rect
    pad = max(8.0, 0.15 * max(x1 - x0, y1 - y0))
    region = (
        max(0, int(x0 - pad)),
        max(0, int(y0 - pad)),
        min(size[0], math.ceil(x1 + pad)),
        min(size[1], math.ceil(y1 + pad)),
    )
    return region, min(
        float(CROP_SCALE), cap / max(region[2] - region[0], region[3] - region[1])
    )


def box_tile(image: Image.Image, box: Box) -> Image.Image:
    region, scale = _fit(box.xyxy, image.size, CROP_CAP)
    new = view_of(
        image, region, scale, [(None, box.xyxy, COLORS[box.status], WHITE)], 2
    )
    if box.old_xyxy is None:
        return tile([f"#{box.id} {box.status}"], [new])
    old = view_of(
        image, region, scale, [(None, box.old_xyxy, COLORS["existing"], WHITE)], 2
    )
    return tile([f"#{box.id} old", f"#{box.id} {box.status}"], [old, new])


def cell_tile(image: Image.Image, cell: str, items: Sequence[Item]) -> Image.Image:
    require(
        len(cell) == 2 and cell[0] in ROWS and cell[1] in "12345678",
        f"Bad grid cell {cell!r}",
    )
    width, height = image.size
    row, col = ROWS.index(cell[0]), int(cell[1]) - 1
    region = (
        col * width // 8,
        row * height // 8,
        (col + 1) * width // 8,
        (row + 1) * height // 8,
    )
    scale = min(
        float(CROP_SCALE), CELL_CAP / max(region[2] - region[0], region[3] - region[1])
    )
    return tile([f"cell {cell}"], [view_of(image, region, scale, items, 2)])


def paginate(title: str, tiles: Sequence[Image.Image]) -> list[Image.Image]:
    """Flow tiles into pages that are PAGE_WIDTH wide and at most PAGE_MAX tall."""
    pages: list[list[tuple[int, int, Image.Image]]] = [[]]
    x, y, row = 0, TITLE_HEIGHT, 0
    for t in tiles:
        if x and x + t.width > PAGE_WIDTH:
            x, y, row = 0, y + row + GAP, 0
        if pages[-1] and y + t.height > PAGE_MAX:
            pages.append([])
            x, y, row = 0, TITLE_HEIGHT, 0
        pages[-1].append((x, y, t))
        x, row = x + t.width + GAP, max(row, t.height)
    if not tiles:
        return []
    rendered: list[Image.Image] = []
    for number, placed in enumerate(pages, 1):
        page = Image.new(
            "RGB", (PAGE_WIDTH, max(py + t.height for _, py, t in placed)), BACKGROUND
        )
        draw = ImageDraw.Draw(page)
        draw.text(
            (2, 4), f"{title} (page {number}/{len(pages)})", font=font(22), fill=WHITE
        )
        for px, py, t in placed:
            page.paste(t, (px, py))
        rendered.append(page)
    return rendered
