"""Review-sheet renderers (triage, proposal pages, before/after) on plain PIL images and box lists.

Grid convention of the proposal pages: the image is split into 8x8 cells; rows are the
letters A-H from top to bottom, columns the digits 1-8 from left to right, so A1 is the
top-left cell and H8 the bottom-right one. Every produced image is at most 1600 px on its
long side (the size an agent reviewer sees without downscaling): proposal_sheet returns
page 1 (main image, numbered boxes, grid) followed by paginated 3x crop pages (old vs
refined per box, capped at 360 px per crop) and optional cell-zoom pages. Text uses Arial
when present, which has no Hangul glyphs: flag text should be ASCII.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from PIL import Image, ImageDraw

from public_label_rework.render_pages import (
    BACKGROUND,
    COLORS,
    GAP,
    MAIN_SIDE,
    MARGIN,
    PAGE_MAX,
    PAGE_WIDTH,
    ROWS,
    TEXT,
    WHITE,
    Box,
    Item,
    Rect,
    box_tile,
    cell_tile,
    check_boxes,
    font,
    paginate,
    tile,
    view_of,
    wrap,
)
from retraining_data.schema import require

TRIAGE_TILE = 480
PAIR_SIDE = 780


@dataclass(frozen=True, slots=True)
class TriageItem:
    image_id: str
    image: Image.Image
    boxes: tuple[Box, ...]
    flag: str


def _cell_labels(size: tuple[int, int]) -> list[tuple[str, tuple[int, int], Rect]]:
    f = font(16)
    labels: list[tuple[str, tuple[int, int], Rect]] = []
    for r in range(8):
        for c in range(8):
            text = f"{ROWS[r]}{c + 1}"
            x, y = (c + 1) * size[0] // 8 - 4, (r + 1) * size[1] // 8 - 4
            labels.append(
                (text, (x, y), (x - f.getlength(text) - 4, y - 20, x + 4, y + 4))
            )
    return labels


def _grid(view: Image.Image) -> Image.Image:
    overlay = Image.new("RGBA", view.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for k in range(1, 8):
        draw.line(
            [(k * view.width // 8, 0), (k * view.width // 8, view.height)],
            fill=(255, 255, 255, 120),
        )
        draw.line(
            [(0, k * view.height // 8), (view.width, k * view.height // 8)],
            fill=(255, 255, 255, 120),
        )
    for text, point, _ in _cell_labels(view.size):
        draw.text(
            point,
            text,
            font=font(16),
            fill=WHITE,
            anchor="rd",
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )
    return Image.alpha_composite(view.convert("RGBA"), overlay).convert("RGB")


def _main_page(rgb: Image.Image, items: Sequence[Item], scale: float) -> Image.Image:
    width, height = rgb.size
    size = (round(width * scale), round(height * scale))
    avoid = [rect for _, _, rect in _cell_labels(size)]
    main = _grid(view_of(rgb, (0, 0, width, height), scale, items, 3, avoid))
    page = Image.new("RGB", (PAGE_WIDTH, MARGIN + main.height + GAP), BACKGROUND)
    left = MARGIN + (MAIN_SIDE - main.width) // 2
    page.paste(main, (left, MARGIN))
    draw = ImageDraw.Draw(page)
    f = font(18)
    for k in range(8):
        draw.text(
            (left + (2 * k + 1) * main.width // 16, MARGIN // 2),
            str(k + 1),
            font=f,
            fill=WHITE,
            anchor="mm",
        )
        draw.text(
            (MARGIN // 2, MARGIN + (2 * k + 1) * main.height // 16),
            ROWS[k],
            font=f,
            fill=WHITE,
            anchor="mm",
        )
    return page


def proposal_sheet(
    image: Image.Image, boxes: Sequence[Box], cells_to_zoom: Sequence[str] = ()
) -> list[Image.Image]:
    """Pages: numbered colour-coded boxes with an A1..H8 grid, then crop and cell pages."""
    rgb = image.convert("RGB")
    require(len(boxes) > 0, "Proposal sheet needs at least one box")
    check_boxes(rgb, boxes)
    items: list[Item] = [
        (str(b.id), b.xyxy, COLORS[b.status], TEXT[b.status]) for b in boxes
    ]
    pages = [
        _main_page(rgb, items, MAIN_SIDE / max(rgb.size)),
        *paginate("Box crops 3x (old | refined)", [box_tile(rgb, b) for b in boxes]),
        *paginate("Cell crops 3x", [cell_tile(rgb, c, items) for c in cells_to_zoom]),
    ]
    require(all(max(p.size) <= PAGE_MAX for p in pages), f"Page exceeds {PAGE_MAX}px")
    return pages


def triage_sheet(items: Sequence[TriageItem]) -> Image.Image:
    """3x2 sheet of exactly 6 images, each tile 480 px wide with id, box count and flag text."""
    require(len(items) == 6, f"Triage sheet needs exactly 6 items, got {len(items)}")
    f = font(18)
    tile_h = 30 + TRIAGE_TILE + 48
    sheet = Image.new(
        "RGB", (3 * TRIAGE_TILE + 4 * GAP, 2 * tile_h + 3 * GAP), BACKGROUND
    )
    draw = ImageDraw.Draw(sheet)
    for index, item in enumerate(items):
        rgb = item.image.convert("RGB")
        check_boxes(rgb, item.boxes)
        ox, oy = (
            GAP + index % 3 * (TRIAGE_TILE + GAP),
            GAP + index // 3 * (tile_h + GAP),
        )
        view = view_of(
            rgb,
            (0, 0, *rgb.size),
            TRIAGE_TILE / max(rgb.size),
            [(None, b.xyxy, COLORS["existing"], WHITE) for b in item.boxes],
            2,
        )
        sheet.paste(view, (ox, oy + 30 + (TRIAGE_TILE - view.height) // 2))
        draw.text(
            (ox + 2, oy + 4),
            f"{item.image_id}  boxes={len(item.boxes)}",
            font=f,
            fill=WHITE,
        )
        for line, text in enumerate(wrap(item.flag, f, TRIAGE_TILE - 4, 2)):
            draw.text(
                (ox + 2, oy + 30 + TRIAGE_TILE + 2 + line * 22),
                text,
                font=f,
                fill=(250, 220, 30),
            )
    return sheet


def before_after(
    image: Image.Image, before: Sequence[Box], after: Sequence[Box]
) -> Image.Image:
    """Side-by-side panels (each long side 780 px): before boxes in red, after in green."""
    rgb = image.convert("RGB")
    require(len(before) + len(after) > 0, "Before/after needs at least one box")
    check_boxes(rgb, before)
    check_boxes(rgb, after)
    scale = PAIR_SIDE / max(rgb.size)
    full = (0, 0, *rgb.size)
    panels = [
        view_of(rgb, full, scale, [(None, b.xyxy, color, WHITE) for b in boxes], 3)
        for boxes, color in ((before, COLORS["existing"]), (after, COLORS["refined"]))
    ]
    return tile(
        [f"before ({len(before)} boxes)", f"after ({len(after)} boxes)"], panels
    )
