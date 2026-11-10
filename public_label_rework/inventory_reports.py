"""PIL contact-sheet rendering for read-only inventory evidence."""

import math
from pathlib import Path

from PIL import Image, ImageDraw

from retraining_data.schema import require


def sheet(tiles: list[tuple[str, Image.Image]], path: Path, columns: int = 3) -> None:
    """Bounded sheets: 480px image cells, titles outside image pixels."""
    require(0 < len(tiles) <= 18, "Sheet must have 1..18 tiles")
    canvas = Image.new(
        "RGB", (480 * columns, 510 * math.ceil(len(tiles) / columns)), "#222222"
    )
    draw = ImageDraw.Draw(canvas)
    for i, (title, image) in enumerate(tiles):
        x, y = i % columns * 480, i // columns * 510
        tile = image.copy()
        tile.thumbnail((480, 480))
        canvas.paste(tile, (x, y + 30))
        draw.text((x + 4, y + 6), title, fill="white")
    canvas.save(path, quality=92)
