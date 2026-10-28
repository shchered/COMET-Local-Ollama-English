#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local deterministic "Entity board" image generation backend.

When cloud image models (DALL-E / Flux services, etc.) are unavailable, use Pillow to render a scene board
containing all entities to be migrated:

  * Each entity occupies its own panel with a unique color + a unique geometric shape;
  * Each panel carries an entity-name tag — the vision model uses it to map the "entity name"
    to an "image position", producing purely positional phrases such as "the device in the top-left of the image";
  * The entity name exists only in image pixels; on the text side it is replaced with a positional description
    by core/entity_migration.py, so the cross-modal entanglement mechanism still holds.

This backend does not require a GPU and produces reproducible results, making it suitable for local runs and ablation experiments.
For more realistic images closer to the paper, use ``IMAGE_GEN_BACKEND=draw_things`` instead.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFilter, ImageFont

# Color scheme for each entity (border + shape), ensuring pairwise distinguishability
_PALETTE: List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]] = [
    ((198, 63, 63), (245, 205, 205)),
    ((54, 116, 181), (206, 226, 245)),
    ((46, 139, 87), (205, 238, 215)),
    ((196, 137, 26), (248, 232, 190)),
    ((124, 76, 178), (228, 214, 244)),
    ((38, 148, 148), (201, 236, 236)),
    ((190, 90, 140), (247, 214, 229)),
    ((110, 110, 110), (223, 223, 223)),
]

_GLYPHS = ("circle", "square", "triangle", "diamond", "hexagon", "star", "cross", "ring")

_FONT_CANDIDATES = (
    "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
)


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in _FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _entity_name(entity: Any) -> str:
    """Extract a readable name from various entity structures."""
    if isinstance(entity, str):
        return entity
    if isinstance(entity, dict):
        for key in ("entity", "name", "canonical", "label"):
            value = entity.get(key)
            if value:
                return str(value)
    return str(entity)


def _draw_glyph(draw: ImageDraw.ImageDraw, kind: str, box: Tuple[int, int, int, int],
                 fill: Tuple[int, int, int], outline: Tuple[int, int, int]) -> None:
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
    r = min(x1 - x0, y1 - y0) // 2 - 4
    width = max(3, r // 6)

    if kind == "circle":
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline, width=width)
    elif kind == "square":
        draw.rectangle((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline, width=width)
    elif kind == "triangle":
        draw.polygon([(cx, cy - r), (cx + r, cy + r), (cx - r, cy + r)],
                     fill=fill, outline=outline)
    elif kind == "diamond":
        draw.polygon([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy)],
                     fill=fill, outline=outline)
    elif kind == "hexagon":
        import math
        pts = [(cx + r * math.cos(math.radians(60 * i - 30)),
                cy + r * math.sin(math.radians(60 * i - 30))) for i in range(6)]
        draw.polygon(pts, fill=fill, outline=outline)
    elif kind == "star":
        import math
        pts = []
        for i in range(10):
            radius = r if i % 2 == 0 else r * 0.45
            angle = math.radians(36 * i - 90)
            pts.append((cx + radius * math.cos(angle), cy + radius * math.sin(angle)))
        draw.polygon(pts, fill=fill, outline=outline)
    elif kind == "cross":
        arm = r // 3
        draw.rectangle((cx - arm, cy - r, cx + arm, cy + r), fill=fill, outline=outline)
        draw.rectangle((cx - r, cy - arm, cx + r, cy + arm), fill=fill, outline=outline)
    else:  # ring
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(255, 255, 255),
                     outline=fill, width=max(6, r // 3))


def _wrap(text: str, font: ImageFont.ImageFont, max_width: int) -> List[str]:
    words = text.split()
    lines: List[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw_text_width(candidate, font) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [text]


def draw_text_width(text: str, font: ImageFont.ImageFont) -> int:
    try:
        return int(font.getbbox(text)[2] - font.getbbox(text)[0])
    except Exception:
        return len(text) * 8


def render_entity_board(entities: List[Any], output_path: str,
                        width: int = 1024, height: int = 768) -> bool:
    """Render an entity board and write it to ``output_path``.

    Args:
        entities: List of entities (dict / str both accepted).
        output_path: Output image path.
        width / height: Canvas dimensions.

    Returns:
        bool: Whether rendering succeeded.
    """
    if not entities:
        return False

    width = max(512, int(width))
    height = max(384, int(height))
    margin = max(24, width // 24)
    gap = max(12, width // 60)

    count = len(entities)
    cols = 1 if count == 1 else (2 if count <= 4 else 3)
    rows = (count + cols - 1) // cols

    panel_w = (width - 2 * margin - (cols - 1) * gap) // cols
    panel_h = (height - 2 * margin - (rows - 1) * gap) // rows

    canvas = Image.new("RGB", (width, height), (238, 241, 245))
    draw = ImageDraw.Draw(canvas)

    # Fine background grid adds depth without interfering with recognition
    for x in range(0, width, max(32, width // 24)):
        draw.line((x, 0, x, height), fill=(229, 233, 239), width=1)
    for y in range(0, height, max(32, height // 18)):
        draw.line((0, y, width, y), fill=(229, 233, 239), width=1)

    shadow = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    shadow_draw = ImageDraw.Draw(shadow)
    title_font = _load_font(max(14, height // 34))

    for index, entity in enumerate(entities):
        row, col = divmod(index, cols)
        x0 = margin + col * (panel_w + gap)
        y0 = margin + row * (panel_h + gap)
        x1, y1 = x0 + panel_w, y0 + panel_h

        border, soft = _PALETTE[index % len(_PALETTE)]
        glyph = _GLYPHS[index % len(_GLYPHS)]

        shadow_draw.rounded_rectangle((x0 + 4, y0 + 6, x1 + 4, y1 + 6),
                                     radius=18, fill=(0, 0, 0, 45))

        draw.rounded_rectangle((x0, y0, x1, y1), radius=18, fill=(255, 255, 255),
                               outline=border, width=4)

        caption_font = _load_font(max(15, height // 30))
        lines = _wrap(_entity_name(entity), caption_font, panel_w - 24)
        text_h = len(lines) * (caption_font.size + 4)
        glyph_box = (x0 + 24, y0 + 20, x1 - 24, y1 - text_h - 28)
        _draw_glyph(draw, glyph, glyph_box, soft, border)

        ty = y1 - text_h - 18
        for line in lines:
            tw = draw_text_width(line, caption_font)
            draw.text(((x0 + x1 - tw) // 2, ty), line, font=caption_font, fill=(40, 45, 55))
            ty += caption_font.size + 4

    canvas = Image.alpha_composite(canvas.convert("RGBA"), shadow).convert("RGB")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    canvas.save(output_path, format="PNG")
    return True


def render_from_prompt(prompt: str, output_path: str, entities: Optional[List[Any]] = None,
                       width: int = 1024, height: int = 768) -> bool:
    """Fall back to a "prompt card" image when there are no structured entities."""
    canvas = Image.new("RGB", (width, height), (240, 243, 247))
    draw = ImageDraw.Draw(canvas)
    font = _load_font(max(16, height // 30))
    lines = _wrap(prompt[:1200], font, width - 80)
    y = margin_y = 50
    for line in lines:
        draw.text((50, y), line, font=font, fill=(40, 45, 55))
        y += font.size + 8
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    canvas.save(output_path, format="PNG")
    return True


__all__ = ["render_entity_board", "render_from_prompt"]
