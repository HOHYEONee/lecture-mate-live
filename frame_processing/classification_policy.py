"""Provisional rules that convert local model evidence into the shared FrameType."""

from __future__ import annotations

from dataclasses import dataclass

from interface_spec.contracts import FrameType

# Both released Paddle label maps contain these content categories. Caption,
# header/footer, seal, and their image variants are excluded as evidence.
TEXT_REGIONS = frozenset({
    "text", "paragraph_title", "doc_title", "abstract", "content", "formula",
    "algorithm", "aside_text", "reference", "reference_content", "footnote",
})
STRUCTURED_VISUAL_REGIONS = frozenset({"table", "chart"})

@dataclass(frozen=True)
class LayoutBox:
    label: str
    score: float
    xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class LayoutRules:
    min_score: float = .5
    min_text_area: float = .015
    min_chart_table_area: float = .035
    min_image_area: float = .10
    min_image_to_text_ratio: float = .4
    edge_margin: float = .15
    edge_area_multiplier: float = 2
    max_image_area: float = .90


def layout_prediction(boxes: tuple[LayoutBox, ...], width: int, height: int,
                      rules: LayoutRules, *, whole_image_min_score: float | None = None
                      ) -> tuple[FrameType, str, dict]:
    text = []
    visuals = []
    for box in boxes:
        if box.score < rules.min_score:
            continue
        x1, y1, x2, y2 = (max(0., min(1., value / limit))
                          for value, limit in zip(box.xyxy, (width, height, width, height)))
        if x2 <= x1 or y2 <= y1:
            continue
        if box.label in TEXT_REGIONS:
            text.append((x1, y1, x2, y2))
        elif box.label in STRUCTURED_VISUAL_REGIONS or box.label == "image":
            area = (x2 - x1) * (y2 - y1)
            x, y = (x1 + x2) / 2, (y1 + y2) / 2
            edge = min(x, y, 1 - x, 1 - y) < rules.edge_margin
            visuals.append((box.label, area, edge, box.score))
    text_area = _union_area(text)
    scores = {"text_coverage": text_area,
              "largest_visual_coverage": max((area for _, area, _, _ in visuals), default=0.)}
    for label, area, edge, confidence in visuals:
        if label in STRUCTURED_VISUAL_REGIONS:
            minimum = rules.min_chart_table_area
        elif (area <= rules.max_image_area
              or (whole_image_min_score is not None and confidence >= whole_image_min_score)):
            minimum = max(rules.min_image_area, text_area * rules.min_image_to_text_ratio)
        else:
            continue
        if area >= minimum * (rules.edge_area_multiplier if edge else 1):
            return FrameType.VISUAL, f"salient_{label}", scores
    if text_area >= rules.min_text_area:
        return FrameType.TEXT, "text_coverage", scores
    return FrameType.UNKNOWN, "insufficient_layout_evidence", scores


def _union_area(rectangles: list[tuple[float, float, float, float]]) -> float:
    """Area of clipped text boxes with overlaps counted once."""
    boundaries = sorted({x for left, _, right, _ in rectangles for x in (left, right)})
    area = 0.
    for left, right in zip(boundaries, boundaries[1:]):
        topmost_end = 0.
        height = 0.
        for top, bottom in sorted((y1, y2) for x1, y1, x2, y2 in rectangles if x1 <= left < x2):
            height += max(0., bottom - max(topmost_end, top))
            topmost_end = max(topmost_end, bottom)
        area += (right - left) * height
    return area
