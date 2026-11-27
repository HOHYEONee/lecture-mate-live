"""Decision rules for experimental model candidates."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from statistics import mean

from interface_spec.contracts import FrameType

from ..classification_policy import LayoutBox, LayoutRules, layout_prediction

_YOLO_REGIONS = {
    "title": "doc_title",
    "plain text": "text",
    "isolate_formula": "formula",
    "figure": "image",
    "table": "table",
}

TEXT_PROMPTS = (
    "a lecture slide primarily explaining the topic through written text",
    "a programming lecture slide whose main content is source code",
    "a lecture screen focused on handwritten notes or mathematical equations",
    "an instructional slide conveying its main ideas in words rather than pictures",
)
VISUAL_PROMPTS = (
    "a lecture slide whose main concept is illustrated by an important graph or chart",
    "a lecture slide explaining a concept using a meaningful diagram",
    "a teaching screen whose main information is a data table",
    "a lecture slide where an illustration or photograph explains the topic",
)


@dataclass(frozen=True)
class YoloLayoutRules(LayoutRules):
    """Frozen DocStructBench rules for the lecture experiment."""

    min_score: float = .45
    min_text_area: float = .025
    min_chart_table_area: float = .045
    min_image_area: float = .12
    min_image_to_text_ratio: float = .5
    edge_margin: float = .12
    edge_area_multiplier: float = 1.8
    max_image_area: float = .92
    whole_figure_min_score: float = .80


@dataclass(frozen=True)
class SiglipRules:
    # Logit units; these defaults are provisional and are not probabilities.
    min_margin: float = .5
    min_winning_logit: float = -12

    def __post_init__(self) -> None:
        if not isfinite(self.min_margin) or self.min_margin < 0 or not isfinite(self.min_winning_logit):
            raise ValueError("SigLIP needs a finite nonnegative margin and finite logit floor")


def yolo_layout_prediction(boxes: tuple[LayoutBox, ...], width: int, height: int,
                           rules: YoloLayoutRules) -> tuple[FrameType, str, dict]:
    mapped = tuple(LayoutBox(_YOLO_REGIONS[box.label], box.score, box.xyxy)
                   for box in boxes if box.label in _YOLO_REGIONS)
    return layout_prediction(mapped, width, height, rules, whole_image_min_score=rules.whole_figure_min_score)


def siglip_prediction(logits: list[float], rules: SiglipRules) -> tuple[FrameType, str, dict]:
    prompts = TEXT_PROMPTS + VISUAL_PROMPTS
    if len(logits) != len(prompts) or not all(isfinite(score) for score in logits):
        raise ValueError("invalid SigLIP prompt scores")
    text = mean(logits[:len(TEXT_PROMPTS)])
    visual = mean(logits[len(TEXT_PROMPTS):])
    margin = abs(text - visual)
    scores = {"mean_logits": {"Text": text, "Visual": visual}, "absolute_margin": margin,
              "prompt_logits": dict(zip(prompts, logits))}
    if max(text, visual) < rules.min_winning_logit:
        return FrameType.UNKNOWN, "weak_prompt_match", scores
    if margin <= rules.min_margin:
        return FrameType.UNKNOWN, "close_prompt_scores", scores
    return (FrameType.TEXT if text > visual else FrameType.VISUAL), "stronger_mean_logit", scores
