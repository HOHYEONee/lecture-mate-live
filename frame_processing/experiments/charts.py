"""Write standalone SVG diagnostics without introducing plotting dependencies."""

from __future__ import annotations

from html import escape
from pathlib import Path

from .labels import CLASSES

COLORS = ("#175cd3", "#b42318", "#067647")


def _text(x: float, y: float, text: str, *, anchor: str = "start") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}">{escape(text)}</text>'


def _save(output: Path, title: str, width: int, height: int, body: list[str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" role="img">', f'<title>{escape(title)}</title>',
           f'<rect width="{width}" height="{height}" fill="white"/>',
           '<g font-family="sans-serif" font-size="13" fill="#20242c">',
           *body, '</g>', '</svg>']
    output.write_text("\n".join(svg) + "\n", encoding="utf-8")


def _line_chart(output: Path, title: str, rows: list[dict],
                series: list[tuple[str, str]], maximum: float | None = None) -> None:
    dual = rows[0]["threshold"].startswith("g")
    if dual:
        x_values = list(range(len(rows)))
    else:
        rows = sorted(rows, key=lambda row: float(row["threshold"]))
        x_values = [float(row["threshold"]) for row in rows]
    low, high = x_values[0], x_values[-1]
    values = [row[key] for row in rows for key, _ in series if row[key] is not None]
    upper = maximum if maximum is not None else (max(values, default=0) or 1)
    left, right, top, bottom = 75, 864, 55, 253

    def x_at(index: int) -> float:
        if high == low:
            return (left + right) / 2
        return left + (right - left) * (x_values[index] - low) / (high - low)

    body = [_text(left, 29, title)]
    for tick in range(5):
        y = bottom - tick * (bottom - top) / 4
        body.append(f'<line x1="{left}" x2="{right}" y1="{y}" y2="{y}" stroke="#d8dde6"/>')
        body.append(_text(left - 9, y + 4, f"{upper * tick / 4:.3g}", anchor="end"))
    # Keep labels readable even for a dense sweep with thousands of values.
    if len(rows) <= 8:
        ticks = range(len(rows))
    else:
        ticks = sorted({round((len(rows) - 1) * index / 7) for index in range(8)})
    last_tick_x = left - 100
    for index in ticks:
        x = x_at(index)
        if x - last_tick_x < 64:
            continue
        body.append(_text(x, bottom + 20, rows[index]["threshold"], anchor="middle"))
        last_tick_x = x
    axis = "SSIM threshold pair (P/C and R/C)" if dual else "SSIM threshold"
    body.append(_text((left + right) / 2, bottom + 42, axis, anchor="middle"))
    if not values:
        body.append(_text((left + right) / 2, 150, "No defined values for these labels", anchor="middle"))
    for n, (key, label) in enumerate(series):
        color = COLORS[n]
        path = []
        active = False
        for index, row in enumerate(rows):
            value = row[key]
            if value is None:
                active = False
                continue
            x, y = x_at(index), bottom - (bottom - top) * value / upper
            path.append(f'{"L" if active else "M"}{x:.2f} {y:.2f}')
            body.append(f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2.8" fill="{color}">'
                        f'<title>{escape(label)} at {escape(row["threshold"])}: {value:.4g}</title></circle>')
            active = True
        body.append(f'<path d="{" ".join(path)}" stroke="{color}" fill="none" stroke-width="1.6"/>')
        legend_x = left + n * 245
        body.append(f'<rect x="{legend_x}" y="310" width="15" height="5" fill="{color}"/>')
        body.append(_text(legend_x + 22, 318, label))
    _save(output, title, 920, 340, body)


def _confusion_chart(output: Path, threshold: str, rows: list[dict]) -> None:
    title = f"Confusion matrix at {threshold}"
    body = [_text(22, 26, title), _text(22, 49, "Rows: actual label · Columns: predicted label")]
    counts = {(row["actual"], row["predicted"]): row["count"] for row in rows}
    upper = max(counts.values()) or 1
    left, top, cell_w, cell_h = 170, 87, 155, 57
    for i, actual in enumerate(CLASSES):
        body.append(_text(left - 12, top + i * cell_h + 34, actual, anchor="end"))
        body.append(_text(left + i * cell_w + cell_w / 2, top - 10, actual, anchor="middle"))
        for j, predicted in enumerate(CLASSES):
            count = counts[actual, predicted]
            shade = round(255 - 90 * count / upper)
            x, y = left + cell_w * j, top + cell_h * i
            body.append(f'<rect x="{x}" y="{y}" width="{cell_w - 3}" height="{cell_h - 3}" '
                        f'fill="rgb({shade},{shade + (255 - shade) // 3},255)"/>')
            body.append(_text(x + cell_w / 2, y + 34, str(count), anchor="middle"))
    _save(output, title, 655, 280, body)


def write_charts(output: Path, metrics: list[dict], confusion: list[dict]) -> None:
    _line_chart(output / "storage.svg", "Stored representative frames", metrics, [("stored", "Stored frames")])
    _line_chart(output / "detection_scores.svg", "Meaningful change detection", metrics,
                [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1")], 1)
    _line_chart(output / "detection_errors.svg", "Frame-level detection errors", metrics,
                [("fp_plot", "False positives"), ("fn_plot", "False negatives")])
    if metrics[0]["events"]:
        _line_chart(output / "preservation.svg", "Important information by threshold", metrics,
                    [("events_preserved", "Confirmed preserved"), ("events_not_preserved", "Not preserved"),
                     ("events_needing_review", "Needs review")])
    rows_by_threshold: dict[str, list[dict]] = {}
    for row in confusion:
        rows_by_threshold.setdefault(row["threshold"], []).append(row)
    for threshold, rows in rows_by_threshold.items():
        _confusion_chart(output / "confusion" / f"threshold_{threshold}.svg", threshold, rows)
