from __future__ import annotations

import argparse
import sqlite3
import statistics
import sys
from pathlib import Path
from typing import Sequence

from context_qa import AnswerRecord, AnswerStatus, ContextStore, format_ms

COLUMNS = ("question_id", "submitted", "type", "status", "first_ms", "total_ms", "in_tok", "out_tok", "model", "strategy", "error")


def session_ids(db_path: Path) -> list[str]:
    conn = sqlite3.connect(f"file:{db_path.as_posix()}?mode=ro", uri=True)
    try:
        rows = conn.execute("SELECT DISTINCT session_id FROM answers ORDER BY session_id").fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def load_records(session_dir: Path) -> dict[str, tuple[AnswerRecord, ...]]:
    db_path = session_dir / "metadata.db"
    if not db_path.is_file():
        raise FileNotFoundError(f"metadata.db not found in {session_dir}")
    result = {}
    for sid in session_ids(db_path):
        with ContextStore(sid, db_path) as store:
            result[sid] = store.answer_records()
    return result


def _cell(value: object) -> str:
    return "-" if value is None else str(value)


def format_table(records: Sequence[AnswerRecord]) -> str:
    rows = [COLUMNS]
    for r in records:
        rows.append(
            (
                r.question_id,
                format_ms(r.submitted_ms) if r.submitted_ms is not None else "-",
                _cell(r.question_type),
                r.status.value,
                _cell(r.first_response_latency_ms),
                str(r.elapsed_ms),
                _cell(r.input_tokens),
                _cell(r.output_tokens),
                _cell(r.model),
                _cell(r.strategy),
                _cell(r.error_code),
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(len(COLUMNS))]
    lines = ["  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)).rstrip() for row in rows]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)


def format_details(records: Sequence[AnswerRecord], max_chars: int) -> str:
    blocks = []
    for r in records:
        answer = r.answer if len(r.answer) <= max_chars else r.answer[:max_chars] + "..."
        blocks.append(f"[{r.question_id}] {r.status.value}\nQ: {_cell(r.question)}\nA: {answer or '-'}")
    return "\n\n".join(blocks)


def summarize(records: Sequence[AnswerRecord]) -> str:
    counts = {s: sum(r.status == s for r in records) for s in AnswerStatus}
    parts = [f"total={len(records)}"] + [f"{s.value}={n}" for s, n in counts.items()]
    done = [r for r in records if r.status == AnswerStatus.COMPLETED]
    firsts = [r.first_response_latency_ms for r in done if r.first_response_latency_ms is not None]
    if firsts:
        parts.append(f"median_first_ms={int(statistics.median(firsts))}")
    if done:
        parts.append(f"median_total_ms={int(statistics.median(r.elapsed_ms for r in done))}")
    in_tok = [r.input_tokens for r in records if r.input_tokens is not None]
    out_tok = [r.output_tokens for r in records if r.output_tokens is not None]
    if in_tok or out_tok:
        parts.append(f"input_tokens={sum(in_tok)} output_tokens={sum(out_tok)}")
    return " ".join(parts)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show answers stored in a CLUE session")
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--full", action="store_true", help="also print question and answer text")
    parser.add_argument("--max-chars", type=int, default=300)
    args = parser.parse_args(argv)

    try:
        sessions = load_records(args.session_dir)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not sessions:
        print("no answers recorded")
        return 0
    for sid, records in sessions.items():
        print(f"session_id={sid}")
        print(format_table(records))
        print(summarize(records))
        if args.full:
            print()
            print(format_details(records, args.max_chars))
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
