"""Run with python -m scripts.analyze_session; no app imports or dependencies."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sqlite3
import sys
import webbrowser

from session_analyzer.analysis import analyze
from session_analyzer.reader import read_session
from session_analyzer.report import stamp, write_report
from session_analyzer.source import ROOT


def discover(root: Path) -> list[dict]:
    if not root.is_dir():
        return []
    found = []
    for db in root.glob("*/metadata.db"):
        if db.is_file():
            found.append({"folder": db.parent, "modified": db.stat().st_mtime})
    return sorted(found, key=lambda r: r["modified"], reverse=True)


def choose(items: list[dict], *, list_only: bool = False) -> Path | None:
    print("\nCLUE 세션 목록 · metadata.db 수정 시각 순 (생성 시각 아님)")
    for i, item in enumerate(items, 1):
        label = datetime.fromtimestamp(item["modified"]).strftime("%Y-%m-%d %H:%M:%S")
        try:
            data = analyze(read_session(item["folder"]))["summary"]
            observed = data["observed_range"]
            line = (f"관측 {stamp(observed['start_ms'])}~{stamp(observed['end_ms'])} | "
                    f"프레임 {data['frame_count']} · 전사 {data['transcript_count']} · 질문 {data['question_count']}")
            if data["synthetic"]:
                line += " | 합성 데이터"
        except (OSError, sqlite3.Error, ValueError):
            line = "읽기 불가 · --session-id 필요 또는 DB 손상/변경 확인"
        print(f"  {i}. {item['folder'].name} | {label} | {line}")
    if list_only:
        return None
    if not sys.stdin.isatty():
        print("\n비대화형 실행은 세션 경로 또는 --latest를 지정해 주세요.")
        return None
    while True:
        try:
            answer = input("\n분석할 번호 [Enter: 가장 최근 / q: 종료]: ").strip()
        except EOFError:
            return None
        if answer.lower() == "q":
            return None
        if not answer:
            return items[0]["folder"]
        if answer.isdecimal() and 1 <= int(answer) <= len(items):
            return items[int(answer) - 1]["folder"]
        print("목록에 있는 번호를 입력해 주세요.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CLUE Session Analyzer · 오프라인 세션 분석 (표준 라이브러리)")
    parser.add_argument("session", type=Path, nargs="?", help="metadata.db가 있는 세션 폴더")
    parser.add_argument("--sessions-root", type=Path, default=ROOT / "sessions")
    parser.add_argument("--output-root", type=Path, default=ROOT / "analysis")
    parser.add_argument("--session-id", help="DB 안에 여러 세션이 있을 때 분석할 ID")
    parser.add_argument("--latest", action="store_true", help="수정 시각이 가장 최근인 세션 선택")
    parser.add_argument("--list", action="store_true", help="세션 목록만 표시")
    parser.add_argument("--no-open", action="store_true", help="브라우저 자동 열기 생략")
    args = parser.parse_args(argv)
    try:
        session = args.session
        if session is None or args.list:
            items = discover(args.sessions_root)
            if not items:
                print("분석할 세션이 없습니다. sessions/SESSION_ID/metadata.db를 가져오거나 세션 경로를 지정해 주세요.")
                print("합성 시연: python -m scripts.generate_analyzer_demo (실제 측정 결과가 아닙니다.)")
                return 0
            if args.list:
                choose(items, list_only=True)
                return 0
            session = items[0]["folder"] if args.latest else choose(items)
        if session is None:
            return 0
        raw = read_session(session, args.session_id)
        data = analyze(raw)
        report = write_report(data, session, args.output_root)
        print(f"\n{data['summary']['data_origin']}")
        print(f"보고서: {report / 'index.html'}")
        print(f"진단 {len(data['warnings'])}건 · 원본 DB/이미지 변경 없음 · API/모델 실행 없음")
        if not args.no_open:
            if not webbrowser.open((report / "index.html").as_uri()):
                print("브라우저를 자동으로 열지 못했습니다. 위 index.html을 직접 열어 주세요.")
        return 0
    except (OSError, sqlite3.Error, ValueError) as exc:
        # Avoid printing raw SQLite/path errors with host names or private paths.
        print(f"분석 실패: {type(exc).__name__}. metadata.db, 출력 권한, 수집 상태를 확인해 주세요.", file=sys.stderr)
        if isinstance(exc, ValueError):
            print(str(exc), file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\n분석을 종료했습니다.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
