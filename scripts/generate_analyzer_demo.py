"""Generate explicitly synthetic local data using the current DB schema."""
from __future__ import annotations

import argparse
from pathlib import Path

from session_analyzer.source import ROOT
from session_analyzer.synthetic import generate


def main(argv=None):
    parser=argparse.ArgumentParser(description="분석기 시연용 합성 세션 생성 · 실제 측정 아님")
    parser.add_argument("--output",type=Path,default=ROOT/"synthetic_sessions"/"합성_멘토링_데모")
    parser.add_argument("--observations",type=int,default=180,help="정기 관측 수 (기본 180, 최대 200000)")
    args=parser.parse_args(argv)
    try:
        path=generate(args.output,args.observations)
    except (OSError,ValueError) as exc:
        print(f"생성 실패: {type(exc).__name__}. 새 출력 폴더와 관측 수를 확인해 주세요.")
        return 2
    print("합성 데이터 생성 완료 · 실제 강의/성능 측정 결과가 아닙니다.")
    print(path)
    print(f'python -m scripts.analyze_session "{path}"')
    return 0


if __name__=="__main__":
    raise SystemExit(main())
