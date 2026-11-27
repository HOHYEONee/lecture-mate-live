# CLUE — lecture-mate-live

온라인 강의의 **화면과 시스템 오디오를 수집**하고, 학습자가 별도의 맥락 설명 없이 강의 내용에 질문할 수 있도록 돕는 Windows 데스크톱 AI 학습 파트너입니다.

P-실무프로젝트 SW1분반 1조 · Python 3.11+ · 모듈 계약 v1.6

## 처음 시작하는 팀원

1. **[개발 시작 가이드](docs/getting-started.md)** — 환경 설정, 담당자별 첫 작업, 독립 테스트, GitHub 협업 방법
2. **[팀 개발 안내](interface_spec/README.md)** — 전체 데이터 흐름, 4개 모듈의 책임과 구현 체크리스트
3. **[공통 Python 계약](interface_spec/contracts.py)** — 실제 구현할 데이터 클래스와 `*API` 메서드 정의

기능을 구현하기 전에 공통 계약을 확인하고, 기존 `core/`의 시계와 세션 코드를 재사용합니다.

멘토링 시연은 [시연 진행서](docs/mentoring-demo.md)에서 설치, 강의 자료 선택, 질문 순서를 확인하세요.

수집한 세션은 **[CLUE Session Analyzer](docs/session-analyzer.md)**로 분석할 수 있습니다.
`python -m scripts.analyze_session` 또는 `CLUE 세션 분석.cmd`를 실행하면
이미지·전사·질문 맥락·응답 성능을 담은 오프라인 HTML 보고서를 만듭니다.
분석기는 Python 표준 라이브러리만 사용하며 별도 AI 패키지나 모델 설치가 필요하지 않습니다.

## 문서 및 코드

| 경로 | 내용 |
|---|---|
| [`docs/getting-started.md`](docs/getting-started.md) | 환경 준비 및 팀원별 개발 착수 절차 |
| [`interface_spec/README.md`](interface_spec/README.md) | 프로젝트 구조와 담당 범위 |
| [`interface_spec/interface_spec_v1.md`](interface_spec/interface_spec_v1.md) | 상세 동작·시간·저장·상태·연동 규칙 (명세 v1.6) |
| [`interface_spec/contracts.py`](interface_spec/contracts.py) | Python 공통 인터페이스 계약 |
| [`interface_spec/proposal_changes.md`](interface_spec/proposal_changes.md) | 기존 제안서 대비 변경·미확정 사항 |
| [`core/README.md`](core/README.md) | 구현된 `SessionClock`·`SessionManager` 사용법 |
| [`tests/test_core.py`](tests/test_core.py) | 공통 기반 단위 테스트 |

## 현재 구현 상태

- **통합됨:** 공통 계약·세션, PySide6 앱, 프레임 캡처/SSIM, 음성 전사, HybridRAGIndex, SQLite·Q&A의 기본 실행 연결
- **기본 실행:** 프레임·음성 GPU 자동 검사, CPU 호환 모드, GUI 대표 프레임 미리보기, CPU RAG
- **실제 질의응답:** 앱 설정에서 OpenAI API 키를 한 번 저장하면 강의 화면과 전사를 사용합니다. 개발용 테스트 답변은 명시적으로 켤 수 있습니다.

GPU 설치는 [Windows 의존성 설치](INSTALL-dependencies-win-py312.md), 실행 옵션과 CPU 개발 모드는 [통합 실행 안내](docs/integration-runtime.md)를 참고합니다.

프로젝트 루트에서 공통 테스트를 실행할 수 있습니다.

```bash
python -m unittest discover -s tests -v
```

## 저장 및 주의 사항

- 기본 세션 구조: `sessions/<session_id>/metadata.db` 및 `sessions/<session_id>/frames/`
- SQLite 쓰기는 맥락 관리 모듈이 전담하며, 프레임 처리와 맥락 관리에는 앱이 생성한 동일한 `session_dir`를 전달합니다.
- `sessions/`, API 키, 허가받지 않은 강의 원본·화면·전사는 GitHub에 올리지 않습니다.
