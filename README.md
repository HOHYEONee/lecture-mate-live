# lecture-mate-live
A real-time multimodal AI learning assistant that understands online lectures through screen capture, audio transcription, and context-aware Q&amp;A.

## 팀 개발 문서

- [프로젝트 개요·모듈별 역할](interface_spec/README.md)
- [모듈 인터페이스 상세 명세](interface_spec/interface_spec_v1.md)
- [공통 Python 데이터·API 계약](interface_spec/contracts.py)
- [기존 제안서 대비 변경 사항](interface_spec/proposal_changes.md)

각 팀원은 `interface_spec/README.md`에서 전체 흐름과 담당 범위를 확인한 뒤 상세 명세와 공통 계약을 참고합니다.

## 공통 실행 기반

- [공통 시계·세션 리소스 최소 구현과 사용법](core/README.md)
- 테스트: `python -m unittest discover -s tests -v`
