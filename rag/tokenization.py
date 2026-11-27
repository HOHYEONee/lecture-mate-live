"""Korean morphological keywords for BM25; preserve technical ASCII tokens."""

# [주요 설정 목록] 위치(함수) | 파라미터: 값 (설명)
# _content_tags: NNG, NNP, NR, VV, VA, MAG, XR, SH (검색에 남기는 품사)
# tokenize | num_workers: 1 (Kiwi 분석 워커 수), VX 예외: {"않", "못하"} (부정 표현 보존)
# tokenize | ASCII 정규식: [a-z0-9_]+ (영문 용어, 숫자, 함수명 보존)

import re
from threading import Lock
import unicodedata


_lock = Lock()
_kiwi = None
# [설정] _content_tags: 명사·수사·동사·형용사·일반부사·어근·한자 (BM25 검색 핵심 품사).
_content_tags = {"NNG", "NNP", "NR", "VV", "VA", "MAG", "XR", "SH"}


def _get_kiwi():
    """Called while holding _lock; share the morphology model across RAG sessions."""
    global _kiwi
    if _kiwi is None:
        from kiwipiepy import Kiwi
        _kiwi = Kiwi(num_workers=1)
    return _kiwi


# [내부] 문장 분할과 BM25 형태소 분석이 하나의 Kiwi 모델을 재사용한다.
def split_sentences(text, *, return_tokens=False):
    with _lock:
        return _get_kiwi().split_into_sents(text, return_tokens=return_tokens)


# [내부] 하나의 Kiwi 분석기를 필요할 때만 준비하고 동시에 호출되지 않도록 한다.
def tokenize(text):
    normalized = unicodedata.normalize("NFKC", text)
    # 함수명·약어·숫자는 형태소 분리에 의해 잘리지 않도록 원문 기준으로 보존한다.
    # [설정] ASCII 정규식: [a-z0-9_]+ (영문·숫자·밑줄로 된 기술 키워드 보존).
    keywords = re.findall(r"[a-z0-9_]+", normalized.casefold())
    if not re.search(r"[가-힣\u3400-\u9fff]", normalized):
        return tuple(keywords)
    with _lock:
        # [외부(Kiwi.tokenize)] 한국어 형태소와 품사를 분석한다.
        tokens = _get_kiwi().tokenize(normalized)
    for token in tokens:
        tag = token.tag.split("-", 1)[0]
        # 조사·어미·기호는 제외하고 부정 의미의 보조 용언은 보존한다.
        # [설정] VX 예외: {"않", "못하"} (검색에서 부정 의미 누락 방지).
        if (tag in _content_tags or (tag == "VX" and token.form in {"않", "못하"})) and re.search(r"[가-힣\u3400-\u9fff]", token.form):
            keywords.append(token.form.casefold())
    return tuple(keywords)
