from pathlib import Path

import numpy as np


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
WINDOWED_SUFFIX = "::windowed-mean-v1"


class SentenceTransformerEmbedder:
    # [내부] 임베딩 모델·장치·캐시 경로·긴 텍스트 처리 방식과 모델 식별자를 준비한다.
    def __init__(self, model_name=DEFAULT_EMBEDDING_MODEL,
                 *, device="cpu", cache_folder=None, window_long_text=True):
        self._model_id = model_name
        self._window_long_text = window_long_text
        # Representation changes must not silently mix with older saved vectors.
        self.model_name = model_name + (WINDOWED_SUFFIX if window_long_text else "")
        self._device = device
        self._cache_folder = str(cache_folder or (
            Path(__file__).resolve().parents[1] / ".cache" / "models" / "embeddings"
        ))
        self._model = None

    # [내부 / 외부(sentence_transformers.SentenceTransformer)] 로컬 임베딩 모델을 준비하며 없으면 캐시에 내려받는다.
    def prepare(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(
                self._model_id, device=self._device, cache_folder=self._cache_folder,
                trust_remote_code=False,
            )

    # [내부 → 반환(벡터 목록)] 텍스트를 인코딩하고 부모 청크의 대표 벡터만 반환한다.
    def encode(self, texts):
        # [같은 파일] SentenceTransformerEmbedder.encode → SentenceTransformerEmbedder.encode_with_passages: 대표 벡터·세부 검색 구간 벡터 생성.
        return self.encode_with_passages(texts)[0]

    # [내부 / 외부(SentenceTransformer.encode)] 긴 글을 토큰 구간으로 나눠 구간별 벡터와 부모 평균 벡터를 반환한다.
    def encode_with_passages(self, texts):
        """Return parent vectors and short retrieval passages from one inference.

        Long lecture cards remain readable as a whole. Searching their shorter
        passages avoids averaging away a relevant sentence in the parent vector.
        """
        # [같은 파일] SentenceTransformerEmbedder.encode_with_passages → SentenceTransformerEmbedder.prepare: 다국어 임베딩 모델 준비.
        self.prepare()
        texts = list(texts)
        if not texts:
            return [], []
        if not self._window_long_text:
            vectors = self._model.encode(texts, normalize_embeddings=True,
                                         show_progress_bar=False).tolist()
            return vectors, [[(text, vector)] for text, vector in zip(texts, vectors)]
        tokenizer = self._model.tokenizer
        limit = self._model.max_seq_length
        special = tokenizer.num_special_tokens_to_add(pair=False)
        if limit <= special:
            raise ValueError("embedding token limit must leave room for text")
        budget = min(64, limit-special)
        pieces, owners, weights = [], [], []
        for owner, text in enumerate(texts):
            ids = tokenizer.encode(text, add_special_tokens=False, verbose=False)
            if len(ids) <= budget:
                pieces.append(text)
                owners.append(owner)
                weights.append(max(1, len(ids)))
                continue
            start = 0
            while start < len(ids):
                end = min(len(ids), start + budget)
                # Decoding a subword boundary can add tokens when encoded again.
                # Verify the actual piece rather than allowing silent truncation.
                while end > start:
                    piece = tokenizer.decode(ids[start:end], skip_special_tokens=True)
                    if len(tokenizer.encode(piece, add_special_tokens=True, verbose=False)) <= limit:
                        break
                    end -= 1
                if end == start:
                    raise ValueError("cannot fit an embedding token window")
                pieces.append(piece)
                owners.append(owner)
                weights.append(end-start)
                start = end
        encoded = self._model.encode(pieces, normalize_embeddings=True,
                                     show_progress_bar=False)
        vectors = np.zeros((len(texts), encoded.shape[1]), dtype=np.float64)
        passages = [[] for _ in texts]
        for piece, vector, owner, weight in zip(pieces, encoded, owners, weights):
            vectors[owner] += vector * weight
            passages[owner].append((piece, tuple(float(v) for v in vector)))
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if not np.all(np.isfinite(norms)) or np.any(norms == 0):
            raise ValueError("embedding windows produced invalid vectors")
        return (vectors / norms).tolist(), passages
