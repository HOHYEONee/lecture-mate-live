"""Optional local Korean cross-encoder; model loading is lazy."""

# [주요 설정 목록] 위치(함수) | 파라미터: 값 (설명)
# KoReranker.__init__ | model_name: "Dongjin-kr/ko-reranker" (Hugging Face ID 또는 완전한 로컬 모델 경로)
# KoReranker.__init__ | device: "auto" (PyTorch CUDA 가능 시 GPU, 아니면 CPU; 테스트 UI는 CPU 지정)
# KoReranker.__init__ | cache_folder: None (기본 프로젝트 .cache/models/reranker), local_files_only: False (다운로드 허용)
# KoReranker.__init__ | batch_size: 5 (한 번에 평가할 질문·청크 쌍), max_length: 512 (쌍의 최대 입력 토큰)
# KoReranker.prepare | use_safetensors: True, trust_remote_code: False (가중치 형식 및 외부 코드 실행 방지)
# KoReranker.score | padding: True, truncation: True (배치 길이 정렬 및 토큰 상한 적용)

from pathlib import Path
from threading import RLock


class KoReranker:
    # [설정] model_name: "Dongjin-kr/ko-reranker", device: "auto" (모델·PyTorch 실행 장치).
    # [설정] cache_folder: None (기본 저장 경로), local_files_only: False (온라인 다운로드 가능).
    # [설정] batch_size: 5 (배치당 후보 수), max_length: 512 (질문+청크 입력 토큰 상한).
    # [외부 호출] 한국어 재정렬 모델의 캐시·장치·배치 설정을 준비한다.
    def __init__(self, model_name="Dongjin-kr/ko-reranker", *, device="auto",
                 cache_folder=None, local_files_only=False, batch_size=5, max_length=512):
        if batch_size < 1 or max_length < 1:
            raise ValueError("batch_size and max_length must be positive")
        self.model_name = model_name
        self.device = device
        # [설정] cache_folder: .cache/models/reranker (프로젝트 루트 기준 모델 캐시 위치).
        self.cache_folder = str(cache_folder or Path(__file__).resolve().parents[1]/".cache/models/reranker")
        self.local_files_only = local_files_only
        self.batch_size = batch_size
        self.max_length = max_length
        self.actual_device = None
        self._model = self._tokenizer = None
        self._lock = RLock()

    # [내부 / 외부(Hugging Face Transformers)] 로컬 모델을 준비하고 필요한 경우 가중치를 내려받는다.
    def prepare(self):
        with self._lock:
            if self._model is not None:
                return
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            # [설정] device: "auto" (torch.cuda.is_available() 기준; Whisper의 CUDA 여부와 별개).
            device = self.device if self.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
            # [설정] trust_remote_code: False, use_safetensors: True (외부 모델 코드 실행·pickle 가중치 사용 방지).
            options = {"cache_dir": self.cache_folder, "local_files_only": self.local_files_only,
                       "trust_remote_code": False}
            tokenizer = AutoTokenizer.from_pretrained(self.model_name, **options)
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name, use_safetensors=True, **options)
            model.to(device).eval()
            self._tokenizer, self._model = tokenizer, model
            self.actual_device = device

    # [외부 호출 → 반환(점수 목록)] 질문과 각 후보 본문을 함께 읽어 관련성 점수를 계산한다.
    def score(self, query, texts):
        texts = list(texts)
        if not texts:
            return ()
        import torch
        with self._lock:
            # [같은 파일] score → prepare: 필요한 경우 모델 로드.
            self.prepare()
            scores = []
            with torch.inference_mode():
                for start in range(0, len(texts), self.batch_size):
                    batch = texts[start:start+self.batch_size]
                    # [외부(tokenizer)] 질문과 본문을 하나의 입력 쌍으로 구성한다.
                    # [설정] padding: True (배치 내 길이 맞춤), truncation: True, max_length: 기본 512토큰 (입력 제한).
                    inputs = self._tokenizer([query]*len(batch), batch, padding=True,
                                             truncation=True, max_length=self.max_length,
                                             return_tensors="pt").to(self.actual_device)
                    logits = self._model(**inputs, return_dict=True).logits
                    if logits.numel() != len(batch):
                        raise ValueError("reranker must return one score per candidate")
                    scores.extend(logits.reshape(-1).float().cpu().tolist())
            return tuple(scores)
