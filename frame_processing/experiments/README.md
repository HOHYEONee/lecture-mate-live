# P/C/R SSIM 임계값 실험

한 번 수집한 1FPS 무손실 PNG를 여러 임계값에서 순서대로 재생합니다. 각 후보는 서비스와 같은 `FramePipeline`과 `SSIMComparator`의 새 인스턴스를 사용합니다. 기본 비교 조건은 서비스와 동일한 긴 변 960px, BGR 3채널 SSIM입니다. 서비스의 잠정 기본값은 P/C **0.90**, R/C **0.98**입니다. 실험에서 이 한 조합을 재현하려면 `--group-thresholds 0.90 --representative-thresholds 0.98`을 지정하세요.

저장소 루트에서 [공통 캡처·SSIM 의존성](../requirements-base.txt)을 사용합니다. 수집·SSIM 평가에는 분류 모델 설치가 필요하지 않습니다. 아래 명령은 PowerShell 기준입니다.

## 1. 원본 화면 수집

```powershell
$python = '.\.venv\Scripts\python.exe'
$dataset = & $python -m frame_processing.experiments capture --name 강의필기 --region 100 100 1280 720 --seconds 180
```

- `--region`은 `(left, top, width, height)` 물리 픽셀입니다. 모니터가 기본 모니터 왼쪽에 있으면 음수 좌표를 쓸 수 있습니다. 먼저 짧게 수집해 고DPI 환경에서도 원하는 화면이 들어오는지 확인하세요.
- `--synthetic`을 추가하면 데스크톱을 읽지 않고 동일한 합성 슬라이드를 반복 수집합니다. 도구의 동작 확인에 쓸 수 있습니다.
- 결과는 `experiments/<시각>_<이름>/` 아래 `raw_frames/`, `capture_metadata.json`, `labels.csv`, `events.csv`에 기록됩니다. 원본 PNG에는 중복 프레임도 포함됩니다. 수집 메타데이터에는 목표 FPS, 취득 경과 시각, 순번, 예정 슬롯과 누락 슬롯 수, PNG 체크섬이 들어갑니다. 기록된 간격을 확인한 뒤 누락이 많은 수집본은 다시 수집하거나 품질 분석에서 구분하세요. 중단·실패한 수집본은 평가에 사용하지 않습니다.
- 허가된 강의 자료만 수집하고 알림이나 개인정보가 영역에 들어가지 않도록 설정하세요. 실험 폴더는 Git에서 제외됩니다.

## 2. 같은 원본으로 임계값 탐색

```powershell
# 1차 탐색. 기본 후보는 0.90, 0.91, …, 0.99입니다.
$run = & $python -m frame_processing.experiments sweep $dataset --write-representatives

# 원본 4~455번만 양 끝을 포함해 재생합니다.
$trimmedRun = & $python -m frame_processing.experiments sweep $dataset --start-frame 4 --end-frame 455 --range 0.80 0.99 0.01 --write-representatives

# 2차 탐색 예시. 1차 결과를 보고 후보를 고른 뒤 실행하세요.
$fineRun = & $python -m frame_processing.experiments sweep $dataset --range 0.970 0.990 0.001
$listRun = & $python -m frame_processing.experiments sweep $dataset --thresholds '0.97,0.975,0.98,0.985,0.99,0.995'
```

`--range START STOP STEP`은 십진수로 후보를 생성합니다. 종료값이 간격에 맞을 때만 포함됩니다. `--comparison-max-side`를 바꿔 해상도를 시험할 수 있으며, `0`이면 축소하지 않습니다. 임계값 비교에서는 모든 후보의 비교 해상도를 동일하게 유지하세요.

`--start-frame`과 `--end-frame`은 원본의 1부터 시작하는 순번입니다. 생략한 경계는 원본의 처음 또는 끝이 됩니다. 각 임계값은 선택한 첫 프레임에서 독립적으로 새 그룹을 시작하고, 해당 프레임을 초기 대표로 저장합니다. 원본 PNG와 수집 메타데이터를 그대로 두고, 결과에 원본 순번과 타임스탬프를 기록합니다. 같은 P/C 또는 R/C 픽셀 쌍은 비교 결과를 재사용할 수 있으며, 각 임계값의 대표 선택과 그룹 상태는 따로 계산합니다.

스윕 중 약 1초 간격으로 현재 임계값·프레임, 전체 진행률, 경과 시간과 ETA를 표준 오류에 표시합니다. `Total`은 선택한 프레임 수 × 임계값 수를 기준으로 셉니다. 여러 임계값이 같은 원본 프레임을 차례로 처리하므로 화면에 표시되는 임계값은 갱신마다 바뀔 수 있습니다. ETA는 평균 처리 속도를 기준으로 추정합니다. 표준 출력에는 완료된 결과 경로만 남습니다. 진행률을 숨기려면 `--no-progress`를 사용하세요. 표준 오류를 파일에 저장하면 각 갱신이 일반 텍스트 한 줄로 기록됩니다.

P/C와 R/C 기준을 분리하려면 두 목록을 함께 지정합니다. 아래 명령은 4×4 조합을 모두 실행합니다. P/C가 그룹 임계값 미만이면 바로 `new_group`이며, 나머지 프레임은 R/C가 대표 임계값 미만일 때 `new_representative`입니다. 두 값이 같으면 단일 임계값의 판정과 일치합니다. 각 조합의 P/C/R 상태는 별도로 초기화됩니다. 기존 `--thresholds`와 `--range`는 단일 임계값 모드에서 사용합니다.

```powershell
$dualRun = & $python -m frame_processing.experiments sweep $dataset --start-frame 4 --end-frame 455 --group-thresholds '0.89,0.91,0.93,0.95' --representative-thresholds '0.95,0.97,0.98,0.99'
```

이중 임계값 결과의 `summary.csv`와 `decisions.csv`에는 `group_threshold`와 `representative_threshold`가 함께 들어갑니다. 예를 들어 `threshold_g0.89_r0.97/`은 P/C 0.89, R/C 0.97 조합입니다.

각 실행은 `results/sweep_.../`에 별도로 기록됩니다. `summary.csv`에 후보별 관측 수, 새 그룹·그룹 내 새 대표·재사용 횟수, 저장 수, 감소율과 그룹당 평균 대표 수가 들어갑니다. `threshold_*/decisions.csv`에는 모든 관측의 판정, SSIM, 그룹 ID, 활성 대표 ID와 원본 순번이 들어갑니다. 첫 프레임의 비교 점수와 P/C가 비유사해 생략한 R/C 점수는 비어 있습니다.

`stored=True`는 서비스에서 이미지가 저장될 판정이라는 뜻입니다. `--write-representatives`를 주면 서비스와 동일한 품질 90 JPEG를 생성하고 `summary.csv`에 파일 크기 합계를 기록합니다. JPEG는 비교가 끝난 뒤에만 생성되며, P와 R은 비교용 픽셀 배열을 메모리에 유지합니다. 작은 필기처럼 JPEG에서도 내용이 읽혀야 하는 항목을 검토할 때는 이 옵션을 사용하세요.

## 3. 정답 라벨과 중요 정보 구간 작성

```powershell
& $python -m frame_processing.experiments review $dataset --frame 42 --reference 38
# 특정 스윕과 같은 구간만 검토합니다.
& $python -m frame_processing.experiments review $dataset --run $trimmedRun --frame 42 --reference 38
```

위 명령은 직전 화면, 현재 화면, 직접 고른 기준 화면을 한 PNG에 표시합니다. 작은 글자는 `raw_frames/` 원본으로 확인하세요. 처음에는 임계값별 판단을 보기 전에 라벨을 정합니다.

`review --run`은 선택 구간 밖의 현재·기준 프레임을 거부하며, 구간 첫 프레임에서는 이전 화면을 표시하지 않습니다. `evaluate`와 `compare`는 지정한 스윕의 구간을 자동으로 사용합니다. 구간 첫 프레임은 저장량과 정보 보존에는 포함하고, 변화 검출 지표와 최소 라벨 기준에서는 제외합니다. 라벨·이벤트 파일은 전체 원본 순번을 유지하므로 다른 구간에서도 재사용할 수 있습니다.

`labels.csv`의 `frame_index`와 `timestamp_ms`를 유지하고 `label`, `notes`를 편집하세요. 가능한 값은 다음과 같습니다.

| 라벨 | 기준 |
|---|---|
| `new_group` | 슬라이드 또는 강의 내용 전환 |
| `new_representative` | 같은 화면에서 의미 있는 필기·수식·시각 자료 추가 또는 변경 |
| `reused` | 기존 대표로 충분한 상태, 포인터 이동이나 작은 잡음 |
| `ambiguous` | 스크롤·애니메이션 등 판정이 모호한 사례. 이유는 `notes`에 기록 |
| 빈 값 | 아직 검토하지 않은 프레임 |

중요 정보는 `events.csv`에 **정보 하나당 한 행**으로 별도 기록합니다. `event_id`는 영문·숫자·`_`·`-`를 사용합니다. **`first_frame`~`last_frame`은 해당 정보가 화면에 보이는 전체 검토 구간을 포함해야 합니다.** 정보가 처음 보이는 원본 순번부터 마지막으로 보이는 순번까지 양 끝을 포함해 적고, 뒤에 다시 등장하는 경우도 포함합니다. 중간에 정보가 보이지 않는 대표는 사진을 확인해 `absent`로 판단합니다. `description`에는 확인할 내용을 구체적으로 씁니다. 예를 들어 `formula_01,42,49,최종 수식에 분모 n-1이 보임`은 42~49번 프레임의 대표에서 해당 수식이 읽히는지 확인하라는 뜻입니다.

## 4. 평가와 보존 검토

```powershell
$evaluation = & $python -m frame_processing.experiments evaluate $dataset $run
& $python -m frame_processing.experiments compare $dataset $run --frame 42

# evidence_to_review.csv의 verdict와 notes를 채운 뒤 평가만 다시 실행합니다.
$checks = Join-Path $evaluation 'evidence_to_review.csv'
$reviewed = & $python -m frame_processing.experiments evaluate $dataset $run --checks $checks
```

`evidence_to_review.csv`는 후보 중 적어도 하나에서 실제로 저장되는 `(event_id, frame_index)`를 모은 검토표입니다. `raw_image`는 수집 폴더 기준, `representative_jpeg`는 sweep 폴더 기준의 상대 경로입니다. JPEG 경로가 있으면 그 이미지에서 정보가 보이는지 확인합니다. `verdict`는 `visible`, `absent`, `uncertain` 또는 빈 값입니다. `verdict`와 `notes`를 편집하고, 데이터셋과 이벤트를 확인하는 `dataset_sha256`, `event_sha256` 열은 유지하세요.

`--checks`로 가져온 판정과 메모는 데이터셋 및 해당 이벤트의 ID·설명·범위가 일치할 때 이어받습니다. 설명이나 범위를 수정한 이벤트는 새 검토표에서 다시 확인합니다. 적용되지 않은 판정과 메모는 사유와 함께 `checks_requiring_review.csv`에 남고, 입력 검토표 전체는 `checks_snapshot.csv`에 보존됩니다. 해시 열이 없는 이전 검토표도 원래 평가 폴더의 `evaluation_metadata.json`과 `events_snapshot.csv`로 출처를 확인할 수 있으면, 변경되지 않은 이벤트의 판정을 이어받습니다.

후보를 선정하기 전에 유효 라벨이 **20개 이상**, 평가 대상의 **50% 이상**, 의미 있는 변화와 `reused`가 **각각 5개 이상**인지 확인합니다. 이는 소수의 라벨만으로 후보가 나오는 것을 막는 운영 기본값입니다. 과학적으로 검증된 표본 크기나 정확도 보장 기준이 아니며, 실험 목적에 맞게 바꿀 수 있습니다. 평가 대상은 최초 프레임을 제외한 전체입니다. 미라벨과 `ambiguous`는 라벨 비율의 분모에 포함되며, 유효 라벨 수에는 포함되지 않습니다. 조건 미달이어도 개별 지표와 보존 검토표는 계속 생성됩니다.

```powershell
& $python -m frame_processing.experiments evaluate $dataset $run --checks $checks --min-labeled-frames 40 --min-labeling-ratio 0.7 --min-labels-per-binary-class 10
```

| 결과 | 읽는 법 |
|---|---|
| `metrics.csv`, `class_metrics.csv`, `confusion_matrix.csv` | `new_group`과 `new_representative`를 묶은 Precision·Recall·F1과 FP/FN, 별도의 클래스별 지표와 혼동행렬. 첫 프레임, 미라벨, `ambiguous`는 정확도 집계에서 제외합니다. 분모가 0인 비율은 빈 값입니다. |
| `errors.csv`, `compare_frame_*.csv` | 검출 오탐·누락과 그룹 오분류를 각각 추적하고, 같은 프레임에서 후보별 판단과 활성 대표를 비교합니다. |
| `preservation_events.csv` | 구간 내 후보 중 하나라도 정보가 확인되면 `preserved`입니다. 최초 등장 저장 여부와 최초 **확인된** 보존까지의 지연을 분리해 기록합니다. 후보가 없거나 모두 `absent`면 `not_preserved`, 미확인 후보가 남으면 `needs_review`입니다. |
| `events_excluded.csv` | 스윕 구간 밖에 있거나 경계에 잘려 전체 검토 구간을 확인할 수 없는 이벤트입니다. 정보 보존 지표에서는 제외합니다. |
| `candidates.json` | 평가 대상 프레임 수, 유효 라벨 수와 비율, 클래스별 라벨 수, 적용한 최소 조건과 미충족 항목을 기록합니다. 조건 미달이면 `insufficient_frame_labels`로 표시하고 후보 목록을 비웁니다. 조건을 충족하고 모든 이벤트 결과가 확정되면 정보 미보존이 가장 적은 후보부터, 프레임 FN·FP·그룹 오분류·저장 수를 기존 방식으로 비교합니다. |
| `charts/` | 저장 수, 검출 점수와 오탐·누락, 임계값별 혼동행렬, 이벤트가 있으면 보존 비교를 SVG로 확인할 수 있습니다. |

보존 결과는 원본에 잡힌 프레임과 직접 지정한 검색 구간에 한정됩니다. 1초 사이에만 나타난 정보는 평가할 수 없습니다. JPEG를 내보내지 않은 평가에서 PNG로 확인한 결과는 대표 선택의 보존 여부를 뜻하며, JPEG의 가독성 평가에는 대표 JPEG가 필요합니다. `candidates.json`은 현재 수집본의 후보를 비교한 결과이므로 최종값은 다른 강의 유형과 별도 수집본에서도 검증하세요.

## 5. 수동 유효 구간으로 스윕 평가

`evaluate-windows`는 기존 스윕에서 지정한 범위만 집계합니다. 앞쪽 프레임에서 만들어진 P/C/R 상태는 기존 판정에 반영되어 있으며, 구간 시작에서 다시 계산하지 않습니다.

```powershell
& $python -m frame_processing.experiments evaluate-windows $dataset $run --start-frame 7 --end-frame 426
```

필수 화면은 기존 `events.csv`에 기록합니다. 유효 구간에 `stored=True`가 하나라도 있으면 적중입니다. 이미지 내용 자체를 확인하는 평가는 앞 절의 보존 검토를 사용하세요. 그 밖의 조건은 `verification_windows.csv`에 `check_id,kind,first_frame,last_frame,event_id,description`으로 기록합니다. `new_representative`는 하나 이상의 실제 갱신을 요구하고, `stored_representative`는 저장 유형 두 가지를 모두 허용합니다. 비연속 후보를 지정할 때는 `candidate_frames`에 공백으로 구분한 원본 번호를 넣습니다. `target_new_representatives`와 `target_count`는 참고용이며 엄격 조건에서 제외합니다. `unnecessary`와 `scroll`은 저장량을 별도로 집계합니다.

결과는 데이터셋 아래 `evaluations/`의 새 폴더에 생성됩니다. `summary.csv`에는 필수 구간 충족률, 저장 수, 중복 제거율, 대표 갱신, 새 그룹 대체 및 누락 구간이 있고, `window_results.csv`에는 각 조건에 해당한 원본 프레임과 판정이 있습니다.

기존 단일 임계값 스윕을 `--baseline`으로 전달하면 같은 평가 범위와 레이블로 `baseline_summary.csv`도 생성합니다. 재생 시작 프레임과 비교 크기는 두 스윕에서 같아야 합니다. 숫자 정답만 기록한 강의는 `--required-frames`를 추가해 기존 `required_frames.csv`의 정답 `N`과 바로 앞 `N−1`을 일대일로 대응시킵니다. 이 경우 미판정 저장을 불필요 저장으로 계산하지 않습니다.

```powershell
& $python -m frame_processing.experiments evaluate-windows $dataset $dualRun --start-frame 7 --end-frame 426 --baseline $run
```

`transition_review.csv`는 수동 `new_group` 라벨 또는 기준 단일 임계값(기본 `0.95`, `--transition-threshold`로 변경 가능)의 `new_group`이 이중 임계값에서 `new_representative`가 된 사례를 모읍니다. 수동 라벨이 없는 행은 실제 슬라이드 전환으로 확인하기 위한 후보입니다. `evaluation_metadata.json`에는 두 임계값을 같은 값으로 둔 조합과 과거 단일 스윕 사이의 프레임별 일치 여부를 기록합니다.

## 프레임 분류 비교

분류 모델 선택과 실행 장치 변경은 실험 CLI에서 사용할 수 있습니다. 기존 비교는 정적
슬라이드 41장과 필기·그래프 혼합 81장의 동일 PNG와 수동 정답을 사용했습니다.

| 모델 | Accuracy | Macro F1 | Visual Recall | 필기 Visual Recall |
|---|---:|---:|---:|---:|
| 연결 요소 휴리스틱 | 52.5% | 31.9% | 98.2% | 96.9% |
| PP-DocLayout-S | 50.0% | 45.9% | 62.5% | 59.4% |
| PP-DocLayout_plus-L | 73.8% | 62.7% | 87.5% | 78.1% |
| SigLIP 2 Large | 59.0% | 47.8% | 92.9% | 100.0% |
| DocLayout-YOLO | 72.1% | 63.8% | 71.4% | 53.1% |

plus-L은 Visual→Text 0장, RTX 4060 Ti에서 추론 P95 35.0ms를 기록했습니다.
DocLayout-YOLO는 첫 1080p 서비스 캡처 간격이 약 3.2초였습니다. 위 결과는 두 강의에
대해 측정한 값입니다.

### 데이터와 라벨링

아래 명령은 기존 스윕에서 저장된 대표 프레임을 골라 새 평가 폴더를 만들고 라벨링 창을
엽니다. 강의마다 `--source <강의_ID> <데이터셋> <스윕> <후보_ID>`를 지정할 수 있습니다.
준비만 하려면 `--prepare-only`를, 기존 폴더를 이어서 열려면 폴더 경로만 전달합니다.
숫자 1·2·3으로 Text·Visual·Unknown을 저장하며, 화살표로 기존 라벨을 수정합니다.

```powershell
$out = 'experiments/classification_run'
& $python -m frame_processing.experiments.labeling $out --source lecture_a $dataset $run 0.95
```

일부 표본만 평가할 때는 `classification prepare`에서 `--per-lecture`와
`--source <강의_ID> <데이터셋> <스윕> <후보_ID> <첫_프레임> <마지막_프레임>`을 지정합니다.
원본 PNG 체크섬과 강의 ID·원본 순번을 매니페스트로 고정합니다.

### 선택형 실험 환경과 명령

- 휴리스틱은 공통 캡처 의존성으로 실행합니다. Paddle 두 모델은 [서비스 설치 환경](../README.md#Windows-설치)을 사용하며, `PP-DocLayout-S`는 이름에 맞는 공식 가중치 폴더를 지정합니다. [PaddleOCR 레이아웃 문서](https://www.paddleocr.ai/latest/en/version3.x/module_usage/layout_detection.html)
- SigLIP 2는 별도 Python 3.12 가상환경에 CUDA PyTorch 2.8.0, torchvision 0.23.0과
  `transformers==4.57.1 sentencepiece opencv-python`을 설치합니다. 모델 리비전
  `1b42688`을 로컬 폴더에 내려받아 사용합니다. [공식 가중치](https://huggingface.co/google/siglip2-large-patch16-384)
- DocLayout-YOLO는 별도 Python 3.12 가상환경에 같은 CUDA PyTorch/torchvision과
  `doclayout-yolo==0.0.4 huggingface-hub==0.36.2`를 설치합니다. 리비전 `6ca1877`의
  `doclayout_yolo_docstructbench_imgsz1024.pt`를 로컬 폴더에 두고, SDK 설정 경로
  `YOLO_CONFIG_DIR`는 쓰기 가능한 로컬 디렉터리로 지정합니다. 체크포인트는 Apache-2.0,
  SDK는 AGPL-3.0으로 표기되어 있습니다. [공식 가중치](https://huggingface.co/juliozhao/DocLayout-YOLO-DocStructBench) · [SDK 라이선스](https://github.com/opendatalab/DocLayout-YOLO/blob/main/LICENSE)

PyTorch GPU 패키지는 `pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126`으로
설치했습니다. `$modelPython`은 해당 모델의 Python, `$weights`는 내려받은 가중치 폴더입니다.

```powershell
& $python -m frame_processing.experiments.classification predict $out --model heuristic
& $modelPython -m frame_processing.experiments.classification predict $out --model PP-DocLayout_plus-L `
    --model-dir $weights --no-cpu-fallback --boxes --cadence-frames 8
& $python -m frame_processing.experiments.classification score $out --models heuristic PP-DocLayout_plus-L
```

나머지 모델은 `--model PP-DocLayout-S`, `--model siglip2-large`, `--model doclayout-yolo`를
선택합니다. `score --models`에는 평가할 예측 폴더의 모델 이름을 전달합니다. 같은 폴더 안의
`labels.csv`에 수동 정답이 입력된 프레임을 채점합니다. `comparison.json`에는 지표와 혼동
행렬, `review_mistakes.html`에는 오분류가 기록됩니다. 기존 실험 이미지는 파일로 공유하지
않고, 판정 정책과 예측·평가 코드만 이 저장소에서 관리합니다.

## 테스트

```powershell
& $python -m unittest discover -s tests -v
```

합성 테스트는 동일한 픽셀과 타임스탬프를 넣은 서비스 파이프라인과 PNG 재생의 판정·그룹 ID·활성 대표 ID를 비교합니다. 대표 JPEG 저장 유무에 따른 비교 점수와 판정, 후보 순서를 바꾼 재현성, 라벨 수정 후 재평가, 프레임 누락과 이후 정보 보존의 구분을 확인합니다.
