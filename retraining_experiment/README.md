# v2 재학습 실험

**실제 학습·선택·성능 비교·두 전체 영상 생성과 검증을 완료했습니다.**
관측 결과와 한계, Windows 실행창용 경로는 `RESULTS.md`를 확인하십시오.

원본 코드·라벨·`dataset_v2`·기존 모델을 변경하지 않는 실행 패키지입니다.
GPU 학습·선택·비교·전체 영상 실행은 chief가 순서대로 수행합니다.
`review/**`는 chief 전용이며 이 패키지는 그 경로에 기록하지 않습니다.

## 실행

저장소 루트에서 고정 환경을 사용합니다. 각 명령은 **새 Python 프로세스**에서
실행해야 합니다. 이미 존재하는 출력 디렉터리는 덮어쓰거나 재사용하지 않습니다.
실패한 실행도 같은 경로로 재시도하지 말고 새로운 `--output`을 지정하십시오.

```bash
cd /mnt/c/Users/USER/Desktop/협력/potato
PY=/home/sc/.cache/potato-baseline/.venv/bin/python

# CPU 검증만 실행: 모델 로드·학습·GPU 추론 없음
$PY -B -m retraining_experiment.smoke

# 이하 GPU 명령은 chief 실행
$PY -B -m retraining_experiment.train
$PY -B -m retraining_experiment.select
$PY -B -m retraining_experiment.evaluate
$PY -B -m retraining_experiment.video
```

기본 입력은 `retraining_data/outputs/dataset_v2`입니다.
`train --dataset 경로 --output 새경로`,
`select --training 학습출력 --dataset 경로 --output 새경로`,
`evaluate --weights 새selected.pt --baseline 기존selected.pt --dataset 경로 --output 새경로`,
`video --weights selected.pt --output 새경로`를 지원합니다.
모든 새 출력은 반드시 `retraining_experiment/outputs` 아래에 있어야 합니다.
사용자 지정 학습 출력을 사용했다면 다음 선택 명령에 그 경로를 전달해야 합니다.
비교·영상 명령의 `--weights`도 사용자 지정 선택 출력에 맞추십시오.

## 학습 격리와 출처

`retraining_data.verify.verify_dataset(path, V2)`로 원본 111장
(train 88 / val 11 / test 12)을 확인합니다. 학습 직전에 train과 val의
이미지·라벨 99쌍만 새 `train/training_dataset`으로 **실제 복사**합니다.
심볼릭 링크와 하드 링크를 사용하지 않습니다. 학습용 YAML에는 train/val만
존재하며 원본·test·외부 데이터 경로는 없습니다.
Ultralytics는 이미지 캐시가 꺼져 있어도 라벨 캐시를 만들 수 있으므로,
캐시 기록은 이 개인 복사본에서만 일어납니다.

`data-provenance.json`에는 원본 111장 snapshot 전체, 원본 snapshot/manifest/YAML
SHA-256, 모든 복사 이미지·라벨의 SHA-256, 개인 YAML 해시를 기록합니다.
`provenance.json`과 `environment.json`에는 코드·두 기준 설정·원본 영상·체크포인트·
Python 실행 파일·환경 해시와 실행 환경을 기록합니다.
학습 종료 후 native `weights/best.pt`, `weights/last.pt` 해시를 기록합니다.

기준 `baseline/config/train.json` 및 `stone_experiment/config/train.json`과
설정 전체가 같은지 확인합니다. 루트의 `yolo11s.pt`를 절대 경로로 로드하고
COCO 80클래스를 확인한 후 새 2클래스 모델을 학습합니다. 기존 학습 모델의
fine-tuning이나 resume을 하지 않습니다. SGD, seed 42, 640 입력, 60 epochs,
patience 15, batch 8, workers 0, CUDA 0, AMP 및 모든 명시된 설정은 그대로입니다.

프로세스별 `runtime/process-PID-UUID` 아래에 YOLO/MPL/TORCH/XDG 및
컴파일 캐시 위치를 **Ultralytics import 전** 설정합니다.
작업 디렉터리도 이 개인 경로로 옮겨 상대 경로의 AMP 검사 가중치 다운로드가
원본 루트에 기록되지 않도록 합니다. 외부 로깅 연동은 끕니다.
고정 버전은 기존 검사와 동일한 Ultralytics 8.4.183, torch 2.9.1+cu128,
torchvision 0.24.1+cu128입니다. 설치·업그레이드를 수행하지 않습니다.
chief의 `review/protocol.json`에 고정된 원본 데이터·COCO 초기값·두 기준 설정·
비교 기준 모델 해시도 실행 전에 검사합니다. `review/**`는 읽기 전용입니다.
기준 환경에 없는 Pydantic/Typer/pytest를 추가하지 않고 기존 저장소 방식의
frozen dataclass, argparse, 실행형 CPU smoke를 사용합니다.

## 사전 고정된 선택·평가

모든 선택·비교·영상은 `postprocess_review.common.infer`와
`Postprocess(confidence=0.45, nms_iou=0.5)`를 공유합니다.
매칭 IoU는 0.5이며 기존 `stone_experiment.metrics`의 confidence 순서,
클래스 비종속 1:1 매칭을 그대로 사용합니다. 파라미터 탐색 옵션은 없습니다.

선택은 native best/last를 각각 **새로 로드**하여 val 11장만 점수화합니다.
macro F1 최대 → potato recall 최대 → native best 순으로 동점을 해소합니다.
`selection/selected.pt`와 `selection.json`에는 기준·후보 점수·체크포인트 해시·
환경 및 데이터 출처 해시를 남깁니다. test/외부 데이터는 선택에 쓰지 않습니다.
v2 검증 자체는 전체 snapshot을 확인하지만 선택 추론에는 val만 전달됩니다.

비교 기준은 이전 40장 2클래스 실험의
`stone_experiment/outputs/selection/selected.pt`입니다.
새 선택 모델과 기존 모델 모두 동일한 val/test/외부 166장에서 평가합니다.
내부 정답은 검증된 v2 YOLO 라벨에서 `Example`로 만들고, 소수점 9자리
반올림의 경계 오차(정규 좌표 1e-9)만 허용합니다.
외부 데이터는 potato-only 정답이므로 **매칭 전** stone 예측을 제거합니다.
외부 stone precision/recall/F1은 계산하지 않습니다.
각 이미지 JSON에는 원시 검출·매칭·미검출·배경 FP를, 각 split summary에는
클래스별 집계와 confusion matrix(행 정답 / 열 예측: potato, stone, background)를
기록합니다. 값은 프레임 관측 수이며 고유 개체 수가 아닙니다.

## 전체 영상

원본 2224프레임 모두 추론합니다. stream과 codec time base 양쪽을 1/90000으로
설정하고 각 프레임의 원본 PTS·time base·duration을 전달합니다.
B-프레임은 끄고(`bf=0`) 원본 PTS에 대응하는 packet duration을 명시하여
마지막 프레임의 구간 길이까지 유지합니다.
H.264/no-audio로 저장한 후 전체 결과를 다시 디코딩하여 프레임 수·PTS·time base·
duration·codec·무음 조건을 확인합니다.
화면 하단의 `ON SCREEN ... (not unique)`는 해당 프레임의 검출 수입니다.
추적·고유 감자 수·구간 누적 개체 수를 계산하지 않습니다.

`video/counts.json`에는 모든 프레임의 클래스별 수, PTS/duration 및 원시 검출을
저장합니다. 20, 35, 65, 70, 71, 78, 80, 83, 130, 134초 각각에 대해 해당 시각
이상의 첫 원본 프레임을 선택하여 `diagnostics`에 원본 PNG·주석 PNG·검출 JSON을
저장합니다. 시각을 성능에 따라 바꾸지 않습니다. 실제 화면 검수는 chief 담당입니다.

## CPU 검사와 실행 한계

```bash
/home/sc/.cache/potato-baseline/.venv/bin/ruff check --no-cache retraining_experiment
/home/sc/.cache/potato-baseline/.venv/bin/ruff format --check --no-cache retraining_experiment
/home/sc/.cache/potato-baseline/.venv/bin/basedpyright \
  --pythonpath /home/sc/.cache/potato-baseline/.venv/bin/python retraining_experiment
/home/sc/.cache/potato-baseline/.venv/bin/python -B -m retraining_experiment.smoke
```

CPU smoke는 실제 v2 입력과 임시 `outputs/cpu-smoke-*`를 사용한 뒤 삭제합니다.
분할 격리·복사 해시·독립 inode·원본 불변성·잘못된 출처 거부·출력 충돌·경계 라벨·
고정 파라미터 전달·선택 동점 규칙·potato-only 매칭 및 CLI import/help를 확인합니다.
GPU 전용 inference/conversion 경계만 mock으로 대체하고 실제 매칭·집계·감사 JSON
기록은 실행합니다. 실제 GPU 학습·선택·비교·전체 영상 결과를 대신 검증하지는 않습니다.
추가 회귀 검사는 명령별 기본 출력 경로, 선택 체크포인트 변조 거부,
실제 CPU 인코딩의 가변 구간과 마지막 프레임 길이를 포함합니다.
AMP 검사용 보조 가중치 또는 폰트가 없으면 고정 Ultralytics가 개인 runtime 경로에
다운로드할 수 있으므로 해당 실행 환경의 네트워크 조건을 확인하십시오.

현재 고정 환경의 basedpyright는 오류 0개, 경고 21개로 종료 코드 1을 반환합니다.
경고는 Ultralytics `settings.update`/`model.train` 및 PyAV
`add_stream`/`encode`/`mux`의 불완전한 타입 정보에서 발생합니다.
경고를 숨기는 설정이나 `type: ignore`는 추가하지 않았습니다.
LSP 도구의 전역 실행 경로에는 서버가 없어서 위 고정 환경의 basedpyright CLI로
검사했습니다. GPU 명령과 실제 영상 화면 검수는 CPU smoke 결과에 포함되지 않습니다.
실제 GPU 명령 및 영상 검수 결과는 별도 `review/verification.json`에 기록했습니다.

선택 전에 native best/last 해시를 학습 종료 기록과 대조합니다.
비교·영상은 검증에서 선택한 체크포인트 해시 또는 고정된 기존 모델 해시를 확인합니다.
외부 166장과 라벨·manifest는 학습 전에 확보한 보존 목록과 파일 구성·해시를
대조한 뒤에만 평가합니다.

코드는 CLI·실행 경계·자료 변환·학습·선택·평가·영상·검사로 책임을 나눴습니다.
타입 억제나 캐스트를 추가하지 않았습니다. `smoke.py`는 순수 코드 236줄로
크기 경고 구간이므로 향후 검사 항목을 크게 늘릴 때 검사 영역별 분리를 먼저 고려해야 합니다.
