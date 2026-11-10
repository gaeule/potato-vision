# 재학습용 데이터 준비 전용

## 명시적 v2 profile

기본 profile은 계속 **v1**이다. 아래 기존 설명의 `label_review/`, 79/11/12,
legacy 44/46/48초 전체 제외 규칙과 기존 dataset은 변경하지 않는다.
`--profile v2`만 새 `label_review_v2/`를 선택하며 동일한 native 검증기와
rectangle 변환기를 사용한다. 모델 로딩이나 학습 통합은 없다.

| v2 split | 프레임 | potato (0) | stone (1) |
|---|---:|---:|---:|
| train | 88 | 473 | 39 |
| val | 11 | 60 | 8 |
| test | 12 | 33 | 7 |

v2 train은 원본 legacy 40장(P222/S16)을 **바이트 그대로 모두 유지**하고
새 후보 48장(P251/S23)을 추가한다. 총 111장이다. v1 입력의 정확히 42 후보/
8 held와 v2 입력의 정확히 48 후보/2 held를 각각 강제한다.

2026-10-26 사용자의 시각 확인은 43, 43.5, 45, 47, 47.5, 49초 slab가
stone이라는 재질 확인이며, 동일한 legacy 44/46/48초도 복원한다.
65초 확인 대상은 **이미 stone으로 라벨된 큰 흰 돌**이다. 위쪽 작은 fragment의
확인이 아니므로 65초와 71초의 top fragment 프레임은 계속 held로 제외한다.
클래스나 박스는 이 도구가 수정하지 않는다.

Chief가 생성하는 `label_review_v2/manifest.json`의 `source_pack`과
`reviewer_type`은 기존 native AI 계약을 유지한다. 재질 확인 provenance는
별도 `label_review_v2/user_confirmation.json`에 두며 v2 입력 inventory에
해시와 JSON 원문을 포함한다. 전체 `human_reviewed=false`, `native_checked=false`,
native `checked=false`는 유지한다. v2 snapshot/manifest는 각각
`retraining-dataset-v2`/`retraining-manifest-v2`로 구분하여 다른 profile 검증을
거부한다. v1의 스키마와 직렬화 결과는 그대로 유지한다.

```bash
PY=/home/sc/.cache/potato-baseline/.venv/bin/python

# source-ready 확인 뒤 임시 smoke만 실행; canonical dataset_v2는 만들지 않음
$PY -B -m retraining_data.smoke --profile v2

# Chief 전용 정식 생성과 독립 검증
$PY -B -m retraining_data.prepare --profile v2
$PY -B -m retraining_data.verify --profile v2
```

v2 기본 출력은 `retraining_data/outputs/dataset_v2`다. `--output`/`--dataset`으로
새 경로를 지정해도 profile은 명시해야 한다. 기존 출력 거부 정책은 같다.
Python API는 `build_plan(V2)`, `prepare(path, V2)`, `verify_dataset(path, V2)`이며
`V2`는 `retraining_data.schema`에서 가져온다. 인자를 생략하면 v1이다.

## 기존 v1 데이터

정식 데이터 생성과 독립 검증을 완료했다. 실제 수량과 검사 결과는
`review/RESULTS.md`에 정리했다. 이번 작업에서 모델 학습은 실행하지 않았다.

학습·모델 로딩·추론·패키지 설치를 수행하지 않는다. 기존 데이터와 검수 파일은
읽기 전용이다. `retraining_data/review/`는 chief 소유이며 이 도구는 쓰지 않는다.

## 포함 범위

| split | 프레임 | potato (0) | stone (1) |
|---|---:|---:|---:|
| train | 79 | 417 | 30 |
| val | 11 | 60 | 8 |
| test | 12 | 33 | 7 |

기존 `stone_experiment/outputs/dataset`의 train 40장 중 **44, 46, 48초 프레임
전체**를 제외하고 37장을 유지한다. Chief의 시각 확인으로 held review 43..49초와
동일한 미확정 slab로 판정된 세 프레임이다. 해당 프레임의 potato 5/7/6개와
stone 각 1개를 모두 제외한다. 다른 legacy 프레임은 제거하지 않는다.
여기에 `label_review/manifest.json`의 eligible AI 후보 **정확히 42장**을 train에
추가한다. `needs_confirmation` 8장과 외부 166장 ZIP은 전부 제외한다.
ZIP을 열거나 해시·압축 해제하지 않는다.

val/test의 이미지와 라벨, 유지한 legacy train의 이미지와 라벨은 원본 바이트 그대로
복사한다. 후보 라벨만 native rectangle에서 소수점 9자리 YOLO로 변환한다.
빈 native shapes는 0바이트 라벨로 유지한다.

## 실행

워크스페이스 루트에서 이미 준비된 인터프리터를 사용한다. 설치 명령은 필요 없다.

```bash
PY=/home/sc/.cache/potato-baseline/.venv/bin/python

# 소규모 결정적 CPU 검증: 임시 출력은 retraining_data/outputs/smoke_* 아래 생성 후 삭제
$PY -B -m retraining_data.smoke

# 실제 정식 생성: chief가 실행할 명령
$PY -B -m retraining_data.prepare

# 별도 검증기: 정식 생성 후 실행
$PY -B -m retraining_data.verify

# 다른 새 출력 경로도 retraining_data/outputs/ 하위만 허용
$PY -B -m retraining_data.prepare --output retraining_data/outputs/run_001
$PY -B -m retraining_data.verify --dataset retraining_data/outputs/run_001
```

기본 출력은 `retraining_data/outputs/dataset`이다. 존재하는 디렉터리/파일은
**무조건 거부**한다. overwrite/regenerate 옵션은 없다. 실패한 부분 출력도
완성본으로 취급하지 않으며 같은 경로로 재시도할 수 없다. 새 경로를 사용한다.
`manifest.json`은 마지막에 생성하며, 성공 여부는 별도 verifier로 확인한다.

## 검증과 기록

- 기존 `stone_experiment.prepare.verify_dataset`을 **그대로** 호출해 63장짜리
  원본의 라벨·이미지·스냅샷·YAML을 확인한다. 기존 검증기의 63장 불변식을
  새 데이터에 맞춰 약화하지 않는다.
- manifest와 native의 eligibility, review status, `needs_confirmation`,
  `model_proposal`, `ai_visual_reviewed`, `human_reviewed`, `checked`, 원본 JSON/
  이미지 SHA-256 및 source pack manifest·동영상 해시를 대조한다.
- 요청/실제 시간과 PTS/time_base를 대조한다. 요청 시간은 소수 초를 지원하며
  파일명에서 시간을 추론하지 않는다. 안전한 PNG basename은 임의 이름을 허용하되
  출력 파일명 중복은 금지한다.
- 모든 입력 PNG의 1280×720 크기, 파일 해시, 디코딩한 RGB 원시 바이트 해시를
  기록·확인한다. rectangle은 유한 좌표, 두 대각 꼭짓점 또는 네 실제 사각 꼭짓점,
  이미지 범위와 양의 면적을 요구한다. 범위 밖 박스를 자동 보정하지 않는다.
- 요청 시간 구간은 train 0..84초, val 92..112초, test 120..147초다.
  원본 추출은 요청 직후 첫 프레임을 사용하므로 실제 시간은 요청 이상, 요청+1초
  미만이어야 한다. 예를 들어 기존 val 112초 프레임의 실제 PTS는 112.041666초다.
  legacy PTS는 baseline manifest와 정확히 대조하며 split 사이 공백은 유지한다.
  split 사이 정확한 RGB 픽셀 중복을 거부한다. 신규 후보는 제외된 3장을 포함한
  **전체 legacy 63장** 및 이미 추가한 후보와 정확한 픽셀이 같으면 거부한다.
- `annotations.json`은 새 스키마의 전체 스냅샷이다. 프레임별 이미지/라벨
  해시·출처·시간·클래스 수, 제외 이유/목록, 전체 legacy 픽셀 해시, 실제 읽은
  입력 파일의 SHA-256을 포함한다. JSON/YAML 원문도 텍스트로 내장하므로 native
  uncertainty/notes/shape metadata와 held-frame 불확실성이 그대로 보존된다.
- `manifest.json`은 스냅샷/YAML/전체 출력 라벨 해시를 묶는다. 별도 verifier는
  원본에서 계획을 다시 구성하고 전체 파일 구성과 실제 바이트를 비교한다.
  검증에는 원본 워크스페이스가 필요하다.
- `data.yaml`은 `path`를 생략해 YAML 파일 위치를 기준으로 split을 찾는
  YOLO 형식이다. 목적지가 달라도 결과 바이트가 같게 유지된다.

## 검수 상태와 한계

`human_reviewed=false`, native `checked=false`, 원본의 `training_inclusion=false`를
변경하거나 사람 검수 완료로 승격하지 않는다. AI 후보를 실험 데이터에 선택한
것이지 사람의 정답 확인이 아니다. 픽셀 중복 검사는 물리적으로 서로 다른 개체라는
증명이 아니다. 클래스 수는 고유 물체 수가 아닌 프레임별 관측 수다.

**기존 `stone_experiment/train.py`는 새 102장 데이터를 받지 못한다.** 기존
63장 불변식에 의해 거부되는 것이 정상이다. 학습 진입점/모델/평가/통합 변경은
다음 단계의 범위이며 여기에는 학습 래퍼나 실행 예제를 추가하지 않는다.
정식 데이터 생성과 독립 검증은 chief가 담당한다.

CPU smoke는 실제 소스 기반 포함/제외, 경계/빈 라벨 변환, 부적격 후보 거부,
기존 출력 거부, 실제 파일 변조 검출, 목적지 간 바이트 재현성을 검사한다.
모델 성능이나 라벨의 의미적 정확성은 검사하지 않는다.
