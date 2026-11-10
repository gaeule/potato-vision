# P0 과거 평가 보고서 (저장 증거 한정)

기준일: 2026-11-05. **G005의 과거 자료 부분만 정리했으며 전체 P0/G005 완료가 아니다.** 신규 촬영 자료 질문의 저장 답변은 `확인이 필요함`이다. 학습·추론·내보내기·설치·위임·모델 역직렬화·임계값 변경 없이 저장 JSON/CSV/라벨/매칭만 읽었다.

## 1. 과제와 NA 계약

- **과제 A**: `0=potato`, `1=stone`, 프레임별 객체 박스 탐지. 반복 관측 수를 고유 물체 수로 해석하지 않는다. baseline은 감자 단일 클래스 모델이다.
- **과제 B**: 감자 개체별 `greening/sprouting/rot/damage` 다중라벨. 각 항목은 present/absent/unknown이며 동시 결함을 허용한다. 정상은 모두 absent이고 판정 가능할 때만 도출한다.
- B는 `NOT_TRAINED/NOT_EVALUATED`: 결함별 AP/P/R/F1/TP/FP/FN/TN, 정상 수, unknown 수, 판정 가능 분모 모두 NA다. A 탐지 지표를 B 성능으로 옮기지 않았다.
- JSON의 `null`은 NA이며 사유를 함께 기록한다. 실제 GT가 있는 클래스에서 관측된 TP=0/recall=0과 미학습·미평가를 구분한다.
- 요청의 `userclass-policy.md`는 저장 트리에 없었다. 실제 정책 `.omo/evidence/potato-p0-20261105/class-policy.md`를 적용했다(M10).

## 2. 모델·데이터·조건 결합

| 모델 ID | 저장 가중치 | 실제 SHA-256 |
|---|---|---|
| current | `retraining_experiment/outputs/selection/selected.pt` | `8b2cb7982fcd11f61a33cab8659de23128d0065cfa8e8966093be297a7e216c8` |
| stone40 | `stone_experiment/outputs/selection/selected.pt` | `06d46e0dc728ffbabf4078f79acaf9d32d809a57a46810acdf39a7927aa2bbd1` |
| baseline | `baseline/runs/train/weights/selected.pt` | `ae991c19df8beae8997af93636cd4e5855513fafbb45087f45a8fa78aa351b18` |
| current_last | `retraining_experiment/outputs/train/weights/last.pt` | `d734a9e5fe7bf8a60e1f406e788f996ba5534dc26f4fb3da00e6bc57bd215cc0` |
| stone_last | `stone_experiment/outputs/train/weights/last.pt` | `d80ebbe97d981cb11328dcb3fc95cdcdf13d2ffca72bcc1a9f3e9c5e76cd8878` |
| baseline_best | `baseline/runs/train/weights/best.pt` | `5d153d5f7d7797125ac2efc3752daac0b31921c06a9afcd564a29bce080fc3b6` |

현재 selected.pt의 실제 해시는 selection.json, 비교 provenance, 저장 무결성 수령증과 일치한다. current=88장 학습 native best(29에폭), stone40=40장 학습 native best(32에폭), baseline=40장 감자 단일 클래스 last(36에폭)이다. 에폭 매핑은 저장 검증 보고서/학습 보고서에 근거하며 모델 객체를 로드하지 않았다. baseline_best는 선택되지 않은 21에폭 체크포인트다. baseline_best의 현재 바이트 해시는 계산했으나 과거 selection.json에 후보 best 해시는 별도로 저장되지 않았다.

| 데이터 버전 | train 이미지/감자/돌 | val 이미지/감자/돌 | test 이미지/감자/돌 |
|---|---|---|---|
| v2 | 88 / 473 / 39 | 11 / 60 / 8 | 12 / 33 / 7 |
| stone | 40 / 222 / 16 | 11 / 60 / 8 | 12 / 33 / 7 |
| baseline 내부 | 40 / 222 / NA | 11 / 60 / NA | 12 / 33 / NA |
| 외부 | 학습 제외 | 선택 제외 | 166 / 192 / NA |

- 내부 학습은 원영상 0~84초, val11은 92~112초, test12는 120~146초의 시간 분할이다. 분할 간 프레임이 달라도 **동일 원영상·장면·반복 물체**이며 신규 촬영 세션 시험이 아니다. v2의 추가 검수 학습 프레임도 같은 영상이다.
- 외부166은 이전 개발에서 이미 본 탐색 자료다. 같은 RGB가 없다는 과거 검사는 다른 촬영일·다른 개체를 보장하지 않는다. 독립 신규 세션으로 승인된 자료가 아니다. 제공 라벨은 감자만이며 돌이 실제 0개라는 뜻이 아니다.
- 내부 라벨/추적 참조는 AI 시각 기록, 외부 라벨은 공급 라벨이며 인간 전문가 정답으로 승격하지 않는다.
- v2 snapshot/manifest/YAML 및 선택/비교 수령증 해시를 실제 파일과 대조했다. 정확한 데이터 해시·각 행의 모델 해시·현재 selected 해시·분할·조건·JSON pointer는 `historical-metrics.json`에 함께 저장했다.
- **운영점 V2**: conf=0.45, NMS IoU=0.5, 정답 매칭 IoU=0.5, imgsz=640, class-aware NMS, confidence 순서의 class-agnostic 1:1 GT 매칭.
- **운영점 Stone**: conf=0.45, NMS IoU=0.7, 매칭 IoU=0.5. 동일한 모델이라도 NMS 조건이 다른 행을 합치지 않는다.
- **운영점 Baseline**: conf=0.25, NMS IoU=0.7, 매칭 IoU=0.5. AP sweep과 별도다.

## 3. 고정 운영점 과거 탐지표

P=TP/(TP+FP), R=TP/(TP+FN), F1=2TP/(2TP+FP+FN). 원시 캐시에서 비교 756 model-frame의 매칭을 독립 재계산하여 aggregate JSON과 대조했다. 저장 GT는 실제 YOLO 라벨의 좌표와 대조했다. baseline conf0.25의 189장도 기존 예측 TXT에서 재계산했다. 아래 수치는 비율이며 퍼센트가 아니다.

| 행 ID (정확한 출처는 JSON의 동일 ID) | 모델/조건 | 감자 TP/FP/FN | 감자 P / R / F1 | 돌 TP/FP/FN | 돌 P / R / F1 | macro F1 |
|---|---|---|---|---|---|---|
| `v2_old_two_class_val` | stone40 / 0.45, NMS 0.5 | 59/3/1 | 0.951613 / 0.983333 / 0.967213 | 7/0/1 | 1.000000 / 0.875000 / 0.933333 | 0.950273 |
| `v2_old_two_class_test` | stone40 / 0.45, NMS 0.5 | 33/6/0 | 0.846154 / 1.000000 / 0.916667 | 6/0/1 | 1.000000 / 0.857143 / 0.923077 | 0.919872 |
| `v2_old_two_class_external` | stone40 / 0.45, NMS 0.5 | 183/72/9 | 0.717647 / 0.953125 / 0.818792 | NA | NA (GT 미보유/미평가) | 0.818792 |
| `v2_new_selected_val` | current / 0.45, NMS 0.5 | 59/0/1 | 1.000000 / 0.983333 / 0.991597 | 7/0/1 | 1.000000 / 0.875000 / 0.933333 | 0.962465 |
| `v2_new_selected_test` | current / 0.45, NMS 0.5 | 33/2/0 | 0.942857 / 1.000000 / 0.970588 | 6/1/1 | 0.857143 / 0.857143 / 0.857143 | 0.913866 |
| `v2_new_selected_external` | current / 0.45, NMS 0.5 | 182/23/10 | 0.887805 / 0.947917 / 0.916877 | NA | NA (GT 미보유/미평가) | 0.916877 |
| `stone_baseline_val` | baseline / 0.45, NMS 0.7 | 59/0/1 | 1.000000 / 0.983333 / 0.991597 | 0/0/8 | NA / 0.000000 / 0.000000 | 0.495798 |
| `stone_baseline_test` | baseline / 0.45, NMS 0.7 | 33/4/0 | 0.891892 / 1.000000 / 0.942857 | 0/0/7 | NA / 0.000000 / 0.000000 | 0.471429 |
| `stone_baseline_external` | baseline / 0.45, NMS 0.7 | 184/102/8 | 0.643357 / 0.958333 / 0.769874 | NA | NA (GT 미보유/미평가) | 0.769874 |
| `stone_new_val` | stone40 / 0.45, NMS 0.7 | 59/3/1 | 0.951613 / 0.983333 / 0.967213 | 7/0/1 | 1.000000 / 0.875000 / 0.933333 | 0.950273 |
| `stone_new_test` | stone40 / 0.45, NMS 0.7 | 33/6/0 | 0.846154 / 1.000000 / 0.916667 | 6/0/1 | 1.000000 / 0.857143 / 0.923077 | 0.919872 |
| `stone_new_external` | stone40 / 0.45, NMS 0.7 | 183/72/9 | 0.717647 / 0.953125 / 0.818792 | NA | NA (GT 미보유/미평가) | 0.818792 |
| `baseline_conf025_val` | baseline / 0.25, NMS 0.7 | 59/5/1 | 0.921875 / 0.983333 / 0.951613 | NA | NA (GT 미보유/미평가) | 0.951613 |
| `baseline_conf025_test` | baseline / 0.25, NMS 0.7 | 33/10/0 | 0.767442 / 1.000000 / 0.868421 | NA | NA (GT 미보유/미평가) | 0.868421 |
| `baseline_conf025_external` | baseline / 0.25, NMS 0.7 | 184/150/8 | 0.550898 / 0.958333 / 0.699620 | NA | NA (GT 미보유/미평가) | 0.699620 |

돌 GT가 있는 stone 비교에서 감자 단일 클래스 baseline의 stone FN=8/7 및 recall=0은 실제 관측이다. stone 예측 헤드가 없어 precision 분모는 0이므로 저장 코드의 0 대신 NA로 정정했다. 해당 macro F1은 stone F1=0을 포함한 과거 비교 산식이지 2클래스 모델의 동일 능력을 뜻하지 않는다. 외부 행의 macro F1은 감자 한 클래스 F1이다.

**현재 모델의 변화**: 동일 conf0.45/NMS0.5에서 외부 감자 FP는 72→23, FN은 9→10이다. 시험12장의 감자 F1은 0.916667→0.970588이지만 돌 F1은 0.923077→0.857143, 두 클래스 macro F1은 0.919872→0.913866이다. 모든 클래스가 개선됐다는 주장은 성립하지 않는다.

### 배경 포함 혼동행렬

행=GT, 열=예측, 내부 순서 `[potato, stone, background]`. current의 실제 계수는 다음과 같다.

| 분할 | 감자 GT 행 | 돌 GT 행 | 배경 FP 행 |
|---|---|---|---|
| val | [59, 0, 1] | [0, 7, 1] | [0, 0, 0] |
| test | [33, 0, 0] | [0, 6, 1] | [2, 1, 0] |

배경-배경의 저장값 0은 유한 TN을 센 값이 아닌 배열 자리표시자다. 정확도/TN은 NA다. 외부는 돌 예측을 **매칭 전에 제거**했으므로 감자/배경 2x2만 유효하다: `[[182,10],[23,NA]]`. 저장 3x3의 stone 행/열 0을 돌 완전검출 또는 돌 0개 증거로 쓰지 않았다. 다른 모델 혼동행렬도 JSON에 보존했다.

## 4. 체크포인트 선택 (시험·외부를 선택에 사용하지 않음)

| 후보 | 모델 | 검증 macro F1 | 선택 | 출처 |
|---|---|---|---|---|
| retraining_experiment_best_val_selection | current | 0.962465 | 예 | `retraining_experiment/outputs/selection/selection.json#/candidates/best` |
| retraining_experiment_last_val_selection | current_last | 0.962465 | 아니오 | `retraining_experiment/outputs/selection/selection.json#/candidates/last` |
| stone_experiment_best_val_selection | stone40 | 0.950273 | 예 | `stone_experiment/outputs/selection/selection.json#/candidates/best` |
| stone_experiment_last_val_selection | stone_last | 0.925103 | 아니오 | `stone_experiment/outputs/selection/selection.json#/candidates/last` |

선택 후보의 저장 매칭 44건을 합산 대조했다. V2 best/last는 동점이어서 사전 규칙(max val macro F1 → 감자 recall → native best)으로 best를 택했다. baseline의 선택은 val11/conf0.25 F1을 사용했지만, 최초 외부 탐색 후 운영점 선택 절차를 추가했다는 저장 경고를 유지한다(`baseline/selection/selection.json`).

## 5. AP sweep과 학습 중 val mAP는 별도

**명시적 클래스별 AP50/AP50-95는 확인한 저장 JSON/CSV에 없다. 모든 모델·분할의 클래스별 AP는 NA**이며 다음 표는 저장 aggregate mAP만 전사한다. baseline은 단일 potato 클래스이지만 집계 필드명을 바꾸지 않았다. baseline AP를 current/stone40 AP로 옮기지 않았다.

### 별도 저장된 baseline 검증 AP

metric confidence floor=0.001, NMS IoU=0.7, imgsz=640. 라이브러리 P/R은 데이터별 best-F1 지점이며 conf0.25/0.45의 운영 P/R이 아니다.

| 모델/분할 | mAP50 | mAP50-95 | 출처 |
|---|---|---|---|
| baseline / val | 0.993571 | 0.793375 | `baseline/evaluations/val/metrics.json#/metrics` |
| baseline / test | 0.995000 | 0.816793 | `baseline/evaluations/test/metrics.json#/metrics` |
| baseline / external | 0.910144 | 0.705800 | `baseline/evaluations/external/metrics.json#/metrics` |

external JSON의 library split 문자열은 `test`지만 dataset은 `external`이다. 내부 test12와 혼동하지 않았다.

### results.csv의 에폭별 trainer validation aggregate

| 모델 | 에폭 | mAP50 | mAP50-95 | CSV 행(헤더 포함) |
|---|---|---|---|---|
| current | 29 | 0.969290 | 0.809960 | `retraining_experiment/outputs/train/results.csv:30` |
| current_last | 44 | 0.988250 | 0.779480 | `retraining_experiment/outputs/train/results.csv:45` |
| stone40 | 32 | 0.952130 | 0.818850 | `stone_experiment/outputs/train/results.csv:33` |
| stone_last | 47 | 0.966760 | 0.686210 | `stone_experiment/outputs/train/results.csv:48` |
| baseline_best | 21 | 0.991380 | 0.826250 | `baseline/runs/train/results.csv:22` |
| baseline | 36 | 0.993570 | 0.794470 | `baseline/runs/train/results.csv:37` |

위 CSV는 전부 **학습 중 val** 결과다. current 29에폭 mAP50=0.96929/mAP50-95=0.80996을 test12 AP나 외부 AP로 쓰지 않는다. train args의 NMS IoU는 0.7이고 conf=null이다. 유효 runtime confidence 기본값은 수령증에 없어 NA다. 에폭별 P/R은 JSON에 원래 library 요약으로만 남겼으며 정수 TP/FP/FN을 역산하지 않았다. 재로딩 selected의 conf0.45/NMS0.5 지표와 동일하지 않다.

## 6. 지원되는 층화와 지원되지 않는 층화

### 크기: 원본 GT 박스 면적 기반의 사후 기술 통계

원본 1280×720 좌표에서 small<32², 32²≤medium<96², large≥96² pixel². 물리적 mm, 실제 물체 크기 또는 COCO AP 결과가 아니다. GT 크기에 따라 TP/FN/recall만 재집계한다. FP 크기 할당 규약이 없으므로 크기별 precision/F1은 NA이며 conf0.45 캐시로 AP를 만들지 않았다.

| current 분할/클래스 | small GT/TP/FN/recall | medium GT/TP/FN/recall | large GT/TP/FN/recall |
|---|---|---|---|
| val/potato | 1/0/1/0.000000 | 58/58/0/1.000000 | 1/1/0/1.000000 |
| val/stone | 1/0/1/0.000000 | 7/7/0/1.000000 | 0/0/0/NA |
| test/potato | 0/0/0/NA | 33/33/0/1.000000 | 0/0/0/NA |
| test/stone | 1/0/1/0.000000 | 6/6/0/1.000000 | 0/0/0/NA |
| external/potato | 0/0/0/NA | 178/168/10/0.943820 | 14/14/0/1.000000 |

### 빈 GT 이미지: 실제 음성 부분집합의 FP만 집계

| current 분할 | 빈 GT 이미지 | FP 박스 | FP 있는 이미지 | P/R/F1/AP/accuracy |
|---|---|---|---|---|
| val | 0 | 0 | 0 | NA (양성 GT/평가 분모 없음) |
| test | 3 | 0 | 0 | NA (양성 GT/평가 분모 없음) |
| external | 87 | 9 | 9 | NA (양성 GT/평가 분모 없음) |

external 87장의 빈 GT는 공급 라벨 기준이다. 누락 정답이 전혀 없다고 보장하지 않는다. 빈 GT 장면과 의미별 배경(장갑/흙/설비)은 다른 층화다. 모든 모델의 빈 GT FP는 JSON에 있다.

**가림별 성능과 의미별 배경 성능은 NA**다. 실제 주석에는 일관된 개체별 가림/가시율 및 배경 유형 태그가 없다. 특정 장면의 서술을 새 가림 라벨로 만들거나 태그 없는 이미지에 추측 라벨을 붙이지 않았다(M05/M06).

## 7. 추적: 한 세션 AI 참조와의 대응

모델은 current `8b2cb7982fcd11f61a33cab8659de23128d0065cfa8e8966093be297a7e216c8`. 원영상 저장 SHA-256은 `1b931cf852c3f4a97652ae94b800447d302ab7a1bebca7ae95905a61e23e3b42`. 검출 캐시는 conf0.45/NMS0.5이며 추적은 y=360, x=220~1000, 위쪽 통과, hysteresis=8px, max_gap=0.5초, max_distance=100, min_hits=2다. 전체2224프레임이며 ID131개가 고유 감자131개를 뜻하지 않는다.

| 저장 대조 | TP/FP/FN | P | R | F1 | 해석 |
|---|---|---|---|---|---|
| `tracking_experiment/review/full_comparison.json` | 100/0/2 | 1.000000 | 0.980392 | 0.990099 | one_to_one_greedy; time within reference bracket +-0.35s; |x| <= 90px |
| `tracking_experiment/review/full_comparison_strict.json` | 99/1/3 | 0.990000 | 0.970588 | 0.980198 | one_to_one_greedy; time within reference bracket +-0.1s; |x| <= 50px |
| 같은 감자 재통과 2건 제외 조건부 | 100/0/0 | 1.000000 | 1.000000 | 1.000000 | E077/E097 제외, 원참조의 distinct estimate=100에 한함 |

원참조에는 감자 통과102건·돌10건이 있고 자동 통과는100건이다. 실제 대응쌍 길이·ID 유일성·원사건 frame/ID·반올림 시각/x를 대조했다. 원사건102건을 그대로 분모로 쓰면 기본 매칭에서도 FN=2이고 recall=100/102다. 원참조는 E077/E097를 재통과로 보되 identity/반전 여부를 ambiguous로 표시한다. 제외 후 100대응은 이 조건 아래의 AI 참조 일치이지 **일반화 정확도100%, 사람 정답 또는 물리적 고유 감자100개 확정**이 아니다. 엄격 매칭은 새로 임계값을 조정한 것이 아닌 저장된 민감도 대조 결과다.

## 8. 음성·미학습·unknown 사례

- 실제 `new_selected/external/extra2_t000000000ms_f000000.json`은 truth=[]와 predictions=[]다. P/R/F1/AP/accuracy는 NA이며 100% 또는 0% 정확도로 만들지 않았다.
- current external 빈 GT87장에는 FP9개(9장)가 있다. FP는 보존하지만 양성 GT 없는 부분집합을 성능0으로 치환하지 않았다.
- B 미학습: 모델/GT/예측 부재이므로 모든 결함 지표 NA와 NOT_TRAINED/NOT_EVALUATED다.
- B unknown: absent/정상 정답으로 세지 않는다. 현재 개체별 B 기록 자체가 없어 unknown 개수도 NA다. 이 항목은 정책 음성 사례이며 실제 관측 표본을 발명한 테스트가 아니다.

## 9. 결측과 다음 검증 (이번에는 수행하지 않음)

| ID | 결측 | 정확한 사유 / 제안 |
|---|---|---|
| M01 | 신규 세션 일반화 (NOT_EVALUATED) | 수락된 새 촬영 세션 없음; 사용자 답변은 확인이 필요함. 외부166장의 세션 독립성도 미확정 / 다음: 사용자에게 새 자료 위치와 세션 provenance 확인 후 기존 학습/선택/평가 가족 제외, 사람 검수 GT로 별도 동결 시험 계획 수립 |
| M02 | 과제 B 네 결함 다중라벨 (NOT_TRAINED/NOT_EVALUATED) | B 체크포인트·개체별 결함 GT·예측 없음; 과제 A 성능 재사용 불가 / 다음: 별도 B 라벨 계약으로 독립 검수 자료를 확보하고 별도 승인된 학습·평가 후 결함별 AP/P/R/F1/TP/FP/FN/TN 보고 |
| M03 | 현재·이전 2클래스의 클래스 AP (NA) | 저장 비교는 고정 conf0.45 매칭뿐이며 CSV는 aggregate val mAP뿐; 모델/분할별 potato/stone AP50/AP50-95 미저장 / 다음: 향후 별도 승인 시 해당 모델·동결 분할의 표준 저-confidence 검증 결과와 클래스별 AP를 저장; 이번에는 캐시에서 AP 계산하지 않음 |
| M04 | 외부166 돌 GT (NA) | 외부 라벨은 potato만; stone GT 없음은 돌 0개 확인이 아님 / 다음: 감자·돌·unknown을 완전 재검수한 별도 정답 버전 확보 전 stone P/R/F1/AP/혼동행렬을 NA로 유지 |
| M05 | 가림별 성능 (NA) | 확인한 v2 프레임 스키마와 YOLO 라벨 및 stone snapshot에 체계적 개체 가림 태그 없음; 서술 사례를 태그로 승격하지 않음 / 다음: 모델 결과를 보지 않고 원자료에 가림 메타데이터를 검수한 뒤 해당 동결 분할로 재집계 |
| M06 | 배경 유형별·크기별 완전 지표 (PARTIAL) | 픽셀 박스 면적과 빈 GT 이미지 지표만 지원; 장갑/흙/설비의 일관된 배경 태그 및 크기별 FP 할당 정책 없음 / 다음: 픽셀 크기별 GT/TP/FN/recall과 빈 GT의 FP만 사용; 추가 성능은 별도 태그와 정책 동결 후 검증 |
| M07 | 사람 전문가 정답 (UNVERIFIED) | 내부 라벨과 추적 참조는 AI 시각 기록; 외부 공급 라벨도 expert gold 아님 / 다음: 사람 검수 정답과 기존 AI 참조를 별도 버전으로 관리하여 새 세션 검증에 사용 |
| M08 | 탐지 TN·정확도 및 빈 GT (NA) | 객체 탐지의 유한 배경 TN 미정의; 빈 GT만으로 0% 또는 100% 정확도 주장 불가 / 다음: 현재는 TP/FP/FN 및 정의된 P/R/F1만 보고; 빈 GT 부분집합은 FP와 이미지 수만 보고 |
| M09 | 학습 CSV 유효 AP 조건·개별 AP (PARTIAL) | args.yaml conf:null, NMS0.7은 확인; 실행 시 유효 conf 기본값과 클래스별 AP/에폭별 매칭은 수령증에 없음 / 다음: CSV를 학습 중 val aggregate로만 유지하고 별도 재로딩 평가와 구분; 차후 승인 검증에서 유효 설정 저장 |
| M10 | 정확한 정책 파일명 (RESOLVED_WITH_ALTERNATE) | 요청의 userclass-policy.md는 없음; 실제 P0 class-policy.md를 읽고 적용 / 다음: 현재 저장된 class-policy.md 계약을 적용하며 다른 정책이 있으면 차후 대조 |
| M11 | 추적의 신규 영상·사람 계수 (NOT_EVALUATED) | 100 대응은 한 원영상의 AI 참조; 두 재통과 identity는 원참조에서도 ambiguous / 다음: 새 영상과 독립 사람 사건 GT를 받아 검증; 기존 strict 비교는 저장된 민감도 결과로만 보고하고 설정 변경하지 않음 |

정확한 부족 메타데이터·영향 범위·출처는 `missing-evidence.json`에 있다. 새 세션 자료의 존재/부재를 임의 확정하지 않았고 다른 자료 수집·학습·평가 실행은 하지 않았다.

## 10. 출처와 검증 범위

- 주 집계: `retraining_experiment/outputs/comparison/comparison.json`, `stone_experiment/outputs/comparison/comparison.json`, 각 하위 split의 실제 프레임 JSON 및 summary.json.
- 선택: 두 실험 `outputs/selection/selection.json`, 후보별 실제 프레임 JSON. baseline 선택은 `baseline/selection/selection.json`.
- AP/고정 PR: `baseline/evaluations/{val,test,external}/metrics.json`과 `fixed_confidence_metrics.json`, 기존 `predictions/labels/*.txt`.
- 학습 CSV: 세 모델의 results.csv 및 args.yaml; hash/epoch 연결은 저장 체크포인트·선택·검증 보고서.
- 추적: `tracking_experiment/outputs/video/{summary,events}.json`, `review/{full_reference,full_comparison,full_comparison_strict}.json`.
- 정책/신규 세션: `.omo/evidence/potato-p0-20261105/{class-policy.md,new-session/inventory.json}`.
- 파일별 SHA-256, 행별 JSON pointer, 데이터 해시, 매칭 캐시 디렉터리/manifest digest는 `historical-metrics.json`의 provenance에 있다. 캐시 digest는 summary.json을 제외한 basename→바이트 SHA256 매핑을 sort_keys=True JSON으로 직렬화한 SHA256이다.
- 문서 검증은 JSON 파싱·행별 집계/수령증/산식 대조·원본 해시 재대조로 수행한다. 새 코드나 prose 테스트는 추가하지 않는다. 모델 실행/전체 MP4 재해싱·새 학습/신규 세션 검증은 범위 밖이다. 자체 하위 프로세스는 시작하지 않았다.

**범위 종료:** 과거 수치와 결측 목록만 준비했다. 신규 세션 질문은 미결이며 G005/P0 전체 완료·현장 배포 승인·일반화 성능 보증을 주장하지 않는다.
