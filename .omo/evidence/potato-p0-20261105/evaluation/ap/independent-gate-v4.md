# AP 행렬 독립 최종 감사 v4

감사일: 2026-11-09. **판정: PASS — 기존 Task A의 AP 6개 단위 완료를 인정한다.** 고유 성공 6개, 보존 실패 1개, 실제 `model.val` 누적 7회다. 회귀가 있으므로 “current가 모든 클래스·분할에서 개선됐다”는 주장은 기각한다. 부모 집계 수치는 승인하며 비차단 문서 주의사항 2개를 아래에 남긴다. 새 세션 일반화, Task B 또는 전체 P0/G005 완료 판정이 아니다.

경로 기준은 이 문서의 `evaluation/ap/`이다. 전체 정밀도 지표, all_ap, 실제 설정, 여섯 실행 영수증·완료 이벤트·해시는 `verified-ap-v4.json`에 있다.

## 1. 동일 조건의 클래스 AP와 회귀

AP는 0~1 비율이며 아래는 소수점 6자리다. 각 셀은 AP50 / AP50:95, 차이는 current − stone40의 **퍼센트포인트**다. 반올림 전 수치로 판정했다.

| 분할 | 클래스 | current AP50 / AP50:95 | stone40 AP50 / AP50:95 | 차이 pp: AP50 / AP50:95 |
|---|---|---:|---:|---:|
| val11 | potato | 0.995000 / 0.881642 | 0.993696 / 0.857951 | +0.130435 / +2.369156 |
| val11 | stone | 0.962273 / 0.754506 | 0.918636 / 0.784903 | +4.363636 / -3.039610 |
| test12 | potato | 0.989706 / 0.890029 | 0.988740 / 0.868657 | +0.096562 / +2.137219 |
| test12 | stone | 0.855000 / 0.716792 | 0.861282 / 0.731468 | -0.628205 / -1.467628 |
| external166 | potato | 0.943964 / 0.786134 | 0.947635 / 0.793546 | -0.367162 / -0.741274 |
| external166 | stone | NA / NA | NA / NA | NA / NA |

감자는 내부 val/test에서 두 AP가 모두 상승했다. 그러나 **val 돌 AP50:95 −3.039610 pp**, **test 돌 AP50 −0.628205 pp 및 AP50:95 −1.467628 pp**, **외부 감자 AP50 −0.367162 pp 및 AP50:95 −0.741274 pp**로 회귀했다. val 돌 AP50은 +4.363636 pp여서 같은 클래스 안에서도 지표 방향이 다르다. 비교 가능한 5개 클래스·분할 중 두 AP가 모두 개선된 것은 2개, 하나 이상 하락한 것은 3개다. 외부 돌은 GT 미보유의 NA이며 0점도, 완전검출도 아니다.

## 2. 실제 실행 및 완료 증거

| 단위 | 최신 outer runner 영수증 | monitor | runner / worker exit |
|---|---|---|---:|
| current__val11 | results-v3/current__val11.json | mon_YQPW6ZVY07W1Z9BS | 0 / 0 |
| current__test12 | results-v4/current__test12.json | mon_9BRGRAGANMZK7VVJ | 0 / 0 |
| current__external166 | results-v4/current__external166.json | mon_BA6V56K3MNP63P1F | 0 / 0 |
| stone40__val11 | results-v4/stone40__val11.json | mon_ZWCNWJMDVFQJV8KC | 0 / 0 |
| stone40__test12 | results-v4/stone40__test12.json | mon_WPZMW7CN094T0WXC | 0 / 0 |
| stone40__external166 | results-v4/stone40__external166.json | mon_2MF98WCEA04FPNCQ | 0 / 0 |

각 단위의 승인 argv, `intent.json`, `exit.json`, `actual-settings.json`, `result.json`, `completed.json`, `events.jsonl` 및 stdout/stderr를 직접 대조했다. 여섯 `validation_completed.result_sha256`가 실제 result 바이트와 completed의 해시에 일치한다. 프로토콜·승인·체크포인트·단위 결속, worker 종료 0/result_exists=true, 로그 해시도 일치한다. 성공별 native 산출물 7개, 총 42개 해시가 최신 실행 영수증과 일치한다. 여섯 성공 stderr는 비어 있고 stdout에 CUDA:0/RTX 5080과 실제 이미지·GT 수 및 지표가 기록됐다. DAG 노드 완료 상태만으로 평가 성공을 세지 않았다.

이벤트는 launch → process_started → model_import → model_val_call → validation_started → validation_completed 순서로 증가한다. 나노초 시각은 문자열/정수로 손실 없이 비교했다. 앞선 성공의 **outer 영수증 mtime이 다음 launch보다 이르다**는 것도 5개 인계에서 확인했다. 부모의 과거 monitor 종료를 이 감사에서 다시 관측한 것은 아니다. 보존된 실제 종료 0 영수증을 native 증거와 교차 검증한 판정이다.

`units/` 전수 원장에는 attempt 7개와 model_val_call 7건, result 6개, 고유 성공 6개뿐이다. current__val11만 최초 실패와 승인된 1회 복구의 두 시도가 있고 나머지는 각 1회다. 최초 실패 `units/current__val11/attempt-001-21893e66ad3440c7957e1e2e73b26ee2`는 exit=1/result 없음, model.val 진입 traceback과 `Offline/no-subprocess guard: subprocess.Popen` 오류를 유지한다. 승인서에 결속된 실패 5개 파일 해시도 일치한다. 실패 exit 해시는 `837fcc5291ca9e4484eec43b01830522b6f50408774714f58118309040199c88`다.

`parent-recovery-v4.md`의 인계와 승인된 단일 명령을 인정한다. 성공 replay, 새 실패 재시도 또는 행렬 컨트롤러 실행 증거는 없다. 현재 runner는 한 단위만 실행하고 완료 결과가 있으면 replay를 거절한다. 다른 다섯 `results-v3/*.json`의 blocked_before_runner_invocation은 **보존된 과거 상태**이며 현재 성공을 부정하는 보고서가 아니다. 과거 v2/v3 감사와 차단 보고서를 수정하지 않았다.

## 3. AP 검산과 P/R의 정확한 의미

각 result의 `ap_class_index`를 순회해 `all_ap`의 실제 행을 클래스에 매핑했다. 내부는 [0,1], 외부는 [0]이다. 유효 10개 클래스 행 모두 유한한 [0,1] 값 10개를 가진다. AP50은 첫 열, AP50:95는 10개 IoU 열의 평균이며 클래스별 값과 aggregate 평균을 독립 검산했다. 최대 오차는 **2.220446049250313e-16**, 허용 오차 1e-12 이내다. 없는 클래스를 aggregate로 채우거나 클래스 ID를 무조건 행 번호로 사용하지 않았다. 근거: 설치 `ultralytics/utils/metrics.py:939-954`, `models/yolo/detect/val.py:59`.

P/R은 설치 `utils/metrics.py:892-893`의 **평활화된 클래스 평균 F1 곡선이 최대인 공통 confidence 지점** 값이다. 클래스별 독립 최적점이나 과거 conf0.45 운영점이 아니다. 모든 P/R은 native와 실행 영수증 간 일치하며 JSON에 별도 필드로 보존한다. 예를 들어 current 외부 best-F1 P/R은 0.9610824735574326 / 0.9322916666666666이고, 과거 conf0.45 P/R은 0.8878048780487805 / 0.9479166666666666이다.

**증거 한계:** p_curve/r_curve/f1_curve/x 및 raw detection/matching 배열(tp/conf/pred_cls/target_cls)은 저장되지 않았다. 여섯 validation 출력 디렉터리는 비어 있고 저장 설정도 false다. 따라서 best-F1 argmax 수치의 독립 재현이나 미저장 원시 검출 배열에서 AP 재생성을 주장하지 않는다. 이번 재계산은 저장된 all_ap 행·평균과 aggregate 검산이다. 과거 conf0.45 캐시나 trainer CSV로 AP를 대체하지 않았다.

## 4. 실제 조건, 멤버십과 외부 클래스 필터

여섯 단위 모두 conf=0.001, NMS IoU=0.5, imgsz=640, batch=1, GPU0, max_det=300, augment=false, seed=42, deterministic=true, workers=0, rect=true, single_cls=false, agnostic_nms=false, quantize=null이다. 모든 intent 설정은 문서화된 half 예외 외에 actual-settings와 일치한다. **실제 half 필드는 없다.** quantize=null과 동결 runner의 FP32 체크, 설치 `engine/validator.py:188-201` 및 `models/yolo/detect/val.py:109`의 float 경로가 FP32를 지지한다. 감사자가 tensor dtype을 새로 관측한 것은 아니다.

| 분할 | 이미지 / 라벨 | potato GT | stone 라벨 수 | 빈 라벨 |
|---|---:|---:|---:|---:|
| val11 | 11 / 11 | 60 | 8 | 0 |
| test12 | 12 / 12 | 33 | 7 | 3 |
| external166 | 166 / 166 | 192 | 0 | 87 |

189개 라벨과 이미지/라벨 파일 소속을 전수 재집계했다. 실제 validation_started의 이미지·GT·cuda:0·결정론 값도 일치한다. 동결 on_start는 실제 dataloader 소속과 GT를 manifest에 비교한 후 이 이벤트를 기록한다. 최대 이미지당 GT는 8/7/6개로 max_det 자동 상향 조건에 해당하지 않는다. 두 모델은 분할별로 같은 입력과 평가 조건을 사용했다.

외부의 라벨 0개는 실제 돌 부재의 증거가 아니다. 외부 YAML은 potato만 정의하고 실제 두 외부 호출은 classes=[0]이다. 동결 `runner.py`의 ExternalValidator.postprocess → external_postprocess는 설치 NMS에 classes=[0]을 명시한다. 설치 `utils/nms.py:129-160`은 클래스 필터를 NMS와 max_det 절단 전에 적용한다. single_cls=false라 돌을 감자로 재명명하지 않는다. 외부 stone AP/P/R은 이유 `external_ground_truth_is_potato_only_no_stone_gt`의 null이다. 필터 경로는 실행 바인딩과 소스로 확인했으며 미저장 검출 배열을 다시 검사했다고 주장하지 않는다.

## 5. 보존과 프로세스 정리

동결 기대값과 실제 바이트를 **1,239회 비교, 불일치 0회**: 원본 419, 설치 소스 355, 설정 3, private 사본 392, 코드/설정 6, Python 1, protocol 1, manifest 1, 승인 결속 과거 증거 17, font manager 1, 성공 native 42, 기존 completed 해시 1. 별도로 완료 6개와 최신 outer 영수증 6개의 현재 해시를 JSON에 기록했다. 설치 소스 소속 355개와 배포판 버전 65개가 일치하고 사본 392개에 symlink/hardlink/동일 원본 inode가 없다. 모델은 역직렬화하지 않고 바이트만 해시했다.

| 결속 | SHA-256 |
|---|---|
| current 원본/사본 | 8b2cb7982fcd11f61a33cab8659de23128d0065cfa8e8966093be297a7e216c8 |
| stone40 원본/사본 | 06d46e0dc728ffbabf4078f79acaf9d32d809a57a46810acdf39a7927aa2bbd1 |
| protocol | 189c4d8b92b0d5d1579dd09621e235821b3d0efd16b313462053ff0b4d242244 |
| approval-v3 | 9d374128f6c5e057c87d8501541c0e8da34694fe2076fc1a6fd4e88858aaa3ad |
| inputs-manifest | 05dc668f51ee0fa8c70840c03bcd5438ac9bd5caea04a2ef3f8cf7fd1f55b30f |

v2 보존 before/after는 바이트 동일하다. 당시 442개 중 439개는 현재도 size/mtime_ns/ctime_ns/inode/mode가 같고, 3개 준비 파일만 이미 승인된 v3 폰트 복구로 변경됐다. 세 이전 원문은 history/preflight-v2에, 최초 실패는 units/current__val11의 첫 attempt에 보존되며 승인서의 해시와 일치한다. 이번 부모 인계에서 소스·설정·모델·입력 변경은 확인되지 않았다. 새 private 라벨 cache는 정상 실행의 부수 출력이며 삭제하지 않았다.

/proc에서 **89개 PID 명령행, 읽기 오류 0개**, 남은 해당 runner/기존 controller 0개를 확인했다. 성공 worker 6개 및 최초 실패 worker PID는 모두 부재이고 active.lock도 없다. outer PID 자체가 영수증에 없으므로 모르는 PID를 만들어 적지 않았다. 종료 영수증과 현재 명령행 스캔을 근거로 한다.

쓰기 직전 기존 증거 737개와 부모 신규 집계 2개, 다른 세션이 추가한 new-session/inventory-refresh-20261109.json을 보존 대상으로 관리했다. 이 감사는 아래 두 파일만 작성하며 기존 보고서·라벨·지표를 덮어쓰지 않는다. 시스템 전체의 모든 과거 쓰기를 추적한 OS 감사는 아니다.

## 6. 부모 통합 보고서 집계 검증

**수치 집계 APPROVED_WITH_NOTES.** `../report.md`와 `../metrics.json`을 실제 AP, `../historical-metrics.json`, `../error-analysis/verified-strata.json`과 대조해 **132개 집계 검사, 수치 불일치 0개**를 확인했다. AP 6행의 4자리 반올림과 원본 all_ap/클래스/P/R/해시, 과거 운영점 15행 및 면적 계층 15행, 표시 운영점 6행의 6자리 반올림, 배경 TN=NA 혼동행렬, 오류 계층 전체가 일치한다. AP 원본 6개 경로·해시와 과거 source JSON pointer 15개도 확인했다.

부모의 돌 AP50:95 및 외부 감자 AP 회귀 서술은 정확하다. test 돌 AP50 감소도 표에 보존돼 있다. 과거 외부 FP 72→23, FN 9→10 및 test macro F1 0.919872→0.913866 역시 원본과 맞으며 AP 개선으로 전용하지 않았다. 13건은 **possible_annotation_issue(FP 6/FN 7)**로 유지됐고 확정 GT 수정이 아니다. 오류만의 계층을 전체 성능 분모로 쓰지 않았다. Task B는 NOT_TRAINED/NOT_EVALUATED와 null을 유지하고 P0/G005=false, 신규 세션 UNCONFIRMED도 일치한다.

비차단 문서 주의사항:
- **AGG-V4-001, 낮음 — `../report.md:56`:** `verified-strata.json`은 문서 디렉터리 기준으로 존재하지 않는 축약 경로다. 정확한 참조는 `error-analysis/verified-strata.json`이며 부모 metrics.json의 source는 올바르다. 해당 실제 파일로 집계를 검증했다.
- **AGG-V4-002, 낮음 — `../report.md:17`:** “원시 배열”은 저장된 `all_ap`를 뜻하는 범위에서만 인정한다. raw detection/matching 및 confidence 곡선까지 보존됐다는 뜻으로 확대하면 안 된다.

부모 문서의 “독립 감사 진행 중”은 작성 당시 상태로 보존한다. 이 감사가 그 상태를 종료하는 별도 근거이며 부모 파일을 수정하지 않는다. 검토한 부모 report SHA-256은 `335cfdbf73fc4101987dba21527d145c9c646b0650c17b2822ed4e976cf9386d`, metrics SHA-256은 `b3922c6dec1ef9892bd8b03a05c5646197101a99dd0219ca60ba8e9baf8cec61`다.

## 7. 범위와 종료

이는 동일 원영상·장면의 시간 상관 holdout과 이미 열람한 외부 자료에 대한 **역사적 Task A 증거**다. 신규 촬영 세션 데이터·GT가 미확정이므로 일반화, 과적합 부재, 배포 준비, 전체 P0/G005 완료를 승인하지 않는다. 과제 B 지표는 NA다. 기존 오류 분석의 13개 라벨 재검토 후보를 이유로 라벨이나 AP/FP/FN을 조정하지 않았다.

감사에서 모델 import/추론/설치/재실행/소스 수정/하위 위임은 0회다. 읽기·해시·원장 및 수치 검산만 수행했다. 산출물은 `independent-gate-v4.md`와 `verified-ap-v4.json` 두 개이며, 원시 배열 부재는 명시된 검증 한계이지 여섯 실제 성공을 미실행으로 바꾸는 사유가 아니다.

저장된 JSON을 다시 파싱해 여섯 고유 행, 누적 일곱 호출, 완료 해시, 외부 돌 NA, 보고서 표와 JSON의 정합성을 포함한 12개 산출물 검사를 통과했다. 보존 대상 740개 파일의 바이트 변경은 0개이며 추가 파일은 허용된 두 개뿐이다. JSON LSP 진단은 기존 Biome 실행 파일이 없어 수행하지 못했고 설치하지 않았다. JSON 파싱과 데이터 검산은 별도로 통과했으며 모델·fixture·preflight는 재실행하지 않았다.

