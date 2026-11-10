# P0 무결성 감사 — 2026-11-05

## 판정

지정 검증 두 개는 각각 한 번 실제 실행되어 모두 종료 0이다. 멤버십·누락·해시 실패는 검출되지 않았다. 모델 및 기록된 환경 출처도 일치한다. 제품·원본·모델·기존 출력은 감사 전후 보존되었고 소유 프로세스는 종료했다. 전체 작업 공간이 완전히 같지는 않다. 부모 병행 작업에 해당하는 `.omo` 파일 네 건의 차이는 아래에 그대로 기록한다. 이는 학습·추론 성능이나 결함 분류 준비 완료를 뜻하지 않는다.

작업 경로는 `/mnt/c/Users/USER/Desktop/협력/potato`이다. 모든 새 파일은 `.omo/evidence/potato-p0-20261105/integrity/` 안에 `apply_patch`로만 작성했다. 제품 코드·원본 데이터·모델·기존 출력·보호 기준은 수정하지 않았다.

## 실제 명령 영수증

실행기는 `/home/sc/.cache/potato-baseline/.venv/bin/python -B .omo/evidence/potato-p0-20261105/integrity/audit.py`이다. `tool.monitor`의 `mon_PRVBTC4VP6G5Q961` / `bash_3`으로 실행했다. 실행기는 자식의 stdout/stderr를 별도 파이프로 수집하고, 종료 코드·PID·UTC 시각·원문·정확한 base64 바이트·스트림 SHA256을 출력했다. 그 실제 결과를 `commands.json`에 옮겼다. 모니터 자체의 stderr 병합을 검증기 stderr가 비었다는 근거로 사용하지 않았다.

| 명령 | 실행 UTC | 종료 | 실제 stdout | 실제 stderr |
|---|---|---:|---|---|
| `/home/sc/.cache/potato-baseline/.venv/bin/python -B -m retraining_data.verify --profile v2` | 08:25:54.262455–08:26:05.498950 | 0 | `Verified 111 frames; source hashes and output bytes match\n` | 빈 바이트 |
| `/home/sc/.cache/potato-baseline/.venv/bin/python -B -m public_label_rework.protect verify` | 08:26:05.499034–08:27:22.524920 | 0 | `CHECKED 4499 files; CHANGED []\n` | 빈 바이트 |
| `git -C /mnt/c/Users/USER/Desktop/협력/potato rev-parse --show-toplevel` | 08:22:09.212078–08:22:09.214977 | 128 | 빈 바이트 | `fatal: not a git repository (or any parent up to mount point /mnt)` 및 파일시스템 경계 안내 |

Git 종료 128은 읽기 전용 탐색에서 이 작업 경로의 저장소를 발견하지 못했다는 결과다. 초기 위치 탐색에서도 같은 결과가 나왔다. Git 초기화·worktree 생성·commit은 하지 않았다.

## 검증 범위와 실패 진단

실행 전 `retraining_data/verify.py`, `sources.py`, `common.py`, `schema.py`, `stone_experiment/prepare.py`, `schema.py`, `common.py`, `scripts/baseline_common.py`, `scripts/run_baseline.py`의 검증·import 경로, `public_label_rework/protect.py`, `common.py`를 확인했다.

v2 검증은 기존 63프레임 검증을 재사용한 뒤 원본에서 계획을 재구성한다. 이미지 바이트·RGB 픽셀·라벨·원본 출처 해시, 스냅샷·YAML, 출력 및 리뷰 폴더 멤버십을 검사한다. 기존 스키마와 v1/v2 격리를 변경하거나 우회하지 않았다. v2는 111프레임으로 실제 통과했다.

보호 검증은 저장된 `public_label_rework/outputs/protected_before.json`과 현재 파일을 비교한다. `.omo`, `.ruff_cache`, `__pycache__`, `public_label_rework` 경로 부분은 기존 규칙상 제외된다. 따라서 이번 증거 작성은 보호 검사 밖이며, 이 검증만으로 public_label_rework 내부 보존을 주장하지 않는다.

`protected-origin-diff.json`의 항목은 빈 배열이다. 감사 전 실측 해시를 같은 보호 제외 규칙으로 대조해도 추가·누락·변경 항목이 없었다. 실패를 정상 변경으로 재분류하거나 기준 스냅샷을 덮어쓴 경우가 없고, 재시도로 초록 결과를 만든 것도 아니다.

## 모델·환경 연결

상세 실측값은 `model-environment-binding.json`에 있다.

- `retraining_experiment/outputs/selection/selected.pt` SHA256: `8b2cb7982fcd11f61a33cab8659de23128d0065cfa8e8966093be297a7e216c8`.
- 사용자 제시 과거 해시, `selection.json.selected_sha256`, 선택된 `best` 체크포인트 및 provenance 체크포인트 해시와 일치한다. `last.pt`도 저장 해시와 일치한다.
- selection이 참조하는 provenance·environment·dataset 영수증의 해시 3개가 모두 일치한다. provenance에 기록된 소스·설정·프로토콜 파일 38개, 원본 비디오 해시, Python 실행 파일 해시도 일치한다.
- Python: `3.12.3 (main, Aug 31 2026, 10:18:26) [GCC 13.3.0]`; 실행 파일 SHA256은 `e50d468e8b0adfb05733f5b87b3cff34829c4a8c1aea50c865aa8bdfe4bb150f`로 저장값과 같다.

| 배포 패키지 | 현재 metadata 버전 | 저장 environment | 판정 |
|---|---|---|---|
| ultralytics | 8.4.183 | 8.4.183 | 일치 |
| torch | 2.9.1+cu128 | 2.9.1+cu128 | 일치 |
| torchvision | 0.24.1+cu128 | 0.24.1+cu128 | 일치 |
| opencv-python | 5.0.0.93 | 기록 없음 | 과거 동일성 미확인 |
| opencv-python-headless / opencv-contrib-python | 설치 metadata 없음 | 기록 없음 | 해당 배포판 없음 |

저장된 나머지 7개 패키지(PyYAML, av, numpy, pillow, ruff, scikit-image, scipy)와 플랫폼 문자열도 일치한다. 버전은 `importlib.metadata`로만 읽었다. 감사 실행기의 `torch`, `torchvision`, `ultralytics`, `cv2` 로드 목록은 비어 있다. 모델 역직렬화·GPU 초기화 없이 바이트 해시와 배포 metadata를 비교했다. 패키지 import 가능성·네이티브 ABI·현재 GPU/CUDA 동작까지 검증한 것은 아니다. 저장 environment에 OpenCV가 빠져 있다는 한계를 버전 불일치나 손상으로 단정하지 않는다.

## 추가 한정 진단: 수확 박스 3901 / 3902

부모가 알린 두 기존 요약의 차이를 이미지 처리 없이 확인했다. `harvest_counts.py`를 별도 `tool.monitor` (`mon_99S9V47C91ACK09C` / `bash_4`)로 한 번 실행했고 종료 0이다. 실제 결과는 `harvest-receipt.json`에 있다. 이 추가 프로브는 모니터 병합 스트림을 보관했으며 별도 stderr 바이트 수집은 하지 않았다.

- 원본 ZIP 안의 `manifest.json`은 `outputs/source/manifest.json`과 바이트 단위로 같다. SHA256은 `c58194285611c98ce1b3844f7a5da8e088a2013a223583a8fb5ad8d874d807a2`이다.
- manifest는 수확 748장·박스 3901개, 공개 2060장·박스 9498개, 합계 2808장·13399개를 요약한다. 수확 record 수도 748개다.
- 원본 ZIP의 수확 JSON 748개만 메모리에서 읽어 합산하면 potato rectangle 3902개·빈 라벨 68개다. 기존 `harvest_convention.json`의 히스토그램은 748장·3902개로 계산되며 원본 JSON의 전체 히스토그램과 같다.
- 따라서 현재 근거는 **원본 manifest에 이미 포함된 요약 1개 부족**을 지지한다. 3902와 공개 요약 9498을 합치면 13400으로, manifest 전체 요약도 같은 1개 차이를 갖는다. 공개 9498은 이번 추가 프로브에서 재계수하지 않고 저장 요약으로만 사용했다.
- manifest record에 개별 박스 수가 없으므로 어떤 단일 라벨 때문에 요약이 달라졌는지와 요약 생성 시점의 원인은 미확인이다. 이미지 손상·보호 실패·라벨 재작성 필요로 자동 해석하지 않는다. 수확 이미지 payload 읽기 0, 추출 0, 원본 쓰기 0이다.

부모의 ZIP 헤더상 수확 748장·공개 2060장·비디오 없음은 부모 제공 사실이며 이 감사의 독립 비디오 전수 검사 결과가 아니다. Task A 감자/돌과 별도 Task B 결함 유형 분류라는 방향을 이 감사 결과가 변경하지 않는다. 부모 전달 당시 Task B 분류 체계는 미정이었고, 감사 중 부모의 후속 `class-policy.md`가 생겼다. 그 정책의 확정·승인은 이 감사가 대신하지 않는다.

## 보존·소유 프로세스 마감

감사 전 owned evidence만 제외한 파일 34,758개의 SHA256·크기·mtime_ns·파일 심볼릭 링크 대상을 수집했다. 제품뿐 아니라 기존 public_label_rework 출력과 기존 `.omo` 증거도 포함한다. 감사 전 인벤토리 SHA256은 `27083f59b9946bd4458d16531a5c913ef220b0459f72d81cfc08a472b09d053e`이다.

감사 후 08:31:21.865550 UTC 인벤토리는 34,760개이며 SHA256은 `0d400257ea918598bce5c6cefc8d431bbb2b1f39887f52e2fe096e98603240f6`이다. 전체 인벤토리 `unchanged`는 **false**이다. 차이는 다음 네 경로뿐이며 원래 값·현재 값·해시·크기·mtime_ns는 `preservation-cleanup.json`에 보존했다.

| 경로 | 실제 차이 | 분류 |
|---|---|---|
| `.omo/evidence/potato-p0-20261105/class-policy.md` | 신규 파일 | 부모의 병행 정책 증거 |
| `.omo/ulw-loop/01a17b9a-16e7-7504-b1d3-e2c5ec87e26e/goals.json` | 내용·mtime 변경 | 부모 소유 루프 상태 |
| `.omo/ulw-loop/01a17b9a-16e7-7504-b1d3-e2c5ec87e26e/ledger.jsonl` | 내용·크기·mtime 변경 | 부모 소유 루프 기록 |
| `.omo/ulw-loop/01a17b9a-16e7-7504-b1d3-e2c5ec87e26e/revisions/00000013.json` | 신규 파일 | 부모 소유 루프 revision |

`class-policy.md`의 사용자 결정·별도 Task A/B 정책과 revision 13의 P0 부모 계획을 읽어 병행 부모 작업과 일치함을 확인했다. 운영체제 수준의 작성자 추적은 하지 않았으므로 작성 주체를 독립적으로 증명한 것은 아니다. 감사 보조 실행기는 파일 쓰기 함수 없이 stdout만 출력하며, 이 네 경로를 생성·편집·되돌리지 않았다. 기존 protect 기준이 `.omo`를 제외하므로 위 차이는 보호 검증 실패가 아니다. 전체 불변으로 결과를 조작하지 않았고, 반대로 이를 제품 손상으로 단정하지도 않았다. `.omo` 밖 및 기존 public_label_rework 출력에는 관측된 차이가 없다.

이는 순차 파일 스캔 두 번 사이의 SHA256·크기·mtime_ns·파일 링크 대상 비교다. 원자적 파일시스템 스냅샷, 접근 시각(atime), 디렉터리 메타데이터, 외부 캐시 불변성까지 보장하지 않는다.

감사 실행기는 08:31:21.865644 UTC cleanup 이벤트 후 종료 0으로 완료했다. Git 및 두 validator 자식은 모두 `communicate()`로 수거되어 남은 자식 목록이 비었다. 추가 읽기 전용 프로세스 확인 모니터 `mon_YNWJRCN6KNJJNAMB` / `bash_5`도 종료 0이다. `1021253`(실행기), `1021254`(Git), `1022142`(v2), `1022194`(protect)는 `/proc`에 없었다. 최초 이름 검색의 awk 열 위치가 하나 밀려 있어 그때의 `AUDIT_VALIDATOR_PROCESSES_NONE`은 증거로 쓰지 않았다. 열 위치를 교정한 `mon_1CY87B14NHZ0NYNF` / `bash_6`이 실제 `CORRECTED_AUDIT_VALIDATOR_PROCESSES_NONE`을 출력하고 종료 0이었다. 교정은 정리 확인에만 해당하며 validator를 재실행하지 않았다. 수확 보조 모니터 역시 이미 종료 0이다. 다른 주체의 프로세스는 종료시키지 않았고 강제 정리는 필요 없었다.

`receipt-validation.json`에는 저장된 `commands.json`을 다시 읽어 검사한 결과를 기록했다. 세 명령 모두 원문과 base64 바이트가 같고 stdout/stderr SHA256이 맞으며, 수집한 실제 영수증과 저장된 명령 항목이 동일했다. 지정 validator 영수증은 정확히 두 개다.

## 후속 증거 전사 교정

독립 감사 `independent-gate.md`는 보존 영수증의 두 나노초 정수가 JavaScript Number 왕복에서 각각 200ns·100ns 작아졌음을 발견하여 NEEDS_FIX로 판정했다. 부모가 원래 터미널 `c5c449eb` 이벤트를 Python 임의 정밀도 정수로 읽고, `preservation-cleanup.json`의 중복된 네 위치만 정확한 값으로 복원했다.

잘못 전사된 원본은 `preservation-cleanup.before-precision-fix.json`에 보존했으며 SHA256은 `d465898f4b4d80ccf22f487fa45d784e21308540da2040feddebb0e900e3b708`이다. `precision-fix-validation.json`은 원래 변경 목록 전체와 수정된 두 목록의 정수 포함 동등성, 정확히 네 잎 값만 바뀌었음, 전후 파일 해시를 기록한다. 원래 NEEDS_FIX 보고서도 보존한다.

이 교정은 감사 실행기 내부의 원래 정수 비교·인벤토리 해시·검증기 결과를 바꾸지 않는다. 검증기 재실행이나 전체 인벤토리 재스캔은 하지 않았다. 이후 이런 영수증은 정수 문자열 또는 임의 정밀도 파서로 대조해야 하며 JavaScript Number 객체 비교만으로 정밀도 보존을 주장하면 안 된다. 교정 후 독립 판정은 별도 `independent-gate-precision-recheck.md`에서 확인한다.

## 실행하지 않은 항목과 한계

학습·추론·모델 로딩·GPU/CUDA 확인·영상 디코딩·성능 재평가·신규 다운로드·설치·위임·재라벨링은 하지 않았다. 기존 validator의 학습·prepare·snapshot 쓰기 경로는 호출하지 않았다. 외부 아카이브는 보존 해시를 위해 바이트로 읽었지만 추출하거나 데이터셋에 편입하지 않았다.

`basedpyright` LSP가 설치되어 있지 않아 보조 Python 파일의 LSP 진단은 실행할 수 없었다. 설치 금지 범위에 따라 설치하지 않았다. 보조 실행기의 실제 실행 성공 여부와 원본 검증기의 두 실제 결과를 정적 타입 검사 완료로 바꾸어 주장하지 않는다.

SDK 상태·goal·criteria는 작성하지 않았다. G002는 이 감사에서 읽거나 판정하지 않았으며 부모 소유 기준을 대신 마감하지 않는다.
