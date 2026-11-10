# 클래스별 AP 평가 실행 전 동결 준비

이 디렉터리는 **평가 준비만** 수행한다. 실제 체크포인트 초기화, 역직렬화,
추론, `model.val`, 학습 및 Task B 모델/지표 생성은 수행하지 않는다.
독립 검토자가 정확한 바이트 바인딩을 승인한 뒤 별도 실행자가 평가한다.
자식 에이전트, 설치, 다운로드는 사용하지 않는다.

## 입력과 동결 범위

- 현재 모델: `retraining_experiment/outputs/selection/selected.pt`,
  SHA256 `8b2cb7982fcd11f61a33cab8659de23128d0065cfa8e8966093be297a7e216c8`.
- stone40: `stone_experiment/outputs/selection/selected.pt`,
  SHA256 `06d46e0dc728ffbabf4078f79acaf9d32d809a57a46810acdf39a7927aa2bbd1`.
- 내부 val11/test12: `retraining_data/outputs/dataset_v2`; 클래스 potato=0, stone=1.
  기존 `retraining_experiment/review/protocol.json`의 manifest/snapshot 해시와
  `annotations.json`의 이미지·라벨 해시/분할을 확인한다.
- 외부 166장: `baseline/external/images/test`, `labels/test`.
  `retraining_experiment/review/protected_before.json`의 원본 바이트와 일치한다.
  GT는 potato만 존재한다. stone GT 0개는 **stone AP=0이라는 뜻이 아니다**.
- GT 인스턴스 수: val potato60/stone8, test potato33/stone7, external potato192.
- 원본 이미지 189개와 동반 라벨 189개, 체크포인트 2개, 기존 비교/데이터
  메타데이터 11개, 로컬 폰트 1개를 복사한다. 합계 392개이며 하드링크가 아니다.
  학습 이미지와 기존 `.cache` 파일은 복사하지 않는다.

`inputs-manifest.json`은 모든 복사본의 원본/비공개 경로와 SHA256,
새 데이터 YAML SHA256, 설치 Ultralytics 소스/설정 355개의 SHA256,
기존 보조 소스 SHA256, Python 실행 파일 SHA256, Python 버전,
설치된 모든 distribution 버전을 기록한다. `protocol.json`은 manifest와
러너·준비 코드·픽스처·설정 템플릿·타입 검사 설정·이 문서의 해시를 바인딩한다.
프로토콜 파일 자체의 SHA256은 사전 점검과 독립 승인서가 바인딩한다.
어느 바이트라도 바뀌면 새 검토가 필요하다.

사용하는 기존 Python:
`/home/sc/.cache/potato-baseline/.venv/bin/python` (3.12.3).
설치된 Ultralytics 8.4.183, torch 2.9.1+cu128,
torchvision 0.24.1+cu128, numpy 2.5.2, matplotlib 3.11.2, PyYAML 6.0.3.
다른 환경으로 자동 전환하지 않는다.

## 정확히 여섯 평가 단위

| ID | 체크포인트 | 데이터 | 라이브러리 split |
|---|---|---|---|
| current__val11 | current.pt | private/val11/data.yaml | val |
| current__test12 | current.pt | private/test12/data.yaml | test |
| current__external166 | current.pt | private/external166/data.yaml | test |
| stone40__val11 | stone40.pt | private/val11/data.yaml | val |
| stone40__test12 | stone40.pt | private/test12/data.yaml | test |
| stone40__external166 | stone40.pt | private/external166/data.yaml | test |

고정값: imgsz=640, batch=1, device=0, conf=0.001,
NMS IoU=0.5, augment=False, max_det=300, half=False, seed=42,
deterministic=True, workers=0, rect=True, single_cls=False,
agnostic_nms=False, cache=False, plots=False, save=False,
save_json=False, save_txt=False, compile=False, channels_last=False.
설치 버전은 `half=False`를 `quantize=None`으로 변환한다.
이를 실제 `get_cfg`와 평가 시작 콜백에서 확인한다.
`rect=True`는 설치된 `Model.val` 기본값을 명시적으로 동결한 것이다.

평가 워커 시작 전에 PYTHONHASHSEED=42를 설정한다.
CUDA 초기화 전에 CUBLAS_WORKSPACE_CONFIG=:4096:8을 설정하고
설치 `init_seeds(42, deterministic=True)` 이후 결정론 연산을
`warn_only=False`로 강제한다. cuDNN benchmark와 TF32는 끈다.
보이는 물리 GPU는 0으로 고정한다. torch가 실제 CUDA를 사용할 수 없으면
차단 사유와 exit 2를 기록한다. CPU 평가, 다른 GPU, 임계값 탐색은 없다.

## 설치 소스에서 확인한 지표 의미

권위는 현재 설치된 `ultralytics` 소스다. 인터넷 문서나 과거 버전의
필드 이름을 근거로 삼지 않는다.

- `utils/metrics.py:939-956`: `box.ap50`는 `all_ap[:,0]`,
  `box.ap`는 `all_ap.mean(1)`이다.
- `utils/metrics.py:835-895`: GT의 고유 클래스 ID 순서로 AP 행을 만들고,
  `ap_class_index`에 실제 ID를 반환한다. `per_class`는 **행 번호와
  클래스 ID를 구분**한다. aggregate mAP를 두 클래스에 복제하지 않는다.
- `box.maps`는 없는 클래스를 aggregate mAP로 채우므로 사용하지 않는다.
  매핑에서 빠진 클래스는 null과 이유를 기록한다.
- 라이브러리 P/R은 smoothed mean F1 최대점의 P/R이다.
  과거 고정 confidence=0.45 운영 지표와 별개 필드에 저장한다.
  과거 결과는 `private/reference/.../comparison.json`으로 보존하고 재계산하지 않는다.
- `models/yolo/detect/val.py:169-180`은 NMS에 `classes`를 넘기지 않는다.
  외부 평가는 그 호출과 동일한 설정의 `ExternalValidator`를 사용하되
  **NMS에 classes=[0]을 추가**한다. 따라서 stone이 max_det를 소비하거나
  potato GT 매칭에 들어가지 않는다. single_cls로 stone을 potato로 바꾸지 않는다.
  외부 stone AP50/AP50-95와 P/R은 null이며 GT 부재 이유를 명시한다.
- `engine/validator.py:185-243`은 독립 평가에서 AutoBackend/캐시/워밍업을
  수행하므로 준비 단계에서 호출하지 않는다. 향후에도 비공개 YAML만 전달한다.
  평가 시작 시 실제 이미지 목록과 라벨 인스턴스 수를 다시 확인한다.

## 격리와 보조 import 검토

기존 `retraining_experiment/common.py`, `evaluate.py`,
`stone_experiment/common.py`, `evaluate.py`, `postprocess_review/common.py`,
`scripts/run_baseline.py`, `baseline_common.py`를 읽었다.
일부 런타임 함수는 원래 outputs를 만들거나 settings를 바꾸고,
과거 평가 함수는 고정 0.45 지표를 계산한다. 이번 러너는 이들을 import하지 않는다.
설치 소스의 import 시 `SettingsManager`, 온라인 DNS 조회, 폰트 검색,
이벤트 전송 초기화가 발생할 수 있음을 확인했다.

모든 scientific import 전에 CWD/HOME/TMPDIR, YOLO_CONFIG_DIR,
MPLCONFIGDIR, TORCH_HOME, XDG 설정/캐시, CUDA/Inductor/Triton 캐시를
해당 실행의 `runtime/` 아래로 격리한다.
설치 `get_user_config_dir`가 `/Ultralytics`를 덧붙이는 경로까지 반영하여
settings 템플릿과 로컬 DejaVuSans 폰트를 Arial 요청 이름으로 미리 복사한다.
sync와 모든 외부 로깅 통합은 꺼져 있다. YOLO_OFFLINE=true,
YOLO_AUTOINSTALL=false, MPLBACKEND=Agg를 설정한다.
Python audit hook은 범위 밖 파일 쓰기, 네트워크 연결/DNS/UDP,
서브프로세스 실행을 차단한다. 이는 별도 OS 보안 샌드박스를 뜻하지 않는다.
실행 컨트롤러의 유일한 자식 프로세스는 명시된 평가 워커다.

사전 점검/픽스처에는 모델 모듈 import 차단기도 설치한다.
torch CUDA 가용성 조회와 실제 CPU 지표/NMS 계산만 허용한다.
체크포인트 바이트의 해시 계산은 모델 로딩이 아니다.

## 독립 승인과 이후 실행 계약

이 준비 노드는 승인서를 만들거나 승인하지 않는다.
`runner.py preflight`는 `checks/preflight-*/preflight.json` 및
**승인되지 않은** `approval-request.json`을 생성한다.
별도 검토자는 그 요청서의 exact digest를 확인하고 같은 AP 디렉터리 안의
별도 승인 파일에 아래 필드를 기록해야 한다.

```json
{
  "status": "approved",
  "independent": true,
  "reviewer": "independent reviewer identity",
  "protocol_sha256": "exact protocol.json SHA256",
  "inputs_manifest_sha256": "exact inputs-manifest.json SHA256",
  "code_hashes": {"every bound filename": "its exact SHA256"},
  "unit_ids": [
    "current__val11", "current__test12", "current__external166",
    "stone40__val11", "stone40__test12", "stone40__external166"
  ],
  "allow_retry_failed": false
}
```

위 코드는 스키마 설명이며 유효한 승인서가 아니다.
`reviewer`는 비어 있거나 `prepared_by`와 같을 수 없다.
로컬 승인서는 독립 검토의 증거이며 신원을 암호학적으로 인증하는 체계는 아니다.
컨트롤러와 워커가 각각 정확한 바인딩을 재검증한 후에만 모델 import에 진입한다.

승인 후 별도 실행자가 사용할 명령 형식:

```text
/home/sc/.cache/potato-baseline/.venv/bin/python -B /absolute/ap/runner.py run current__val11 /absolute/ap/approval.json
```

`/absolute/ap`는 이 문서가 있는 실제 절대 경로로 대체한다.
한 명령은 한 단위만 실행한다. 자동 전체 실행·재시도·나머지 이어달리기는 없다.
`units/UNIT/attempt-NNN-UUID/`에 요청/실제 argv, 고정 kwargs, gate/protocol/data
바인딩, 시작 이벤트, 실제 라이브러리 설정, stdout/stderr, exit와 출력 해시를 남긴다.
모델 import, `model.val` 호출, 실제 검증 시작/완료는 별도 이벤트다.
사전 점검은 저장된 이벤트에서 실제 시도 수를 집계한다.

완료 표식 또는 결과 파일이 있으면 재실행을 거부한다.
실패 이력은 지우지 않는다. 실패 재시도에는 `--retry-failed`와
`allow_retry_failed=true`인 독립 승인이 모두 필요하다.
활성 lock은 동시 실행을 거부하며 프로세스 강제 종료로 남은 lock은
자동 복구하지 않는다. exit가 없는 시도는 미완료로 보존된다.

## 준비 검증

`prepare_inputs.py`는 한 번만 실행하며 원본과 복사본을 해시로 확인한다.
재실행 시 기존 입력 manifest를 덮어쓰지 않는다.
코드 검사는 기존 Python의 AST/compile과 기존 basedpyright를 사용한다.
LSP는 PATH에서 서버를 찾지 못하므로 설치하지 않고 설치된 CLI를 직접 사용한다.
`pyrightconfig.json`은 숨김 `.omo` 경로의 세 파일을 명시적으로 포함한다.
파일 0개를 검사한 출력은 검증 근거로 인정하지 않는다.

`check_fixtures.py`는 CPU에서 실제 `DetMetrics.process`로 서로 다른
클래스 AP, class1만 있는 매핑, 외부 stone null, 300개 초과 stone 후보보다
먼저 적용되는 potato 필터, 여섯 설정의 설치 타입/FP32 변환,
미승인/오래된 gate 거부를 확인한다.
픽스처 값은 합성 데이터의 검사 결과일 뿐 모델 평가 지표가 아니다.
문서 문구를 테스트하지 않는다.

최종 준비 증거는 `checks/`의 타입/구문 검사, preflight 및 fixtures 영수증에 있다.
GPU 확인은 실행 시 다시 수행한다. 독립 승인은 아직 없으며,
평가 결과와 실제 모델 AP는 아직 생성되지 않았다.
