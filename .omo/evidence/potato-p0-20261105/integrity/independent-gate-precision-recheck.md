# P0 무결성 증거 정밀도 수정 독립 재검증

작성일: 2026-11-05

## 판정: APPROVE

기존 `independent-gate.md`의 유일한 차단 사항인 `mtime_ns` 정수 전사 오류가 해소됐다. 실제 원본 이벤트, 수정된 영수증, 수정 전 백업을 Python `json`의 임의 정밀도 정수로 독립 대조했다. 수정된 `after` 객체 전체가 원본 이벤트와 일치하며, 백업 대비 차이는 지정된 네 정수 leaf뿐이다.

이는 기존 독립 감사 결과에 이번 한정 재검증을 합친 **수정된 무결성 증거의 승인**이다. 기존 `independent-gate.md`의 `NEEDS_FIX`는 당시 판정 이력으로 그대로 보존한다. 다른 기존 검사에는 미해결 차단 사항이 없으며, 이번 수정에서 그 결과를 뒤집는 변경도 발견되지 않았다. 기존 보고서의 검증 범위와 한계는 계속 적용된다.

## 실제 수정 후 해시

파일을 바이너리로 열어 `hashlib.file_digest(..., "sha256")`로 직접 계산했다. 부모 기록의 해시를 복사해 판정하지 않았다.

| 파일 | 실제 SHA256 |
|---|---|
| `preservation-cleanup.json` | `42d641c3cf2b845d41cfe2a0f01a87168707cdd4e9ce383a5feaefd6cafc16bb` |
| `preservation-cleanup.before-precision-fix.json` | `d465898f4b4d80ccf22f487fa45d784e21308540da2040feddebb0e900e3b708` |
| `precision-fix-validation.json` | `80d10f5311c56ecbaa1dded45ac07fb1574de501db1ca1123a8e8b964d973b24` |
| 기존 `independent-gate.md` | `01d7a1ccbdf5f145b2865a78c169932606be22ce771128bdb71b6f42c9e7b7c2` |

백업 해시는 사용자와 부모 검증 기록이 제시한 수정 전 해시와 일치한다. 수정 후 해시 역시 부모 검증 기록과 일치한다.

## 원본 및 대조 방법

원본은 아래 실제 세션 JSONL의 **104행**, 이벤트 **`c5c449eb`**다.

```text
/home/sc/.omo/agent/projects/potato-274033cae687/senpi-task/children/st_01a1aacb/sessions/st_01a1aacb/2026-11-05T08-19-59.007Z_7baf1586-4a7e-43b5-9167-60dc5fd6211b.jsonl
```

이벤트 시각은 `2026-11-05T08:31:23.960Z`, 내부 `preservation_after` 시각은 `2026-11-05T08:31:21.865550+00:00`다. 해당 행을 직접 읽고 이벤트의 `content`에서 원래 `preservation_after` JSON을 추출했다.

원본 이벤트, 수정 영수증, 백업, 부모 검증 JSON을 Python `json.loads`로 파싱했다. 큰 정수는 Python `int`로 유지했고 JavaScript `Number` 변환이나 부동소수점 비교를 사용하지 않았다. 백업과 수정 영수증 전체를 재귀 비교하여 변경 leaf를 독립 산출했다. 읽은 텍스트의 UTF-8 재인코딩 해시가 실제 파일 바이트 해시와 같은 것도 확인했다.

## 독립 관측 결과

| 검사 | 결과 |
|---|---|
| 수정 영수증의 `after` 전체 객체와 실제 원본 `preservation_after` 객체 동일성 | PASS |
| `after.changes` 전체와 원본 `changes` 동일성 | PASS |
| `concurrent_changes` 각 항목에서 `path`, `before`, `after`만 남긴 목록과 원본 `changes` 동일성 | PASS |
| 백업 대비 재귀 비교의 변경 leaf 수 | 정확히 4개 |
| 변경 leaf 목록과 부모 `precision-fix-validation.json` 기록 일치 | PASS |
| 백업 텍스트에 아래 두 정수 치환만 적용하면 수정 파일 전체와 동일 | PASS |
| 백업 SHA256과 지정된 수정 전 SHA256 일치 | PASS |
| 부모가 기록한 수정 후 SHA256과 실제 수정 파일 SHA256 일치 | PASS |

실제 네 변경은 다음과 같다.

| JSON 경로 | 백업 값 | 수정 값 = 실제 원본 값 |
|---|---|---|
| `/after/changes/2/before/2` | `1793866892310702000` | `1793866892310702200` |
| `/after/changes/3/after/2` | `1793867071601420300` | `1793867071601420400` |
| `/concurrent_changes/2/before/2` | `1793866892310702000` | `1793866892310702200` |
| `/concurrent_changes/3/after/2` | `1793867071601420300` | `1793867071601420400` |

대상은 각각 기존 루프 `ledger.jsonl`의 before mtime와 `revisions/00000013.json`의 after mtime이며, 두 목록에 중복 기록된 값이다. 내용 해시, 크기, 네 변경 경로, 인벤토리 digest, 정리 기록 및 전체 불변 `false`에는 변경이 없다. 정수 치환 외의 공백·서식 등 텍스트 변경도 없다.

## 정확한 범위와 기존 판정의 유지

- 이번 재검증은 위 전사 오류의 수정 여부만 다뤘다. 기존 validator 2개, 모델·환경 등 47개 대상 검사, 전체 인벤토리 스캔, ZIP 계수는 다시 실행하지 않았다. 기존 검사 결과는 원래 독립 게이트의 근거를 유지하며 현재 시점의 새 전수 검증 결과로 표현하지 않는다.
- 기존 게이트의 다른 검사에는 추가 차단 사항이 없었다. 이번 수정의 네 leaf 외에는 영수증이 그대로이므로 기존 통과 결과를 취소할 근거도 발견되지 않았다. 따라서 이전 유일한 차단 사항은 닫힌다.
- OpenCV 과거 버전 미기록, 수확 3901/3902의 기존 요약 불일치와 원인 미확인, 네 `.omo` 동시 변경의 작성자 독립 증명 부재, 순차 스캔의 비원자성, 전체 원본 인벤토리 미보관, 모델 추론·GPU·제품 품질 미검증 등 원래 한계는 해소됐다고 주장하지 않는다.
- 부모의 `validator_reruns: 0`, `new_full_inventory_scans: 0`은 부모 기록이다. 이 재검증 자체의 해당 실행 횟수도 각각 0이지만, 부모의 모든 활동을 새로 전수 감사한 것은 아니다.
- 이 재검증은 메모리 내 파싱·비교와 대상 파일 해시 계산만 수행했다. 새 프로세스 실행, GPU, 모델 import, 설치, 다운로드, 위임 및 prose 테스트를 하지 않았다.
- 새로 작성한 파일은 이 `independent-gate-precision-recheck.md`뿐이다. 기존 `independent-gate.md`, 수정 영수증, 백업, 부모 검증 기록 및 다른 파일은 변경하지 않았다. 부모 goal/criteria 또는 제품 배포 승인을 대신하지 않는다.
