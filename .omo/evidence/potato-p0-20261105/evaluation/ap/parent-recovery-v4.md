# AP 단위별 실행 인계 복구

2026-11-09. v3 독립 감사는 current__val11 성공 1개와 미호출 5개를 확인했다. current__test12가 앞선 바깥 runner 종료 영수증 작성 전에 검사해 중단했고, 나머지도 그 중단을 따랐다. 모델의 실패나 0점 AP가 아니다.

부모 세션이 기존 approval-v3.json의 명시적 단위 실행을 이어받는다. runner, protocol, 승인, 모델, 데이터와 평가 조건은 변경하지 않는다. 자동 행렬 반복문이나 새로운 실패 재시도는 없다. 각 명령을 별도 monitor로 시작하고, 실제 바깥 runner 종료0을 관측한 후 worker 종료, result/completed/event 해시, 실제 설정, 로그 해시를 검증하고 새 영수증을 작성한다. 그 후에만 다음 단위를 시작한다.

최초 current__val11 실패와 성공, v2/v3 감사, 다섯 기존 미호출 보고서는 보존한다. 이미 성공한 current__val11은 재실행하지 않는다. 새 성공 영수증은 `results-v4/<unit>.json`에 저장되며 과거 미호출 보고서를 덮어쓰지 않는다.

## 현재 인계 기록

| 단위 | 실제 상태 | 부모 monitor |
|---|---|---|
| current__val11 | 기존 성공 보존, 재실행 없음 | 기존 results-v3/current__val11.json |
| current__test12 | 종료0, native/로그/완료 이벤트 검증, results-v4 영수증 저장 | mon_9BRGRAGANMZK7VVJ |
| current__external166 | 종료0, native/로그/완료 이벤트 검증, results-v4 영수증 저장 | mon_BA6V56K3MNP63P1F |
| stone40__val11 | 종료0, native/로그/완료 이벤트 검증, results-v4 영수증 저장 | mon_ZWCNWJMDVFQJV8KC |
| stone40__test12 | 종료0, native/로그/완료 이벤트 검증, results-v4 영수증 저장 | mon_WPZMW7CN094T0WXC |
| stone40__external166 | 종료0, native/로그/완료 이벤트 검증, results-v4 영수증 저장 | mon_2MF98WCEA04FPNCQ |

6개 고유 단위 모두 실제 종료0과 결과를 확보했다. 과거 실패1회와 성공6회를 합친 실제 model.val 호출은 7회다. 기존 current__val11은 보존했고 부모가 미호출5개만 각각 한 번 실행했다. 완료된 성공을 반복한 일은 없다.

부모의 단위별 검증 후 `independent-gate-v4.md`와 `verified-ap-v4.json`에서 독립 최종 PASS를 받았다. 외부 돌 지표는 정답 미보유로 NA이며, 이 AP는 기존 자료 평가이지 신규 세션 또는 과제 B 성능이 아니다. 과거 conf0.45 운영점의 FP 감소를 AP 전 구간 개선으로 해석해서는 안 된다. 두 모델의 동일 조건 비교에서 일부 클래스·분할의 회귀를 확인했으며 통합 보고서에 그대로 기록했다.
