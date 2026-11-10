"""Korean package README rendering from parsed inputs."""

from public_label_rework.package_inputs import (
    PackageMetrics,
    SourceCredit,
    StatusRow,
    source_id,
)


def readme(
    rows: tuple[StatusRow, ...],
    metrics: PackageMetrics,
    credits: tuple[SourceCredit, ...],
) -> str:
    """Render status-derived counts and metrics without claiming human review."""
    by_source: dict[str, dict[str, int]] = {}
    for row in rows:
        counts = by_source.setdefault(source_id(row["image"]), {})
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    source_lines = [
        f"| {source} | {counts.get('kept', 0)} | {counts.get('excluded', 0)} | {counts.get('needs_human', 0)} |"
        for source, counts in sorted(by_source.items())
    ]
    source_totals = {
        status: sum(counts.get(status, 0) for counts in by_source.values())
        for status in ("kept", "excluded", "needs_human")
    }
    metric_lines = [
        f"| {name} | {token} |" for name, token in sorted(metrics["thresholds"].items())
    ]
    per_source_lines = [
        f"| {source} | {values['background_potatoes']} | {values['watermark_stock']} | {name} | {token} |"
        for source, values in sorted(metrics["per_source"].items())
        for name, token in sorted(values["thresholds"].items())
    ]
    credit_lines = [
        f"- {credit['source']}: {credit['url']} — {credit['license']}"
        for credit in credits
    ]
    mode_label = {"pilot": "파일럿 패키지", "final": "최종 패키지"}[metrics["mode"]]
    if metrics["fixture_only"]:
        fixture_label = "합성 테스트 픽스처이며 실제 검수 결과가 아닙니다."
    else:
        fixture_label = ""
    return f"""# 공개 감자 라벨 재작업 패키지

> {mode_label}. 이 결과는 AI 시각 검수 자료이며 사람의 확정 검수가 아닙니다.
> {fixture_label}

## 작업 기준

- 감자의 보이는 덩이 부분만 표시합니다. 싹은 제외하고, 식별 가능한 가림·가장자리 감자는 보이는 범위만 표시합니다.
- 흐릿한 배경 감자와 긴 변 16px 미만인 배경 감자는 표시하지 않고 배경 감자 플래그로 기록합니다.
- 분리된 절단 조각은 각각 표시합니다. 이미지 픽셀은 수정하지 않았습니다.

## 수량

| 출처 | 유지 | 제외 | 사람 확인 필요 |
|---|---:|---:|---:|
{chr(10).join(source_lines)}
| 전체 | {source_totals["kept"]} | {source_totals["excluded"]} | {source_totals["needs_human"]} |
배경 감자 표시 이미지: {metrics["background_potatoes"]}
워터마크 표시 이미지: {metrics["watermark_stock"]}

## 품질 지표

| 지표 | 결과 |
|---|---|
{chr(10).join(metric_lines)}
G-full 상태: {metrics["verification_status"]}; 평가 횟수: {metrics["gfull_evaluations"]}

| 출처 | 배경 감자 표시 이미지 | 워터마크 표시 이미지 | 지표 | 결과 |
|---|---:|---:|---|---|
{chr(10).join(per_source_lines)}

## 라이선스 및 출처

{chr(10).join(credit_lines)}

`sources/`의 원본 출처 파일을 변경 없이 포함했습니다. 해당 CC BY 4.0 데이터의 라벨을 수정했으며 이미지 픽셀은 수정하지 않았습니다.

## 한계

- AI 검수 결과이며 사람이 확정한 정답이 아닙니다.
- X-AnyLabeling GUI에서 여는 동작은 테스트하지 않았습니다.
- 원본의 늘림·레터박스 전처리 때문에 픽셀 크기를 서로 직접 비교할 때 주의해야 합니다.
- 수확 데이터와 공개 데이터의 박스 기준 차이: {metrics["harvest_convention"]}

## 사용 방법

압축을 풀고 `images/public/`을 학습용 이미지·라벨 쌍으로 사용하세요. `needs_human/`은 사람 확인 전 학습에 넣지 말고, 제외 사유는 `exclusions.csv`, 변경 내역은 `changes.csv`에서 확인하세요. `before_after/`는 참고 자료이며 학습 입력이 아닙니다.
"""
