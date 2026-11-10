"""CPU-only behavioral fixtures using the installed metrics/NMS/config, no models."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import runner


def check_event_timestamp(output: Path) -> None:
    """Preserve nanoseconds beyond JavaScript's safe integer range."""
    # Given a fixed nanosecond value that is not exactly representable as Number.
    timestamp = 1793869200123456789
    # When the real event writer serializes it.
    with patch.object(runner.time, "time_ns", return_value=timestamp):
        runner.event(output, "timestamp_fixture")
    # Then the machine-consumed field is an exact decimal string.
    value = runner.json.loads((output / "events.jsonl").read_text())
    assert value["time_ns"] == str(timestamp), "time_ns must be an exact decimal string"


def main() -> None:
    """Check real installed field mapping, NMS filtering and configuration types."""
    runner.bindings()
    for source in ("runner.py", "prepare_inputs.py", "check_fixtures.py"):
        path = runner.HERE / source
        compile(path.read_bytes(), str(path), "exec")
    sys.meta_path.insert(0, runner.NoModelImports())
    output = runner.HERE / "checks" / f"fixtures-{uuid4().hex}"
    output.mkdir(parents=True)
    check_event_timestamp(output)
    runner.isolated(output / "runtime")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch
    from ultralytics.cfg import get_cfg
    from ultralytics.utils.checks import check_font
    from ultralytics.utils.metrics import DetMetrics

    # Given a fresh private cache and the real no-subprocess/network guard;
    # When dataset validation requests its font; Then no fontconfig subprocess
    # or download is needed, and the existing private font is returned.
    font = check_font("Arial.ttf")
    assert font == output / "runtime/yolo/Ultralytics/Arial.ttf"
    assert runner.sha(font) == runner.sha(runner.HERE / "private/font.ttf")

    def metric(predicted, target, true_positive):
        value = DetMetrics(names=runner.NAMES)
        value.stats = {
            "tp": [np.asarray(true_positive, dtype=bool)],
            "conf": [np.asarray([0.9] * len(predicted))],
            "pred_cls": [np.asarray(predicted)],
            "target_cls": [np.asarray(target)],
            "target_img": [np.asarray(sorted(set(target)))],
        }
        value.process(plot=False)
        return value

    # Given unequal real AP rows; When extracted; Then classes must not share mAP.
    unequal = metric([0, 1], [0, 1], [[True] * 10, [False] * 10])
    rows = runner.per_class(unequal, False)
    potato_ap = rows["potato"]["AP50"]
    assert potato_ap is not None and potato_ap > 0.99
    assert rows["stone"]["AP50"] == 0
    assert rows["potato"]["AP50"] != float(unequal.box.map50)
    # Given only GT class 1; When mapped; Then row 0 belongs to stone, not potato.
    stone_only = metric([1], [1], [[True] * 10])
    stone_rows = runner.per_class(stone_only, False)
    assert list(stone_only.box.ap_class_index) == [1]
    stone_ap = stone_rows["stone"]["AP50_95"]
    assert stone_ap is not None and stone_ap > 0.99
    assert stone_rows["potato"]["AP50"] is None
    # Given external potato GT; When exported; Then unscored stone is null, not 0.
    external = runner.per_class(metric([0], [0], [[True] * 10]), True)
    assert external["stone"]["AP50"] is None
    assert external["stone"]["AP50_95"] is None
    assert external["stone"]["reason"]
    # Given >300 high-confidence stone boxes; When NMS runs; Then potato survives
    # class filtering before the global max_det limit, not just after matching.
    preds = torch.zeros((1, 6, 302), dtype=torch.float32)
    preds[0, 0, :] = torch.arange(302, dtype=torch.float32) * 10
    preds[0, 1:4, :] = 2
    preds[0, 5, :301] = 0.99
    preds[0, 4, 301] = 0.9
    detections = runner.external_postprocess(preds)
    assert len(detections) == 1 and detections[0]["cls"].tolist() == [0]
    assert detections[0]["bboxes"].shape == (1, 4)
    # Given the frozen kwargs; When installed get_cfg parses; Then exact types and
    # deprecated half=False -> quantize=None semantics remain valid for all units.
    normalized = {}
    for unit in runner.UNITS:
        kwargs = runner.settings_for(unit, output)
        args = get_cfg(overrides=kwargs)
        assert args.quantize is None
        assert args.batch == 1 and args.device == 0 and args.imgsz == 640
        assert args.conf == 0.001 and args.iou == 0.5
        assert args.augment is False and args.single_cls is False
        normalized[unit] = vars(args)
    # Given no review approval; When gate checks; Then no model import is reachable.
    rejected = output / "unapproved.json"
    runner.save(rejected, {"status": "not_approved"})
    try:
        runner.approved(rejected, runner.load(runner.PROTOCOL))
    except RuntimeError:
        pass
    else:
        raise AssertionError("Unapproved gate accepted")
    # Given a stale protocol digest; When checked; Then independent=true is not enough.
    stale = output / "stale-approval.json"
    protocol = runner.load(runner.PROTOCOL)
    runner.save(stale, {
        "status": "approved", "independent": True, "reviewer": "fixture-reviewer",
        "protocol_sha256": "0" * 64, "inputs_manifest_sha256": runner.sha(runner.MANIFEST),
        "code_hashes": protocol["code_hashes"], "unit_ids": list(runner.UNITS),
    })
    try:
        runner.approved(stale, protocol)
    except RuntimeError:
        pass
    else:
        raise AssertionError("Stale gate accepted")
    imported = [
        n for n in sys.modules
        if n.startswith(("ultralytics.models", "ultralytics.nn", "ultralytics.engine"))
    ]
    assert imported == []
    receipt = {
        "passed": True, "device": "cpu", "model_modules_imported": imported,
        "evaluation_attempts": 0, "protocol_sha256": runner.sha(runner.PROTOCOL),
        "inputs_manifest_sha256": runner.sha(runner.MANIFEST),
        "checks": [
            "event_time_ns_exact_decimal_string",
            "fresh_cache_check_font_under_offline_subprocess_guard",
            "three_python_sources_compile_without_bytecode",
            "real_DetMetrics_unequal_AP_rows", "nonzero_class_id_row_mapping",
            "external_stone_null", "external_filter_before_max_det",
            "six_installed_get_cfg_types_FP32", "unapproved_gate_rejected",
            "stale_binding_rejected",
        ],
        "unequal_rows": rows, "stone_only_rows": stone_rows,
        "external_rows": external, "normalized_settings": normalized,
    }
    runner.save(output / "fixtures.json", receipt)
    print(f"CPU_FIXTURES_PASS {output / 'fixtures.json'}")


if __name__ == "__main__":
    main()
