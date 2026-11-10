"""Frozen AP evaluation runner. Default is model-free preflight, never evaluation."""

from __future__ import annotations

import hashlib
import importlib.abc
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final, TypedDict
from uuid import uuid4

if TYPE_CHECKING:
    from torch import Tensor
    from ultralytics.utils.metrics import DetMetrics

HERE: Final = Path(__file__).resolve().parent
PROTOCOL: Final = HERE / "protocol.json"
MANIFEST: Final = HERE / "inputs-manifest.json"
NAMES: Final = {0: "potato", 1: "stone"}
UNITS: Final = tuple(
    f"{model}__{split}"
    for model in ("current", "stone40")
    for split in ("val11", "test12", "external166")
)
type Json = None | bool | int | float | str | list[Json] | dict[str, Json]


class ClassAP(TypedDict):
    """One actual metric row or explicitly unavailable class."""

    class_id: int
    AP50: float | None
    AP50_95: float | None
    reason: str | None
    library_best_f1_precision: float | None
    library_best_f1_recall: float | None


class ModelImportBlocked(ImportError):
    """Preparation is not authorized to import model implementation modules."""


class NoModelImports(importlib.abc.MetaPathFinder):
    """Prevent accidental model imports during preflight and CPU fixtures."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("ultralytics.models", "ultralytics.nn", "ultralytics.engine")):
            raise ModelImportBlocked(fullname)


def sha(path: Path) -> str:
    """Hash bytes, never deserialize models."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load(path: Path):
    """Read a JSON document at the file boundary."""
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, value) -> None:
    """Create an immutable receipt, refusing overwrite."""
    with path.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def event(path: Path, kind: str, **fields) -> None:
    """Append an actual lifecycle event; prior events are never truncated."""
    with (path / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"event": kind, "time_ns": str(time.time_ns()), **fields}) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def inside(path: Path) -> Path:
    """Reject symlinks or paths escaping this node's only writable directory."""
    absolute = path.absolute()
    if absolute != absolute.resolve() or not absolute.is_relative_to(HERE):
        raise RuntimeError(f"Non-private path refused: {path}")
    return absolute


def bindings():
    """Recompute every frozen input, code, source, config and version binding."""
    protocol = load(PROTOCOL)
    manifest = load(MANIFEST)
    if sha(MANIFEST) != protocol["inputs_manifest_sha256"]:
        raise RuntimeError("Input manifest changed")
    if sys.version != manifest["python"] or sys.executable != manifest["executable"]:
        raise RuntimeError("Use the exact frozen Python executable/version")
    if sha(Path(sys.executable).resolve()) != manifest["executable_sha256"]:
        raise RuntimeError("Python executable changed")
    for name, expected in manifest["installed_versions"].items():
        if importlib.metadata.version(name) != expected:
            raise RuntimeError(f"Installed version changed: {name}")
    for group in ("source_hashes", "installed_source_hashes", "config_hashes"):
        for filename, expected in manifest[group].items():
            if sha(Path(filename)) != expected:
                raise RuntimeError(f"Frozen {group} changed: {filename}")
    for filename, expected in protocol["code_hashes"].items():
        if sha(inside(HERE / filename)) != expected:
            raise RuntimeError(f"Frozen code/config changed: {filename}")
    installed = Path(manifest["ultralytics_root"])
    current_sources = {
        str(p) for p in installed.rglob("*")
        if p.is_file() and p.suffix in (".py", ".yaml", ".yml")
    }
    if current_sources != set(manifest["installed_source_hashes"]):
        raise RuntimeError("Installed source membership changed")
    for row in manifest["private_copies"]:
        private = inside(Path(row["private"]))
        if private.stat().st_nlink != 1:
            raise RuntimeError(f"Hardlinked private input refused: {private}")
        if sha(private) != row["sha256"] or sha(Path(row["source"])) != row["sha256"]:
            raise RuntimeError(f"Original/private bytes changed: {private}")
    for split, members in manifest["membership"].items():
        data_split = "val" if split == "val11" else "test"
        base = HERE / "private" / split
        images = sorted((base / "images" / data_split).glob("*"))
        labels = sorted((base / "labels" / data_split).glob("*.txt"))
        if [p.name for p in images] != members:
            raise RuntimeError(f"Private image membership changed: {split}")
        if {p.stem for p in labels} != {Path(p).stem for p in members}:
            raise RuntimeError(f"Private label membership changed: {split}")
    if tuple(p["id"] for p in protocol["matrix"]) != UNITS:
        raise RuntimeError("Expected exactly two checkpoints times three splits")
    return protocol, manifest


def isolated(runtime: Path) -> None:
    """Set all mutable locations and offline guards before scientific imports."""
    inside(runtime).mkdir(parents=True, exist_ok=False)
    locations = {
        "YOLO_CONFIG_DIR": "yolo", "MPLCONFIGDIR": "mpl", "TORCH_HOME": "torch",
        "XDG_CACHE_HOME": "cache", "XDG_CONFIG_HOME": "config",
        "TORCHINDUCTOR_CACHE_DIR": "inductor", "TRITON_CACHE_DIR": "triton",
        "CUDA_CACHE_PATH": "cuda", "TMPDIR": "tmp", "HOME": "home",
    }
    for key, name in locations.items():
        directory = runtime / name
        directory.mkdir()
        os.environ[key] = str(directory)
    os.environ.update({
        "YOLO_OFFLINE": "true", "YOLO_AUTOINSTALL": "false",
        "MPLBACKEND": "Agg", "PYTHONDONTWRITEBYTECODE": "1",
        "MPL_IGNORE_SYSTEM_FONTS": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "WANDB_MODE": "disabled", "HF_HUB_OFFLINE": "1",
        "CUDA_VISIBLE_DEVICES": "0", "PYTHONHASHSEED": "42",
    })
    for key in ("WORLD_SIZE", "RANK", "LOCAL_RANK", "ULTRALYTICS_API_KEY",
                "ULTRALYTICS_PLATFORM_API_KEY", "COMET_API_KEY", "WANDB_API_KEY"):
        os.environ.pop(key, None)
    sys.dont_write_bytecode = True
    config = runtime / "yolo/Ultralytics"  # installed get_user_config_dir appends this
    config.mkdir()
    settings = load(HERE / "settings-template.json")
    for key, name in (("datasets_dir", "datasets"), ("weights_dir", "weights"), ("runs_dir", "runs")):
        settings[key] = str(runtime / name)
    save(config / "settings.json", settings)
    shutil.copyfile(HERE / "private/font.ttf", config / "Arial.ttf")
    shutil.copyfile(HERE / "private/font.ttf", config / "Arial.Unicode.ttf")
    os.chdir(runtime)

    def audit(name, args):
        # Guard Python writes even if a library ignores its configured cache location.
        if name == "open":
            filename, mode, flags = args
            writing = (mode is not None and any(c in mode for c in "wax+")) or (
                flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
            )
            if writing and not isinstance(filename, int):
                inside(Path(os.fsdecode(filename)))
        elif name in ("os.mkdir", "os.remove", "os.rmdir", "os.chmod", "os.utime"):
            inside(Path(os.fsdecode(args[0])))
        elif name in ("os.rename", "os.link", "os.symlink"):
            inside(Path(os.fsdecode(args[0])))
            inside(Path(os.fsdecode(args[1])))
        elif name in ("socket.connect", "socket.connect_ex", "socket.getaddrinfo",
                      "socket.sendto", "socket.bind", "subprocess.Popen", "os.system",
                      "os.posix_spawn"):
            raise RuntimeError(f"Offline/no-subprocess guard: {name}")

    sys.addaudithook(audit)


def settings_for(unit: str, output: Path):
    """Construct the sole frozen evaluation call for a named unit."""
    model, split = unit.split("__")
    data = inside(HERE / "private" / split / "data.yaml")
    return {
        **load(PROTOCOL)["settings"],
        "model": str(inside(HERE / "private/models" / f"{model}.pt")),
        "data": str(data), "split": "val" if split == "val11" else "test",
        "classes": [0] if split == "external166" else None,
        "project": str(output), "name": "validation", "exist_ok": False,
    }


def approved(path: Path, protocol) -> None:
    """Require independent attestation bound to the exact ready artifact bytes."""
    gate = load(inside(path))
    expected = {
        "status": "approved", "independent": True,
        "protocol_sha256": sha(PROTOCOL),
        "inputs_manifest_sha256": sha(MANIFEST),
        "code_hashes": protocol["code_hashes"],
        "unit_ids": list(UNITS),
    }
    if any(gate.get(k) != v for k, v in expected.items()):
        raise RuntimeError("Independent approval missing or has stale exact bindings")
    if not gate.get("reviewer") or gate["reviewer"] == protocol["prepared_by"]:
        raise RuntimeError("Independent named reviewer required")


def attempt_inventory():
    """Report persisted actual launches/calls, including failed and incomplete runs."""
    attempts = []
    for path in sorted((HERE / "units").glob("*/attempt-*")):
        log = path / "events.jsonl"
        events = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
        attempts.append({
            "path": str(path), "unit": path.parent.name,
            "launched": any(e["event"] == "process_started" for e in events),
            "model_val_calls": sum(e["event"] == "model_val_call" for e in events),
            "completed": (path / "result.json").exists(),
            "exit": load(path / "exit.json") if (path / "exit.json").exists() else None,
        })
    return attempts


def per_class(metrics: DetMetrics, external: bool) -> dict[str, ClassAP]:
    """Map actual AP array rows to class IDs; never use aggregate-filled box.maps."""
    indices = [int(i) for i in metrics.box.ap_class_index]
    if len(indices) != len(set(indices)) or any(i not in NAMES for i in indices):
        raise RuntimeError("Unexpected metric class-index mapping")
    if not all(len(v) == len(indices) for v in (
        metrics.box.ap50, metrics.box.ap, metrics.box.p, metrics.box.r,
    )):
        raise RuntimeError("Metric field length mismatch")
    result: dict[str, ClassAP] = {}
    for class_id, name in NAMES.items():
        row: ClassAP = {
            "class_id": class_id, "AP50": None, "AP50_95": None, "reason": None,
            "library_best_f1_precision": None, "library_best_f1_recall": None,
        }
        if external and class_id == 1:
            row["reason"] = "external_ground_truth_is_potato_only_no_stone_gt"
        elif class_id not in indices:
            row["reason"] = "class_absent_from_installed_ap_class_index"
        else:
            i = indices.index(class_id)
            values = [float(v[i]) for v in (
                metrics.box.ap50, metrics.box.ap, metrics.box.p, metrics.box.r,
            )]
            if not all(math.isfinite(v) and 0 <= v <= 1 for v in values):
                raise RuntimeError("Non-finite/out-of-range installed metric")
            row["AP50"], row["AP50_95"] = values[:2]
            row["library_best_f1_precision"], row["library_best_f1_recall"] = values[2:]
        result[name] = row
    return result


def external_postprocess(preds: Tensor) -> list[dict[str, Tensor]]:
    """Installed validator NMS with the missing classes=[0] argument made explicit."""
    from ultralytics.utils import nms

    outputs = nms.non_max_suppression(
        preds, conf_thres=0.001, iou_thres=0.5, classes=[0],
        nc=0, multi_label=True, agnostic=False, max_det=300,
        end2end=False, rotated=False,
    )
    assert isinstance(outputs, list), "Installed NMS return_idxs=False must return a list"
    return [
        {"bboxes": x[:, :4], "conf": x[:, 4], "cls": x[:, 5], "extra": x[:, 6:]}
        for x in outputs
    ]


def preflight() -> int:
    """Emit input/intent/GPU evidence without importing any model module."""
    protocol, manifest = bindings()
    sys.meta_path.insert(0, NoModelImports())
    output = HERE / "checks" / f"preflight-{uuid4().hex}"
    output.mkdir(parents=True)
    isolated(output / "runtime")
    import torch

    cuda = torch.cuda.is_available()
    attempts = attempt_inventory()
    report = {
        "protocol_sha256": sha(PROTOCOL), "inputs_manifest_sha256": sha(MANIFEST),
        "bindings_verified": True, "units": list(UNITS),
        "intents": [
            {"unit": unit, "argv": [sys.executable, "-B", str(HERE / "runner.py"),
                                   "run", unit, "<independent-approval.json>"],
             "settings": settings_for(unit, HERE / "units" / unit / "<attempt>")}
            for unit in UNITS
        ],
        "cuda_available": cuda, "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if cuda else None,
        "blockers": [] if cuda else ["CUDA device 0 unavailable; no fallback permitted"],
        "gate_status": "independent_approval_required",
        "model_modules_imported": [
            n for n in sys.modules
            if n.startswith(("ultralytics.models", "ultralytics.nn", "ultralytics.engine"))
        ],
        "actual_attempts": attempts,
        "evaluation_attempts": sum(a["model_val_calls"] for a in attempts),
        "private_copies": len(manifest["private_copies"]),
        "gt_instances": manifest["gt_instances"],
    }
    if report["model_modules_imported"]:
        raise RuntimeError("Preflight imported model implementation")
    save(output / "approval-request.json", {
        "status": "pending_independent_review", "independent": False, "reviewer": None,
        "protocol_sha256": sha(PROTOCOL), "inputs_manifest_sha256": sha(MANIFEST),
        "code_hashes": protocol["code_hashes"], "unit_ids": list(UNITS),
        "allow_retry_failed": False,
    })
    save(output / "preflight.json", report)
    print(json.dumps({"receipt": str(output / "preflight.json"), **report}, ensure_ascii=False))
    return 0 if cuda else 2


def worker(unit: str, gate: Path, attempt: Path) -> int:
    """Run exactly one approved evaluation; called only by the recording controller."""
    protocol, manifest = bindings()
    approved(gate, protocol)  # must precede isolation's scientific imports and model import
    intent = load(attempt / "intent.json")
    if intent["unit"] != unit or intent["gate_sha256"] != sha(gate):
        raise RuntimeError("Worker intent/gate binding mismatch")
    if attempt.parent != HERE / "units" / unit:
        raise RuntimeError("Worker output mismatch")
    if intent["protocol_sha256"] != sha(PROTOCOL) or intent["settings"] != settings_for(unit, attempt):
        raise RuntimeError("Worker protocol/settings binding mismatch")
    isolated(attempt / "runtime")
    import torch

    if not torch.cuda.is_available():
        event(attempt, "blocked", reason="CUDA device 0 unavailable; no fallback")
        return 2
    event(attempt, "model_import", checkpoint_sha256=sha(Path(intent["settings"]["model"])))
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect.val import DetectionValidator
    from ultralytics.utils.torch_utils import init_seeds

    init_seeds(42, deterministic=True)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    kwargs = settings_for(unit, attempt)
    model = YOLO(kwargs.pop("model"), task="detect")
    native = model.model
    assert isinstance(native, torch.nn.Module), "Frozen .pt checkpoint must load a torch module"
    if model.names != NAMES or next(native.parameters()).dtype != torch.float32:
        raise RuntimeError("Expected FP32 potato0/stone1 checkpoint")
    if getattr(model.model, "end2end", False):
        raise RuntimeError("Frozen YOLO11 external NMS adapter requires non-end2end detection")

    class ExternalValidator(DetectionValidator):
        """Filter predictions before metric matching and before max_det truncation."""

        def postprocess(self, preds):
            return external_postprocess(preds)

    def on_start(validator):
        actual = vars(validator.args).copy()
        save(attempt / "actual-settings.json", actual)
        for key, expected in kwargs.items():
            if key != "half" and actual[key] != expected:
                raise RuntimeError(f"Effective setting changed: {key}")
        if actual["quantize"] is not None or validator.device.type != "cuda":
            raise RuntimeError("FP32 CUDA contract changed")
        split = unit.split("__")[1]
        members = sorted(Path(p).name for p in validator.dataloader.dataset.im_files)
        if members != sorted(manifest["membership"][split]):
            raise RuntimeError("Validator dropped or added images")
        instances = [0, 0]
        for label in validator.dataloader.dataset.labels:
            for class_id in label["cls"].reshape(-1):
                instances[int(class_id)] += 1
        if instances != [manifest["gt_instances"][split][str(i)] for i in (0, 1)]:
            raise RuntimeError("Validator changed GT instances")
        event(attempt, "validation_started", images=len(members), instances=instances,
              device=str(validator.device), deterministic=torch.are_deterministic_algorithms_enabled())

    model.add_callback("on_val_start", on_start)
    event(attempt, "model_val_call", settings=kwargs)
    metrics = model.val(
        validator=ExternalValidator if unit.endswith("external166") else DetectionValidator,
        **kwargs,
    )
    external = unit.endswith("external166")
    result = {
        "unit": unit, "protocol_sha256": sha(PROTOCOL), "gate_sha256": sha(gate),
        "checkpoint_sha256": sha(Path(intent["settings"]["model"])),
        "ap_class_index": [int(i) for i in metrics.box.ap_class_index],
        "all_ap": metrics.box.all_ap.tolist(),
        "classes": per_class(metrics, external),
        "aggregate_library_metrics": {k: float(v) for k, v in metrics.results_dict.items()},
        "precision_recall_definition": "library_smoothed_mean_best_F1_not_fixed_confidence_0.45",
        "historical_fixed_conf_0_45": "separate_reference_only_not_recomputed",
    }
    bindings()  # fail closed if inputs changed during evaluation
    save(attempt / "result.json", result)
    event(attempt, "validation_completed", result_sha256=sha(attempt / "result.json"))
    return 0


def run(unit: str, gate: Path, retry: bool) -> int:
    """Record one requested attempt; never auto-run remaining units or replay success."""
    if unit not in UNITS:
        raise RuntimeError(f"Unknown unit: {unit}")
    protocol, _ = bindings()
    approved(gate, protocol)
    directory = inside(HERE / "units" / unit)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "completed.json").exists() or any(directory.glob("attempt-*/result.json")):
        raise RuntimeError("Completed unit is immutable; replay refused")
    previous = sorted(directory.glob("attempt-*"))
    if previous and not retry:
        raise RuntimeError("Prior failed/incomplete attempt exists; explicit retry approval required")
    if previous and not load(gate).get("allow_retry_failed", False):
        raise RuntimeError("Approval does not authorize failed-attempt retry")
    lock = directory / "active.lock"
    with lock.open("x", encoding="utf-8") as stream:
        stream.write(str(os.getpid()))
    try:
        attempt = directory / f"attempt-{len(previous) + 1:03d}-{uuid4().hex}"
        attempt.mkdir()
        command = [sys.executable, "-B", str(HERE / "runner.py"),
                   "_worker", unit, str(gate), str(attempt)]
        save(attempt / "intent.json", {
            "unit": unit, "argv": command, "requested_argv": sys.argv,
            "settings": settings_for(unit, attempt), "protocol_sha256": sha(PROTOCOL),
            "gate_sha256": sha(gate), "inputs_manifest_sha256": sha(MANIFEST),
            "prior_attempts": [p.name for p in previous], "automatic_retry": False,
        })
        event(attempt, "process_launch_intent", argv=command)
        environment = os.environ.copy()
        environment.update({"PYTHONHASHSEED": "42", "PYTHONDONTWRITEBYTECODE": "1"})
        with (attempt / "stdout.log").open("xb") as stdout, (attempt / "stderr.log").open("xb") as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr, env=environment, cwd=HERE)
            event(attempt, "process_started", pid=process.pid)
            code = process.wait()
        save(attempt / "exit.json", {
            "exit_code": code, "result_exists": (attempt / "result.json").exists(),
            "stdout_sha256": sha(attempt / "stdout.log"),
            "stderr_sha256": sha(attempt / "stderr.log"),
        })
        if code == 0 and (attempt / "result.json").exists():
            save(directory / "completed.json", {
                "attempt": attempt.name, "result_sha256": sha(attempt / "result.json"),
                "protocol_sha256": sha(PROTOCOL), "gate_sha256": sha(gate),
            })
        print(json.dumps({"unit": unit, "attempt": str(attempt), "exit_code": code}))
        return code
    finally:
        lock.unlink()


def main() -> int:
    """Use explicit commands; no evaluation is reachable from the default command."""
    args = sys.argv[1:]
    if not args or args == ["preflight"]:
        return preflight()
    if args[0] == "run" and len(args) in (3, 4):
        if len(args) == 4 and args[3] != "--retry-failed":
            raise RuntimeError("Only --retry-failed is accepted")
        return run(args[1], inside(Path(args[2])), len(args) == 4)
    if args[0] == "_worker" and len(args) == 4 and args[1] in UNITS:
        return worker(args[1], inside(Path(args[2])), inside(Path(args[3])))
    raise RuntimeError("Usage: runner.py preflight | run UNIT /absolute/approval.json [--retry-failed]")


if __name__ == "__main__":
    raise SystemExit(main())
