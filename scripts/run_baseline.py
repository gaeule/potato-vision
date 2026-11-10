"""Pinned training, fixed-checkpoint evaluation, and full-video visualization."""

import argparse
import importlib.metadata
import json
import platform
import sys
from typing import TYPE_CHECKING, Final, TypedDict, assert_never

from .baseline_common import (
    BASE,
    SOURCE,
    PipelineError,
    output_directory,
    sha256,
    write_json,
)

if TYPE_CHECKING:
    from ultralytics import YOLO

BEST: Final = BASE / "runs/train/weights/best.pt"
SELECTED: Final = BASE / "runs/train/weights/selected.pt"
CONF: Final = 0.25
IOU: Final = 0.7


class Environment(TypedDict):
    """Recorded runtime details attached to training and evaluation results."""

    python: str
    platform: str
    packages: dict[str, str]
    cuda_runtime: str | None
    cudnn: int | None
    cuda_available: bool
    gpu: str | None


def environment() -> Environment:
    """Record package versions and the actual CUDA device/runtime."""
    import torch

    packages = ("torch", "torchvision", "ultralytics", "av", "pillow", "numpy",
                "scipy", "scikit-image", "PyYAML", "ruff")
    return {
        "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "cuda_runtime": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def pinned_environment() -> None:
    """Refuse silently different training/inference frameworks."""
    for name, expected in (("ultralytics", "8.4.183"), ("torch", "2.9.1+cu128"),
                           ("torchvision", "0.24.1+cu128")):
        actual = importlib.metadata.version(name)
        if actual != expected:
            raise PipelineError(f"{name}: expected {expected}, installed {actual}")


def best_model() -> "YOLO":
    """Load the validation-selected operational checkpoint."""
    from ultralytics import YOLO

    if not SELECTED.is_file():
        raise PipelineError("Run python -m scripts.select_checkpoint after training")
    return YOLO(str(SELECTED))


def train() -> None:
    """Train on internal frames only; validation selects the checkpoint."""
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise PipelineError("Pinned device=0 requires a CUDA GPU")
    dataset = BASE / "internal/data.yaml"
    if not dataset.is_file():
        raise PipelineError("Run prepare_data.py internal first")
    # This branch never opens, imports, or enumerates the external ZIP/dataset.
    output = output_directory(BASE / "runs/train", regenerate=False)
    config = json.loads((BASE / "config/train.json").read_text(encoding="utf-8"))
    write_json(output / "environment.json", environment())
    write_json(output / "input.json", {
        "config": config, "data_yaml_sha256": sha256(dataset),
        "dataset_manifest_sha256": sha256(BASE / "internal/manifest.json"),
    })
    model = YOLO(config.pop("model"))
    model.train(**config, data=str(dataset), project=str(output.parent), name=output.name, exist_ok=True)
    # The model above may retain last weights; explicitly reopen validation-selected best.
    write_json(output / "checkpoint.json", {
        "checkpoint": str(BEST), "sha256": sha256(BEST),
        "selection": "internal_validation_fitness_only",
    })


def evaluate(dataset: str) -> None:
    """Evaluate the selected checkpoint without fitting on holdouts."""
    from .select_checkpoint import score_predictions
    match dataset:
        case "val":
            data, split = BASE / "internal/data.yaml", "val"
        case "test":
            data, split = BASE / "internal/data.yaml", "test"
        case "external":
            data, split = BASE / "external/data.yaml", "test"
        case _:
            raise PipelineError(f"Unknown evaluation dataset: {dataset}")
    if not data.is_file():
        raise PipelineError(f"Prepare dataset first: {data}")
    model = best_model()
    output = output_directory(BASE / "evaluations" / dataset, regenerate=False)
    metrics = model.val(
        data=str(data), split=split, imgsz=640, batch=8, workers=0, device=0,
        conf=0.001, iou=IOU, seed=42, deterministic=True,
        project=str(output.parent), name=output.name, exist_ok=True, plots=True,
    )
    write_json(output / "metrics.json", {
        "dataset": dataset, "split": split, "checkpoint": str(SELECTED),
        "checkpoint_sha256": sha256(SELECTED), "data_yaml_sha256": sha256(data),
        "dataset_manifest_sha256": sha256(data.parent / "manifest.json"),
        "environment": environment(), "imgsz": 640, "metric_conf": 0.001,
        "visualization_conf": CONF, "nms_iou": IOU,
        "precision_recall_definition": "dataset_best_F1_operating_point_not_fixed_visualization_conf",
        "metrics": {key: float(value) for key, value in metrics.results_dict.items()},
        "speed_ms": metrics.speed,
    })
    # Separate explicitly fixed-confidence visualizations, not metric tuning.
    model.predict(
        source=str(data.parent / "images" / split), imgsz=640, device=0,
        conf=CONF, iou=IOU, save=True, save_txt=True, save_conf=True,
        project=str(output), name="predictions", exist_ok=True,
    )
    write_json(output / "fixed_confidence_metrics.json", score_predictions(
        data.parent / "labels" / split, output / "predictions/labels",
    ))


def infer_video() -> None:
    """Encode H.264/no-audio video with per-frame on-screen detection counts."""
    import av
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from torch import Tensor
    from ultralytics.engine.results import Results

    model = best_model()
    output = output_directory(BASE / "inference", regenerate=False)
    counts = []
    font = ImageFont.load_default(size=30)
    with av.open(str(SOURCE)) as source, av.open(str(output / "annotated.mp4"), "w") as target:
        source_stream = source.streams.video[0]
        if source_stream.average_rate is None:
            raise PipelineError("Video has no frame rate")
        stream = target.add_stream("libx264", rate=source_stream.average_rate)
        # Preserve VFR timestamps rather than rounding onto the average-rate grid.
        stream.time_base = source_stream.time_base
        stream.codec_context.time_base = source_stream.time_base
        stream.width, stream.height = source_stream.width, source_stream.height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "20", "preset": "medium"}
        for index, frame in enumerate(source.decode(video=0)):
            if frame.pts is None or frame.time_base is None:
                raise PipelineError("Source frame lacks timestamp")
            prediction = next(iter(model.predict(
                frame.to_ndarray(format="bgr24"), imgsz=640,
                device=0, conf=CONF, iou=IOU, verbose=False,
            )))
            match prediction:
                case Results():
                    result = prediction
                case Tensor():
                    raise PipelineError("Detection inference returned embeddings instead of Results")
                case unexpected:
                    assert_never(unexpected)
            if result.boxes is None:
                raise PipelineError("Detection inference returned no box container")
            count = len(result.boxes)
            image = Image.fromarray(result.plot()[..., ::-1])
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 0, 610, 48), fill="black")
            draw.text((10, 6), f"ON SCREEN: {count} (not unique count)", font=font, fill="white")
            annotated = av.VideoFrame.from_ndarray(np.asarray(image), format="rgb24")
            annotated.pts, annotated.time_base = frame.pts, frame.time_base
            for packet in stream.encode(annotated):
                target.mux(packet)
            counts.append({"frame": index, "second": float(frame.pts * frame.time_base),
                           "on_screen_count": count})
        for packet in stream.encode():
            target.mux(packet)
    write_json(output / "counts.json", {
        "meaning": "per_frame_visible_detections_not_unique_potatoes",
        "source_sha256": sha256(SOURCE), "checkpoint_sha256": sha256(SELECTED),
        "conf": CONF, "nms_iou": IOU, "audio": False, "codec": "h264",
        "environment": environment(), "frames": counts,
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train")
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--dataset", choices=("val", "test", "external"), required=True)
    commands.add_parser("video")
    args = parser.parse_args()
    pinned_environment()
    match args.command:
        case "train":
            train()
        case "evaluate":
            evaluate(args.dataset)
        case "video":
            infer_video()
