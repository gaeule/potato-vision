"""Train fresh COCO-pretrained YOLO11s with the unchanged baseline preset."""

import argparse
from pathlib import Path

from stone_experiment.common import (
    BASE,
    DATASET,
    EXPERIMENT,
    OUTPUTS,
    ROOT,
    Options,
    PipelineError,
    environment,
    new_output,
    runtime,
    sha256,
    write_json,
)
from stone_experiment.prepare import verify_dataset
from stone_experiment.schema import document


def train(dataset: Path, output: Path) -> None:
    """Never import external images/ZIP or fine-tune the single-class checkpoint."""
    _ = verify_dataset(dataset)
    config_path = EXPERIMENT / "config/train.json"
    config = document(config_path)
    if config != document(BASE / "config/train.json") or config["model"] != "yolo11s.pt":
        raise PipelineError("Training preset must exactly match baseline/config/train.json")
    pretrained = ROOT / "yolo11s.pt"
    if not pretrained.is_file():
        raise PipelineError(f"Fresh root COCO-pretrained weights required: {pretrained}")
    output = new_output(output)
    runtime()
    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise PipelineError("Comparable training requires device 0 CUDA")
    model = YOLO(str(pretrained))
    if len(model.names) != 80:
        raise PipelineError("Root yolo11s.pt must be fresh COCO-pretrained, not potato fine-tuned")
    write_json(output / "environment.json", environment())
    write_json(output / "inputs.json", {
        "config": config, "config_sha256": sha256(config_path),
        "baseline_config_sha256": sha256(BASE / "config/train.json"),
        "pretrained_sha256": sha256(pretrained),
        "data_yaml_sha256": sha256(dataset / "data.yaml"),
        "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
        "annotation_snapshot_sha256": sha256(dataset / "annotations.json"),
    })
    parameters = {key: value for key, value in config.items() if key != "model"}
    _ = model.train(**parameters, data=str(dataset / "data.yaml"),
                project=str(output.parent), name=output.name, exist_ok=True)
    write_json(output / "checkpoints.json", {
        name: sha256(output / "weights" / f"{name}.pt") for name in ("best", "last")
    })


def main() -> None:
    """Start only the explicitly requested training run."""
    parser = argparse.ArgumentParser(description=__doc__)
    _ = parser.add_argument("--dataset", type=Path, default=DATASET)
    _ = parser.add_argument("--output", type=Path, default=OUTPUTS / "train")
    args = parser.parse_args(namespace=Options())
    train(args.dataset, args.output)


if __name__ == "__main__":
    main()
