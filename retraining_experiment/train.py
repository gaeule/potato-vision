"""Fresh COCO YOLO11s training on an isolated train88/val11 private dataset."""

from dataclasses import asdict, dataclass
from pathlib import Path

from retraining_data.schema import require
from retraining_experiment.common import (
    DATASET,
    OUTPUTS,
    ROOT,
    check_frozen,
    new_output,
    record_context,
    runtime,
    sha256,
    write_json,
)
from retraining_experiment.data import private_copy, verified
from stone_experiment.schema import document


@dataclass(frozen=True, slots=True)
class TrainConfig:
    """Exact unchanged baseline preset parsed at the file boundary."""

    model: str = "yolo11s.pt"
    imgsz: int = 640
    seed: int = 42
    deterministic: bool = True
    epochs: int = 60
    patience: int = 15
    batch: int = 8
    workers: int = 0
    device: int = 0
    pretrained: bool = True
    optimizer: str = "SGD"
    lr0: float = 0.01
    lrf: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    amp: bool = True
    val: bool = True
    save: bool = True
    plots: bool = True


def config() -> TrainConfig:
    """Refuse preset drift between both reference configurations."""
    baseline = ROOT / "baseline/config/train.json"
    stone = ROOT / "stone_experiment/config/train.json"
    value = TrainConfig()
    expected = asdict(value)
    require(document(baseline) == document(stone) == expected, "Preset drift")
    check_frozen(baseline, "train_config_sha256")
    check_frozen(stone, "train_config_sha256")
    return value


def train(dataset: Path = DATASET, output: Path = OUTPUTS / "train") -> None:
    """No test/external paths ever reach model.train or the label-cache loader."""
    dataset = dataset.resolve()
    snapshot, preset = verified(dataset), config()
    pretrained = ROOT / "yolo11s.pt"
    require(pretrained.is_file(), "Root COCO yolo11s.pt is required")
    check_frozen(pretrained, "pretrained_sha256")
    output = new_output(output)
    yaml = private_copy(dataset, snapshot, output)
    with runtime(output):
        import torch
        from ultralytics import YOLO

        require(torch.cuda.is_available(), "Comparable training requires CUDA device 0")
        model = YOLO(str(pretrained))
        require(len(model.names) == 80, "Expected fresh 80-class root COCO checkpoint")
        record_context(output, (pretrained,))
        write_json(output / "config.json", asdict(preset))
        _ = model.train(
            imgsz=preset.imgsz,
            seed=preset.seed,
            deterministic=preset.deterministic,
            epochs=preset.epochs,
            patience=preset.patience,
            batch=preset.batch,
            workers=preset.workers,
            device=preset.device,
            pretrained=preset.pretrained,
            optimizer=preset.optimizer,
            lr0=preset.lr0,
            lrf=preset.lrf,
            momentum=preset.momentum,
            weight_decay=preset.weight_decay,
            amp=preset.amp,
            val=preset.val,
            save=preset.save,
            plots=preset.plots,
            data=str(yaml),
            project=str(output.parent),
            name=output.name,
            exist_ok=True,
        )
        write_json(
            output / "checkpoints.json",
            {
                name: sha256(output / "weights" / f"{name}.pt")
                for name in ("best", "last")
            },
        )


if __name__ == "__main__":
    from retraining_experiment.cli import arguments

    args = arguments("train")
    train(args.dataset, args.output)
