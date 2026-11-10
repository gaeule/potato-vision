"""Extract training-only frames and write explicitly unreviewed X-AnyLabeling drafts."""

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import av
from PIL import Image

from postprocess_review.common import (
    BASE,
    DATASET,
    NAMES,
    OUTPUT,
    ROOT,
    SOURCE,
    WEIGHTS,
    PipelineError,
    Postprocess,
    infer,
    load_model,
    sha256,
    write_json,
)
from stone_experiment.evaluate import detections


def main() -> None:
    """Create proposals, never verified ground truth or a training dataset."""
    plan = json.loads((BASE / "protocol.json").read_text(encoding="utf-8"))
    selection = json.loads((OUTPUT / "nms/selection.json").read_text(encoding="utf-8"))
    requests = plan["review_requests"]
    times = [float(row["second"]) for row in requests]
    old_manifest = json.loads((ROOT / "baseline/internal/manifest.json").read_text())
    old_pts = {row["pts"] for row in old_manifest["frames"]}
    if len(times) != 50 or times != sorted(set(times)) or not all(0 <= t <= 84 for t in times):
        raise PipelineError("Expected 50 distinct training-only requested times")
    if old_manifest["source_sha256"] != sha256(SOURCE):
        raise PipelineError("Original source changed")
    output = OUTPUT / "review_pack"
    data = output / "data"
    data.mkdir(parents=True, exist_ok=False)
    previews = output / "previews"
    previews.mkdir()
    model = load_model()
    parameters = Postprocess(float(plan["draft_confidence"]), float(selection["selected_nms_iou"]))
    records = []
    seen_pts: set[int] = set()
    old_pixels: set[str] = set()
    for path in DATASET.glob("images/*/*.png"):
        with Image.open(path) as image:
            old_pixels.add(hashlib.sha256(image.convert("RGB").tobytes()).hexdigest())
    pixel_hashes: set[str] = set()
    index = 0
    with av.open(str(SOURCE)) as video:
        for frame_index, frame in enumerate(video.decode(video=0)):
            if frame.pts is None or frame.time_base is None:
                raise PipelineError("Source frame lacks a timestamp")
            second = float(frame.pts * frame.time_base)
            if second < times[index]:
                continue
            if second > 84 or frame.pts in old_pts or frame.pts in seen_pts:
                raise PipelineError("Review frame overlaps old samples or lies outside training time")
            seen_pts.add(frame.pts)
            image = frame.to_image().convert("RGB")
            digest = hashlib.sha256(image.tobytes()).hexdigest()
            duplicate = digest in old_pixels or digest in pixel_hashes
            pixel_hashes.add(digest)
            filename = f"train_t{round(times[index]*1000):09d}ms.png"
            image.save(data / filename)
            prediction = infer(model, image, parameters)
            proposals = detections(prediction, potato_only=False)
            shapes = []
            for proposal in proposals:
                x1, y1, x2, y2 = proposal.xyxy
                if not (0 <= x1 < x2 <= image.width and 0 <= y1 < y2 <= image.height):
                    raise PipelineError(f"Proposal outside image: {filename}")
                shapes.append({
                    "label": NAMES[proposal.class_id], "score": proposal.confidence,
                    "points": [[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                    "group_id": None, "description": "MODEL PROPOSAL - UNREVIEWED",
                    "difficult": False, "shape_type": "rectangle", "flags": {},
                    "attributes": {}, "kie_linking": [],
                })
            write_json(data / Path(filename).with_suffix(".json"), {
                "version": "4.2.0", "flags": {"model_proposal": True},
                "checked": False, "shapes": shapes, "imagePath": filename,
                "imageData": None, "imageHeight": image.height, "imageWidth": image.width,
                "description": "Unreviewed model proposals; manually add, remove, and correct every object.",
            })
            Image.fromarray(prediction.plot()[..., ::-1]).save(previews / Path(filename).with_suffix(".jpg"))
            records.append({
                "filename": filename, "requested_second": times[index], "actual_second": second,
                "frame_index": frame_index, "pts": frame.pts, "time_base": str(frame.time_base),
                "reason": requests[index]["reason"], "pixel_sha256": digest,
                "exact_pixel_duplicate": duplicate,
                "potato_proposals": sum(p.class_id == 0 for p in proposals),
                "stone_proposals": sum(p.class_id == 1 for p in proposals),
                "review_status": "pending", "reviewer": "", "notes": "",
            })
            index += 1
            if index == len(times):
                break
    if index != len(times):
        raise PipelineError("Source ended before all review frames were extracted")
    write_json(output / "manifest.json", {
        "source_video": SOURCE.name, "source_sha256": sha256(SOURCE),
        "checkpoint_sha256": sha256(WEIGHTS), "protocol_sha256": sha256(BASE / "protocol.json"),
        "proposal_confidence": parameters.confidence, "proposal_nms_iou": parameters.nms_iou,
        "ground_truth": False, "human_review_required": True, "training_inclusion": False,
        "frame_count": len(records), "frames": records,
    })
    with (output / "review.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    _ = shutil.copyfile(BASE / "REVIEW_GUIDE.md", output / "README.md")
    archive = OUTPUT / "potato_stone_review_pack.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=3) as zipped:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                zipped.write(path, Path("review_pack") / path.relative_to(output))
    print("REVIEW_PACK_COMPLETE", len(records), archive, flush=True)


if __name__ == "__main__":
    main()
