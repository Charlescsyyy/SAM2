#!/usr/bin/env python3
"""Generate SAM-style prompts from ground-truth masks.

This utility scans a dataset that follows the DAVIS-style layout produced by
``tools/convert_testdata_to_sam2.py`` (i.e. ``images/<video>/<frame>.jpg`` and
``masks/<video>/<frame>.png``). For every foreground object (unique non-zero
value in the mask), it creates a simple prompt consisting of a bounding box and
positive / negative point prompts. The resulting prompts are saved in a JSON
file that can be consumed by custom inference scripts.

Example:
    python tools/generate_prompts_from_masks.py \
        --dataset-root /home/jyu197/DIWork/sam2/Testdata_sam2_perimg \
        --output prompts.json \
        --num-positive 3 --num-negative 2
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - handled via user message
    raise SystemExit(
        "Pillow is required to run this script. Install it via 'pip install pillow'."
    ) from exc


@dataclass
class PromptSample:
    """Container that describes prompts for a single object instance."""

    obj_id: int
    area: int
    bbox: List[int]
    bbox_normalized: List[float]
    point_coords: List[List[int]]
    point_coords_normalized: List[List[float]]
    point_labels: List[int]
    num_positive: int
    num_negative: int

    def to_dict(self) -> Dict[str, object]:
        return {
            "obj_id": self.obj_id,
            "area": self.area,
            "bbox": self.bbox,
            "bbox_normalized": self.bbox_normalized,
            "point_coords": self.point_coords,
            "point_coords_normalized": self.point_coords_normalized,
            "point_labels": self.point_labels,
            "num_positive": self.num_positive,
            "num_negative": self.num_negative,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Path that contains images/ and masks/ directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Where to write the generated prompts JSON file.",
    )
    parser.add_argument(
        "--num-positive",
        type=int,
        default=1,
        help="Foreground point prompts per object (>=1).",
    )
    parser.add_argument(
        "--num-negative",
        type=int,
        default=1,
        help="Background point prompts per object (>=0).",
    )
    parser.add_argument(
        "--negative-margin",
        type=int,
        default=20,
        help="Restrict negative samples to a band around the bbox (pixels).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output JSON file if it already exists.",
    )
    return parser.parse_args()


def load_mask(mask_path: Path) -> np.ndarray:
    mask = Image.open(mask_path)
    if mask.mode != "L":
        mask = mask.convert("L")
    return np.array(mask)


def iter_mask_files(mask_root: Path) -> Iterable[Path]:
    for video_dir in sorted(mask_root.iterdir()):
        if not video_dir.is_dir():
            continue
        for mask_path in sorted(video_dir.glob("*.png")):
            yield mask_path


def get_unique_object_ids(mask: np.ndarray) -> List[int]:
    unique = np.unique(mask)
    return [int(v) for v in unique if v > 0]


def compute_bbox(fg_mask: np.ndarray) -> List[int]:
    ys, xs = np.nonzero(fg_mask)
    y0, y1 = int(ys.min()), int(ys.max())
    x0, x1 = int(xs.min()), int(xs.max())
    return [x0, y0, x1, y1]


def normalize_point(x: float, y: float, w: int, h: int) -> List[float]:
    denom_x = max(w - 1, 1)
    denom_y = max(h - 1, 1)
    return [x / denom_x, y / denom_y]


def normalize_bbox(bbox: List[int], width: int, height: int) -> List[float]:
    x0, y0, x1, y1 = bbox
    return [
        float(x0) / max(width - 1, 1),
        float(y0) / max(height - 1, 1),
        float(x1) / max(width - 1, 1),
        float(y1) / max(height - 1, 1),
    ]


def _seed_from_array(arr: np.ndarray) -> int:
    digest = hashlib.sha1(arr.view(np.uint8)).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False)


def sample_positive_points(
    fg_mask: np.ndarray, num_points: int, rng: np.random.Generator
) -> np.ndarray:
    coords = np.column_stack(np.nonzero(fg_mask))  # (N, 2) -> (y, x)
    if coords.shape[0] == 0 or num_points <= 0:
        return np.empty((0, 2), dtype=np.int32)

    # Start from the mask centroid to get a stable click.
    centroid = np.round(coords.mean(axis=0)).astype(int)
    centroid = np.clip(centroid, [0, 0], np.array(fg_mask.shape) - 1)
    if not fg_mask[tuple(centroid)]:
        # Fallback: pick the middle coordinate if centroid fell outside after rounding.
        centroid = coords[len(coords) // 2]

    chosen = [centroid]
    if coords.shape[0] > 1 and num_points > 1:
        # Avoid repeating the centroid when sampling.
        mask_idxs = np.arange(coords.shape[0])
        centroid_idx = np.argmin(np.sum((coords - centroid) ** 2, axis=1))
        mask_idxs = np.delete(mask_idxs, centroid_idx)
        if mask_idxs.size > 0:
            extra = min(num_points - 1, mask_idxs.size)
            picked = rng.choice(mask_idxs, size=extra, replace=False)
            for idx in picked:
                chosen.append(coords[idx])

    while len(chosen) < num_points:
        chosen.append(centroid)

    return np.stack(chosen[:num_points], axis=0)


def sample_negative_points(
    fg_mask: np.ndarray,
    num_points: int,
    bbox: List[int],
    margin: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if num_points <= 0:
        return np.empty((0, 2), dtype=np.int32)

    background = ~fg_mask
    if not np.any(background):
        return np.empty((0, 2), dtype=np.int32)

    h, w = fg_mask.shape
    x0, y0, x1, y1 = bbox
    x0b = max(0, x0 - margin)
    y0b = max(0, y0 - margin)
    x1b = min(w - 1, x1 + margin)
    y1b = min(h - 1, y1 + margin)

    # Candidate negatives: ring around the bbox intersection with background
    candidate = np.zeros_like(background)
    candidate[y0b : y1b + 1, x0b : x1b + 1] = True
    candidate[y0 : y1 + 1, x0 : x1 + 1] = False
    candidate &= background

    if not np.any(candidate):
        candidate = background

    coords = np.column_stack(np.nonzero(candidate))
    if coords.shape[0] <= num_points:
        return coords.astype(np.int32)

    idxs = rng.choice(coords.shape[0], size=num_points, replace=False)
    return coords[idxs].astype(np.int32)


def mask_to_prompt_samples(
    mask: np.ndarray,
    num_positive: int,
    num_negative: int,
    negative_margin: int,
) -> List[PromptSample]:
    height, width = mask.shape
    prompt_samples: List[PromptSample] = []

    for obj_id in get_unique_object_ids(mask):
        fg_mask = mask == obj_id
        if not np.any(fg_mask):
            continue

        bbox = compute_bbox(fg_mask)
        seed = _seed_from_array(fg_mask)
        rng = np.random.default_rng(seed)

        pos_pts = sample_positive_points(fg_mask, num_positive, rng)
        neg_pts = sample_negative_points(fg_mask, num_negative, bbox, negative_margin, rng)

        # Convert from (y, x) to (x, y) order expected by SAM.
        pos_xy = pos_pts[:, ::-1] if pos_pts.size else np.empty((0, 2), dtype=np.int32)
        neg_xy = neg_pts[:, ::-1] if neg_pts.size else np.empty((0, 2), dtype=np.int32)

        if pos_xy.size or neg_xy.size:
            all_xy = np.concatenate([pos_xy, neg_xy], axis=0)
        else:
            all_xy = np.empty((0, 2), dtype=np.int32)
        labels = [1] * len(pos_xy) + [0] * len(neg_xy)

        prompt_samples.append(
            PromptSample(
                obj_id=obj_id,
                area=int(np.sum(fg_mask)),
                bbox=bbox,
                bbox_normalized=normalize_bbox(bbox, width, height),
                point_coords=all_xy.astype(int).tolist(),
                point_coords_normalized=[
                    normalize_point(float(x), float(y), width, height) for x, y in all_xy
                ],
                point_labels=labels,
                num_positive=len(pos_xy),
                num_negative=len(neg_xy),
            )
        )

    return prompt_samples

def find_corresponding_image(image_root: Path, video: str, frame_stem: str) -> Optional[Path]:
    img_dir = image_root / video
    if not img_dir.is_dir():
        return None
    candidates = [img_dir / f"{frame_stem}{ext}" for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff")]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    # Fallback: pick the first file with matching stem ignoring case.
    for img_path in img_dir.glob(f"{frame_stem}.*"):
        if img_path.is_file():
            return img_path
    return None


def main() -> None:
    args = parse_args()
    dataset_root: Path = args.dataset_root.resolve()
    mask_root = dataset_root / "masks"
    image_root = dataset_root / "images"

    if not mask_root.is_dir() or not image_root.is_dir():
        raise SystemExit(
            f"Expected both 'images/' and 'masks/' under {dataset_root}, got {image_root.exists()} and {mask_root.exists()}."
        )

    output_path: Path = args.output
    if output_path.exists() and not args.overwrite:
        raise SystemExit(
            f"Output file {output_path} already exists. Pass --overwrite if you want to replace it."
        )

    data: Dict[str, Dict[str, object]] = {}

    for mask_path in iter_mask_files(mask_root):
        mask = load_mask(mask_path)
        height, width = mask.shape
        video = mask_path.parent.name
        frame_stem = mask_path.stem
        key = f"{video}/{frame_stem}"

        samples = mask_to_prompt_samples(
            mask,
            num_positive=max(args.num_positive, 1),
            num_negative=max(args.num_negative, 0),
            negative_margin=max(args.negative_margin, 0),
        )
        if not samples:
            continue

        image_path = find_corresponding_image(image_root, video, frame_stem)
        rel_image = str(image_path.relative_to(dataset_root)) if image_path else None
        rel_mask = str(mask_path.relative_to(dataset_root))

        data[key] = {
            "image_path": rel_image,
            "mask_path": rel_mask,
            "height": height,
            "width": width,
            "objects": [sample.to_dict() for sample in samples],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Wrote {len(data)} prompt entries to {output_path}")


if __name__ == "__main__":
    main()
