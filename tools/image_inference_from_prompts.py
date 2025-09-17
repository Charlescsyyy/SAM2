#!/usr/bin/env python3
"""Run SAM2 image inference from pre-generated prompts and report metrics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image

import sam2
from hydra.utils import instantiate
from omegaconf import OmegaConf
from sam2.build_sam import build_sam2, _load_checkpoint
from sam2.sam2_image_predictor import SAM2ImagePredictor
from training.utils.train_utils import register_omegaconf_resolvers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True, help="Path to prompts.json")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Root directory containing images/ and masks/ as referenced in prompts.json",
    )
    parser.add_argument("--config", type=str, required=True, help="SAM2 config YAML path")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint (.pt)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to save predicted masks (mirrors dataset structure).",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference")
    parser.add_argument(
        "--multimask-output",
        action="store_true",
        help="Return multi-mask predictions and pick the highest IoU per object.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Log additional information for the first few samples.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict[str, dict]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_numpy_coords(coords: List[List[float]]) -> np.ndarray:
    if not coords:
        return np.empty((0, 2), dtype=np.float32)
    return np.asarray(coords, dtype=np.float32)


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray, obj_id: int) -> float:
    pred = pred_mask == obj_id
    gt = gt_mask == obj_id
    inter = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(inter) / float(union)


def compute_dice(pred_mask: np.ndarray, gt_mask: np.ndarray, obj_id: int) -> float:
    pred = pred_mask == obj_id
    gt = gt_mask == obj_id
    inter = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0 if inter == 0 else 0.0
    return 2.0 * float(inter) / float(denom)


def save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8)).save(path)


def select_mask(masks: np.ndarray, ious: np.ndarray) -> np.ndarray:
    if masks.ndim == 3:
        idx = int(np.argmax(ious)) if ious.size > 0 else 0
        return masks[idx]
    return masks.squeeze(axis=0)


def normalize_config_path(config: str) -> str:
    cfg_path = Path(config)
    if not cfg_path.is_absolute():
        return config
    candidates = [Path(__file__).resolve().parents[1], Path(sam2.__path__[0])]
    for base in candidates:
        try:
            rel = cfg_path.relative_to(base)
        except ValueError:
            continue
        else:
            rel_parts = rel.parts
            if len(rel_parts) >= 2 and rel_parts[0] == "sam2" and rel_parts[1] == "configs":
                return "/".join(rel_parts[1:])
            return str(rel.as_posix())
    return str(cfg_path)
    return config


def run_inference(args: argparse.Namespace) -> None:
    try:
        register_omegaconf_resolvers()
    except Exception:
        pass
    data = load_json(args.prompts)
    cfg_path = Path(args.config)
    if cfg_path.is_absolute() and cfg_path.exists():
        cfg = OmegaConf.load(cfg_path)
        OmegaConf.resolve(cfg)
        if "model" in cfg:
            model = instantiate(cfg.model, _recursive_=True)
        elif "lightning" in cfg and "model" in cfg.lightning:
            model = instantiate(cfg.lightning.model, _recursive_=True)
        elif "trainer" in cfg and "model" in cfg.trainer:
            model = instantiate(cfg.trainer.model, _recursive_=True)
        else:
            raise ValueError("Configuration file must contain a 'model' definition.")
        _load_checkpoint(model, args.ckpt)
        model = model.to(args.device)
        model.eval()
    else:
        config_name = normalize_config_path(args.config)
        model = build_sam2(config_name, args.ckpt, device=args.device)
    predictor = SAM2ImagePredictor(model)
    predictor.model.to(args.device)

    results: Dict[str, List[float]] = {"iou": [], "dice": []}
    processed = 0

    for key, entry in data.items():
        image_path = args.dataset_root / entry["image_path"]
        mask_path = args.dataset_root / entry["mask_path"]
        if not image_path.exists() or not mask_path.exists():
            print(f"[WARN] Missing image or mask for {key}, skipping.")
            continue

        image = np.array(Image.open(image_path).convert("RGB"))
        gt_mask = np.array(Image.open(mask_path)).astype(np.uint8)

        predictor.set_image(image)

        pred_mask = np.zeros_like(gt_mask, dtype=np.uint8)

        for obj_idx, obj in enumerate(
            sorted(entry.get("objects", []), key=lambda o: int(o["obj_id"]))
        ):
            obj_id = int(obj["obj_id"])
            point_coords = to_numpy_coords(obj.get("point_coords", []))
            point_labels = np.array(obj.get("point_labels", []), dtype=np.int32)
            bbox = np.array(obj.get("bbox", []), dtype=np.float32) if obj.get("bbox") else None

            masks, scores, _ = predictor.predict(
                point_coords=point_coords if point_coords.size else None,
                point_labels=point_labels if point_labels.size else None,
                box=bbox if bbox is not None and bbox.size == 4 else None,
                multimask_output=args.multimask_output,
            )

            mask_sel = select_mask(masks, scores)
            pred_mask[mask_sel > 0] = obj_id
            if args.debug and processed == 0:
                print(
                    f"  obj {obj_id}: raw_scores={scores.tolist()}, mask_sum={int(mask_sel.sum())}"
                )

        if args.output_dir is not None:
            out_path = args.output_dir / entry["mask_path"]
            save_mask(out_path, pred_mask)

        unique_ids = [oid for oid in np.unique(gt_mask) if oid > 0]
        if not unique_ids:
            continue

        sample_ious: List[float] = []
        sample_dices: List[float] = []
        for oid in unique_ids:
            iou = compute_iou(pred_mask, gt_mask, oid)
            dice = compute_dice(pred_mask, gt_mask, oid)
            results["iou"].append(iou)
            results["dice"].append(dice)
            sample_ious.append(iou)
            sample_dices.append(dice)

        print(
            f"{key}: mean IoU={np.mean(sample_ious):.4f}, mean Dice={np.mean(sample_dices):.4f}"
        )
        processed += 1

    if not results["iou"]:
        print("No valid samples processed.")
        return

    mean_iou = float(np.mean(results["iou"]))
    mean_dice = float(np.mean(results["dice"]))
    print("\n=== Aggregate Metrics ===")
    print(f"Samples evaluated: {processed}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Dice: {mean_dice:.4f}")


def main() -> None:
    args = parse_args()
    run_inference(args)


if __name__ == "__main__":
    main()
