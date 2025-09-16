import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Convert a simple images/labels dataset to SAM2 input format (DAVIS-like or SA-V).\n"
            "Source layout expected: <src>/images/*.jpg and <src>/labels/*.png with shared stems.\n"
            "Frames are grouped by prefix before the last underscore (e.g., test01_1 -> video=test01)."
        )
    )
    p.add_argument("--src", required=True, help="Source root with images/ and labels/")
    p.add_argument("--dst", required=True, help="Destination root to create images/ and masks/")
    p.add_argument(
        "--format",
        choices=["davis", "sav"],
        default="davis",
        help="Output mask format: 'davis' = packed per-frame PNGs; 'sav' = per-object subfolders with binary masks.",
    )
    p.add_argument(
        "--class_ids",
        type=str,
        default=None,
        help="Comma-separated class IDs to treat as foreground (e.g., '1,2,3'). If omitted, uses non-zero as foreground.",
    )
    p.add_argument(
        "--grouping",
        choices=["prefix", "per_image"],
        default="prefix",
        help=(
            "Grouping strategy for building videos: "
            "'prefix' groups by prefix before the last underscore (default); "
            "'per_image' creates one single-frame video per image (video name = full stem)."
        ),
    )
    p.add_argument(
        "--video_name_override",
        type=str,
        default=None,
        help="Override grouping to a single video name (e.g., 'test01'). If set, all frames go into this video.",
    )
    p.add_argument(
        "--copy_images",
        action="store_true",
        help="Copy image files instead of symlinking. Default is symlink when possible.",
    )
    return p.parse_args()


def stem_to_group_and_frame(stem: str) -> Tuple[str, int]:
    """
    Split filename stem by the last underscore to get (video, frame_idx).
    E.g., 'test01_4' -> ('test01', 4). If no underscore or non-integer suffix, fallback to lexicographic ordering with frame=0.
    """
    if "_" in stem:
        prefix, suffix = stem.rsplit("_", 1)
        try:
            return prefix, int(suffix)
        except ValueError:
            return prefix, 0
    return stem, 0


def collect_pairs(src_images: Path, src_labels: Path, grouping: str) -> Dict[str, List[Tuple[Path, Path, int]]]:
    """
    Return mapping: video_name -> list of (img_path, label_path, original_frame_idx).
    Only include items where both image and label exist.
    """
    pairs_by_video: Dict[str, List[Tuple[Path, Path, int]]] = {}
    image_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
    for img in sorted(src_images.iterdir()):
        if not img.is_file() or img.suffix.lower() not in image_exts:
            continue
        stem = img.stem
        label = src_labels / f"{stem}.png"
        if not label.exists():
            print(f"[WARN] Missing label for {img.name}, skipping.")
            continue
        if grouping == "per_image":
            video, frame = stem, 1
        else:
            video, frame = stem_to_group_and_frame(stem)
        pairs_by_video.setdefault(video, []).append((img, label, frame))

    # sort frames and ensure stable order
    for video in pairs_by_video:
        pairs_by_video[video].sort(key=lambda t: (t[2], str(t[0])))
    return pairs_by_video


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_mask_from_label(label_path: Path, out_path: Path, class_ids: List[int] | None, binary: bool):
    lab = np.array(Image.open(label_path))
    if class_ids is None:
        fg = (lab != 0)
    else:
        mask = np.zeros_like(lab, dtype=bool)
        for cid in class_ids:
            mask |= (lab == cid)
        fg = mask

    if binary:
        out = fg.astype(np.uint8)  # 0 or 1
    else:
        # DAVIS-like packed with single object id 1
        out = (fg.astype(np.uint8))
        out[out > 0] = 1

    Image.fromarray(out).save(out_path)


def link_or_copy(src: Path, dst: Path, copy: bool):
    if dst.exists():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)


def main():
    args = parse_args()
    src = Path(args.src)
    dst = Path(args.dst)
    src_images = src / "images"
    src_labels = src / "labels"
    assert src_images.is_dir() and src_labels.is_dir(), "Source must contain images/ and labels/"

    class_ids = None
    if args.class_ids:
        class_ids = [int(x) for x in args.class_ids.split(",") if x.strip()]

    pairs_by_video = collect_pairs(src_images, src_labels, args.grouping)
    if args.video_name_override and pairs_by_video:
        # merge all into one video
        merged = []
        for video, items in pairs_by_video.items():
            merged.extend(items)
        pairs_by_video = {args.video_name_override: sorted(merged, key=lambda t: (t[2], str(t[0])))}

    out_images_root = dst / "images"
    out_masks_root = dst / "masks"
    ensure_dir(out_images_root)
    ensure_dir(out_masks_root)

    total_frames = 0
    for video, items in pairs_by_video.items():
        # reindex frames 1..N in sorted order
        vid_img_dir = out_images_root / video
        ensure_dir(vid_img_dir)
        if args.format == "sav":
            vid_mask_dir = out_masks_root / video / "obj_001"
        else:
            vid_mask_dir = out_masks_root / video
        ensure_dir(vid_mask_dir)

        for new_idx, (img_p, lab_p, _orig_idx) in enumerate(items, start=1):
            # write image
            img_ext = img_p.suffix.lower()
            out_img = vid_img_dir / f"{new_idx}{img_ext}"
            link_or_copy(img_p, out_img, copy=args.copy_images)

            # write mask
            out_mask = vid_mask_dir / f"{new_idx}.png"
            save_mask_from_label(lab_p, out_mask, class_ids, binary=(args.format == "sav"))
            total_frames += 1

        print(f"[OK] Video '{video}': {len(items)} frames -> {vid_img_dir} and {vid_mask_dir}")

    print(f"Done. Wrote {total_frames} frames across {len(pairs_by_video)} video(s) to {dst}")
    print("You can now run vos_inference.py using these images/ and masks/ roots.")


if __name__ == "__main__":
    main()
