#!/usr/bin/env python3
"""
Standalone preprocessing / augmentation script for MEDR-Net.

Loads GT / NoisyLR .npy pairs, normalizes them, generates flip/rotation
augmentations, and writes the results to an output directory.

Usage:
    python preprocess.py --dataset-dir train --augmentations 1
    python preprocess.py --dataset-dir train --no-preview
"""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe; no display needed to save figures
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

AUG_SUFFIXES = {
    0: "original",
    1: "hflip",
    2: "vflip",
    3: "rot90",
    4: "rot180",
    5: "rot270",
}


def load_npy(path):
    image = np.load(path)
    image = np.squeeze(image)
    if image.ndim == 3 and image.shape[0] in (1, 3):
        image = np.transpose(image, (1, 2, 0))
    return image


def normalize_image(image):
    image = image.astype(np.float32)
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(image, 0.0, 1.0)


def augment_pair(gt, lr, mode):
    if mode == 0:
        return gt, lr
    if mode == 1:
        return np.fliplr(gt).copy(), np.fliplr(lr).copy()
    if mode == 2:
        return np.flipud(gt).copy(), np.flipud(lr).copy()
    if mode == 3:
        return np.rot90(gt, k=1).copy(), np.rot90(lr, k=1).copy()
    if mode == 4:
        return np.rot90(gt, k=2).copy(), np.rot90(lr, k=2).copy()
    if mode == 5:
        return np.rot90(gt, k=3).copy(), np.rot90(lr, k=3).copy()
    return gt.copy(), lr.copy()


def display_normalize(img):
    img = img.astype(np.float32)
    min_val, max_val = img.min(), img.max()
    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    return np.clip(img, 0, 1)


def run_preprocessing(dataset_dir: Path, augmentations: int):
    gt_dir = dataset_dir / "GT"
    lr_dir = dataset_dir / "NoisyLR"
    output_dir = dataset_dir / "Preprocessed_Augmented"
    gt_out = output_dir / "GT"
    lr_out = output_dir / "NoisyLR"
    gt_out.mkdir(parents=True, exist_ok=True)
    lr_out.mkdir(parents=True, exist_ok=True)

    gt_files = sorted(gt_dir.glob("*.npy"))
    if not gt_files:
        raise FileNotFoundError(f"No GT .npy files found in {gt_dir}")

    total_saved = 0
    for gt_path in tqdm(gt_files, desc="Processing"):
        image_name = gt_path.stem
        lr_path = lr_dir / f"{image_name}.npy"
        if not lr_path.exists():
            print(f"\nMissing LR image: {image_name}")
            continue
        try:
            gt = normalize_image(load_npy(gt_path))
            lr = normalize_image(load_npy(lr_path))
            for aug_id in range(augmentations + 1):
                gt_aug, lr_aug = augment_pair(gt, lr, aug_id)
                suffix = AUG_SUFFIXES.get(aug_id, f"aug{aug_id}")
                np.save(gt_out / f"{image_name}_{suffix}.npy", gt_aug)
                np.save(lr_out / f"{image_name}_{suffix}.npy", lr_aug)
                total_saved += 1
        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    print(f"Saved {total_saved} augmented pairs to {output_dir}")
    return gt_out, lr_out


def save_preview(gt_out: Path, lr_out: Path, output_dir: Path, num_samples: int = 10, seed: int = 42):
    augmented_gt_files = sorted(gt_out.glob("*.npy"))
    if not augmented_gt_files:
        print("No augmented files found; skipping preview.")
        return

    rng = random.Random(seed)
    num_samples = min(num_samples, len(augmented_gt_files))
    sample_gt_files = rng.sample(augmented_gt_files, num_samples)

    fig, axes = plt.subplots(num_samples, 2, figsize=(10, 4 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for row, gt_path in enumerate(sample_gt_files):
        image_name = gt_path.stem
        lr_path = lr_out / f"{image_name}.npy"
        if not lr_path.exists():
            continue

        gt_img = np.load(gt_path)
        lr_img = np.load(lr_path)
        gt_display = display_normalize(gt_img)
        lr_display = display_normalize(lr_img)

        axes[row, 0].imshow(gt_display, cmap="gray" if gt_display.ndim == 2 else None)
        axes[row, 0].set_title(f"GT\n{image_name}\nShape: {gt_img.shape}")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(lr_display, cmap="gray" if lr_display.ndim == 2 else None)
        axes[row, 1].set_title(f"NoisyLR\nShape: {lr_img.shape}")
        axes[row, 1].axis("off")

    fig.suptitle("Augmented GT-NoisyLR Image Pairs", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    preview_path = output_dir / "augmentation_preview.png"
    plt.savefig(preview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved preview to {preview_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess and augment GT/NoisyLR .npy pairs.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("train"),
                         help="Directory containing GT/ and NoisyLR/ subfolders.")
    parser.add_argument("--augmentations", type=int, default=1,
                         help="Number of extra augmentations per image (0-5, in addition to the original).")
    parser.add_argument("--preview-samples", type=int, default=10,
                         help="Number of sample pairs to include in the preview image.")
    parser.add_argument("--no-preview", action="store_true", help="Skip generating the preview image.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for preview sampling.")
    return parser.parse_args()


def main():
    args = parse_args()
    gt_out, lr_out = run_preprocessing(args.dataset_dir, args.augmentations)
    if not args.no_preview:
        save_preview(gt_out, lr_out, args.dataset_dir / "Preprocessed_Augmented",
                      num_samples=args.preview_samples, seed=args.seed)


if __name__ == "__main__":
    main()
