#!/usr/bin/env python3
"""
Standalone inference script for MEDR-Net.

Loads a trained checkpoint and restores every .npy / image file in an input
directory, saving results as both .npy and .png, plus an optional
side-by-side comparison preview.

Usage:
    python run.py --input-dir NoisyLR --checkpoint weights/medr_net_best.h5
    python run.py --input-dir NoisyLR --checkpoint weights/medr_net_best.h5 --no-preview
"""
import argparse
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Model, layers

LR_SIZE = 128
HR_SIZE = 256
CHANNELS = 1


# ============================================================
# MODEL ARCHITECTURE (identical to train.py, kept standalone here)
# ============================================================
def degradation_aware_extraction(x, filters=64, name="dafe"):
    feat = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv1")(x)
    feat = layers.LeakyReLU(0.2, name=f"{name}_act1")(feat)

    gate = layers.GlobalAveragePooling2D(name=f"{name}_gap")(feat)
    gate = layers.Dense(filters // 4, activation="relu", name=f"{name}_fc1")(gate)
    gate = layers.Dense(filters, activation="sigmoid", name=f"{name}_fc2")(gate)
    gate = layers.Reshape((1, 1, filters), name=f"{name}_reshape")(gate)

    feat = layers.Multiply(name=f"{name}_gate_mul")([feat, gate])
    feat = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv2")(feat)
    feat = layers.LeakyReLU(0.2, name=f"{name}_act2")(feat)
    return feat


def multi_scale_residual_block(x, filters=64, block_id=0):
    name = f"msrl_{block_id}"
    branch1 = layers.Conv2D(filters, 3, padding="same", dilation_rate=1, name=f"{name}_b1")(x)
    branch2 = layers.Conv2D(filters, 3, padding="same", dilation_rate=2, name=f"{name}_b2")(x)
    branch3 = layers.Conv2D(filters, 3, padding="same", dilation_rate=4, name=f"{name}_b3")(x)

    fused = layers.Concatenate(name=f"{name}_concat")([branch1, branch2, branch3])
    fused = layers.Conv2D(filters, 1, padding="same", name=f"{name}_fuse")(fused)
    fused = layers.LeakyReLU(0.2, name=f"{name}_act")(fused)

    return layers.Add(name=f"{name}_residual")([x, fused])


def multi_scale_residual_learning(x, filters=64, num_blocks=6):
    for i in range(num_blocks):
        x = multi_scale_residual_block(x, filters=filters, block_id=i)
    return x


def sobel_edge_map(x):
    edges = tf.image.sobel_edges(x)
    return tf.sqrt(tf.reduce_sum(tf.square(edges), axis=-1) + 1e-8)


def edge_aware_enhancement(x, filters=64, name="ease"):
    edge_map = layers.Lambda(sobel_edge_map, name=f"{name}_sobel")(x)
    edge_feat = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv1")(edge_map)
    edge_feat = layers.LeakyReLU(0.2, name=f"{name}_act1")(edge_feat)

    attention = layers.Conv2D(filters, 1, padding="same", activation="sigmoid", name=f"{name}_attn")(edge_feat)
    enhanced = layers.Multiply(name=f"{name}_apply")([x, attention])
    enhanced = layers.Add(name=f"{name}_residual")([x, enhanced])
    return enhanced, edge_feat


def learned_super_resolution(x, filters=64, scale=2, name="lsrr"):
    x = layers.Conv2D(filters * (scale ** 2), 3, padding="same", name=f"{name}_pre_conv")(x)
    x = layers.Lambda(lambda t: tf.nn.depth_to_space(t, scale), name=f"{name}_pixel_shuffle")(x)
    x = layers.Conv2D(filters, 3, padding="same", name=f"{name}_post_conv")(x)
    x = layers.LeakyReLU(0.2, name=f"{name}_act")(x)
    return x


def adaptive_feature_fusion(denoised, structural, high_freq, filters=64, name="aff"):
    concat = layers.Concatenate(name=f"{name}_concat")([denoised, structural, high_freq])
    weights = layers.Conv2D(3, 1, padding="same", activation="softmax", name=f"{name}_weights")(concat)

    w0 = layers.Lambda(lambda t: t[..., 0:1], name=f"{name}_w0")(weights)
    w1 = layers.Lambda(lambda t: t[..., 1:2], name=f"{name}_w1")(weights)
    w2 = layers.Lambda(lambda t: t[..., 2:3], name=f"{name}_w2")(weights)

    fused = layers.Add(name=f"{name}_fuse")([
        layers.Multiply()([denoised, w0]),
        layers.Multiply()([structural, w1]),
        layers.Multiply()([high_freq, w2]),
    ])
    fused = layers.Conv2D(filters, 3, padding="same", name=f"{name}_conv")(fused)
    fused = layers.LeakyReLU(0.2, name=f"{name}_act")(fused)
    return fused


def build_medr_net(lr_size=LR_SIZE, channels=CHANNELS, filters=64):
    inputs = layers.Input(shape=(lr_size, lr_size, channels), name="lr_input")

    dafe_feat = degradation_aware_extraction(inputs, filters=filters)
    denoised_feat = multi_scale_residual_learning(dafe_feat, filters=filters, num_blocks=6)
    structural_feat, edge_feat = edge_aware_enhancement(denoised_feat, filters=filters)

    upsampled_denoised = learned_super_resolution(denoised_feat, filters=filters, scale=2, name="lsrr_denoised")
    upsampled_structural = learned_super_resolution(structural_feat, filters=filters, scale=2, name="lsrr_structural")
    upsampled_edge = learned_super_resolution(edge_feat, filters=filters, scale=2, name="lsrr_edge")

    fused = adaptive_feature_fusion(upsampled_denoised, upsampled_structural, upsampled_edge, filters=filters)

    fused = layers.Conv2D(filters, 3, padding="same", name="refine_conv1")(fused)
    fused = layers.LeakyReLU(0.2, name="refine_act1")(fused)
    fused = layers.Conv2D(filters // 2, 3, padding="same", name="refine_conv2")(fused)
    fused = layers.LeakyReLU(0.2, name="refine_act2")(fused)

    output = layers.Conv2D(channels, 3, padding="same", activation="sigmoid", name="restored_output")(fused)
    return Model(inputs=inputs, outputs=output, name="MEDR_Net")


# ============================================================
# I/O HELPERS
# ============================================================
def load_input_array(path: Path, channels=CHANNELS):
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    pil_img = Image.open(path)
    pil_img = pil_img.convert("L") if channels == 1 else pil_img.convert("RGB")
    return np.array(pil_img).astype(np.float32)


def normalize(img):
    img = img.astype(np.float32)
    min_val, max_val = img.min(), img.max()
    if max_val > min_val:
        img = (img - min_val) / (max_val - min_val)
    else:
        img = np.zeros_like(img)
    return np.clip(img, 0, 1)


def prepare_model_input(img, lr_size=LR_SIZE):
    img = normalize(img)
    if img.ndim == 2:
        img = img[..., np.newaxis]
    if img.shape[0] != lr_size or img.shape[1] != lr_size:
        img_tensor = tf.convert_to_tensor(img[np.newaxis, ...])
        img_tensor = tf.image.resize(img_tensor, [lr_size, lr_size], method="bicubic")
        img = img_tensor.numpy()[0]
    return img[np.newaxis, ...]  # add batch dim


def run_inference(model, input_files, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    for idx, input_path in enumerate(input_files, start=1):
        try:
            raw_input = load_input_array(input_path)
            model_input = prepare_model_input(raw_input)

            restored = model.predict(model_input, verbose=0)[0]
            restored = np.clip(restored, 0, 1)

            stem = input_path.stem
            np.save(output_dir / f"{stem}_restored.npy", restored)

            restored_display = restored[..., 0] if restored.shape[-1] == 1 else restored
            restored_uint8 = (restored_display * 255.0).astype(np.uint8)
            Image.fromarray(restored_uint8).save(output_dir / f"{stem}_restored.png")

            print(f"[{idx}/{len(input_files)}] {stem} -> saved")
        except Exception as e:
            print(f"[{idx}/{len(input_files)}] {input_path.name} -> ERROR: {e}")


def save_preview(input_files, output_dir: Path, num_samples: int, seed: int):
    pairs = []
    for input_path in input_files:
        restored_path = output_dir / f"{input_path.stem}_restored.npy"
        if restored_path.exists():
            pairs.append((input_path, restored_path))

    if not pairs:
        print("No restored outputs found; skipping preview.")
        return

    rng = random.Random(seed)
    selected_pairs = rng.sample(pairs, min(num_samples, len(pairs)))

    n_samples = len(selected_pairs)
    fig, axes = plt.subplots(n_samples, 2, figsize=(10, 4 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, (input_path, restored_path) in enumerate(selected_pairs):
        raw_input = load_input_array(input_path)
        model_input = prepare_model_input(raw_input)
        input_display = model_input[0]
        input_display = input_display[..., 0] if input_display.shape[-1] == 1 else input_display

        restored = np.clip(np.load(restored_path), 0, 1)
        restored_display = restored[..., 0] if restored.shape[-1] == 1 else restored

        cmap = "gray" if CHANNELS == 1 else None
        axes[i, 0].imshow(input_display, cmap=cmap)
        axes[i, 0].set_title(
            f"Input / Degraded\n{input_display.shape[0]}x{input_display.shape[1]}\n{input_path.stem}",
            fontsize=10,
        )
        axes[i, 0].axis("off")

        axes[i, 1].imshow(restored_display, cmap=cmap)
        axes[i, 1].set_title(
            f"Restored (MEDR-Net)\n{restored_display.shape[0]}x{restored_display.shape[1]}",
            fontsize=10,
        )
        axes[i, 1].axis("off")

    fig.suptitle(f"MEDR-Net Restoration Results - {n_samples} Samples", fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    comparison_path = output_dir / f"{n_samples}_sample_restoration_comparison.png"
    plt.savefig(comparison_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison preview to {comparison_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run MEDR-Net inference on a directory of images.")
    parser.add_argument("--input-dir", type=Path, required=True,
                         help="Directory of .npy/.png/.jpg inputs to restore.")
    parser.add_argument("--checkpoint", type=Path, default=Path("train/medr_net_best.h5"))
    parser.add_argument("--output-dir", type=Path, default=Path("Restoration_Results"))
    parser.add_argument("--pattern", type=str, default="*.npy",
                         help="Glob pattern for selecting input files (default: *.npy).")
    parser.add_argument("--preview-samples", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-preview", action="store_true", help="Skip generating the comparison preview.")
    return parser.parse_args()


def main():
    args = parse_args()

    input_files = sorted(args.input_dir.glob(args.pattern))
    print(f"Found {len(input_files)} input images.")
    if not input_files:
        return

    print(f"Rebuilding MEDR-Net architecture and loading weights from:\n  {args.checkpoint}")
    model = build_medr_net()
    model.load_weights(args.checkpoint)
    print("Weights loaded successfully.")
    model.summary()

    run_inference(model, input_files, args.output_dir)

    if not args.no_preview:
        save_preview(input_files, args.output_dir, args.preview_samples, args.seed)


if __name__ == "__main__":
    main()
