#!/usr/bin/env python3
"""
Standalone evaluation script for MEDR-Net.

Loads a trained checkpoint, computes PSNR / SSIM / VGG-perceptual-proxy
metrics on the held-out test split, and saves a qualitative comparison grid.

Usage:
    python evaluate.py --dataset-dir train --checkpoint train/medr_net_best.h5
"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
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
# DATA PIPELINE
# ============================================================
def _load_npy_pair(lr_path, gt_path):
    lr_img = np.load(lr_path.numpy().decode("utf-8")).astype(np.float32)
    gt_img = np.load(gt_path.numpy().decode("utf-8")).astype(np.float32)

    def _normalize(img):
        min_val, max_val = img.min(), img.max()
        if max_val > min_val:
            return (img - min_val) / (max_val - min_val)
        return np.zeros_like(img)

    lr_img, gt_img = _normalize(lr_img), _normalize(gt_img)
    if lr_img.ndim == 2:
        lr_img = lr_img[..., np.newaxis]
    if gt_img.ndim == 2:
        gt_img = gt_img[..., np.newaxis]
    return lr_img, gt_img


def _tf_load_pair(lr_path, gt_path):
    lr_img, gt_img = tf.py_function(func=_load_npy_pair, inp=[lr_path, gt_path],
                                     Tout=[tf.float32, tf.float32])
    lr_img.set_shape([LR_SIZE, LR_SIZE, CHANNELS])
    gt_img.set_shape([HR_SIZE, HR_SIZE, CHANNELS])
    return lr_img, gt_img


def build_dataset(pairs, batch_size, shuffle=False):
    lr_paths = [p[0] for p in pairs]
    gt_paths = [p[1] for p in pairs]
    ds = tf.data.Dataset.from_tensor_slices((lr_paths, gt_paths))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(pairs), reshuffle_each_iteration=True)
    ds = ds.map(_tf_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_test_pairs(dataset_dir: Path, seed: int, test_size: float):
    gt_dir = dataset_dir / "GT"
    lr_dir = dataset_dir / "NoisyLR"
    gt_files = sorted(gt_dir.glob("*.npy"))
    pairs = [(str(lr_dir / p.name), str(p)) for p in gt_files if (lr_dir / p.name).exists()]
    if not pairs:
        raise FileNotFoundError(f"No matching GT/NoisyLR pairs found under {dataset_dir}")
    _, test_pairs = train_test_split(pairs, test_size=test_size, random_state=seed)
    return test_pairs


# ============================================================
# METRICS
# ============================================================
def build_perceptual_extractor():
    vgg = tf.keras.applications.VGG16(include_top=False, weights="imagenet",
                                       input_shape=(HR_SIZE, HR_SIZE, 3))
    vgg.trainable = False
    return Model(inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output,
                 name="vgg_perceptual_extractor")


def make_metrics(perceptual_extractor):
    def perceptual_loss(y_true, y_pred):
        y_true_rgb = tf.image.grayscale_to_rgb(y_true) if y_true.shape[-1] == 1 else y_true
        y_pred_rgb = tf.image.grayscale_to_rgb(y_pred) if y_pred.shape[-1] == 1 else y_pred
        y_true_rgb = tf.keras.applications.vgg16.preprocess_input(y_true_rgb * 255.0)
        y_pred_rgb = tf.keras.applications.vgg16.preprocess_input(y_pred_rgb * 255.0)
        feat_true = tf.nn.l2_normalize(perceptual_extractor(y_true_rgb), axis=-1)
        feat_pred = tf.nn.l2_normalize(perceptual_extractor(y_pred_rgb), axis=-1)
        return tf.reduce_mean(tf.square(feat_true - feat_pred))

    def psnr_metric(y_true, y_pred):
        return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

    def ssim_metric(y_true, y_pred):
        return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

    # VGG-feature-space distance used as a perceptual-similarity proxy.
    # NOT the official LPIPS metric (linear-calibrated AlexNet/VGG trained
    # on human judgments) -- use the `lpips` PyPI package for publication
    # numbers. This proxy is for monitoring only.
    return psnr_metric, ssim_metric, perceptual_loss


def evaluate(args):
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    test_pairs = build_test_pairs(dataset_dir, args.seed, args.test_size)
    test_dataset = build_dataset(test_pairs, args.batch_size, shuffle=False)

    model = build_medr_net()
    model.load_weights(args.checkpoint)
    model.summary()

    perceptual_extractor = build_perceptual_extractor()
    psnr_metric, ssim_metric, lpips_style_metric = make_metrics(perceptual_extractor)

    psnr_tracker = tf.keras.metrics.Mean(name="test_psnr")
    ssim_tracker = tf.keras.metrics.Mean(name="test_ssim")
    lpips_tracker = tf.keras.metrics.Mean(name="test_lpips_style")

    sample_images, sample_gt, sample_pred = [], [], []

    for lr_batch, gt_batch in test_dataset:
        pred_batch = model(lr_batch, training=False)
        psnr_tracker.update_state(psnr_metric(gt_batch, pred_batch))
        ssim_tracker.update_state(ssim_metric(gt_batch, pred_batch))
        lpips_tracker.update_state(lpips_style_metric(gt_batch, pred_batch))

        if len(sample_images) < args.num_samples:
            for i in range(lr_batch.shape[0]):
                if len(sample_images) >= args.num_samples:
                    break
                sample_images.append(lr_batch[i].numpy())
                sample_gt.append(gt_batch[i].numpy())
                sample_pred.append(pred_batch[i].numpy())

    print("\n=== Test Set Results ===")
    print(f"PSNR: {psnr_tracker.result().numpy():.3f} dB")
    print(f"SSIM: {ssim_tracker.result().numpy():.4f}")
    print(f"LPIPS-style (VGG proxy): {lpips_tracker.result().numpy():.4f}")

    if sample_images and not args.no_preview:
        save_comparison_grid(sample_images, sample_gt, sample_pred, output_dir)


def save_comparison_grid(sample_images, sample_gt, sample_pred, output_dir: Path):
    n_samples = len(sample_images)
    fig, axes = plt.subplots(n_samples, 3, figsize=(14, 4 * n_samples))
    if n_samples == 1:
        axes = np.expand_dims(axes, axis=0)

    for i in range(n_samples):
        noisy, gt, recon = sample_images[i], sample_gt[i], sample_pred[i]
        cmap = "gray" if noisy.shape[-1] == 1 else None
        noisy_d = noisy.squeeze(-1) if noisy.shape[-1] == 1 else noisy
        gt_d = gt.squeeze(-1) if gt.shape[-1] == 1 else gt
        recon_d = recon.squeeze(-1) if recon.shape[-1] == 1 else recon

        psnr = tf.image.psnr(tf.convert_to_tensor(gt), tf.convert_to_tensor(recon), max_val=1.0).numpy()
        ssim = tf.image.ssim(tf.convert_to_tensor(gt), tf.convert_to_tensor(recon), max_val=1.0).numpy()

        axes[i, 0].imshow(np.clip(noisy_d, 0, 1), cmap=cmap)
        axes[i, 0].set_title(f"Input / Noisy\nSample {i + 1}", fontsize=11)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(np.clip(gt_d, 0, 1), cmap=cmap)
        axes[i, 1].set_title(f"Ground Truth\nSample {i + 1}", fontsize=11)
        axes[i, 1].axis("off")

        axes[i, 2].imshow(np.clip(recon_d, 0, 1), cmap=cmap)
        axes[i, 2].set_title(f"Reconstructed\nPSNR: {psnr:.3f} dB | SSIM: {ssim:.4f}", fontsize=11)
        axes[i, 2].axis("off")

    plt.tight_layout()
    grid_path = output_dir / "test_comparison_grid.png"
    plt.savefig(grid_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison grid to {grid_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate MEDR-Net on the held-out test split.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("train"),
                         help="Directory containing GT/ and NoisyLR/ .npy pairs.")
    parser.add_argument("--checkpoint", type=Path, default=Path("train/medr_net_best.h5"))
    parser.add_argument("--output-dir", type=Path, default=None,
                         help="Where to save the comparison grid (default: checkpoint's directory).")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20,
                         help="Fraction of pairs held out as the test split (same seed reproduces the split).")
    parser.add_argument("--num-samples", type=int, default=12, help="Number of samples in the comparison grid.")
    parser.add_argument("--no-preview", action="store_true", help="Skip saving the comparison grid.")
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
