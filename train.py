#!/usr/bin/env python3
"""
Standalone training script for MEDR-Net.

Trains (or resumes training of) the MEDR-Net image restoration model on
GT/NoisyLR .npy pairs, tracking a composite pixel + SSIM + edge + perceptual
loss, with early stopping and LR decay on plateau.

Usage:
    python train.py --dataset-dir train/Preprocessed_Augmented --epochs 10
    python train.py --dataset-dir train/Preprocessed_Augmented --resume
"""
import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras import Model, layers
from tqdm import tqdm

LR_SIZE = 128
HR_SIZE = 256
CHANNELS = 1

LOG_COLUMNS = ["epoch", "train_loss", "val_loss", "train_psnr", "val_psnr",
               "train_ssim", "val_ssim", "epoch_time_sec"]


# ============================================================
# MODEL ARCHITECTURE
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


def build_dataset(pairs, batch_size, seed, shuffle=False, cache_path=None):
    lr_paths = [p[0] for p in pairs]
    gt_paths = [p[1] for p in pairs]
    ds = tf.data.Dataset.from_tensor_slices((lr_paths, gt_paths))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(pairs), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_tf_load_pair, num_parallel_calls=tf.data.AUTOTUNE)
    if cache_path is not None:
        ds = ds.cache(cache_path)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def build_pairs(dataset_dir: Path, seed: int):
    gt_dir = dataset_dir / "GT"
    lr_dir = dataset_dir / "NoisyLR"
    gt_files = sorted(gt_dir.glob("*.npy"))
    pairs = [(str(lr_dir / p.name), str(p)) for p in gt_files if (lr_dir / p.name).exists()]
    if not pairs:
        raise FileNotFoundError(f"No matching GT/NoisyLR pairs found under {dataset_dir}")
    train_pairs, temp_pairs = train_test_split(pairs, test_size=0.20, random_state=seed)
    val_pairs, test_pairs = train_test_split(temp_pairs, test_size=0.50, random_state=seed)
    return train_pairs, val_pairs, test_pairs


# ============================================================
# LOSSES / METRICS
# ============================================================
def make_loss_and_metrics(loss_weights):
    vgg = tf.keras.applications.VGG16(include_top=False, weights="imagenet",
                                       input_shape=(HR_SIZE, HR_SIZE, 3))
    vgg.trainable = False
    perceptual_extractor = Model(inputs=vgg.input, outputs=vgg.get_layer("block3_conv3").output,
                                  name="vgg_perceptual_extractor")

    def pixel_loss(y_true, y_pred):
        return tf.reduce_mean(tf.abs(y_true - y_pred))

    def ssim_loss(y_true, y_pred):
        return 1.0 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

    def edge_loss(y_true, y_pred):
        return tf.reduce_mean(tf.abs(sobel_edge_map(y_true) - sobel_edge_map(y_pred)))

    def perceptual_loss(y_true, y_pred):
        y_true_rgb = tf.image.grayscale_to_rgb(y_true) if y_true.shape[-1] == 1 else y_true
        y_pred_rgb = tf.image.grayscale_to_rgb(y_pred) if y_pred.shape[-1] == 1 else y_pred
        y_true_rgb = tf.keras.applications.vgg16.preprocess_input(y_true_rgb * 255.0)
        y_pred_rgb = tf.keras.applications.vgg16.preprocess_input(y_pred_rgb * 255.0)
        feat_true = tf.nn.l2_normalize(perceptual_extractor(y_true_rgb), axis=-1)
        feat_pred = tf.nn.l2_normalize(perceptual_extractor(y_pred_rgb), axis=-1)
        return tf.reduce_mean(tf.square(feat_true - feat_pred))

    def composite_loss(y_true, y_pred):
        l_pixel = pixel_loss(y_true, y_pred)
        l_ssim = ssim_loss(y_true, y_pred)
        l_edge = edge_loss(y_true, y_pred)
        l_perc = perceptual_loss(y_true, y_pred)
        total = (loss_weights["pixel"] * l_pixel + loss_weights["ssim"] * l_ssim
                 + loss_weights["edge"] * l_edge + loss_weights["perceptual"] * l_perc)
        return total, {"pixel": l_pixel, "ssim": l_ssim, "edge": l_edge, "perceptual": l_perc}

    def psnr_metric(y_true, y_pred):
        return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1.0))

    def ssim_metric(y_true, y_pred):
        return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0))

    return composite_loss, psnr_metric, ssim_metric


# ============================================================
# TRAINING LOOP
# ============================================================
def train(args):
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "medr_net_best.h5"
    log_csv_path = output_dir / "training_log.csv"

    loss_weights = {"pixel": 1.0, "ssim": 0.6, "edge": 0.3, "perceptual": 0.1}

    train_pairs, val_pairs, test_pairs = build_pairs(Path(args.dataset_dir), args.seed)
    train_dataset = build_dataset(train_pairs, args.batch_size, args.seed, shuffle=True,
                                   cache_path=str(output_dir / "train_cache"))
    val_dataset = build_dataset(val_pairs, args.batch_size, args.seed, shuffle=False,
                                 cache_path=str(output_dir / "val_cache"))

    medr_net = build_medr_net()
    if args.resume and checkpoint_path.exists():
        medr_net.load_weights(checkpoint_path)
        print(f"Resumed weights from {checkpoint_path}")
    medr_net.summary()

    composite_loss, psnr_metric, ssim_metric = make_loss_and_metrics(loss_weights)
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, clipnorm=1.0)

    trackers = {
        name: tf.keras.metrics.Mean(name=name)
        for name in [
            "train_loss", "val_loss", "train_psnr", "val_psnr", "train_ssim", "val_ssim",
            "train_pixel", "train_ssim_term", "train_edge", "train_perceptual",
            "val_pixel", "val_ssim_term", "val_edge", "val_perceptual",
        ]
    }

    @tf.function
    def train_step(lr_batch, gt_batch):
        with tf.GradientTape() as tape:
            pred_batch = medr_net(lr_batch, training=True)
            loss, terms = composite_loss(gt_batch, pred_batch)
        gradients = tape.gradient(loss, medr_net.trainable_variables)
        optimizer.apply_gradients(zip(gradients, medr_net.trainable_variables))

        trackers["train_loss"].update_state(loss)
        trackers["train_psnr"].update_state(psnr_metric(gt_batch, pred_batch))
        trackers["train_ssim"].update_state(ssim_metric(gt_batch, pred_batch))
        trackers["train_pixel"].update_state(terms["pixel"])
        trackers["train_ssim_term"].update_state(terms["ssim"])
        trackers["train_edge"].update_state(terms["edge"])
        trackers["train_perceptual"].update_state(terms["perceptual"])
        return loss

    @tf.function
    def val_step(lr_batch, gt_batch):
        pred_batch = medr_net(lr_batch, training=False)
        loss, terms = composite_loss(gt_batch, pred_batch)

        trackers["val_loss"].update_state(loss)
        trackers["val_psnr"].update_state(psnr_metric(gt_batch, pred_batch))
        trackers["val_ssim"].update_state(ssim_metric(gt_batch, pred_batch))
        trackers["val_pixel"].update_state(terms["pixel"])
        trackers["val_ssim_term"].update_state(terms["ssim"])
        trackers["val_edge"].update_state(terms["edge"])
        trackers["val_perceptual"].update_state(terms["perceptual"])
        return loss

    # Resume/append to the existing CSV log instead of assuming a fresh run.
    if log_csv_path.exists():
        log_df = pd.read_csv(log_csv_path)
    else:
        log_df = pd.DataFrame(columns=LOG_COLUMNS)

    history = {k: list(log_df[k]) for k in
               ["train_loss", "val_loss", "train_psnr", "val_psnr", "train_ssim", "val_ssim"]} \
        if not log_df.empty else {k: [] for k in
               ["train_loss", "val_loss", "train_psnr", "val_psnr", "train_ssim", "val_ssim"]}

    best_val_loss = float(log_df["val_loss"].min()) if not log_df.empty else np.inf
    patience_counter = 0
    lr_patience_counter = 0
    log_rows = []

    train_steps_per_epoch = max(1, (len(train_pairs) + args.batch_size - 1) // args.batch_size)
    val_steps_per_epoch = max(1, (len(val_pairs) + args.batch_size - 1) // args.batch_size)

    total_start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        for tracker in trackers.values():
            tracker.reset_state()

        total_steps = train_steps_per_epoch + val_steps_per_epoch
        epoch_bar = tqdm(total=total_steps, desc=f"Epoch {epoch}/{args.epochs}", unit="batch", leave=False)

        for step, (lr_batch, gt_batch) in enumerate(train_dataset, start=1):
            train_step(lr_batch, gt_batch)
            epoch_bar.update(1)
            if step % 10 == 0 or step == train_steps_per_epoch:
                epoch_bar.set_postfix(phase="train", loss=f"{trackers['train_loss'].result().numpy():.4f}")

        for step, (lr_batch, gt_batch) in enumerate(val_dataset, start=1):
            val_step(lr_batch, gt_batch)
            epoch_bar.update(1)
            if step % 10 == 0 or step == val_steps_per_epoch:
                epoch_bar.set_postfix(phase="val", loss=f"{trackers['val_loss'].result().numpy():.4f}")

        epoch_bar.close()

        train_loss = float(trackers["train_loss"].result().numpy())
        val_loss = float(trackers["val_loss"].result().numpy())
        train_psnr = float(trackers["train_psnr"].result().numpy())
        val_psnr = float(trackers["val_psnr"].result().numpy())
        train_ssim = float(trackers["train_ssim"].result().numpy())
        val_ssim = float(trackers["val_ssim"].result().numpy())
        epoch_time = time.perf_counter() - epoch_start_time

        for key, val in zip(
            ["train_loss", "val_loss", "train_psnr", "val_psnr", "train_ssim", "val_ssim"],
            [train_loss, val_loss, train_psnr, val_psnr, train_ssim, val_ssim],
        ):
            history[key].append(val)

        log_rows.append([epoch, train_loss, val_loss, train_psnr, val_psnr, train_ssim, val_ssim, epoch_time])

        saved_flag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            lr_patience_counter = 0
            medr_net.save(checkpoint_path)
            saved_flag = " | saved best"
        else:
            patience_counter += 1
            lr_patience_counter += 1
            if lr_patience_counter >= args.lr_patience:
                current_lr = float(optimizer.learning_rate.numpy())
                new_lr = max(current_lr * args.lr_decay_factor, args.min_lr)
                if new_lr < current_lr:
                    optimizer.learning_rate.assign(new_lr)
                    saved_flag += f" | lr reduced {current_lr:.2e} -> {new_lr:.2e}"
                lr_patience_counter = 0

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} | "
            f"train_psnr={train_psnr:.2f} val_psnr={val_psnr:.2f} | "
            f"train_ssim={train_ssim:.4f} val_ssim={val_ssim:.4f} | "
            f"time={epoch_time:.1f}s{saved_flag}"
        )

        if patience_counter >= args.patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    total_training_time = time.perf_counter() - total_start_time
    print(f"Total training time: {total_training_time / 60:.2f} minutes "
          f"({total_training_time / 3600:.2f} hours)")

    new_log_df = pd.DataFrame(log_rows, columns=LOG_COLUMNS)
    log_df = pd.concat([log_df, new_log_df], ignore_index=True)
    log_df["epoch"] = range(1, len(log_df) + 1)
    log_df.to_csv(log_csv_path, index=False)
    print(f"Saved training log to {log_csv_path}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].plot(history["train_loss"], label="Train Loss")
    axes[0].plot(history["val_loss"], label="Val Loss")
    axes[0].set_title("Composite Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history["train_psnr"], label="Train PSNR")
    axes[1].plot(history["val_psnr"], label="Val PSNR")
    axes[1].set_title("PSNR (dB)")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(history["train_ssim"], label="Train SSIM")
    axes[2].plot(history["val_ssim"], label="Val SSIM")
    axes[2].set_title("SSIM")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150)
    plt.close(fig)
    print(f"Saved training curves to {output_dir / 'training_curves.png'}")
    print(f"Best checkpoint: {checkpoint_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train MEDR-Net.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("train/Preprocessed_Augmented"),
                         help="Directory containing GT/ and NoisyLR/ .npy pairs.")
    parser.add_argument("--output-dir", type=Path, default=Path("train"),
                         help="Directory to write checkpoints, logs, and plots.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience (epochs).")
    parser.add_argument("--lr-patience", type=int, default=3, help="Epochs without improvement before LR decay.")
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--resume", action="store_true",
                         help="Resume from the existing checkpoint in --output-dir if present.")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
