"""
train.py  –  Training script for the floorplan segmentation / depth CNN.

Usage::

    python train.py                             # defaults (multi-task if depth/ dir exists)
    python train.py --model seg --epochs 100    # segmentation-only, 100 epochs
    python train.py --resume Data/checkpoints/last_model.pth

Environment variables::

    FLOORPLAN_DATASET  – path to the dataset root (default: Data/floorplan_dataset)
    CHECKPOINT_DIR     – path for saving checkpoints  (default: Data/checkpoints)
"""

import os
import sys
import math
import time
import json
import random
import logging
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler, autocast

from FloorplanToBlenderLib.model import (
    MultiTaskFloorplanNet,
    FloorplanSegmentationNet,
    FloorplanSegmentationNetV2,
    DepthEstimationNet,
    MultiTaskLoss,
    UncertaintyMultiTaskLoss,
    SegmentationLoss,
    DepthGradientLoss,
    BerHuLoss,
    SegmentationMetrics,
    DepthMetrics,
    save_model,
    count_parameters,
    freeze_encoder,
    unfreeze_all,
    get_learning_rate_groups,
    NUM_CLASSES,
    FLOORPLAN_CLASSES,
)
from FloorplanToBlenderLib.dataset import (
    FloorplanDataset,
    BalancedSampler,
    build_dataloaders,
    compute_class_weights,
    stratified_split,
    verify_dataset_structure,
    floorplan_collate,
)

logger = logging.getLogger("train")


# ===================================================================== #
#  DEFAULT  HYPER-PARAMETERS                                             #
# ===================================================================== #

HPARAMS = {
    "input_size": (512, 512),
    "batch_size": 8,
    "learning_rate": 1e-3,
    "encoder_lr": 1e-4,
    "weight_decay": 1e-4,
    "epochs": 80,
    "warmup_epochs": 3,
    "scheduler_patience": 5,
    "scheduler_factor": 0.5,
    "early_stop_patience": 12,
    "train_split": 0.85,
    "num_workers": 2,
    "grad_clip_norm": 5.0,
    "use_mixed_precision": True,
    "balanced_sampling": False,
    "use_class_weights": True,
    "seg_loss_weight": 1.0,
    "dep_loss_weight": 0.5,
    "uncertainty_loss": False,
    "freeze_encoder_epochs": 0,
}


# ===================================================================== #
#  LEARNING-RATE  SCHEDULE  HELPERS                                      #
# ===================================================================== #

def warmup_cosine_schedule(optimizer, warmup_steps, total_steps):
    """Linear warmup → cosine decay schedule factory.

    Returns a ``torch.optim.lr_scheduler.LambdaLR``.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def build_scheduler(optimizer, hp, steps_per_epoch):
    """Build the LR scheduler according to *hp* settings.

    If warmup_epochs > 0 → cosine with warmup.
    Otherwise          → ReduceLROnPlateau.
    """
    if hp["warmup_epochs"] > 0:
        warmup_steps = hp["warmup_epochs"] * steps_per_epoch
        total_steps  = hp["epochs"] * steps_per_epoch
        return warmup_cosine_schedule(optimizer, warmup_steps, total_steps), "step"
    else:
        sched = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=hp["scheduler_patience"],
            factor=hp["scheduler_factor"],
        )
        return sched, "epoch"


# ===================================================================== #
#  METRIC  HELPERS                                                       #
# ===================================================================== #

def pixel_accuracy(pred, target):
    """Per-pixel accuracy (pred is logits ``(B, C, H, W)``)."""
    labels = pred.argmax(dim=1)
    valid = target != 255   # ignore label
    correct = ((labels == target) & valid).sum().item()
    return correct / max(valid.sum().item(), 1)


def mean_iou(pred, target, num_classes=NUM_CLASSES):
    """Batch-averaged mean IoU (classes present in target only)."""
    labels = pred.argmax(dim=1)
    ious = []
    for cls in range(num_classes):
        p = (labels == cls)
        t = (target == cls)
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union > 0:
            ious.append(inter / union)
    return float(np.mean(ious)) if ious else 0.0


def per_class_iou(pred, target, num_classes=NUM_CLASSES):
    """Return an ndarray of shape ``(num_classes,)`` with per-class IoU."""
    labels = pred.argmax(dim=1)
    result = np.zeros(num_classes, dtype=np.float64)
    for cls in range(num_classes):
        p = (labels == cls)
        t = (target == cls)
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        result[cls] = inter / max(union, 1)
    return result


# ===================================================================== #
#  LOGGING  /  CHECKPOINTING                                             #
# ===================================================================== #

class TrainingLogger:
    """Accumulates per-epoch statistics and writes them to a JSON log."""

    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history = []

    def log_epoch(self, epoch, train_metrics, val_metrics, lr):
        record = {
            "epoch": epoch,
            "lr": lr,
            "train": train_metrics,
            "val": val_metrics,
            "timestamp": datetime.now().isoformat(),
        }
        self.history.append(record)

        path = self.log_dir / "train_log.json"
        with open(path, "w") as fh:
            json.dump(self.history, fh, indent=2)

    def print_epoch(self, epoch, total_epochs, train_m, val_m, lr, elapsed):
        logger.info(
            "Epoch %3d/%d  lr=%.2e  "
            "train_loss=%.4f  train_acc=%.4f | "
            "val_loss=%.4f  val_acc=%.4f  val_mIoU=%.4f  [%.1fs]",
            epoch, total_epochs, lr,
            train_m["loss"], train_m["acc"],
            val_m["loss"], val_m["acc"], val_m.get("miou", 0.0),
            elapsed,
        )


class EarlyStopping:
    """Stop training when validation loss has not improved for *patience*
    consecutive epochs.
    """

    def __init__(self, patience=12, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0

    def step(self, val_loss):
        if val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


# ===================================================================== #
#  TRAINING  LOOP                                                        #
# ===================================================================== #

def train_one_epoch(model, loader, criterion, optimizer, device,
                    multitask=True, scaler=None, grad_clip=5.0,
                    scheduler=None, sched_mode="step"):
    """Single training epoch with optional mixed-precision and gradient
    clipping.

    Returns a dict with ``loss`` and ``acc``.
    """
    model.train()
    running_loss = 0.0
    running_acc  = 0.0
    num_samples  = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask"].to(device, non_blocking=True)
        bs = images.size(0)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:
            with autocast():
                if multitask:
                    depths = batch["depth"].to(device, non_blocking=True)
                    seg_out, dep_out = model(images)
                    loss, _, _ = criterion(seg_out, masks, dep_out, depths)
                else:
                    seg_out = model(images)
                    loss = criterion(seg_out, masks)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if multitask:
                depths = batch["depth"].to(device, non_blocking=True)
                seg_out, dep_out = model(images)
                loss, _, _ = criterion(seg_out, masks, dep_out, depths)
            else:
                seg_out = model(images)
                loss = criterion(seg_out, masks)

            loss.backward()
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        if scheduler is not None and sched_mode == "step":
            scheduler.step()

        running_loss += loss.item() * bs
        running_acc  += pixel_accuracy(seg_out.detach(), masks) * bs
        num_samples  += bs

    return {
        "loss": running_loss / max(num_samples, 1),
        "acc":  running_acc  / max(num_samples, 1),
    }


@torch.no_grad()
def validate(model, loader, criterion, device, multitask=True):
    """Validation pass – returns ``loss``, ``acc``, ``miou``."""
    model.eval()
    running_loss = 0.0
    running_acc  = 0.0
    running_iou  = 0.0
    num_samples  = 0

    seg_metrics = SegmentationMetrics(num_classes=NUM_CLASSES)

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask"].to(device, non_blocking=True)
        bs = images.size(0)

        if multitask:
            depths = batch["depth"].to(device, non_blocking=True)
            seg_out, dep_out = model(images)
            loss, _, _ = criterion(seg_out, masks, dep_out, depths)
        else:
            seg_out = model(images)
            loss = criterion(seg_out, masks)

        seg_metrics.update(seg_out, masks)

        running_loss += loss.item() * bs
        running_acc  += pixel_accuracy(seg_out, masks) * bs
        running_iou  += mean_iou(seg_out, masks) * bs
        num_samples  += bs

    n = max(num_samples, 1)
    return {
        "loss": running_loss / n,
        "acc":  running_acc  / n,
        "miou": running_iou  / n,
        "pixel_accuracy": seg_metrics.pixel_accuracy,
        "mean_iou_full": seg_metrics.mean_iou,
        "per_class_iou": {
            FLOORPLAN_CLASSES[i]: float(v)
            for i, v in enumerate(seg_metrics.per_class_iou)
        },
    }


@torch.no_grad()
def full_evaluation(model, loader, device, multitask=True):
    """Detailed evaluation using accumulating metric objects."""
    model.eval()
    seg_m = SegmentationMetrics(num_classes=NUM_CLASSES)
    dep_m = DepthMetrics()

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask"].to(device, non_blocking=True)

        if multitask:
            depths = batch["depth"].to(device, non_blocking=True)
            seg_out, dep_out = model(images)
            dep_m.update(dep_out, depths)
        else:
            seg_out = model(images)

        seg_m.update(seg_out, masks)

    report = {
        "segmentation": {
            "pixel_accuracy": seg_m.pixel_accuracy,
            "mean_accuracy":  seg_m.mean_accuracy,
            "mean_iou":       seg_m.mean_iou,
            "freq_weighted_iou": seg_m.frequency_weighted_iou,
            "per_class_iou": {
                FLOORPLAN_CLASSES[i]: float(v)
                for i, v in enumerate(seg_m.per_class_iou)
            },
        }
    }
    if multitask:
        report["depth"] = dep_m.compute()

    return report


# ===================================================================== #
#  MODEL  FACTORY                                                        #
# ===================================================================== #

def build_model(model_name, multitask, hp, device):
    """Instantiate the requested model variant."""
    if multitask:
        model = MultiTaskFloorplanNet(
            num_classes=NUM_CLASSES,
            dropout=0.15,
            use_attention=True,
        ).to(device)
    elif model_name == "seg_v2":
        model = FloorplanSegmentationNetV2(
            num_classes=NUM_CLASSES,
            dropout=0.15,
        ).to(device)
    elif model_name == "depth":
        model = DepthEstimationNet(dropout=0.1).to(device)
    else:
        model = FloorplanSegmentationNet(
            num_classes=NUM_CLASSES,
            dropout=0.1,
            use_attention=True,
        ).to(device)
    return model


def build_criterion(multitask, hp, class_weights=None):
    """Build the loss function according to hyperparameters."""
    if multitask:
        if hp["uncertainty_loss"]:
            return UncertaintyMultiTaskLoss(
                class_weights=class_weights.tolist() if class_weights is not None else None
            )
        return MultiTaskLoss(
            seg_weight=hp["seg_loss_weight"],
            dep_weight=hp["dep_loss_weight"],
            class_weights=class_weights.tolist() if class_weights is not None else None,
        )
    return SegmentationLoss(
        class_weights=class_weights.tolist() if class_weights is not None else None
    )


# ===================================================================== #
#  MAIN  ENTRY-POINT                                                     #
# ===================================================================== #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train floorplan segmentation / depth CNN"
    )
    parser.add_argument(
        "--dataset", default=None,
        help="Path to dataset root (default: env FLOORPLAN_DATASET or Data/floorplan_dataset)",
    )
    parser.add_argument(
        "--checkpoint-dir", default=None,
        help="Directory for checkpoints (default: env CHECKPOINT_DIR or Data/checkpoints)",
    )
    parser.add_argument(
        "--model", default="multitask",
        choices=["multitask", "seg", "seg_v2", "depth"],
        help="Model variant to train",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    parser.add_argument("--balanced", action="store_true", help="Use balanced sampler")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()
    hp = dict(HPARAMS)

    # Override hyper-parameters from CLI
    if args.epochs is not None:
        hp["epochs"] = args.epochs
    if args.batch_size is not None:
        hp["batch_size"] = args.batch_size
    if args.lr is not None:
        hp["learning_rate"] = args.lr
    if args.no_amp:
        hp["use_mixed_precision"] = False
    if args.balanced:
        hp["balanced_sampling"] = True

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    dataset_root = args.dataset or os.environ.get(
        "FLOORPLAN_DATASET", "Data/floorplan_dataset"
    )
    checkpoint_dir = args.checkpoint_dir or os.environ.get(
        "CHECKPOINT_DIR", "Data/checkpoints"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)

    # ── Validate dataset ──────────────────────────────────────────────
    ds_info = verify_dataset_structure(dataset_root)
    if not ds_info["valid"]:
        logger.error("Dataset validation failed: %s", ds_info["errors"])
        sys.exit(1)
    logger.info("Dataset stats: %s", ds_info["stats"])

    multitask = (args.model == "multitask") and ds_info["stats"].get("has_depth", False)

    # ── Class weights ─────────────────────────────────────────────────
    class_weights = None
    if hp["use_class_weights"]:
        try:
            class_weights = compute_class_weights(
                dataset_root, hp["input_size"], NUM_CLASSES, max_samples=100
            )
            logger.info("Class weights: %s", class_weights.tolist())
        except Exception as exc:
            logger.warning("Could not compute class weights: %s", exc)

    # ── Build data loaders ────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        dataset_root,
        input_size=hp["input_size"],
        batch_size=hp["batch_size"],
        num_workers=hp["num_workers"],
        train_ratio=hp["train_split"],
        depth=multitask,
        balanced_sampling=hp["balanced_sampling"],
        seed=args.seed,
    )

    logger.info("Training batches  : %d", len(train_loader))
    logger.info("Validation batches: %d", len(val_loader))
    logger.info("Multi-task (depth): %s", multitask)

    # ── Build model, criterion, optimiser ─────────────────────────────
    model = build_model(args.model, multitask, hp, device)
    criterion = build_criterion(multitask, hp, class_weights)
    logger.info("Model: %s  (%s params)",
                type(model).__name__, f"{count_parameters(model):,}")

    # Differential LR for encoder vs decoder
    param_groups = get_learning_rate_groups(
        model, encoder_lr=hp["encoder_lr"], decoder_lr=hp["learning_rate"]
    )
    optimizer = optim.AdamW(
        param_groups, lr=hp["learning_rate"], weight_decay=hp["weight_decay"]
    )

    scheduler, sched_mode = build_scheduler(optimizer, hp, len(train_loader))

    # Mixed precision
    scaler = None
    if hp["use_mixed_precision"] and device.type == "cuda":
        scaler = GradScaler()
        logger.info("Mixed-precision training enabled (AMP)")

    # ── Resume from checkpoint ────────────────────────────────────────
    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("metrics", {}).get("val_loss", float("inf"))
        logger.info("Resumed from %s (epoch %d)", args.resume, start_epoch - 1)

    # ── Logger + early stopping ───────────────────────────────────────
    train_logger = TrainingLogger(checkpoint_dir)
    early_stop = EarlyStopping(patience=hp["early_stop_patience"])

    # ── Encoder freezing (transfer-learning warm-start) ───────────────
    freeze_epochs = hp["freeze_encoder_epochs"]
    if freeze_epochs > 0:
        freeze_encoder(model)
        logger.info("Encoder frozen for first %d epochs", freeze_epochs)

    # ── Training loop ─────────────────────────────────────────────────
    for epoch in range(start_epoch, hp["epochs"] + 1):
        t0 = time.time()

        # Unfreeze encoder after warm-up phase
        if freeze_epochs > 0 and epoch == start_epoch + freeze_epochs:
            unfreeze_all(model)
            logger.info("Encoder unfrozen at epoch %d", epoch)

        train_m = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            multitask=multitask, scaler=scaler,
            grad_clip=hp["grad_clip_norm"],
            scheduler=scheduler if sched_mode == "step" else None,
            sched_mode=sched_mode,
        )
        val_m = validate(model, val_loader, criterion, device, multitask)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # Step epoch-level scheduler
        if sched_mode == "epoch":
            scheduler.step(val_m["loss"])

        train_logger.log_epoch(epoch, train_m, val_m, current_lr)
        train_logger.print_epoch(epoch, hp["epochs"], train_m, val_m,
                                 current_lr, elapsed)

        # ── Checkpointing ─────────────────────────────────────────────
        metrics_record = {
            "val_loss": val_m["loss"],
            "val_acc": val_m["acc"],
            "val_miou": val_m["miou"],
        }

        # Save latest
        save_model(
            os.path.join(checkpoint_dir, "last_model.pth"),
            model, optimizer, epoch, metrics_record,
        )

        # Save best
        if val_m["loss"] < best_val_loss:
            best_val_loss = val_m["loss"]
            save_model(
                os.path.join(checkpoint_dir, "best_model.pth"),
                model, optimizer, epoch, metrics_record,
            )
            logger.info("  ★ New best model (val_loss=%.4f)", best_val_loss)

        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_model(
                os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pth"),
                model, optimizer, epoch, metrics_record,
            )

        # ── Early stopping ────────────────────────────────────────────
        if early_stop.step(val_m["loss"]):
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    # ── Final evaluation ──────────────────────────────────────────────
    logger.info("Running final evaluation on validation set …")
    final_report = full_evaluation(model, val_loader, device, multitask)

    report_path = os.path.join(checkpoint_dir, "final_eval.json")
    with open(report_path, "w") as fh:
        json.dump(final_report, fh, indent=2)
    logger.info("Final evaluation saved to %s", report_path)

    # Print per-class results
    seg_report = final_report["segmentation"]
    logger.info("Final Pixel Accuracy : %.4f", seg_report["pixel_accuracy"])
    logger.info("Final Mean IoU       : %.4f", seg_report["mean_iou"])
    for name, iou in seg_report["per_class_iou"].items():
        logger.info("  %-20s IoU = %.4f", name, iou)

    if "depth" in final_report:
        dep = final_report["depth"]
        logger.info("Depth RMSE: %.4f  MAE: %.4f  d<1.25: %.4f",
                     dep["rmse"], dep["mae"], dep["delta_1.25"])

    logger.info("Training complete.")


# ===================================================================== #
#  LEARNING  RATE  FINDER                                                #
# ===================================================================== #

def lr_range_test(model, loader, criterion, device, multitask=True,
                  start_lr=1e-7, end_lr=10.0, num_steps=100,
                  smooth_factor=0.05):
    """Run a learning-rate range test (Smith, 2017).

    Increases the learning rate exponentially from *start_lr* to *end_lr*
    over *num_steps* mini-batches and records the loss at each step.

    Returns
    -------
    lrs : list of float
    losses : list of float
    suggested_lr : float  – LR at steepest loss descent.
    """
    model.train()
    optimizer = optim.SGD(model.parameters(), lr=start_lr)
    factor = (end_lr / start_lr) ** (1.0 / max(num_steps, 1))

    best_loss = float("inf")
    lrs = []
    losses = []
    avg_loss = 0.0

    data_iter = iter(loader)
    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        images = batch["image"].to(device)
        masks  = batch["mask"].to(device)

        optimizer.zero_grad()
        if multitask:
            depths = batch["depth"].to(device)
            seg_out, dep_out = model(images)
            loss, _, _ = criterion(seg_out, masks, dep_out, depths)
        else:
            seg_out = model(images)
            loss = criterion(seg_out, masks)

        # Smooth the loss
        avg_loss = smooth_factor * loss.item() + (1 - smooth_factor) * avg_loss

        if step > 0 and avg_loss > 4 * best_loss:
            break  # loss is diverging
        if avg_loss < best_loss:
            best_loss = avg_loss

        loss.backward()
        optimizer.step()

        current_lr = optimizer.param_groups[0]["lr"]
        lrs.append(current_lr)
        losses.append(avg_loss)

        # Increase LR
        for pg in optimizer.param_groups:
            pg["lr"] *= factor

    # Find LR with steepest negative gradient
    if len(losses) > 10:
        gradients = np.gradient(losses)
        suggested_idx = int(np.argmin(gradients))
        suggested_lr = lrs[suggested_idx]
    else:
        suggested_lr = start_lr

    return lrs, losses, suggested_lr


# ===================================================================== #
#  EXPORT  UTILITIES                                                     #
# ===================================================================== #

def export_training_config(hp, checkpoint_dir, model_name, multitask):
    """Dump the full training configuration to a JSON file alongside
    the checkpoints for reproducibility.
    """
    config = {
        "hyperparameters": hp,
        "model": model_name,
        "multitask": multitask,
        "num_classes": NUM_CLASSES,
        "classes": FLOORPLAN_CLASSES,
        "timestamp": datetime.now().isoformat(),
    }
    path = os.path.join(checkpoint_dir, "train_config.json")
    with open(path, "w") as fh:
        json.dump(config, fh, indent=2)
    logger.info("Training config saved to %s", path)


def log_per_class_results(val_metrics):
    """Pretty-print per-class IoU from validation metrics."""
    per_class = val_metrics.get("per_class_iou", {})
    if not per_class:
        return
    logger.info("Per-class IoU:")
    for name, iou in per_class.items():
        bar = "█" * int(iou * 40)
        logger.info("  %-20s  %.4f  %s", name, iou, bar)


# ===================================================================== #
#  GRADIENT  STATISTICS                                                  #
# ===================================================================== #

def log_gradient_stats(model):
    """Log basic gradient statistics (mean, max) per parameter group.

    Useful for diagnosing vanishing / exploding gradients.
    """
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            mean_val = grad.abs().mean().item()
            max_val  = grad.abs().max().item()
            if max_val > 100.0:
                logger.warning("Large gradient in %s: mean=%.4f max=%.4f",
                               name, mean_val, max_val)


def compute_gradient_norm(model):
    """Return total L2 gradient norm across all parameters."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5


# ===================================================================== #
#  DATA  SANITY  CHECKS                                                  #
# ===================================================================== #

def check_dataloader(loader, multitask=True, num_batches=2):
    """Iterate a few batches and print shape / range diagnostics.

    Catches common issues like NaN images or mismatched mask shapes
    before spending time on a full training run.
    """
    logger.info("Running data sanity check …")
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        img = batch["image"]
        msk = batch["mask"]
        logger.info("  batch %d  image=%s  mask=%s  img_range=[%.3f, %.3f]",
                     i, list(img.shape), list(msk.shape),
                     img.min().item(), img.max().item())
        if torch.isnan(img).any():
            logger.error("  NaN detected in images!")
        if multitask and "depth" in batch:
            dep = batch["depth"]
            logger.info("            depth=%s  range=[%.3f, %.3f]",
                         list(dep.shape), dep.min().item(), dep.max().item())
    logger.info("Sanity check passed.")


if __name__ == "__main__":
    main()
