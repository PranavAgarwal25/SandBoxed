"""
dataset.py  –  PyTorch Dataset implementations for the floorplan segmentation
               and depth-estimation pipeline.

Provides:
    * ``FloorplanDataset``        – primary training / validation dataset
    * ``DetectionResultDataset``  – wraps cloud API JSON results
    * ``FloorplanTileDataset``    – large-image tiling for high-res inputs
    * ``BalancedSampler``         – class-balanced batch sampler
    * Augmentation helpers, data-splitting, integrity checks, and a colour
      palette shared by all visualisation utilities.
"""

import os
import math
import json
import hashlib
import logging
import random
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, Sampler, DataLoader, random_split

import torchvision.transforms as T
import torchvision.transforms.functional as TF

logger = logging.getLogger(__name__)

# ===================================================================== #
#  CONSTANTS                                                             #
# ===================================================================== #

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

FLOORPLAN_CLASSES = [
    "background",       # 0
    "space_balconi",     # 1
    "space_bedroom",     # 2
    "space_corridor",    # 3
    "space_dining",      # 4
    "space_kitchen",     # 5
    "space_laundry",     # 6
    "space_living",      # 7
    "space_lobby",       # 8
    "space_office",      # 9
    "space_other",       # 10
    "space_parking",     # 11
    "space_staircase",   # 12
    "space_toilet",      # 13
]
NUM_CLASSES = len(FLOORPLAN_CLASSES)

# Colour palette (RGB) used throughout all visualisation helpers.
CLASS_COLOURS = [
    (0,   0,   0),        # background
    (102, 178, 255),       # space_balconi
    (255, 178, 102),       # space_bedroom
    (204, 204,   0),       # space_corridor
    (0,   204, 102),       # space_dining
    (255, 102, 102),       # space_kitchen
    (153, 153, 255),       # space_laundry
    (255, 255, 102),       # space_living
    (178, 102, 255),       # space_lobby
    (102, 255, 178),       # space_office
    (200, 200, 200),       # space_other
    (128, 128, 128),       # space_parking
    (255, 153, 204),       # space_staircase
    (102, 255, 255),       # space_toilet
]


# ===================================================================== #
#  AUGMENTATION  HELPERS                                                 #
# ===================================================================== #

def random_horizontal_flip(image, mask, depth=None, prob=0.5):
    """Randomly flip image and all annotations horizontally."""
    if random.random() < prob:
        image = np.fliplr(image).copy()
        mask  = np.fliplr(mask).copy()
        if depth is not None:
            depth = np.fliplr(depth).copy()
    return image, mask, depth


def random_vertical_flip(image, mask, depth=None, prob=0.5):
    """Randomly flip image and all annotations vertically."""
    if random.random() < prob:
        image = np.flipud(image).copy()
        mask  = np.flipud(mask).copy()
        if depth is not None:
            depth = np.flipud(depth).copy()
    return image, mask, depth


def random_rotation_90(image, mask, depth=None):
    """Random 0 / 90 / 180 / 270 degree rotation."""
    k = random.randint(0, 3)
    if k == 0:
        return image, mask, depth
    image = np.rot90(image, k).copy()
    mask  = np.rot90(mask, k).copy()
    if depth is not None:
        depth = np.rot90(depth, k).copy()
    return image, mask, depth


def random_brightness_contrast(image, brightness_range=(-30, 30),
                                contrast_range=(0.8, 1.2)):
    """Adjust brightness and contrast with random factors."""
    alpha = random.uniform(*contrast_range)
    beta  = random.uniform(*brightness_range)
    image = image.astype(np.float32) * alpha + beta
    image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def random_hsv_shift(image, h_range=(-10, 10), s_range=(-25, 25),
                     v_range=(-25, 25)):
    """Random shift in HSV colour-space."""
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[:, :, 0] = np.clip(hsv[:, :, 0] + random.randint(*h_range), 0, 179)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] + random.randint(*s_range), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] + random.randint(*v_range), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


def random_gaussian_blur(image, ksize_range=(3, 7), prob=0.3):
    """Gaussian blur with random kernel size."""
    if random.random() < prob:
        k = random.choice(range(ksize_range[0], ksize_range[1] + 1, 2))
        image = cv2.GaussianBlur(image, (k, k), 0)
    return image


def random_gaussian_noise(image, sigma_range=(5, 25), prob=0.3):
    """Additive Gaussian noise."""
    if random.random() < prob:
        sigma = random.uniform(*sigma_range)
        noise = np.random.randn(*image.shape) * sigma
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image


def apply_clahe(image, clip_limit=2.0, grid_size=8):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalisation) to the
    luminance channel of the image."""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                             tileGridSize=(grid_size, grid_size))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def random_cutout(image, mask, depth=None, num_holes=2, hole_size_range=(20, 60)):
    """Randomly erase rectangular regions (Cutout / Random Erasing).

    Filled with the per-channel image mean to avoid distribution shift.
    Mask and depth pixels inside holes are set to ignore-label (255) and
    zero respectively so the loss ignores them.
    """
    h, w, _ = image.shape
    result_img  = image.copy()
    result_mask = mask.copy()
    result_dep  = depth.copy() if depth is not None else None

    mean_pixel = image.mean(axis=(0, 1)).astype(np.uint8)

    for _ in range(num_holes):
        hs = random.randint(*hole_size_range)
        y0 = random.randint(0, max(h - hs, 0))
        x0 = random.randint(0, max(w - hs, 0))
        y1 = min(y0 + hs, h)
        x1 = min(x0 + hs, w)

        result_img[y0:y1, x0:x1, :] = mean_pixel
        result_mask[y0:y1, x0:x1] = 255  # ignore label
        if result_dep is not None:
            result_dep[y0:y1, x0:x1] = 0.0

    return result_img, result_mask, result_dep


def random_elastic_deform(image, mask, depth=None, alpha=80, sigma=10,
                          prob=0.25):
    """Light elastic deformation for data augmentation.

    Generates a random displacement field, smooths it with a Gaussian
    kernel, and applies it to both the image and annotations.
    """
    if random.random() > prob:
        return image, mask, depth

    h, w = image.shape[:2]
    dx = cv2.GaussianBlur(
        (np.random.rand(h, w).astype(np.float32) * 2 - 1) * alpha,
        (0, 0), sigma,
    )
    dy = cv2.GaussianBlur(
        (np.random.rand(h, w).astype(np.float32) * 2 - 1) * alpha,
        (0, 0), sigma,
    )
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (grid_x + dx).astype(np.float32)
    map_y = (grid_y + dy).astype(np.float32)

    image = cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REFLECT)
    mask  = cv2.remap(mask, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                      borderMode=cv2.BORDER_REFLECT)
    if depth is not None:
        depth = cv2.remap(depth, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REFLECT)
    return image, mask, depth


def random_scale_crop(image, mask, depth=None, scale_range=(0.75, 1.25),
                      target_size=(512, 512)):
    """Random scale the image then random-crop back to *target_size*."""
    th, tw = target_size
    h, w = image.shape[:2]
    scale = random.uniform(*scale_range)
    new_h = int(h * scale)
    new_w = int(w * scale)

    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    mask  = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    if depth is not None:
        depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # Pad if necessary
    pad_h = max(th - new_h, 0)
    pad_w = max(tw - new_w, 0)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w,
                                   cv2.BORDER_REFLECT)
        mask  = cv2.copyMakeBorder(mask, 0, pad_h, 0, pad_w,
                                   cv2.BORDER_REFLECT)
        if depth is not None:
            depth = cv2.copyMakeBorder(depth, 0, pad_h, 0, pad_w,
                                       cv2.BORDER_REFLECT)

    crop_h, crop_w = image.shape[:2]
    y0 = random.randint(0, max(crop_h - th, 0))
    x0 = random.randint(0, max(crop_w - tw, 0))

    image = image[y0:y0+th, x0:x0+tw]
    mask  = mask[y0:y0+th, x0:x0+tw]
    if depth is not None:
        depth = depth[y0:y0+th, x0:x0+tw]

    return image, mask, depth


def compose_augmentations(image, mask, depth=None, target_size=(512, 512)):
    """Full augmentation pipeline applied during training.

    Sequentially applies geometric transforms, colour jitter, noise, and
    region erasing.  Each transform is applied stochastically so that
    not every sample receives every augmentation.
    """
    image, mask, depth = random_horizontal_flip(image, mask, depth)
    image, mask, depth = random_vertical_flip(image, mask, depth, prob=0.3)
    image, mask, depth = random_rotation_90(image, mask, depth)
    image, mask, depth = random_scale_crop(image, mask, depth,
                                           target_size=target_size)
    image, mask, depth = random_elastic_deform(image, mask, depth)

    # Colour augmentations (do not affect mask / depth)
    image = random_brightness_contrast(image)
    image = random_hsv_shift(image)
    image = random_gaussian_blur(image)
    image = random_gaussian_noise(image)

    # Random cutout
    if random.random() < 0.25:
        image, mask, depth = random_cutout(image, mask, depth)

    return image, mask, depth


# ===================================================================== #
#  DATASET  INTEGRITY  HELPERS                                           #
# ===================================================================== #

def file_md5(path, chunk_size=65536):
    """Compute MD5 checksum of a file for integrity verification."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def verify_dataset_structure(root):
    """Check that the expected directories and at least one image exist.

    Returns a dict with keys ``valid`` (bool), ``errors`` (list of str),
    and ``stats`` (dict of counts).
    """
    root = Path(root)
    errors = []
    stats = {}

    img_dir = root / "images"
    mask_dir = root / "masks"

    if not img_dir.is_dir():
        errors.append(f"Missing images directory: {img_dir}")
    if not mask_dir.is_dir():
        errors.append(f"Missing masks directory: {mask_dir}")

    if not errors:
        images = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])
        masks = sorted([
            f for f in os.listdir(mask_dir)
            if f.lower().endswith(".png")
        ])
        stats["num_images"] = len(images)
        stats["num_masks"]  = len(masks)

        img_stems = {os.path.splitext(f)[0] for f in images}
        mask_stems = {os.path.splitext(f)[0] for f in masks}

        missing_masks = img_stems - mask_stems
        orphan_masks  = mask_stems - img_stems

        if missing_masks:
            errors.append(f"{len(missing_masks)} images have no mask: "
                          f"{list(missing_masks)[:5]}...")
        if orphan_masks:
            errors.append(f"{len(orphan_masks)} masks have no image: "
                          f"{list(orphan_masks)[:5]}...")

        depth_dir = root / "depth"
        stats["has_depth"] = depth_dir.is_dir()
        if stats["has_depth"]:
            stats["num_depth"] = len(list(depth_dir.glob("*.png")))

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "stats": stats,
    }


def compute_class_distribution(root, input_size=(512, 512),
                                num_classes=NUM_CLASSES, max_samples=200):
    """Scan up to *max_samples* masks and return per-class pixel counts.

    Returns
    -------
    counts : np.ndarray of shape ``(num_classes,)``
    fractions : np.ndarray of shape ``(num_classes,)``
    """
    mask_dir = os.path.join(root, "masks")
    files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])
    files = files[:max_samples]

    counts = np.zeros(num_classes, dtype=np.int64)
    for name in files:
        mask = cv2.imread(os.path.join(mask_dir, name), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (input_size[1], input_size[0]),
                          interpolation=cv2.INTER_NEAREST)
        for cid in range(num_classes):
            counts[cid] += int((mask == cid).sum())

    total = counts.sum()
    fractions = counts / max(total, 1)
    return counts, fractions


def compute_class_weights(root, input_size=(512, 512), num_classes=NUM_CLASSES,
                          max_samples=200, method="inverse_freq"):
    """Compute per-class weights for use with cross-entropy loss.

    Parameters
    ----------
    method : str
        ``"inverse_freq"`` – 1 / frequency (clamped).
        ``"effective_samples"`` – Effective number of samples (Cui et al.)
    """
    counts, fractions = compute_class_distribution(
        root, input_size, num_classes, max_samples
    )

    if method == "inverse_freq":
        weights = np.where(fractions > 0, 1.0 / fractions, 0.0)
        weights = weights / weights.sum() * num_classes
    elif method == "effective_samples":
        beta = 0.9999
        effective = 1.0 - np.power(beta, counts.astype(np.float64))
        weights = np.where(effective > 0, (1.0 - beta) / effective, 0.0)
        weights = weights / weights.sum() * num_classes
    else:
        raise ValueError(f"Unknown weighting method: {method}")

    return weights.astype(np.float32)


def compute_dataset_statistics(root, input_size=(512, 512), max_samples=100):
    """Compute per-channel mean and std across training images.

    Returns ``(mean, std)`` each as a list of three floats (RGB).
    """
    img_dir = os.path.join(root, "images")
    files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])
    files = files[:max_samples]

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0

    for name in files:
        img = cv2.imread(os.path.join(img_dir, name), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_size[1], input_size[0]))
        img = img.astype(np.float64) / 255.0

        channel_sum += img.sum(axis=(0, 1))
        channel_sq_sum += (img ** 2).sum(axis=(0, 1))
        pixel_count += img.shape[0] * img.shape[1]

    mean = channel_sum / pixel_count
    std  = np.sqrt(channel_sq_sum / pixel_count - mean ** 2)
    return mean.tolist(), std.tolist()


# ===================================================================== #
#  TRAIN / VAL  SPLIT  UTILITIES                                         #
# ===================================================================== #

def stratified_split(root, train_ratio=0.85, seed=42):
    """Split image stems into train / val sets maintaining approximate
    class-distribution balance via round-robin assignment.

    Returns ``(train_stems, val_stems)`` each as a sorted list of str.
    """
    rng = random.Random(seed)
    mask_dir = os.path.join(root, "masks")
    stems = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(mask_dir) if f.endswith(".png")
    ])
    rng.shuffle(stems)

    n_train = int(len(stems) * train_ratio)
    return sorted(stems[:n_train]), sorted(stems[n_train:])


def k_fold_split(root, k=5, seed=42):
    """Generator yielding ``(train_stems, val_stems)`` for each fold."""
    rng = random.Random(seed)
    mask_dir = os.path.join(root, "masks")
    stems = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(mask_dir) if f.endswith(".png")
    ])
    rng.shuffle(stems)

    fold_size = math.ceil(len(stems) / k)
    for fold_idx in range(k):
        start = fold_idx * fold_size
        end = min(start + fold_size, len(stems))
        val_stems = stems[start:end]
        train_stems = stems[:start] + stems[end:]
        yield sorted(train_stems), sorted(val_stems)


# ===================================================================== #
#  PRIMARY  DATASET                                                      #
# ===================================================================== #

class FloorplanDataset(Dataset):
    """
    Dataset that loads RGB floorplan images together with their per-pixel
    segmentation masks and (optionally) depth maps.

    Expected directory layout::

        root/
            images/          # RGB .png or .jpg
            masks/           # Single-channel .png, pixel value = class id
            depth/           # Single-channel .png, 0-255 relative depth (opt)

    Parameters
    ----------
    root : str
        Path to the dataset root.
    input_size : tuple[int, int]
        (H, W) to resize every sample.
    augment : bool
        Whether to apply training-time augmentations.
    depth : bool
        Whether to load depth maps.
    stems : list of str or None
        If provided, restrict the dataset to these image names (without ext).
    """

    def __init__(self, root, input_size=(512, 512), augment=False,
                 depth=False, stems=None):
        self.root = root
        self.input_size = input_size
        self.augment = augment
        self.depth = depth

        img_dir = os.path.join(root, "images")
        all_files = sorted([
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ])

        if stems is not None:
            stem_set = set(stems)
            self.image_files = [
                f for f in all_files
                if os.path.splitext(f)[0] in stem_set
            ]
        else:
            self.image_files = all_files

        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
        logger.info("FloorplanDataset: %d images from %s (augment=%s, depth=%s)",
                     len(self.image_files), root, augment, depth)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        name = self.image_files[idx]
        stem = os.path.splitext(name)[0]

        # ---- Load RGB image ------------------------------------------------
        img_path = os.path.join(self.root, "images", name)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size[1], self.input_size[0]),
                           interpolation=cv2.INTER_LINEAR)

        # ---- Load segmentation mask ----------------------------------------
        mask_path = os.path.join(self.root, "masks", stem + ".png")
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Failed to read mask: {mask_path}")
        mask = cv2.resize(mask, (self.input_size[1], self.input_size[0]),
                          interpolation=cv2.INTER_NEAREST)

        # ---- Optional depth map --------------------------------------------
        depth_map = None
        if self.depth:
            dep_path = os.path.join(self.root, "depth", stem + ".png")
            if os.path.isfile(dep_path):
                depth_map = cv2.imread(dep_path, cv2.IMREAD_GRAYSCALE)
                depth_map = cv2.resize(
                    depth_map,
                    (self.input_size[1], self.input_size[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                depth_map = depth_map.astype(np.float32) / 255.0

        # ---- Augmentations -------------------------------------------------
        if self.augment:
            image, mask, depth_map = compose_augmentations(
                image, mask, depth_map, target_size=self.input_size
            )

        # ---- To tensor -----------------------------------------------------
        image_t = torch.from_numpy(
            image.astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        image_t = self.normalize(image_t)
        mask_t  = torch.from_numpy(mask.astype(np.int64))

        sample = {"image": image_t, "mask": mask_t, "name": stem}

        if self.depth:
            if depth_map is not None:
                dep_t = torch.from_numpy(depth_map).unsqueeze(0)
            else:
                dep_t = torch.zeros(1, self.input_size[0], self.input_size[1])
            sample["depth"] = dep_t

        return sample


# ===================================================================== #
#  DETECTION  JSON  DATASET                                              #
# ===================================================================== #

class DetectionResultDataset(Dataset):
    """
    Wraps cloud API JSON prediction files and corresponding images so
    they can be consumed by the local CNN for refinement or evaluation.

    Each JSON file is expected to conform to the standard detection output
    schema (``image``, ``predictions`` keys).

    Parameters
    ----------
    image_dir : str
        Folder containing resized images.
    json_dir : str
        Folder containing ``result_*.json`` files.
    input_size : tuple[int, int]
        Resize target.
    """

    def __init__(self, image_dir, json_dir, input_size=(512, 512)):
        self.image_dir = image_dir
        self.json_dir  = json_dir
        self.input_size = input_size
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        self.json_files = sorted([
            f for f in os.listdir(json_dir)
            if f.startswith("result_") and f.endswith(".json")
        ])

    def __len__(self):
        return len(self.json_files)

    def __getitem__(self, idx):
        json_name = self.json_files[idx]
        index = os.path.splitext(json_name)[0].split("_")[-1]

        with open(os.path.join(self.json_dir, json_name), "r") as fh:
            data = json.load(fh)

        predictions = data.get("predictions", [])

        img_path = os.path.join(self.image_dir, f"resized_{index}.jpg")
        if not os.path.isfile(img_path):
            img_path = os.path.join(self.image_dir, f"resized_{index}.png")

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found for result_{index}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.input_size[1], self.input_size[0]),
                           interpolation=cv2.INTER_LINEAR)

        # Build pseudo-mask from bounding-box predictions
        h, w = self.input_size
        pseudo_mask = np.zeros((h, w), dtype=np.uint8)
        img_h = data.get("image", {}).get("height", h)
        img_w = data.get("image", {}).get("width", w)
        scale_y = h / max(img_h, 1)
        scale_x = w / max(img_w, 1)

        for pred in predictions:
            cx = pred["x"] * scale_x
            cy = pred["y"] * scale_y
            bw = pred["width"] * scale_x
            bh = pred["height"] * scale_y

            x1 = int(max(cx - bw / 2, 0))
            y1 = int(max(cy - bh / 2, 0))
            x2 = int(min(cx + bw / 2, w))
            y2 = int(min(cy + bh / 2, h))

            cid = pred.get("class_id", 0)
            pseudo_mask[y1:y2, x1:x2] = cid

        image_t = torch.from_numpy(
            image.astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        image_t = self.normalize(image_t)
        mask_t  = torch.from_numpy(pseudo_mask.astype(np.int64))

        return {
            "image": image_t,
            "mask": mask_t,
            "name": f"detection_{index}",
            "predictions": predictions,
        }


# ===================================================================== #
#  TILE  DATASET  (LARGE-IMAGE  SUPPORT)                                 #
# ===================================================================== #

class FloorplanTileDataset(Dataset):
    """Dataset that yields overlapping tiles from large floorplan images.

    For inference on very high-resolution scans (> 2048 px), processing
    tiles independently and reassembling is more memory-efficient than
    resizing the entire image.

    Parameters
    ----------
    image_path : str
        Path to a single large image.
    tile_size : int
        Edge length of each square tile.
    overlap : int
        Pixel overlap between adjacent tiles (used for blending).
    """

    def __init__(self, image_path, tile_size=512, overlap=64):
        self.image_path = image_path
        self.tile_size  = tile_size
        self.overlap    = overlap
        self.normalize  = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Cannot read: {image_path}")
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.img_h, self.img_w = self.image.shape[:2]

        self.tiles = self._compute_tile_positions()

    def _compute_tile_positions(self):
        """Pre-compute ``(y, x)`` top-left corner for each tile."""
        step = self.tile_size - self.overlap
        positions = []
        y = 0
        while y < self.img_h:
            x = 0
            while x < self.img_w:
                positions.append((y, x))
                x += step
                if x + self.tile_size > self.img_w and x < self.img_w:
                    x = self.img_w - self.tile_size
            y += step
            if y + self.tile_size > self.img_h and y < self.img_h:
                y = self.img_h - self.tile_size
        # Deduplicate
        return list(dict.fromkeys(positions))

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        y, x = self.tiles[idx]
        ts = self.tile_size
        tile = self.image[y:y+ts, x:x+ts].copy()

        # Pad if the tile extends beyond the image boundary
        pad_h = ts - tile.shape[0]
        pad_w = ts - tile.shape[1]
        if pad_h > 0 or pad_w > 0:
            tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w,
                                      cv2.BORDER_REFLECT)

        tile_t = torch.from_numpy(
            tile.astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        tile_t = self.normalize(tile_t)

        return {"image": tile_t, "y": y, "x": x}

    def reassemble(self, tile_masks):
        """Stitch per-tile predictions back into a full-resolution mask.

        Parameters
        ----------
        tile_masks : list of np.ndarray
            Predicted masks (H_tile, W_tile) in the same order as the tiles.

        Returns
        -------
        full_mask : np.ndarray (H, W) of int
        """
        ts = self.tile_size
        overlap = self.overlap
        accum = np.zeros((self.img_h, self.img_w, NUM_CLASSES), dtype=np.float32)
        count = np.zeros((self.img_h, self.img_w), dtype=np.float32)

        for (y, x), mask in zip(self.tiles, tile_masks):
            h = min(ts, self.img_h - y)
            w = min(ts, self.img_w - x)

            if mask.ndim == 2:
                one_hot = np.eye(NUM_CLASSES, dtype=np.float32)[mask[:h, :w]]
            else:
                one_hot = mask[:h, :w]

            accum[y:y+h, x:x+w] += one_hot
            count[y:y+h, x:x+w] += 1.0

        count = np.maximum(count, 1.0)
        avg = accum / count[:, :, None]
        return avg.argmax(axis=-1).astype(np.uint8)


# ===================================================================== #
#  BALANCED  SAMPLER                                                     #
# ===================================================================== #

class BalancedSampler(Sampler):
    """Over-samples images containing rare classes so each training epoch
    exposes the model to a balanced class distribution.

    Scans masks once at construction time to build per-image dominant class
    statistics, then samples indices with inverse-frequency weighting.
    """

    def __init__(self, dataset, num_classes=NUM_CLASSES,
                 samples_per_epoch=None):
        self.dataset = dataset
        self.num_classes = num_classes
        self.samples_per_epoch = samples_per_epoch or len(dataset)

        self.weights = self._compute_weights()

    def _compute_weights(self):
        """Assign a sampling weight to each image based on its dominant class."""
        class_counts = Counter()
        dominant = []

        for idx in range(len(self.dataset)):
            sample = self.dataset[idx]
            mask = sample["mask"]
            if isinstance(mask, torch.Tensor):
                mask = mask.numpy()
            hist = np.bincount(mask.ravel(), minlength=self.num_classes)
            dom_cls = hist[1:].argmax() + 1  # skip background
            dominant.append(dom_cls)
            class_counts[dom_cls] += 1

        # Inverse frequency weight
        total = sum(class_counts.values())
        class_weight = {
            c: total / max(n, 1) for c, n in class_counts.items()
        }
        return [class_weight.get(d, 1.0) for d in dominant]

    def __len__(self):
        return self.samples_per_epoch

    def __iter__(self):
        indices = torch.multinomial(
            torch.tensor(self.weights, dtype=torch.float64),
            num_samples=self.samples_per_epoch,
            replacement=True,
        ).tolist()
        return iter(indices)


# ===================================================================== #
#  COLLATE  FUNCTION                                                     #
# ===================================================================== #

def floorplan_collate(batch):
    """Custom collate that stacks images, masks, and optional depth maps
    while keeping metadata (``name``, ``predictions``) as plain lists.
    """
    images = torch.stack([s["image"] for s in batch])
    masks  = torch.stack([s["mask"] for s in batch])
    names  = [s["name"] for s in batch]

    result = {"image": images, "mask": masks, "name": names}

    if "depth" in batch[0]:
        result["depth"] = torch.stack([s["depth"] for s in batch])
    if "predictions" in batch[0]:
        result["predictions"] = [s["predictions"] for s in batch]

    return result


# ===================================================================== #
#  VISUALISATION  HELPERS                                                #
# ===================================================================== #

def colourise_mask(mask):
    """Convert a class-id mask ``(H, W)`` to an RGB image ``(H, W, 3)``."""
    h, w = mask.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, colour in enumerate(CLASS_COLOURS):
        rgb[mask == cid] = colour
    return rgb


def overlay_mask_on_image(image_rgb, mask, alpha=0.45):
    """Blend a colourised mask onto an image with transparency."""
    colour = colourise_mask(mask)
    blended = cv2.addWeighted(image_rgb, 1.0 - alpha, colour, alpha, 0)
    return blended


def visualise_sample(sample, save_path=None):
    """Render an image + mask (+ depth) side-by-side and optionally save."""
    image = sample["image"]
    mask  = sample["mask"]

    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
        # Undo normalisation
        image = image * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
        image = (image * 255).clip(0, 255).astype(np.uint8)
    if isinstance(mask, torch.Tensor):
        mask = mask.numpy()

    colour_mask = colourise_mask(mask)
    overlay = overlay_mask_on_image(image, mask)
    panels = [image, colour_mask, overlay]

    if "depth" in sample:
        depth = sample["depth"]
        if isinstance(depth, torch.Tensor):
            depth = depth.squeeze().numpy()
        depth_vis = (depth * 255).clip(0, 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
        depth_vis = cv2.cvtColor(depth_vis, cv2.COLOR_BGR2RGB)
        panels.append(depth_vis)

    canvas = np.concatenate(panels, axis=1)

    if save_path is not None:
        cv2.imwrite(save_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    return canvas


# ===================================================================== #
#  DATA-LOADER  FACTORY                                                  #
# ===================================================================== #

def build_dataloaders(root, input_size=(512, 512), batch_size=8,
                      num_workers=2, train_ratio=0.85, depth=False,
                      balanced_sampling=False, seed=42):
    """Convenience factory that returns ``(train_loader, val_loader)`` ready
    for use in the training loop.

    Parameters
    ----------
    root : str
        Dataset root directory.
    balanced_sampling : bool
        Use :class:`BalancedSampler` for the training set to mitigate class
        imbalance.
    """
    train_stems, val_stems = stratified_split(root, train_ratio, seed)

    train_ds = FloorplanDataset(root, input_size, augment=True,
                                depth=depth, stems=train_stems)
    val_ds   = FloorplanDataset(root, input_size, augment=False,
                                depth=depth, stems=val_stems)

    if balanced_sampling:
        sampler = BalancedSampler(train_ds)
        shuffle = False
    else:
        sampler = None
        shuffle = True

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler,
        num_workers=num_workers, pin_memory=True, collate_fn=floorplan_collate,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=floorplan_collate,
    )

    return train_loader, val_loader
