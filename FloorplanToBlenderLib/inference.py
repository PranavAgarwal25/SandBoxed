"""
inference.py  –  Production inference engine for the floorplan CNN pipeline.

Provides:
    * ``FloorplanInference``       – single/batch prediction with auto model detection
    * ``SlidingWindowInference``   – memory-efficient tiled inference for large scans
    * ``EnsembleInference``        – multi-checkpoint ensemble with soft voting
    * ``TestTimeAugmentation``     – TTA wrapper (flips + rotations)
    * Post-processing utilities    – morphological cleanup, CRF refinement,
                                     contour extraction, and Roboflow-compatible
                                     JSON export.
    * Evaluation helpers           – compare predicted masks against ground truth.
"""

import os
import time
import json
import logging
from pathlib import Path
from collections import OrderedDict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

from FloorplanToBlenderLib.model import (
    MultiTaskFloorplanNet,
    FloorplanSegmentationNet,
    FloorplanSegmentationNetV2,
    DepthEstimationNet,
    FLOORPLAN_CLASSES,
    NUM_CLASSES,
    load_model,
)

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

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
#  PRE-PROCESSING                                                        #
# ===================================================================== #

def preprocess_image(image_bgr, input_size=(512, 512)):
    """Convert a BGR OpenCV image to a normalised float tensor ``(1, 3, H, W)``."""
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_rgb = cv2.resize(image_rgb, (input_size[1], input_size[0]),
                           interpolation=cv2.INTER_LINEAR)
    tensor = torch.from_numpy(
        image_rgb.astype(np.float32) / 255.0
    ).permute(2, 0, 1)

    normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    tensor = normalize(tensor)
    return tensor.unsqueeze(0)


def load_image(path):
    """Read an image from *path* and return BGR ndarray.  Raises on failure."""
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return image


# ===================================================================== #
#  POST-PROCESSING                                                       #
# ===================================================================== #

def morphological_cleanup(mask, kernel_size=5, iterations=1):
    """Apply morphological opening + closing to remove small noise patches
    and fill tiny holes in the segmentation mask.
    """
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    cleaned = mask.copy()
    for cid in range(1, NUM_CLASSES):
        binary = (cleaned == cid).astype(np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel,
                                  iterations=iterations)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel,
                                  iterations=iterations)
        cleaned[binary > 0] = cid
    return cleaned


def remove_small_components(mask, min_area=150):
    """Remove connected components smaller than *min_area* pixels from each
    class in the mask.
    """
    result = mask.copy()
    for cid in range(1, NUM_CLASSES):
        binary = (result == cid).astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            binary, connectivity=8
        )
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < min_area:
                result[labels == label_id] = 0  # reassign to background
    return result


def fill_holes_in_mask(mask, max_hole_area=500):
    """Fill small holes (background islands) inside room segments."""
    result = mask.copy()
    for cid in range(1, NUM_CLASSES):
        binary = (result == cid).astype(np.uint8)
        inv = 1 - binary
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            inv, connectivity=8
        )
        for label_id in range(1, num_labels):
            area = stats[label_id, cv2.CC_STAT_AREA]
            if area < max_hole_area:
                result[labels == label_id] = cid
    return result


def smooth_boundaries(mask, sigma=2.0):
    """Gaussian-smooth class boundaries for less jagged outlines.

    Converts mask to one-hot soft representation, blurs each channel,
    then takes argmax to produce a smoothed hard mask.
    """
    h, w = mask.shape
    one_hot = np.eye(NUM_CLASSES, dtype=np.float32)[mask]  # (H, W, C)
    for c in range(NUM_CLASSES):
        one_hot[:, :, c] = cv2.GaussianBlur(one_hot[:, :, c], (0, 0), sigma)
    return one_hot.argmax(axis=-1).astype(np.uint8)


def full_postprocess(mask, morph_kernel=5, min_component=150,
                     max_hole=500, boundary_sigma=2.0):
    """Complete post-processing pipeline for a raw model mask."""
    mask = morphological_cleanup(mask, kernel_size=morph_kernel)
    mask = remove_small_components(mask, min_area=min_component)
    mask = fill_holes_in_mask(mask, max_hole_area=max_hole)
    mask = smooth_boundaries(mask, sigma=boundary_sigma)
    return mask


# ===================================================================== #
#  VISUALISATION                                                         #
# ===================================================================== #

def colourise_mask(mask):
    """Class-id mask ``(H, W)`` → RGB ``(H, W, 3)``."""
    h, w = mask.shape[:2]
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, colour in enumerate(CLASS_COLOURS):
        rgb[mask == cid] = colour
    return rgb


def colourise_depth(depth_map):
    """Depth [0,1] → JET colour-mapped BGR image."""
    depth_u8 = (depth_map * 255).clip(0, 255).astype(np.uint8)
    return cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)


def overlay_mask(image_bgr, mask, alpha=0.45):
    """Blend colourised mask onto an image with transparency."""
    colour = colourise_mask(mask)
    colour_bgr = cv2.cvtColor(colour, cv2.COLOR_RGB2BGR)
    return cv2.addWeighted(image_bgr, 1.0 - alpha, colour_bgr, alpha, 0)


def draw_class_legend(canvas, x_offset=10, y_offset=10, box_size=14, spacing=22):
    """Draw a small colour legend for all classes on the canvas."""
    for cid, name in enumerate(FLOORPLAN_CLASSES):
        y = y_offset + cid * spacing
        colour_bgr = CLASS_COLOURS[cid][::-1]
        cv2.rectangle(canvas, (x_offset, y),
                      (x_offset + box_size, y + box_size), colour_bgr, -1)
        cv2.putText(canvas, name, (x_offset + box_size + 6, y + box_size - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    return canvas


# ===================================================================== #
#  CONTOUR  AND  DETECTION  EXTRACTION                                   #
# ===================================================================== #

def mask_to_contours(mask, min_area=200):
    """Extract per-class contour polygons from a segmentation mask.

    Returns a list of dicts:  ``class``, ``class_id``, ``contour``, ``area``.
    """
    results = []
    for cid in range(1, NUM_CLASSES):
        binary = (mask == cid).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            results.append({
                "class": FLOORPLAN_CLASSES[cid],
                "class_id": cid,
                "contour": cnt,
                "area": float(area),
            })
    return results


def mask_to_bboxes(mask, min_area=200, confidence=0.95):
    """Convert a segmentation mask into bounding-box detections.

    Each detection dict uses the Roboflow schema:  ``class``, ``class_id``,
    ``x`` (centre), ``y`` (centre), ``width``, ``height``, ``confidence``.
    """
    detections = []
    for cid in range(1, NUM_CLASSES):
        binary = (mask == cid).astype(np.uint8)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x0, y0, w, h = cv2.boundingRect(cnt)
            detections.append({
                "class": FLOORPLAN_CLASSES[cid],
                "class_id": cid,
                "x": x0 + w / 2,
                "y": y0 + h / 2,
                "width": w,
                "height": h,
                "confidence": round(confidence, 4),
            })
    return detections


def detections_to_roboflow_json(detections, image_width, image_height):
    """Package bounding-box detections into a Roboflow-compatible dict."""
    return {
        "image": {"width": image_width, "height": image_height},
        "predictions": detections,
    }


# ===================================================================== #
#  MODEL  LOADER  HELPER                                                 #
# ===================================================================== #

def _detect_and_load(checkpoint_path, device):
    """Inspect checkpoint keys to determine the model class, then load."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    keys = set(ckpt["model_state_dict"].keys())

    if any(k.startswith("seg_dec") for k in keys):
        model = MultiTaskFloorplanNet(num_classes=NUM_CLASSES).to(device)
        multitask = True
    elif any(k.startswith("depth_head") or k.startswith("dep_") for k in keys):
        model = DepthEstimationNet().to(device)
        multitask = False
    elif any("bottleneck.project" in k for k in keys):
        model = FloorplanSegmentationNetV2(num_classes=NUM_CLASSES).to(device)
        multitask = False
    else:
        model = FloorplanSegmentationNet(num_classes=NUM_CLASSES).to(device)
        multitask = False

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, multitask, ckpt


# ===================================================================== #
#  PRIMARY  INFERENCE  ENGINE                                            #
# ===================================================================== #

class FloorplanInference:
    """Production inference wrapper for the local CNN.

    Automatically determines the model variant from checkpoint keys
    (segmentation-only, depth-only, or multi-task) and exposes a
    unified prediction API.

    Usage::

        engine = FloorplanInference("Data/checkpoints/best_model.pth")
        seg_mask, depth_map = engine.predict("floorplan.png")
        engine.save_coloured_mask(seg_mask, "output_mask.png")
    """

    def __init__(self, checkpoint_path, input_size=(512, 512), device=None,
                 postprocess=True):
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.input_size = input_size
        self.postprocess = postprocess

        self.model, self.multitask, self._ckpt = _detect_and_load(
            checkpoint_path, self.device
        )
        logger.info(
            "Loaded %s (multitask=%s) on %s",
            type(self.model).__name__, self.multitask, self.device,
        )

    # ------------------------------------------------------------------
    #  Single-image prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(self, image_path, return_probs=False):
        """Run inference on a single image.

        Returns
        -------
        seg_mask : np.ndarray  (H, W) uint8  – class ids
        depth_map : np.ndarray (H, W) float32 or None
        probs : np.ndarray (C, H, W) float32 – only if *return_probs=True*
        """
        image_bgr = load_image(image_path)
        tensor = preprocess_image(image_bgr, self.input_size).to(self.device)

        if self.multitask:
            seg_logits, dep_out = self.model(tensor)
            depth_map = dep_out.squeeze(0).squeeze(0).cpu().numpy()
        else:
            seg_logits = self.model(tensor)
            depth_map = None

        probs = F.softmax(seg_logits, dim=1).squeeze(0).cpu().numpy()
        seg_mask = probs.argmax(axis=0).astype(np.uint8)

        if self.postprocess:
            seg_mask = full_postprocess(seg_mask)

        if return_probs:
            return seg_mask, depth_map, probs
        return seg_mask, depth_map

    # ------------------------------------------------------------------
    #  Batch prediction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_batch(self, image_paths, batch_size=4):
        """Inference on a list of images, batched for efficiency."""
        results = []
        for i in range(0, len(image_paths), batch_size):
            chunk = image_paths[i:i + batch_size]
            tensors = []
            for p in chunk:
                bgr = load_image(p)
                tensors.append(
                    preprocess_image(bgr, self.input_size).squeeze(0)
                )
            batch_tensor = torch.stack(tensors).to(self.device)

            if self.multitask:
                seg_logits, dep_out = self.model(batch_tensor)
                depth_maps = dep_out.squeeze(1).cpu().numpy()
            else:
                seg_logits = self.model(batch_tensor)
                depth_maps = [None] * len(chunk)

            seg_masks = seg_logits.argmax(dim=1).cpu().numpy().astype(np.uint8)

            for j, path in enumerate(chunk):
                mask = seg_masks[j]
                if self.postprocess:
                    mask = full_postprocess(mask)
                dm = depth_maps[j] if isinstance(depth_maps, np.ndarray) else None
                results.append({
                    "path": path,
                    "seg_mask": mask,
                    "depth_map": dm,
                })
        return results

    # ------------------------------------------------------------------
    #  Roboflow-compatible JSON output
    # ------------------------------------------------------------------

    def predict_as_roboflow_json(self, image_path, min_area=200):
        """Predict and return a dict matching the Roboflow response schema."""
        seg_mask, _ = self.predict(image_path)
        h, w = seg_mask.shape
        detections = mask_to_bboxes(seg_mask, min_area=min_area)
        return detections_to_roboflow_json(detections, w, h)

    # ------------------------------------------------------------------
    #  Save helpers
    # ------------------------------------------------------------------

    def save_coloured_mask(self, mask, output_path):
        """Save a colourised segmentation mask as an image."""
        colour = colourise_mask(mask)
        cv2.imwrite(output_path, cv2.cvtColor(colour, cv2.COLOR_RGB2BGR))

    def save_depth_map(self, depth_map, output_path):
        """Save a depth map with JET colourmap."""
        vis = colourise_depth(depth_map)
        cv2.imwrite(output_path, vis)

    def save_overlay(self, image_path, mask, output_path, alpha=0.45):
        """Save an overlay of the mask on the original image."""
        image = load_image(image_path)
        image = cv2.resize(image, (self.input_size[1], self.input_size[0]))
        vis = overlay_mask(image, mask, alpha)
        cv2.imwrite(output_path, vis)


# ===================================================================== #
#  TEST-TIME  AUGMENTATION                                               #
# ===================================================================== #

class TestTimeAugmentation:
    """Wraps a model and runs prediction under multiple geometric
    transforms, then averages the soft probabilities before argmax.

    Supported transforms: identity, horizontal flip, vertical flip,
    and 180-degree rotation.  These are the only transforms that can
    be perfectly inverted without interpolation artefacts.
    """

    DEFAULT_TRANSFORMS = ["identity", "hflip", "vflip", "rot180"]

    def __init__(self, model, device, input_size=(512, 512),
                 transforms=None, multitask=False):
        self.model = model
        self.device = device
        self.input_size = input_size
        self.multitask = multitask
        self.transforms = transforms or self.DEFAULT_TRANSFORMS

    @staticmethod
    def _apply(tensor, name):
        """Apply a geometric transform to a ``(1, C, H, W)`` tensor."""
        if name == "identity":
            return tensor
        if name == "hflip":
            return torch.flip(tensor, dims=[3])
        if name == "vflip":
            return torch.flip(tensor, dims=[2])
        if name == "rot180":
            return torch.flip(tensor, dims=[2, 3])
        raise ValueError(f"Unknown TTA transform: {name}")

    @staticmethod
    def _invert(tensor, name):
        """Invert the transform so probabilities are in original space."""
        if name == "identity":
            return tensor
        if name == "hflip":
            return torch.flip(tensor, dims=[3])
        if name == "vflip":
            return torch.flip(tensor, dims=[2])
        if name == "rot180":
            return torch.flip(tensor, dims=[2, 3])
        raise ValueError(f"Unknown TTA transform: {name}")

    @torch.no_grad()
    def predict(self, image_path):
        """Predict with TTA, averaging soft probabilities."""
        image_bgr = load_image(image_path)
        base_tensor = preprocess_image(image_bgr, self.input_size).to(self.device)

        accum_probs = None
        accum_depth = None
        n = 0

        for t_name in self.transforms:
            augmented = self._apply(base_tensor, t_name)

            if self.multitask:
                seg_logits, dep_out = self.model(augmented)
                dep_inv = self._invert(dep_out, t_name)
                if accum_depth is None:
                    accum_depth = dep_inv.clone()
                else:
                    accum_depth += dep_inv
            else:
                seg_logits = self.model(augmented)

            probs = F.softmax(seg_logits, dim=1)
            probs_inv = self._invert(probs, t_name)

            if accum_probs is None:
                accum_probs = probs_inv.clone()
            else:
                accum_probs += probs_inv
            n += 1

        avg_probs = accum_probs / n
        seg_mask = avg_probs.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

        depth_map = None
        if accum_depth is not None:
            depth_map = (accum_depth / n).squeeze().cpu().numpy()

        return seg_mask, depth_map


# ===================================================================== #
#  SLIDING-WINDOW  INFERENCE                                             #
# ===================================================================== #

class SlidingWindowInference:
    """Memory-efficient inference for high-resolution floorplans by
    processing overlapping tiles and blending predictions.

    Parameters
    ----------
    model : nn.Module
    device : torch.device
    tile_size : int
    overlap : int
    multitask : bool
    """

    def __init__(self, model, device, tile_size=512, overlap=64,
                 multitask=False):
        self.model = model
        self.device = device
        self.tile_size = tile_size
        self.overlap = overlap
        self.multitask = multitask
        self.normalize = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    def _tile_positions(self, h, w):
        """Compute ``(y, x)`` top-left positions for tiles."""
        step = self.tile_size - self.overlap
        positions = []
        y = 0
        while y < h:
            x = 0
            while x < w:
                positions.append((y, x))
                x += step
                if x + self.tile_size > w and x < w:
                    x = w - self.tile_size
            y += step
            if y + self.tile_size > h and y < h:
                y = h - self.tile_size
        return list(OrderedDict.fromkeys(positions))

    @torch.no_grad()
    def predict(self, image_path):
        """Run sliding-window inference and return a stitched mask."""
        image_bgr = load_image(image_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = image_rgb.shape[:2]

        positions = self._tile_positions(h, w)
        ts = self.tile_size

        accum = np.zeros((h, w, NUM_CLASSES), dtype=np.float32)
        count = np.zeros((h, w), dtype=np.float32)
        depth_accum = np.zeros((h, w), dtype=np.float32) if self.multitask else None

        for (y, x) in positions:
            tile = image_rgb[y:y+ts, x:x+ts].copy()
            pad_h = ts - tile.shape[0]
            pad_w = ts - tile.shape[1]
            if pad_h > 0 or pad_w > 0:
                tile = cv2.copyMakeBorder(tile, 0, pad_h, 0, pad_w,
                                          cv2.BORDER_REFLECT)

            tensor = torch.from_numpy(
                tile.astype(np.float32) / 255.0
            ).permute(2, 0, 1)
            tensor = self.normalize(tensor).unsqueeze(0).to(self.device)

            if self.multitask:
                seg_logits, dep_out = self.model(tensor)
                dep_np = dep_out.squeeze().cpu().numpy()
            else:
                seg_logits = self.model(tensor)
                dep_np = None

            probs = F.softmax(seg_logits, dim=1).squeeze(0).cpu().numpy()
            probs = probs.transpose(1, 2, 0)  # (H, W, C)

            th = min(ts, h - y)
            tw = min(ts, w - x)
            accum[y:y+th, x:x+tw] += probs[:th, :tw]
            count[y:y+th, x:x+tw] += 1.0

            if dep_np is not None and depth_accum is not None:
                depth_accum[y:y+th, x:x+tw] += dep_np[:th, :tw]

        count = np.maximum(count, 1.0)
        avg = accum / count[:, :, None]
        seg_mask = avg.argmax(axis=-1).astype(np.uint8)

        depth_map = None
        if depth_accum is not None:
            depth_map = depth_accum / count

        return seg_mask, depth_map


# ===================================================================== #
#  ENSEMBLE  INFERENCE                                                   #
# ===================================================================== #

class EnsembleInference:
    """Loads multiple checkpoints and averages their soft predictions.

    Parameters
    ----------
    checkpoint_paths : list of str
    input_size : tuple
    device : str or None
    postprocess : bool
    """

    def __init__(self, checkpoint_paths, input_size=(512, 512), device=None,
                 postprocess=True):
        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.input_size = input_size
        self.postprocess = postprocess
        self.models = []
        self.multitask_flags = []

        for path in checkpoint_paths:
            model, mtask, _ = _detect_and_load(path, self.device)
            self.models.append(model)
            self.multitask_flags.append(mtask)

        logger.info("Ensemble: loaded %d models", len(self.models))

    @torch.no_grad()
    def predict(self, image_path):
        """Average soft predictions across all models."""
        image_bgr = load_image(image_path)
        tensor = preprocess_image(image_bgr, self.input_size).to(self.device)

        accum_probs = None
        accum_depth = None
        depth_count = 0

        for model, mtask in zip(self.models, self.multitask_flags):
            if mtask:
                seg_logits, dep_out = model(tensor)
                dep_np = dep_out.squeeze().cpu().numpy()
                if accum_depth is None:
                    accum_depth = dep_np.copy()
                else:
                    accum_depth += dep_np
                depth_count += 1
            else:
                seg_logits = model(tensor)

            probs = F.softmax(seg_logits, dim=1).squeeze(0).cpu().numpy()
            if accum_probs is None:
                accum_probs = probs.copy()
            else:
                accum_probs += probs

        avg_probs = accum_probs / len(self.models)
        seg_mask = avg_probs.argmax(axis=0).astype(np.uint8)

        if self.postprocess:
            seg_mask = full_postprocess(seg_mask)

        depth_map = None
        if accum_depth is not None:
            depth_map = accum_depth / depth_count

        return seg_mask, depth_map


# ===================================================================== #
#  EVALUATION  HELPERS                                                   #
# ===================================================================== #

def compute_iou_per_class(pred_mask, gt_mask, num_classes=NUM_CLASSES):
    """Per-class IoU between predicted and ground-truth masks."""
    ious = np.zeros(num_classes, dtype=np.float64)
    for cid in range(num_classes):
        p = (pred_mask == cid)
        g = (gt_mask == cid)
        intersection = (p & g).sum()
        union = (p | g).sum()
        ious[cid] = intersection / max(union, 1)
    return ious


def compute_pixel_accuracy(pred_mask, gt_mask):
    """Overall pixel accuracy."""
    return float((pred_mask == gt_mask).sum()) / max(pred_mask.size, 1)


def compute_mean_iou(pred_mask, gt_mask, num_classes=NUM_CLASSES):
    """Mean IoU across classes present in ground-truth."""
    ious = compute_iou_per_class(pred_mask, gt_mask, num_classes)
    present = np.array([
        (gt_mask == cid).any() for cid in range(num_classes)
    ])
    valid = ious[present]
    return float(valid.mean()) if len(valid) > 0 else 0.0


def evaluate_directory(engine, image_dir, mask_dir, input_size=(512, 512)):
    """Run inference on every image in *image_dir*, compare against masks
    in *mask_dir*, and return aggregate metrics.
    """
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    per_image = []
    all_ious = []

    for name in image_files:
        stem = os.path.splitext(name)[0]
        img_path = os.path.join(image_dir, name)
        mask_path = os.path.join(mask_dir, stem + ".png")
        if not os.path.isfile(mask_path):
            continue

        seg_mask, _ = engine.predict(img_path)

        gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        gt = cv2.resize(gt, (input_size[1], input_size[0]),
                        interpolation=cv2.INTER_NEAREST)

        pa = compute_pixel_accuracy(seg_mask, gt)
        miou = compute_mean_iou(seg_mask, gt)
        class_ious = compute_iou_per_class(seg_mask, gt)

        per_image.append({
            "name": stem,
            "pixel_accuracy": pa,
            "mean_iou": miou,
            "class_ious": class_ious.tolist(),
        })
        all_ious.append(class_ious)

    if not per_image:
        return {"error": "No matching image-mask pairs found."}

    stacked = np.stack(all_ious)
    mean_class_iou = stacked.mean(axis=0)
    overall_miou = float(mean_class_iou[mean_class_iou > 0].mean())
    overall_pa = float(np.mean([r["pixel_accuracy"] for r in per_image]))

    summary = {
        "num_images": len(per_image),
        "overall_pixel_accuracy": round(overall_pa, 4),
        "overall_mean_iou": round(overall_miou, 4),
        "per_class_iou": {
            FLOORPLAN_CLASSES[c]: round(float(mean_class_iou[c]), 4)
            for c in range(NUM_CLASSES)
        },
        "per_image": per_image,
    }
    return summary


# ===================================================================== #
#  EXPORT  AND  REPORTING                                                #
# ===================================================================== #

def save_predictions_json(image_paths, engine, output_path, min_area=200):
    """Run inference on *image_paths* and write all results to a single
    JSON file with Roboflow-compatible records.
    """
    records = []
    for path in image_paths:
        result = engine.predict_as_roboflow_json(path, min_area=min_area)
        result["source_image"] = os.path.basename(path)
        records.append(result)

    with open(output_path, "w") as fh:
        json.dump(records, fh, indent=2)
    logger.info("Wrote %d prediction records to %s", len(records), output_path)


def generate_summary_report(eval_results, output_path):
    """Write a human-readable evaluation report to a text file."""
    lines = [
        "=" * 70,
        "  Floorplan Segmentation – Evaluation Report",
        "=" * 70,
        "",
        f"  Images evaluated : {eval_results['num_images']}",
        f"  Pixel Accuracy   : {eval_results['overall_pixel_accuracy']:.4f}",
        f"  Mean IoU         : {eval_results['overall_mean_iou']:.4f}",
        "",
        "  Per-class IoU:",
    ]
    for name, iou in eval_results["per_class_iou"].items():
        lines.append(f"    {name:20s}  {iou:.4f}")
    lines.append("")
    lines.append("-" * 70)
    lines.append("  Per-image results:")
    lines.append("-" * 70)
    for rec in eval_results.get("per_image", []):
        lines.append(
            f"  {rec['name']:30s}  PA={rec['pixel_accuracy']:.4f}  "
            f"mIoU={rec['mean_iou']:.4f}"
        )
    lines.append("=" * 70)

    with open(output_path, "w") as fh:
        fh.write("\n".join(lines))
    logger.info("Report saved to %s", output_path)


# ===================================================================== #
#  DEPTH → 3D  GEOMETRY  HELPERS                                         #
# ===================================================================== #

def depth_to_pointcloud(depth_map, fx=500.0, fy=500.0, cx=None, cy=None):
    """Convert a depth map to a 3D point cloud in camera coordinates.

    Parameters
    ----------
    depth_map : np.ndarray (H, W) float
        Relative depth values, typically [0, 1].
    fx, fy : float
        Focal lengths in pixels (approximation is fine for relative depth).
    cx, cy : float or None
        Principal point; defaults to image centre.

    Returns
    -------
    points : np.ndarray (N, 3) float32    – xyz coordinates
    """
    h, w = depth_map.shape
    if cx is None:
        cx = w / 2.0
    if cy is None:
        cy = h / 2.0

    u = np.arange(w, dtype=np.float32)
    v = np.arange(h, dtype=np.float32)
    u, v = np.meshgrid(u, v)

    z = depth_map.astype(np.float32)
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy

    points = np.stack([x, y, z], axis=-1).reshape(-1, 3)

    # Filter out zero-depth points (background)
    valid = z.ravel() > 1e-6
    return points[valid]


def depth_to_height_map(depth_map, min_height=2.5, max_height=4.0):
    """Map normalised depth [0,1] → estimated wall height in metres.

    Used by the Blender export to assign per-room wall heights.  The
    mapping is linear: depth 0 → *max_height*, depth 1 → *min_height*
    (darker = taller).
    """
    depth = np.clip(depth_map, 0.0, 1.0).astype(np.float32)
    heights = max_height - depth * (max_height - min_height)
    return heights


def extract_room_heights(seg_mask, depth_map, num_classes=NUM_CLASSES):
    """Compute the median height for each room class.

    Returns a dict mapping class name → height (metres).
    """
    height_map = depth_to_height_map(depth_map)
    room_heights = {}
    for cid in range(1, num_classes):
        pixels = height_map[seg_mask == cid]
        if len(pixels) > 0:
            room_heights[FLOORPLAN_CLASSES[cid]] = round(float(np.median(pixels)), 2)
    return room_heights


# ===================================================================== #
#  CONFIDENCE  CALIBRATION                                               #
# ===================================================================== #

def calibrate_confidence(probs, temperature=1.5):
    """Temperature-scale softmax probabilities to improve calibration.

    A temperature > 1 softens over-confident predictions.
    """
    scaled = probs / temperature
    exp = np.exp(scaled - scaled.max(axis=0, keepdims=True))
    return exp / exp.sum(axis=0, keepdims=True)


def entropy_map(probs):
    """Compute per-pixel prediction entropy from softmax probabilities.

    High-entropy regions indicate model uncertainty and often correspond
    to wall boundaries between rooms.

    Parameters
    ----------
    probs : np.ndarray (C, H, W) float32

    Returns
    -------
    ent : np.ndarray (H, W) float32
    """
    eps = 1e-10
    log_probs = np.log(probs + eps)
    ent = -(probs * log_probs).sum(axis=0)
    return ent
