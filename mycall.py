
import os
import sys
import json
import time
import hashlib
import logging
import argparse
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient

from FloorplanToBlenderLib.model import (
    MultiTaskFloorplanNet,
    FLOORPLAN_CLASSES,
    NUM_CLASSES,
)
from FloorplanToBlenderLib.inference import (
    FloorplanInference,
    TestTimeAugmentation,
    EnsembleInference,
    preprocess_image,
    load_image,
    colourise_mask,
    colourise_depth,
    overlay_mask,
    mask_to_bboxes,
    detections_to_json,
    full_postprocess,
    draw_class_legend,
)

logger = logging.getLogger("mycall")



CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="2mJz8J9Jx7NWoco65LK3",
)
ROBOFLOW_MODEL_ID = "builderformer-4/2"


# ===================================================================== #
#  PATHS  AND  DEFAULTS                                                  #
# ===================================================================== #

LOCAL_CHECKPOINT    = os.path.join("Data", "checkpoints", "best_model.pth")
OUTPUT_DIR          = "output_images"
SUPPORTED_FORMATS   = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEFAULT_INPUT_SIZE  = (512, 512)
MAX_DIMENSION       = 2048


# ===================================================================== #
#  IMAGE  PREPROCESSING  PIPELINE                                        #
# ===================================================================== #

def validate_image(image_path):
    """Check that *image_path* exists and can be decoded.

    Returns ``(True, image_bgr)`` or ``(False, error_message)``.
    """
    path = Path(image_path)
    if not path.is_file():
        return False, f"File does not exist: {image_path}"
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        return False, f"Unsupported format: {path.suffix}"
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        return False, f"OpenCV could not decode: {image_path}"
    return True, img


def auto_orient(image_bgr):
    """Rotate the image so that the longer side is horizontal, matching the
    typical floorplan orientation.
    """
    h, w = image_bgr.shape[:2]
    if h > w * 1.3:
        image_bgr = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    return image_bgr


def resize_for_inference(image_bgr, max_dim=MAX_DIMENSION):
    """Down-scale so neither dimension exceeds *max_dim*, preserving aspect."""
    h, w = image_bgr.shape[:2]
    if max(h, w) <= max_dim:
        return image_bgr
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def enhance_contrast(image_bgr, clip_limit=2.0):
    """Apply CLAHE on the luminance channel to improve wall visibility."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def denoise(image_bgr, strength=7):
    """Light denoising to clean up scanned floorplans."""
    return cv2.fastNlMeansDenoisingColored(image_bgr, None, strength, strength, 7, 21)


def sharpen(image_bgr):
    """Unsharp-mask sharpening to accentuate wall edges."""
    blurred = cv2.GaussianBlur(image_bgr, (0, 0), 3)
    return cv2.addWeighted(image_bgr, 1.5, blurred, -0.5, 0)


def full_preprocess(image_bgr, max_dim=MAX_DIMENSION, enhance=True):
    """Complete preprocessing pipeline before inference.

    Steps: orient → resize → (optional) contrast + sharpen.
    """
    image_bgr = auto_orient(image_bgr)
    image_bgr = resize_for_inference(image_bgr, max_dim)
    if enhance:
        image_bgr = enhance_contrast(image_bgr)
        image_bgr = sharpen(image_bgr)
    return image_bgr


# ===================================================================== #
#  BACK-END  INFERENCE  FUNCTIONS                                        #
# ===================================================================== #

def infer_api(image_path, confidence=0):
    """Call the cloud-hosted detection model and return the raw JSON dict."""
    model_id = f"{CLOUD_MODEL_ID}?confidence={confidence}"
    result = CLIENT.infer(image_path, model_id=model_id)
    return result


def infer_local(image_path, checkpoint=LOCAL_CHECKPOINT,
                use_tta=False, postprocess=True):
    """Run the local CNN and return an API-compatible dict.

    Parameters
    ----------
    use_tta : bool
        Enable test-time augmentation (slower but more accurate).
    postprocess : bool
        Apply morphological cleanup to the segmentation mask.
    """
    engine = FloorplanInference(
        checkpoint, input_size=DEFAULT_INPUT_SIZE, postprocess=postprocess
    )

    if use_tta:
        tta = TestTimeAugmentation(
            engine.model, engine.device,
            input_size=DEFAULT_INPUT_SIZE,
            multitask=engine.multitask,
        )
        seg_mask, depth_map = tta.predict(image_path)
    else:
        seg_mask, depth_map = engine.predict(image_path)

    h, w = seg_mask.shape
    detections = mask_to_bboxes(seg_mask, min_area=200)

    result = detections_to_json(detections, w, h)
    result["_seg_mask"] = seg_mask
    result["_depth_map"] = depth_map
    return result


def infer(image_path, use_local_model=None, use_tta=False):
    """High-level inference dispatcher.

    Parameters
    ----------
    image_path : str
        Path to the floorplan image.
    use_local_model : bool or None
        ``True`` – force local CNN.
        ``False`` – force cloud API.
        ``None`` – auto (local if checkpoint exists, else API).
    use_tta : bool
        Enable test-time augmentation (local only).

    Returns
    -------
    dict – API-compatible JSON with ``image`` and ``predictions``.
    """
    if use_local_model is None:
        use_local_model = os.path.isfile(LOCAL_CHECKPOINT)

    if use_local_model:
        logger.info("[mycall] Using local CNN  (%s)", LOCAL_CHECKPOINT)
        return infer_local(image_path, use_tta=use_tta)
    else:
        logger.info("[mycall] Using cloud API  (%s)", CLOUD_MODEL_ID)
        return infer_api(image_path)


# ===================================================================== #
#  RESULT  NORMALISATION                                                 #
# ===================================================================== #

def normalise_predictions(result):
    """Ensure all prediction dicts have consistent keys and types.

    Both cloud API and local CNN produce slightly different schemata;
    this function unifies them.
    """
    predictions = result.get("predictions", [])
    normalised = []
    for pred in predictions:
        normalised.append({
            "class":      str(pred.get("class", "unknown")),
            "class_id":   int(pred.get("class_id", 0)),
            "x":          float(pred.get("x", 0)),
            "y":          float(pred.get("y", 0)),
            "width":      float(pred.get("width", 0)),
            "height":     float(pred.get("height", 0)),
            "confidence": round(float(pred.get("confidence", 0)), 4),
        })
    result["predictions"] = normalised
    return result


def filter_by_confidence(result, threshold=0.25):
    """Remove predictions below a confidence threshold."""
    result["predictions"] = [
        p for p in result["predictions"]
        if p["confidence"] >= threshold
    ]
    return result


def filter_by_area(result, min_area=100):
    """Remove predictions whose bounding box is smaller than *min_area* px²."""
    result["predictions"] = [
        p for p in result["predictions"]
        if p["width"] * p["height"] >= min_area
    ]
    return result


def merge_overlapping(predictions, iou_threshold=0.5):
    """Non-maximum suppression: merge overlapping boxes of the same class."""
    if not predictions:
        return predictions

    by_class = {}
    for p in predictions:
        by_class.setdefault(p["class"], []).append(p)

    kept = []
    for cls_preds in by_class.values():
        cls_preds.sort(key=lambda p: p["confidence"], reverse=True)
        while cls_preds:
            best = cls_preds.pop(0)
            kept.append(best)
            remaining = []
            for other in cls_preds:
                if _box_iou(best, other) < iou_threshold:
                    remaining.append(other)
            cls_preds = remaining
    return kept


def _box_iou(a, b):
    """Compute IoU between two centre-format boxes."""
    ax1 = a["x"] - a["width"] / 2
    ay1 = a["y"] - a["height"] / 2
    ax2 = a["x"] + a["width"] / 2
    ay2 = a["y"] + a["height"] / 2

    bx1 = b["x"] - b["width"] / 2
    by1 = b["y"] - b["height"] / 2
    bx2 = b["x"] + b["width"] / 2
    by2 = b["y"] + b["height"] / 2

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    return inter / max(union, 1e-6)


# ===================================================================== #
#  VISUALISATION  /  ANNOTATION                                          #
# ===================================================================== #

def draw_detections(image_bgr, predictions, line_width=3, font_scale=0.6):
    """Draw bounding boxes and labels on *image_bgr* (in-place).

    Generates a stable, distinct colour per class rather than random, so
    visualisations are reproducible across runs.
    """
    label_colours = _build_colour_map(predictions)

    for pred in predictions:
        x1 = int(pred["x"] - pred["width"] / 2)
        y1 = int(pred["y"] - pred["height"] / 2)
        x2 = int(pred["x"] + pred["width"] / 2)
        y2 = int(pred["y"] + pred["height"] / 2)

        colour = label_colours.get(pred["class"], (0, 255, 0))
        conf = int(pred["confidence"] * 100)
        text = f"{pred['class']} {conf}%"

        cv2.rectangle(image_bgr, (x1, y1), (x2, y2), colour, line_width)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, 2)
        overlay = image_bgr.copy()
        cv2.rectangle(overlay, (x1, y1 - th - 10), (x1 + tw, y1), colour, -1)
        cv2.addWeighted(overlay, 0.6, image_bgr, 0.4, 0, image_bgr)
        cv2.putText(
            image_bgr, text, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2,
        )

    return image_bgr


def _build_colour_map(predictions):
    """Generate a deterministic colour per unique class label using a hash."""
    colours = {}
    for pred in predictions:
        label = pred["class"]
        if label not in colours:
            h = int(hashlib.md5(label.encode()).hexdigest()[:6], 16)
            colours[label] = ((h >> 16) & 0xFF, (h >> 8) & 0xFF, h & 0xFF)
    return colours


def create_comparison_image(image_bgr, api_preds, local_preds):
    """Side-by-side comparison: cloud API vs. local CNN detections."""
    left = image_bgr.copy()
    right = image_bgr.copy()

    draw_detections(left, api_preds)
    draw_detections(right, local_preds)

    cv2.putText(left, "Cloud API", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    cv2.putText(right, "Local CNN", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return np.concatenate([left, right], axis=1)


# ===================================================================== #
#  RESULT  PERSISTENCE                                                   #
# ===================================================================== #

def save_result_json(result, output_path):
    """Persist inference results to a JSON file (strip numpy arrays)."""
    serialisable = {
        k: v for k, v in result.items()
        if not isinstance(v, np.ndarray)
    }
    with open(output_path, "w") as fh:
        json.dump(serialisable, fh, indent=2)
    logger.info("Saved JSON: %s", output_path)


def save_annotated_image(image_bgr, predictions, output_path):
    """Draw detections and save the annotated image."""
    annotated = draw_detections(image_bgr.copy(), predictions)
    cv2.imwrite(output_path, annotated)
    logger.info("Saved annotated image: %s", output_path)


def save_segmentation_outputs(result, image_path, output_dir):
    """Save colourised mask + depth map if available from local CNN."""
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(image_path).stem

    seg_mask = result.get("_seg_mask")
    if seg_mask is not None:
        colour = colourise_mask(seg_mask)
        cv2.imwrite(
            os.path.join(output_dir, f"{stem}_seg.png"),
            cv2.cvtColor(colour, cv2.COLOR_RGB2BGR),
        )

        image_bgr = load_image(image_path)
        image_bgr = cv2.resize(
            image_bgr, (seg_mask.shape[1], seg_mask.shape[0])
        )
        overlay = overlay_mask(image_bgr, seg_mask)
        cv2.imwrite(os.path.join(output_dir, f"{stem}_overlay.png"), overlay)

    depth_map = result.get("_depth_map")
    if depth_map is not None:
        depth_vis = colourise_depth(depth_map)
        cv2.imwrite(os.path.join(output_dir, f"{stem}_depth.png"), depth_vis)


# ===================================================================== #
#  SUMMARY  REPORT                                                       #
# ===================================================================== #

def generate_summary(result, image_path, elapsed_s):
    """Return a dict summarising the inference run."""
    preds = result.get("predictions", [])
    class_counts = {}
    for p in preds:
        class_counts[p["class"]] = class_counts.get(p["class"], 0) + 1

    return {
        "image": os.path.basename(image_path),
        "backend": "local_cnn" if "_seg_mask" in result else "cloud_api",
        "num_detections": len(preds),
        "class_counts": class_counts,
        "elapsed_seconds": round(elapsed_s, 3),
        "timestamp": datetime.now().isoformat(),
    }


def print_summary(summary):
    """Pretty-print inference summary to stdout."""
    print(f"\n{'='*60}")
    print(f"  Image   : {summary['image']}")
    print(f"  Backend : {summary['backend']}")
    print(f"  Detections: {summary['num_detections']}")
    print(f"  Time    : {summary['elapsed_seconds']:.2f}s")
    print(f"  Classes :")
    for cls, cnt in sorted(summary["class_counts"].items()):
        print(f"    {cls:20s}  × {cnt}")
    print(f"{'='*60}\n")


# ===================================================================== #
#  BATCH  MODE                                                           #
# ===================================================================== #

def batch_infer(image_paths, use_local_model=None, use_tta=False,
                save_dir=None, confidence_threshold=0.0):
    """Run inference on multiple images and optionally save outputs."""
    results = []
    for idx, path in enumerate(image_paths):
        logger.info("[%d/%d] Processing %s", idx + 1, len(image_paths), path)
        t0 = time.time()

        result = infer(path, use_local_model=use_local_model, use_tta=use_tta)
        result = normalise_predictions(result)
        if confidence_threshold > 0:
            result = filter_by_confidence(result, confidence_threshold)
        result["predictions"] = merge_overlapping(result["predictions"])

        elapsed = time.time() - t0
        summary = generate_summary(result, path, elapsed)
        result["_summary"] = summary

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            stem = Path(path).stem
            save_result_json(result, os.path.join(save_dir, f"{stem}.json"))
            img = load_image(path)
            save_annotated_image(img, result["predictions"],
                                 os.path.join(save_dir, f"{stem}_annotated.png"))
            save_segmentation_outputs(result, path, save_dir)

        results.append(result)
    return results


# ===================================================================== #
#  CLI  ENTRY-POINT                                                      #
# ===================================================================== #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Floorplan element detection  (cloud API or local CNN)"
    )
    parser.add_argument(
        "images", nargs="+",
        help="Path(s) to floorplan image(s)",
    )
    parser.add_argument(
        "--local", action="store_true", default=False,
        help="Force local CNN inference",
    )
    parser.add_argument(
        "--api", action="store_true", default=False,
        help="Force cloud API inference",
    )
    parser.add_argument(
        "--tta", action="store_true", default=False,
        help="Enable test-time augmentation (local only)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.0,
        help="Minimum detection confidence (0-1)",
    )
    parser.add_argument(
        "--output-dir", default=OUTPUT_DIR,
        help="Directory for saved outputs",
    )
    parser.add_argument(
        "--save", action="store_true", default=False,
        help="Save annotated images + JSON results",
    )
    parser.add_argument(
        "--preprocess", action="store_true", default=False,
        help="Apply contrast enhancement and sharpening before inference",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    use_local = None
    if args.local:
        use_local = True
    elif args.api:
        use_local = False

    save_dir = args.output_dir if args.save else None

    # Preprocess images if requested
    image_paths = args.images
    if args.preprocess:
        preprocessed = []
        os.makedirs(args.output_dir, exist_ok=True)
        for path in image_paths:
            valid, payload = validate_image(path)
            if not valid:
                logger.error(payload)
                continue
            processed = full_preprocess(payload)
            out_path = os.path.join(
                args.output_dir, f"preprocessed_{Path(path).name}"
            )
            cv2.imwrite(out_path, processed)
            preprocessed.append(out_path)
        image_paths = preprocessed

    # Run inference
    if len(image_paths) == 1:
        t0 = time.time()
        result = infer(image_paths[0], use_local_model=use_local,
                       use_tta=args.tta)
        result = normalise_predictions(result)
        if args.confidence > 0:
            result = filter_by_confidence(result, args.confidence)
        result["predictions"] = merge_overlapping(result["predictions"])

        elapsed = time.time() - t0
        summary = generate_summary(result, image_paths[0], elapsed)
        print_summary(summary)

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            stem = Path(image_paths[0]).stem
            save_result_json(result, os.path.join(save_dir, f"{stem}.json"))
            img = load_image(image_paths[0])
            save_annotated_image(img, result["predictions"],
                                 os.path.join(save_dir, f"{stem}_annotated.png"))
            save_segmentation_outputs(result, image_paths[0], save_dir)

        # Print JSON to stdout
        serialisable = {
            k: v for k, v in result.items()
            if not k.startswith("_") and not isinstance(v, np.ndarray)
        }
        print(json.dumps(serialisable, indent=2))
    else:
        results = batch_infer(
            image_paths,
            use_local_model=use_local,
            use_tta=args.tta,
            save_dir=save_dir,
            confidence_threshold=args.confidence,
        )
        for r in results:
            if "_summary" in r:
                print_summary(r["_summary"])


if __name__ == "__main__":
    main()


# ===================================================================== #
#  RESULT  CACHING                                                       #
# ===================================================================== #

_result_cache = {}


def cached_infer(image_path, use_local_model=None, use_tta=False):
    """Like :func:`infer` but caches results keyed by file hash.

    Prevents redundant API calls / inference runs when the same image is
    processed more than once (e.g. during interactive exploration or
    re-runs of the pipeline).
    """
    file_hash = _file_md5(image_path)
    if file_hash in _result_cache:
        logger.info("[mycall] Cache hit for %s", image_path)
        return _result_cache[file_hash]

    result = infer(image_path, use_local_model=use_local_model, use_tta=use_tta)
    _result_cache[file_hash] = result
    return result


def _file_md5(path, chunk_size=65536):
    """Compute MD5 digest for a file."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ===================================================================== #
#  DETECTION  COMPARISON  UTILITIES                                      #
# ===================================================================== #

def compare_backends(image_path, confidence_threshold=0.0):
    """Run both cloud API and local CNN on the same image and return a
    comparison dict showing per-class detection counts and overlap.

    Useful for validating that the local model matches cloud performance.
    """
    api_result = infer_api(image_path)
    api_result = normalise_predictions(api_result)
    if confidence_threshold > 0:
        api_result = filter_by_confidence(api_result, confidence_threshold)

    local_result = infer_local(image_path)
    local_result = normalise_predictions(local_result)
    if confidence_threshold > 0:
        local_result = filter_by_confidence(local_result, confidence_threshold)

    api_classes   = Counter(p["class"] for p in api_result["predictions"])
    local_classes = Counter(p["class"] for p in local_result["predictions"])

    all_classes = sorted(set(api_classes) | set(local_classes))
    comparison = {}
    for cls in all_classes:
        comparison[cls] = {
            "api_count": api_classes.get(cls, 0),
            "local_count": local_classes.get(cls, 0),
        }

    return {
        "api_total": len(api_result["predictions"]),
        "local_total": len(local_result["predictions"]),
        "per_class": comparison,
        "api_result": api_result,
        "local_result": local_result,
    }


def print_comparison(comp):
    """Pretty-print the output of :func:`compare_backends`."""
    print(f"\n{'='*60}")
    print(f"  API detections   : {comp['api_total']}")
    print(f"  Local detections : {comp['local_total']}")
    print(f"\n  {'Class':20s}  {'API':>5s}  {'Local':>5s}")
    print(f"  {'-'*35}")
    for cls, counts in comp["per_class"].items():
        print(f"  {cls:20s}  {counts['api_count']:5d}  {counts['local_count']:5d}")
    print(f"{'='*60}\n")


# ===================================================================== #
#  AREA-BASED  ROOM  STATISTICS                                          #
# ===================================================================== #

def compute_room_areas(predictions, pixel_scale=1.0):
    """Compute the area (in scaled units) for each detected room.

    Parameters
    ----------
    predictions : list of dict
        Detection dicts with ``class``, ``width``, ``height``.
    pixel_scale : float
        Conversion factor from pixels to real-world units (e.g. m²/px²).

    Returns
    -------
    list of dict   – each with ``class``, ``area_px``, ``area_scaled``.
    """
    rooms = []
    for p in predictions:
        area_px = p["width"] * p["height"]
        rooms.append({
            "class": p["class"],
            "area_px": round(area_px, 2),
            "area_scaled": round(area_px * pixel_scale, 4),
        })
    rooms.sort(key=lambda r: r["area_px"], reverse=True)
    return rooms


def aggregate_room_areas(predictions, pixel_scale=1.0):
    """Sum room areas by class.

    Returns a dict: class_name → total_area_scaled.
    """
    per_class = {}
    for p in predictions:
        area = p["width"] * p["height"] * pixel_scale
        per_class[p["class"]] = per_class.get(p["class"], 0.0) + area
    return {k: round(v, 4) for k, v in sorted(per_class.items())}


# ===================================================================== #
#  INPUT  VALIDATION  HELPERS                                            #
# ===================================================================== #

def check_image_quality(image_bgr, min_size=128, max_size=8192):
    """Return a list of warning strings for potential quality issues."""
    warnings = []
    h, w = image_bgr.shape[:2]

    if h < min_size or w < min_size:
        warnings.append(
            f"Image is very small ({w}×{h}).  Detection quality may suffer."
        )
    if h > max_size or w > max_size:
        warnings.append(
            f"Image is very large ({w}×{h}).  Consider resizing before inference."
        )

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    if variance < 50.0:
        warnings.append(
            f"Image appears blurry (Laplacian variance={variance:.1f}).  "
            "Wall edges may not be detected reliably."
        )

    mean_brightness = gray.mean()
    if mean_brightness < 40:
        warnings.append("Image is very dark.  Consider enhancing contrast.")
    elif mean_brightness > 230:
        warnings.append("Image is very bright / washed out.")

    return warnings
