"""
mod.py  –  Batch floorplan processing module.

Orchestrates the full pipeline for a folder of floorplan images:

    1. Scan input directory for images (PNG, JPG, SVG)
    2. Pre-process each image (SVG→PNG conversion, resize, enhance)
    3. Run element detection (Roboflow API or local CNN)
    4. Post-process and normalise detections
    5. Annotate images with bounding boxes and segmentation overlays
    6. Persist per-image JSON, annotated images, and a summary report

Parallelism is achieved via ``multiprocessing.Pool``.  Progress and
diagnostics are written to the standard logger as well as to a per-run
JSON log in the output directory.
"""

import os
import sys
import json
import time
import shutil
import logging
import hashlib
import argparse
import multiprocessing
from pathlib import Path
from datetime import datetime
from collections import Counter, OrderedDict

import cv2
import numpy as np
import cairosvg
from inference_sdk import InferenceHTTPClient

from FloorplanToBlenderLib.inference import (
    FloorplanInference,
    colourise_mask,
    colourise_depth,
    overlay_mask,
    mask_to_bboxes,
    detections_to_roboflow_json,
    full_postprocess,
    draw_class_legend,
)
from FloorplanToBlenderLib.model import FLOORPLAN_CLASSES, NUM_CLASSES

logger = logging.getLogger("mod")


# ===================================================================== #
#  CONFIGURATION                                                         #
# ===================================================================== #

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="2mJz8J9Jx7NWoco65LK3",
)
ROBOFLOW_MODEL_ID = "builderformer-4/2"

INPUT_FOLDER       = "input_images"
OUTPUT_FOLDER      = "output_images"
LOCAL_CHECKPOINT   = os.path.join("Data", "checkpoints", "best_model.pth")
MAX_IMAGES         = 50
MAX_DIMENSION      = 1024
SUPPORTED_FORMATS  = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".svg"}
ANNOTATION_COLOURS = {
    name: tuple(
        int(c)
        for c in [
            (int(hashlib.md5(name.encode()).hexdigest()[:2], 16)),
            (int(hashlib.md5(name.encode()).hexdigest()[2:4], 16)),
            (int(hashlib.md5(name.encode()).hexdigest()[4:6], 16)),
        ]
    )
    for name in FLOORPLAN_CLASSES
}


# ===================================================================== #
#  IMAGE  CONVERSION  AND  PREPROCESSING                                 #
# ===================================================================== #

def convert_svg_to_png(svg_path, output_path, dpi=150):
    """Rasterise an SVG floorplan to PNG via cairosvg."""
    cairosvg.svg2png(url=svg_path, write_to=output_path, dpi=dpi)
    logger.info("SVG→PNG: %s → %s", svg_path, output_path)
    return output_path


def resize_if_needed(image, max_dim=MAX_DIMENSION):
    """Scale so neither dimension exceeds *max_dim*, preserving aspect."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def auto_orient_image(image):
    """Rotate so the longer edge is horizontal (common floorplan layout)."""
    h, w = image.shape[:2]
    if h > w * 1.3:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    return image


def enhance_scan_quality(image):
    """CLAHE + bilateral filter to improve wall/edge contrast on scans."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    enhanced = cv2.bilateralFilter(enhanced, d=5, sigmaColor=50, sigmaSpace=50)
    return enhanced


def validate_image_file(path):
    """Return ``True`` if *path* is a readable, supported image file."""
    p = Path(path)
    if not p.is_file():
        return False
    if p.suffix.lower() not in SUPPORTED_FORMATS:
        return False
    if p.suffix.lower() == ".svg":
        return True
    img = cv2.imread(str(p), cv2.IMREAD_COLOR)
    return img is not None


def compute_image_hash(path, chunk_size=65536):
    """SHA-256 hash of the file for deduplication / caching."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# ===================================================================== #
#  DETECTION  BACK-END                                                   #
# ===================================================================== #

def run_detection(image_path, use_local=False, confidence=0):
    """Run floorplan element detection on a single image.

    Returns a Roboflow-compatible dict with ``predictions``.
    """
    if use_local and os.path.isfile(LOCAL_CHECKPOINT):
        engine = FloorplanInference(LOCAL_CHECKPOINT)
        return engine.predict_as_roboflow_json(image_path)
    else:
        model_id = f"{ROBOFLOW_MODEL_ID}?confidence={confidence}"
        return CLIENT.infer(image_path, model_id=model_id)


def normalise_result(result):
    """Ensure every prediction dict has consistent keys/types."""
    preds = result.get("predictions", [])
    clean = []
    for p in preds:
        clean.append({
            "class":      str(p.get("class", "unknown")),
            "class_id":   int(p.get("class_id", 0)),
            "x":          float(p.get("x", 0)),
            "y":          float(p.get("y", 0)),
            "width":      float(p.get("width", 0)),
            "height":     float(p.get("height", 0)),
            "confidence": round(float(p.get("confidence", 0)), 4),
        })
    result["predictions"] = clean
    return result


def filter_detections(result, min_confidence=0.0, min_area=100):
    """Remove low-confidence and tiny detections."""
    result["predictions"] = [
        p for p in result["predictions"]
        if p["confidence"] >= min_confidence
           and p["width"] * p["height"] >= min_area
    ]
    return result


# ===================================================================== #
#  ANNOTATION  /  VISUALISATION                                          #
# ===================================================================== #

def draw_detections(image, predictions, line_width=3, font_scale=0.55):
    """Draw bounding boxes and labels on *image* with deterministic colours."""
    for pred in predictions:
        label = pred["class"]
        colour = ANNOTATION_COLOURS.get(label, (0, 255, 0))
        conf = int(pred["confidence"] * 100)
        text = f"{label} {conf}%"

        x1 = int(pred["x"] - pred["width"] / 2)
        y1 = int(pred["y"] - pred["height"] / 2)
        x2 = int(pred["x"] + pred["width"] / 2)
        y2 = int(pred["y"] + pred["height"] / 2)

        cv2.rectangle(image, (x1, y1), (x2, y2), colour, line_width)

        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                       font_scale, 2)
        overlay = image.copy()
        cv2.rectangle(overlay, (x1, y1 - th - 10), (x1 + tw, y1), colour, -1)
        cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)
        cv2.putText(
            image, text, (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2,
        )

    return image


def create_summary_tile(image, predictions, seg_mask=None, depth_map=None):
    """Create a multi-panel summary image showing detection boxes,
    segmentation overlay, and depth map side-by-side.
    """
    panels = [draw_detections(image.copy(), predictions)]

    if seg_mask is not None:
        overlay = overlay_mask(image.copy(), seg_mask, alpha=0.45)
        panels.append(overlay)

    if depth_map is not None:
        depth_vis = colourise_depth(depth_map)
        if depth_vis.shape[:2] != image.shape[:2]:
            depth_vis = cv2.resize(depth_vis, (image.shape[1], image.shape[0]))
        panels.append(depth_vis)

    # Ensure all panels have the same height
    max_h = max(p.shape[0] for p in panels)
    aligned = []
    for p in panels:
        if p.shape[0] < max_h:
            pad = np.zeros((max_h - p.shape[0], p.shape[1], 3), dtype=np.uint8)
            p = np.vstack([p, pad])
        aligned.append(p)

    return np.concatenate(aligned, axis=1)


# ===================================================================== #
#  STATISTICS  AND  REPORTING                                            #
# ===================================================================== #

def compute_batch_statistics(all_results):
    """Aggregate detection statistics across all processed images."""
    total_images = len(all_results)
    total_detections = 0
    class_counter = Counter()
    confidence_values = []

    for record in all_results:
        preds = record.get("predictions", [])
        total_detections += len(preds)
        for p in preds:
            class_counter[p["class"]] += 1
            confidence_values.append(p["confidence"])

    avg_confidence = (
        float(np.mean(confidence_values)) if confidence_values else 0.0
    )

    return {
        "total_images": total_images,
        "total_detections": total_detections,
        "avg_detections_per_image": round(total_detections / max(total_images, 1), 2),
        "avg_confidence": round(avg_confidence, 4),
        "class_distribution": dict(class_counter.most_common()),
    }


def generate_text_report(stats, output_path):
    """Write a human-readable batch processing report."""
    lines = [
        "=" * 65,
        "  Floorplan Batch Processing Report",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 65,
        "",
        f"  Images processed      : {stats['total_images']}",
        f"  Total detections      : {stats['total_detections']}",
        f"  Average per image     : {stats['avg_detections_per_image']}",
        f"  Average confidence    : {stats['avg_confidence']:.2%}",
        "",
        "  Class distribution:",
    ]
    for cls, cnt in stats["class_distribution"].items():
        lines.append(f"    {cls:20s}  × {cnt}")
    lines.append("")
    lines.append("=" * 65)

    with open(output_path, "w") as fh:
        fh.write("\n".join(lines))
    logger.info("Report written: %s", output_path)


def generate_json_report(stats, per_image_records, output_path):
    """Write a machine-readable JSON batch report."""
    report = {
        "summary": stats,
        "per_image": per_image_records,
        "timestamp": datetime.now().isoformat(),
    }
    with open(output_path, "w") as fh:
        json.dump(report, fh, indent=2)
    logger.info("JSON report: %s", output_path)


# ===================================================================== #
#  PER-IMAGE  PROCESSING                                                 #
# ===================================================================== #

def process_image(image_path, index, output_folder=OUTPUT_FOLDER,
                  use_local=False, enhance=False, save_tiles=False):
    """Full pipeline for a single image.

    1. Convert SVG if needed
    2. Read + resize + optionally enhance
    3. Detect elements
    4. Normalise + filter detections
    5. Annotate + save outputs
    6. Return per-image record

    Parameters
    ----------
    image_path : str
    index : int
        Sequential index used for naming output files.
    output_folder : str
    use_local : bool
    enhance : bool
        Apply scan-quality enhancement before detection.
    save_tiles : bool
        Also save the segmentation mask and depth map separately.
    """
    os.makedirs(output_folder, exist_ok=True)
    record = {"index": index, "source": os.path.basename(image_path)}
    t0 = time.time()

    # ── SVG handling ──────────────────────────────────────────────────
    if image_path.lower().endswith(".svg"):
        png_path = os.path.join(output_folder, f"converted_{index}.png")
        try:
            image_path = convert_svg_to_png(image_path, png_path)
        except Exception as exc:
            record["error"] = f"SVG conversion failed: {exc}"
            logger.error(record["error"])
            return record

    # ── Read & preprocess ─────────────────────────────────────────────
    image = cv2.imread(image_path)
    if image is None:
        record["error"] = f"Could not read: {image_path}"
        logger.warning(record["error"])
        return record

    original_h, original_w = image.shape[:2]
    record["original_size"] = {"width": original_w, "height": original_h}

    image = auto_orient_image(image)
    image = resize_if_needed(image)
    if enhance:
        image = enhance_scan_quality(image)

    resized_path = os.path.join(output_folder, f"resized_{index}.jpg")
    cv2.imwrite(resized_path, image)

    # ── Detection ─────────────────────────────────────────────────────
    try:
        result = run_detection(resized_path, use_local=use_local)
    except Exception as exc:
        record["error"] = f"Detection failed: {exc}"
        logger.error(record["error"])
        return record

    result = normalise_result(result)
    result = filter_detections(result)
    predictions = result.get("predictions", [])
    record["num_detections"] = len(predictions)

    class_counts = Counter(p["class"] for p in predictions)
    record["class_counts"] = dict(class_counts)

    # ── Persist raw JSON ──────────────────────────────────────────────
    json_path = os.path.join(output_folder, f"result_{index}.json")
    serialisable = {
        k: v for k, v in result.items()
        if not isinstance(v, np.ndarray)
    }
    with open(json_path, "w") as fh:
        json.dump(serialisable, fh, indent=4)

    # ── Annotated image ───────────────────────────────────────────────
    annotated = draw_detections(image.copy(), predictions)
    annotated_path = os.path.join(output_folder, f"annotated_{index}.jpg")
    cv2.imwrite(annotated_path, annotated)

    # ── Segmentation / depth tiles (local CNN only) ───────────────────
    seg_mask = result.get("_seg_mask")
    depth_map = result.get("_depth_map")

    if save_tiles and seg_mask is not None:
        colour_mask = colourise_mask(seg_mask)
        cv2.imwrite(
            os.path.join(output_folder, f"seg_{index}.png"),
            cv2.cvtColor(colour_mask, cv2.COLOR_RGB2BGR),
        )
        tile = create_summary_tile(image, predictions, seg_mask, depth_map)
        cv2.imwrite(os.path.join(output_folder, f"tile_{index}.jpg"), tile)

    if save_tiles and depth_map is not None:
        depth_vis = colourise_depth(depth_map)
        cv2.imwrite(os.path.join(output_folder, f"depth_{index}.png"), depth_vis)

    record["elapsed_seconds"] = round(time.time() - t0, 3)
    record["outputs"] = {
        "resized": resized_path,
        "json": json_path,
        "annotated": annotated_path,
    }
    return record


# ===================================================================== #
#  BATCH  ORCHESTRATION                                                  #
# ===================================================================== #

def collect_image_paths(folder, max_images=MAX_IMAGES):
    """Gather supported image files from *folder*, up to *max_images*."""
    folder = Path(folder)
    if not folder.is_dir():
        logger.error("Input folder does not exist: %s", folder)
        return []

    paths = sorted([
        str(p)
        for p in folder.iterdir()
        if p.suffix.lower() in SUPPORTED_FORMATS
    ])

    if len(paths) > max_images:
        logger.warning(
            "Found %d images but max_images=%d – truncating.",
            len(paths), max_images,
        )
        paths = paths[:max_images]

    return paths


def _worker(args):
    """Wrapper for multiprocessing.Pool.map."""
    image_path, idx, output_folder, use_local, enhance, save_tiles = args
    return process_image(image_path, idx, output_folder, use_local, enhance,
                         save_tiles)


def run_batch(input_folder=INPUT_FOLDER, output_folder=OUTPUT_FOLDER,
              use_local=False, enhance=False, save_tiles=False,
              max_images=MAX_IMAGES, num_workers=None):
    """Process all images in *input_folder* with optional parallelism.

    Parameters
    ----------
    num_workers : int or None
        Number of parallel workers.  ``None`` = sequential,
        ``0`` = ``cpu_count()``.
    """
    paths = collect_image_paths(input_folder, max_images)
    if not paths:
        logger.error("No images found in %s", input_folder)
        return

    os.makedirs(output_folder, exist_ok=True)
    logger.info("Processing %d images → %s", len(paths), output_folder)

    t0 = time.time()

    if num_workers is not None:
        workers = num_workers if num_workers > 0 else multiprocessing.cpu_count()
        args_list = [
            (p, i, output_folder, use_local, enhance, save_tiles)
            for i, p in enumerate(paths)
        ]
        with multiprocessing.Pool(processes=workers) as pool:
            records = pool.map(_worker, args_list)
    else:
        records = []
        for i, p in enumerate(paths):
            rec = process_image(p, i, output_folder, use_local, enhance,
                                save_tiles)
            records.append(rec)

    elapsed_total = time.time() - t0
    logger.info("Batch complete in %.1fs", elapsed_total)

    # ── Aggregate statistics ──────────────────────────────────────────
    valid_results = [
        r for r in records if "error" not in r
    ]
    # Build pseudo-results for stats
    all_result_jsons = []
    for r in valid_results:
        json_path = r.get("outputs", {}).get("json")
        if json_path and os.path.isfile(json_path):
            with open(json_path) as fh:
                all_result_jsons.append(json.load(fh))

    stats = compute_batch_statistics(all_result_jsons)
    stats["total_elapsed_seconds"] = round(elapsed_total, 2)
    stats["errors"] = len(records) - len(valid_results)

    # ── Reports ───────────────────────────────────────────────────────
    generate_text_report(stats, os.path.join(output_folder, "report.txt"))
    generate_json_report(
        stats,
        records,
        os.path.join(output_folder, "batch_report.json"),
    )

    # Print summary
    print(f"\n{'='*60}")
    print(f"  Processed {stats['total_images']} images  "
          f"({stats['errors']} errors)")
    print(f"  Total detections : {stats['total_detections']}")
    print(f"  Average per image: {stats['avg_detections_per_image']}")
    print(f"  Elapsed          : {elapsed_total:.1f}s")
    print(f"{'='*60}\n")


# ===================================================================== #
#  CLEANUP  UTILITIES                                                    #
# ===================================================================== #

def clean_output_folder(folder=OUTPUT_FOLDER, keep_json=True):
    """Remove generated artefacts from the output folder.

    Useful when re-running the pipeline to avoid mixing results.
    """
    folder = Path(folder)
    if not folder.is_dir():
        return

    removed = 0
    for f in folder.iterdir():
        if keep_json and f.suffix == ".json":
            continue
        if f.is_file():
            f.unlink()
            removed += 1
    logger.info("Cleaned %d files from %s", removed, folder)


def archive_results(output_folder=OUTPUT_FOLDER, archive_name=None):
    """Zip the entire output folder for archival / sharing."""
    if archive_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        archive_name = f"results_{ts}"
    shutil.make_archive(archive_name, "zip", output_folder)
    logger.info("Archived → %s.zip", archive_name)


# ===================================================================== #
#  CLI  ENTRY-POINT                                                      #
# ===================================================================== #

def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch floorplan detection pipeline"
    )
    parser.add_argument(
        "--input", default=INPUT_FOLDER,
        help="Input image folder (default: input_images/)",
    )
    parser.add_argument(
        "--output", default=OUTPUT_FOLDER,
        help="Output folder for results (default: output_images/)",
    )
    parser.add_argument(
        "--local", action="store_true", default=False,
        help="Use local CNN instead of Roboflow API",
    )
    parser.add_argument(
        "--enhance", action="store_true", default=False,
        help="Apply contrast/denoising enhancement before detection",
    )
    parser.add_argument(
        "--tiles", action="store_true", default=False,
        help="Save segmentation + depth tiles (local only)",
    )
    parser.add_argument(
        "--max-images", type=int, default=MAX_IMAGES,
        help="Maximum number of images to process",
    )
    parser.add_argument(
        "--workers", type=int, default=None,
        help="Number of parallel workers (None=sequential, 0=cpu_count)",
    )
    parser.add_argument(
        "--clean", action="store_true", default=False,
        help="Clean output folder before processing",
    )
    parser.add_argument(
        "--archive", action="store_true", default=False,
        help="Zip results after processing",
    )
    return parser.parse_args()


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    if args.clean:
        clean_output_folder(args.output)

    run_batch(
        input_folder=args.input,
        output_folder=args.output,
        use_local=args.local,
        enhance=args.enhance,
        save_tiles=args.tiles,
        max_images=args.max_images,
        num_workers=args.workers,
    )

    if args.archive:
        archive_results(args.output)


if __name__ == "__main__":
    main()


# ===================================================================== #
#  IMAGE  DEDUPLICATION                                                  #
# ===================================================================== #

def deduplicate_images(folder, threshold=0.98):
    """Detect near-duplicate images in *folder* using average-hash comparison.

    Returns a list of ``(kept_path, duplicate_path)`` pairs.
    """
    from collections import defaultdict

    def _avg_hash(image, size=8):
        resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        mean = gray.mean()
        return (gray > mean).flatten()

    paths = collect_image_paths(folder, max_images=9999)
    hashes = {}
    duplicates = []

    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h = _avg_hash(img)
        key = tuple(h.tolist())

        found_dup = False
        for existing_key, existing_path in hashes.items():
            similarity = np.mean(np.array(key) == np.array(existing_key))
            if similarity >= threshold:
                duplicates.append((existing_path, p))
                found_dup = True
                break
        if not found_dup:
            hashes[key] = p

    if duplicates:
        logger.info("Found %d near-duplicate images", len(duplicates))
    return duplicates


# ===================================================================== #
#  RESULT  AGGREGATION                                                   #
# ===================================================================== #

def load_all_results(output_folder=OUTPUT_FOLDER):
    """Load all ``result_*.json`` files from the output folder.

    Returns a list of ``(index, data)`` tuples sorted by index.
    """
    folder = Path(output_folder)
    results = []
    for f in sorted(folder.glob("result_*.json")):
        idx = int(f.stem.split("_")[-1])
        with open(f) as fh:
            data = json.load(fh)
        results.append((idx, data))
    return results


def merge_results_to_single_json(output_folder=OUTPUT_FOLDER):
    """Concatenate all per-image result JSONs into one combined file."""
    all_results = load_all_results(output_folder)
    combined = {
        "num_images": len(all_results),
        "results": [
            {"index": idx, "data": data}
            for idx, data in all_results
        ],
        "timestamp": datetime.now().isoformat(),
    }
    out_path = os.path.join(output_folder, "combined_results.json")
    with open(out_path, "w") as fh:
        json.dump(combined, fh, indent=2)
    logger.info("Combined results: %s", out_path)
    return out_path


def compute_class_heatmap(output_folder=OUTPUT_FOLDER, canvas_size=(512, 512)):
    """Build a spatial heatmap of detection centres across all images.

    Useful for understanding which areas of floorplans are most commonly
    occupied by each room type.

    Returns a dict  class_name → np.ndarray(H, W) float32.
    """
    h, w = canvas_size
    heatmaps = {
        name: np.zeros((h, w), dtype=np.float32)
        for name in FLOORPLAN_CLASSES[1:]  # skip background
    }

    all_results = load_all_results(output_folder)
    for _, data in all_results:
        img_w = data.get("image", {}).get("width", w)
        img_h = data.get("image", {}).get("height", h)
        sx = w / max(img_w, 1)
        sy = h / max(img_h, 1)

        for pred in data.get("predictions", []):
            cls = pred.get("class", "")
            if cls not in heatmaps:
                continue
            cx = int(pred["x"] * sx)
            cy = int(pred["y"] * sy)
            bw = int(pred["width"] * sx / 2)
            bh = int(pred["height"] * sy / 2)

            y1 = max(cy - bh, 0)
            y2 = min(cy + bh, h)
            x1 = max(cx - bw, 0)
            x2 = min(cx + bw, w)
            heatmaps[cls][y1:y2, x1:x2] += 1.0

    return heatmaps


def save_class_heatmaps(heatmaps, output_folder=OUTPUT_FOLDER):
    """Visualise and save each class heatmap as a colour-mapped image."""
    out_dir = os.path.join(output_folder, "heatmaps")
    os.makedirs(out_dir, exist_ok=True)

    for cls, hmap in heatmaps.items():
        if hmap.max() > 0:
            normalised = (hmap / hmap.max() * 255).astype(np.uint8)
        else:
            normalised = np.zeros_like(hmap, dtype=np.uint8)
        coloured = cv2.applyColorMap(normalised, cv2.COLORMAP_HOT)
        cv2.imwrite(os.path.join(out_dir, f"heatmap_{cls}.png"), coloured)

    logger.info("Heatmaps saved to %s", out_dir)


# ===================================================================== #
#  QUALITY  ASSESSMENT                                                   #
# ===================================================================== #

def assess_image_quality(image_path):
    """Return a dict with quality metrics for a single image.

    * ``brightness``     – mean pixel intensity [0, 255]
    * ``contrast``       – standard deviation of intensity
    * ``sharpness``      – Laplacian variance (higher = sharper)
    * ``size``           – ``(width, height)``
    * ``aspect_ratio``   – width / height
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        return {"error": f"Cannot read {image_path}"}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    return {
        "path": str(image_path),
        "size": (w, h),
        "aspect_ratio": round(w / max(h, 1), 3),
        "brightness": round(float(gray.mean()), 2),
        "contrast": round(float(gray.std()), 2),
        "sharpness": round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2),
    }


def batch_quality_report(input_folder=INPUT_FOLDER):
    """Run quality assessment on every image in *input_folder*.

    Returns a list of quality dicts and prints a warning for outliers.
    """
    paths = collect_image_paths(input_folder, max_images=9999)
    reports = []
    for p in paths:
        q = assess_image_quality(p)
        reports.append(q)

        if "error" in q:
            logger.warning(q["error"])
            continue
        if q["sharpness"] < 50:
            logger.warning("Low sharpness (%.1f): %s", q["sharpness"], p)
        if q["brightness"] < 40 or q["brightness"] > 230:
            logger.warning("Extreme brightness (%.1f): %s", q["brightness"], p)

    return reports
