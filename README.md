# FloorplanToBlender3D

Convert 2D floorplan images into 3D models using computer vision and Blender.

The pipeline detects structural elements — walls, doors, windows, rooms and floors — from a scanned or digital floorplan image, then reconstructs the layout as a full 3D object that can be opened, edited and rendered in Blender.

---

## What It Produces

| Output | Description |
|--------|-------------|
| **3D Blender file** (`.blend`) | A scene with extruded walls, floors, rooms, doors and windows ready for rendering or further modelling. |
| **OBJ / FBX export** | Optional mesh export via Blender's format-conversion scripts. |
| **Detection JSON** | Per-image JSON with bounding boxes, class labels and confidence scores for every detected element. |
| **Annotated images** | The original floorplan overlaid with colour-coded bounding boxes and segmentation masks. |
| **Batch reports** | Aggregated statistics, class distribution and quality metrics when processing entire folders. |

---

## Project Structure

```
├── main.py                        # Interactive single-image entry point
├── mycall.py                      # Unified inference (cloud API or local CNN)
├── mod.py                         # Batch processing with multiprocessing
├── train.py                       # Model training script
│
├── FloorplanToBlenderLib/
│   ├── model.py                   # CNN architectures (U-Net, multi-task)
│   ├── dataset.py                 # Dataset classes, augmentations, loaders
│   ├── inference.py               # Production inference engine (TTA, ensemble)
│   ├── floorplan.py               # Floorplan data model
│   ├── execution.py               # Orchestration (single & stacked runs)
│   ├── generate.py                # 3D geometry generation
│   ├── detect.py                  # Wall detection (OpenCV morphological ops)
│   ├── transform.py               # Spatial transforms
│   ├── config.py                  # Configuration management
│   └── ...                        # IO, drawing, stacking utilities
│
├── Blender/
│   ├── floorplan_to_3dObject_in_blender.py   # Blender mesh builder (bpy)
│   ├── blender_export_any.py                 # Format conversion
│   └── ...
│
├── Configs/
│   ├── default.ini                # Image & feature settings
│   └── system.ini                 # Blender path, output format
│
├── Images/Examples/               # Sample floorplan images
└── Stacking/                      # Multi-floorplan arrangement examples
```

---

## How It Works

1. **Input** — A 2D floorplan image (PNG, JPG or SVG).
2. **Detection** — Structural elements are detected using either:
   - A local multi-task CNN (segmentation + depth estimation) built with PyTorch.
3. **Geometry generation** — Detected elements are converted into vertices, edges and faces by the generation pipeline (`generate.py` → `generator.py`).
4. **3D reconstruction** — Blender reads the generated geometry data and builds a mesh scene with walls, floors, rooms, doors and windows.
5. **Export** — The result is saved as a `.blend` file (default) or exported to OBJ, FBX, etc.

---

## Getting Started

### Prerequisites

- **Python 3.8+**
- **Blender 2.80+** (must be installed; the path is configured in `Configs/system.ini`)
- Python packages:

```
numpy
opencv-python
scipy
torch
torchvision
inference-sdk
```

### Run a single floorplan

```bash
python main.py
```

You will be prompted for:
- The path to your Blender installation (auto-detected on first run).
- The path to the floorplan image.

The 3D model is written to the `Data/` directory.

### Run inference only (no Blender)

```bash
python mycall.py --input path/to/image.png --output output_images/
```

Flags:
- `--api` / `--local` — force cloud or local CNN backend.
- `--confidence 0.4` — minimum detection confidence.
- `--batch` — process an entire folder.

### Batch processing

```bash
python mod.py --input input_images/ --output output_images/ --workers 4
```

Processes every image in the folder in parallel and generates per-image JSONs, annotated images and a summary report.

### Train the local CNN

```bash
python train.py --data_root datasets/floorplans --epochs 80 --batch_size 8
```

Supports mixed-precision training, cosine-annealing schedule, early stopping and periodic checkpointing.

---

## Configuration

### `Configs/default.ini`

| Section | Key | Description |
|---------|-----|-------------|
| `IMAGE` | `image_path` | Default input image |
| `TRANSFORM` | `position`, `rotation`, `scale`, `margin` | 3D placement of the generated model |
| `FEATURES` | `floors`, `rooms`, `walls`, `doors`, `windows` | Toggle which elements to detect and generate |
| `EXTRA_SETTINGS` | `remove_noise`, `rescale_image` | Pre-processing flags |
| `WALL_CALIBRATION` | `wall_size_calibration` | Reference-based wall thickness correction |

### `Configs/system.ini`

| Key | Description |
|-----|-------------|
| `blender_installation_path` | Absolute path to the Blender executable |
| `out_format` | Output format (`.blend`, `.obj`, `.fbx`, etc.) |
| `overwrite_data` | Whether to overwrite previous output |

---

## Multi-Floorplan Stacking

Multiple floorplans can be combined into a single scene (e.g. one per storey). Example configuration files are in `Stacking/`:

| File | Layout |
|------|--------|
| `simple_example.txt` | Basic vertical stack |
| `axis_example.txt` | Aligned along an axis with offsets |
| `multiple_example.txt` | Several plans with independent transforms |
| `cylinder_example.txt` | Circular arrangement |
| `all_separated_example.txt` | Each plan placed individually |

---

## Detection Classes

The model recognises 14 classes:

| ID | Class |
|----|-------|
| 0 | background |
| 1 | space_balconi |
| 2 | space_bedroom |
| 3 | space_corridor |
| 4 | space_dining |
| 5 | space_kitchen |
| 6 | space_laundry |
| 7 | space_living |
| 8 | space_lobby |
| 9 | space_office |
| 10 | space_other |
| 11 | space_parking |
| 12 | space_staircase |
| 13 | space_toilet |

---

## Supported Input Formats

- PNG
- JPG / JPEG
- SVG (automatically rasterised to PNG)

---

## License

See repository for licence details.
