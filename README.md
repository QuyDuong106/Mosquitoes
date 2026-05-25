# Mosquitoes — YOLO training

Train **Ultralytics YOLO** on the [Kaggle mosquitoes dataset](https://www.kaggle.com/datasets/duongnguyenquy/mosquitoes-compsci760) (`duongnguyenquy/mosquitoes-compsci760`).

Label splits come from `manual_labels.csv` (70 / 15 / 15 by image, seed 42).

For **RF-DETR**, see the sibling repo: [`mosquitoes-rf-detr`](../mosquitoes-rf-detr) on Desktop.

---

## Project structure

```
mosquitoes-yolo/
├── scripts/
│   ├── convert_to_coco.py                  # CSV → COCO JSON splits
│   ├── dataset_images.py                   # Image path resolution
│   ├── convert_to_yolo.py                  # COCO → yolo_dataset/
│   ├── detection_metrics.py                # Shared overlap / P/R/F1 helpers
│   ├── train_yolo_model.py                 # End-to-end: convert + train
│   ├── test_yolo_model.py                  # End-to-end: test evaluation
│   ├── export_yolo_detection_dataset.py    # Detection-only: COCO + yolo_dataset_det/
│   ├── train_yolo_model_det.py             # Detection-only training
│   ├── test_yolo_model_det.py              # Detection-only evaluation
│   ├── run_yolo_conversion.sh              # SLURM: end-to-end COCO only (CPU)
│   ├── run_yolo_training.sh                # SLURM: end-to-end convert + train (GPU)
│   ├── run_yolo_testing.sh                 # SLURM: end-to-end evaluation (GPU)
│   ├── run_yolo_export_detection.sh        # SLURM: detection-only export (CPU)
│   ├── run_yolo_training_det_only.sh       # SLURM: detection-only train (GPU)
│   └── run_yolo_testing_det_only.sh        # SLURM: detection-only evaluation (GPU)
│
├── logs/                                   # SLURM stdout / stderr (see naming below)
├── manual_labels.csv                       # Canonical train/val/test splits
├── requirements.txt
├── .gitignore
└── README.md
```

**Run all commands from the repo root.** Python scripts write outputs relative to the current working directory. SLURM wrappers `cd` to the repo root automatically.

### SLURM log naming

All batch jobs write paired logs under `logs/` using:

`yolo-{mode}-{step}-{output|error}.log`

Test predictions (repo root or SLURM submit dir): `test_yolo_predictions-{mode}.json` where mode is `end-to-end` or `detection-only`.

| Mode | Step | stdout | stderr |
|------|------|--------|--------|
| `end-to-end` | `convert` | `logs/yolo-end-to-end-convert-output.log` | `...-error.log` |
| `end-to-end` | `train` | `logs/yolo-end-to-end-train-output.log` | `...-error.log` |
| `end-to-end` | `test` | `logs/yolo-end-to-end-test-output.log` | `...-error.log` |
| `detection-only` | `export` | `logs/yolo-detection-only-export-output.log` | `...-error.log` |
| `detection-only` | `train` | `logs/yolo-detection-only-train-output.log` | `...-error.log` |
| `detection-only` | `test` | `logs/yolo-detection-only-test-output.log` | `...-error.log` |

### Generated at runtime (repo root)

| Path | Created by |
|------|------------|
| `<dataset>/labels/train_coco.json` etc. | `convert_to_coco.py` (end-to-end) |
| `<dataset>/labels/train_coco_det.json` etc. | `convert_to_coco.py --detection-only` or `export_yolo_detection_dataset.py` |
| `yolo_dataset/` | `convert_to_yolo.py` / `train_yolo_model.py` |
| `yolo_dataset_det/` | `export_yolo_detection_dataset.py` |
| `runs/mosquito/yolo_train/` | End-to-end YOLO training |
| `runs/mosquito/yolo_train_det/` | Detection-only YOLO training |
| `test_yolo_predictions-end-to-end.json` | End-to-end testing (SLURM submit dir) |
| `test_yolo_predictions-detection-only.json` | Detection-only testing (SLURM submit dir) |

---

## Setup

**Python ≥ 3.10** and a **GPU** are recommended for training and evaluation.

```bash
cd /path/to/mosquitoes-yolo
pip install -r requirements.txt
```

### Dataset access

Provide the dataset in one of two ways:

1. **Local copy (recommended on HPC)** — point at the Kaggle Hub cache root (parent of `versions/` and `labels/`):

   ```bash
   export MOSQUITOES_DATASET=/path/to/mosquitoes-compsci760
   export MOSQUITOES_DATASET_VERSION=3   # images from versions/3/ (default)
   ```

2. **Kaggle Hub download** — set a token and let `kagglehub` fetch or use the cache:

   ```bash
   export KAGGLE_API_TOKEN="your_token_here"
   ```

Do not commit API tokens. For SLURM, use `sbatch --export=ALL` or a local `.env.slurm` file (gitignored).

### Label CSV resolution

`convert_to_coco.py` looks for `manual_labels.csv` in this order:

1. `MOSQUITOES_LABELS_CSV` environment variable
2. `./manual_labels.csv` (repo root)
3. `<dataset_root>/labels/manual_labels.csv`

---

## Data pipeline

```
manual_labels.csv  →  train/val/test COCO JSON  →  yolo_dataset/ + data.yaml
```

**Conversion** (run once, or let the train script regenerate it):

```bash
python3 scripts/convert_to_coco.py
python3 scripts/convert_to_coco.py --dataset /path/to/mosquitoes-compsci760
python3 scripts/convert_to_yolo.py --dataset /path/to/mosquitoes-compsci760
```

---

## YOLO (end-to-end, multi-class)

Same COCO splits as RF-DETR; exports `yolo_dataset/` with symlinks, YOLO `.txt` labels, and `data.yaml`.

### Local

```bash
cd /path/to/mosquitoes-yolo
export MOSQUITOES_DATASET=/path/to/dataset_root

# Train (convert → build yolo_dataset/ → YOLO.train)
python3 scripts/train_yolo_model.py
python3 scripts/train_yolo_model.py --model yolo11n.pt --epochs 80 --patience 15

# YOLO export only
python3 scripts/convert_to_yolo.py --dataset /path/to/dataset_root

# Evaluate
python3 scripts/test_yolo_model.py --weights runs/mosquito/yolo_train/weights/best.pt
python3 scripts/test_yolo_model.py --save-predictions test_yolo_predictions-end-to-end.json
```

### SLURM

```bash
cd /path/to/mosquitoes-yolo
export MOSQUITOES_DATASET=/path/to/dataset_root
export CONDA_BASE=/path/to/miniconda3

sbatch --export=ALL scripts/run_yolo_conversion.sh
sbatch --export=ALL scripts/run_yolo_training.sh
sbatch --export=ALL scripts/run_yolo_testing.sh
```

| Script | SLURM job | Logs |
|--------|-----------|------|
| `run_yolo_conversion.sh` | `mosq-yolo-convert` | `logs/yolo-end-to-end-convert-*.log` |
| `run_yolo_training.sh` | `mosq-yolo` | `logs/yolo-end-to-end-train-*.log` |
| `run_yolo_testing.sh` | `mosq-yolo-test` | `logs/yolo-end-to-end-test-*.log` |

Extra CLI args are forwarded to the Python script:

```bash
sbatch --export=ALL scripts/run_yolo_training.sh -- --model yolo11n.pt --epochs 80
```

---

## Detection-only (single mosquito class)

Same `manual_labels.csv` and train/val/test **images** as the multi-class run; only the COCO class is unified (`category_id=0`). Artifacts are separate so `yolo_dataset/` and `runs/mosquito/yolo_train/` are not overwritten.

| End-to-end (multi-class) | Detection-only |
|--------------------------|----------------|
| `train_coco.json` | `train_coco_det.json` |
| `yolo_dataset/` | `yolo_dataset_det/` |
| `runs/mosquito/yolo_train/` | `runs/mosquito/yolo_train_det/` |
| `test_yolo_predictions-end-to-end.json` | `test_yolo_predictions-detection-only.json` |

```bash
export MOSQUITOES_DATASET=/path/to/mosquitoes-compsci760
export MOSQUITOES_DATASET_VERSION=3

# Export (CPU)
python3 scripts/export_yolo_detection_dataset.py
sbatch --export=ALL scripts/run_yolo_export_detection.sh

# Train (GPU) — requires prior export
python3 scripts/train_yolo_model_det.py --skip-dataset-export
sbatch --export=ALL scripts/run_yolo_training_det_only.sh --skip-dataset-export

# Test (GPU)
python3 scripts/test_yolo_model_det.py
sbatch --export=ALL scripts/run_yolo_testing_det_only.sh
```

| Script | SLURM job | Logs |
|--------|-----------|------|
| `run_yolo_export_detection.sh` | `mosq-yolo-export-det` | `logs/yolo-detection-only-export-*.log` |
| `run_yolo_training_det_only.sh` | `mosq-yolo-train-det` | `logs/yolo-detection-only-train-*.log` |
| `run_yolo_testing_det_only.sh` | `mosq-yolo-test-det` | `logs/yolo-detection-only-test-*.log` |

---

## Environment variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `MOSQUITOES_DATASET` | All train/convert scripts | Kaggle cache root (parent of `versions/` and `labels/`) |
| `MOSQUITOES_DATASET_VERSION` | `convert_to_coco.py` | Image folder under `versions/` (default: `3`) |
| `MOSQUITOES_LABELS_CSV` | `convert_to_coco.py` | Override path to `manual_labels.csv` |
| `KAGGLE_API_TOKEN` | `convert_to_coco.py` | Download dataset via Kaggle Hub |
| `CONDA_BASE` | SLURM scripts | Miniconda root (activates `CONDA_ENV`) |
| `CONDA_ENV` | SLURM scripts | Conda env name (default: `Mosquitoes_env`) |
| `PYTHON_BIN` | SLURM scripts | Python executable (default: `python3`) |

Optional `.env.slurm` at the repo root is sourced by the SLURM scripts:

```bash
export MOSQUITOES_DATASET=/path/to/dataset_root
export KAGGLE_API_TOKEN=...   # only if needed
```

---

## HPC tips

**Disk quota** — if home is full, Matplotlib and PyTorch may warn or fall back to `/tmp`:

```bash
export MPLCONFIGDIR=/path/to/scratch/mplconfig
export XDG_CACHE_HOME=/path/to/scratch/cache
```

**SLURM env inheritance** — login-shell exports are not passed to batch jobs by default. Use `sbatch --export=ALL` or `.env.slurm`.

**Logs directory** — SLURM writes to `logs/` relative to the submit directory. The folder must exist before submitting (it is included in this repo).

---

## Script reference

| Script | Description |
|--------|-------------|
| `convert_to_coco.py` | Build COCO JSON splits from `manual_labels.csv` |
| `dataset_images.py` | Image index / path resolution (imported by others) |
| `convert_to_yolo.py` | COCO → `yolo_dataset/` |
| `detection_metrics.py` | Overlap and pooled P/R/F1 (used by `test_yolo_model.py`) |
| `train_yolo_model.py` | Convert + export + `YOLO.train()` |
| `test_yolo_model.py` | Test-split mAP + overlap metrics |
| `export_yolo_detection_dataset.py` | Detection-only COCO + `yolo_dataset_det/` |
| `train_yolo_model_det.py` | Detection-only training wrapper |
| `test_yolo_model_det.py` | Detection-only evaluation wrapper |

For YOLO, pass `--model yolo11n.pt`, `yolo11s.pt`, etc.

---

## License and data

Model code depends on **ultralytics** — governed by its own license. Mosquito images and labels come from the [Kaggle dataset](https://www.kaggle.com/datasets/duongnguyenquy/mosquitoes-compsci760); use according to its terms on Kaggle.
