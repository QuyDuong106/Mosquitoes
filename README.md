# Mosquitoes — RF-DETR training

Train **RF-DETR Small** on the [Kaggle mosquitoes dataset](https://www.kaggle.com/datasets/duongnguyenquy/mosquitoes-compsci760) (`duongnguyenquy/mosquitoes-compsci760`).

Label splits come from `manual_labels.csv` (70 / 15 / 15 by image, seed 42).

For **Ultralytics YOLO**, see the sibling repo: [`mosquitoes-yolo`](../mosquitoes-yolo) on Desktop.

---

## Project structure

```
mosquitoes-rf-detr/
├── scripts/
│   ├── convert_to_coco.py                  # CSV → COCO JSON splits
│   ├── dataset_images.py                   # Image path resolution
│   ├── train_rfdetr_model.py             # End-to-end: convert + train
│   ├── test_rfdetr_model.py                # End-to-end: test evaluation
│   ├── export_rfdetr_detection_dataset.py  # Detection-only: COCO + rfdetr_dataset_det/
│   ├── train_rfdetr_model_det.py           # Detection-only training
│   ├── test_rfdetr_model_det.py            # Detection-only evaluation
│   ├── run_rfdetr_conversion.sh            # SLURM: end-to-end COCO only (CPU)
│   ├── run_rfdetr_training.sh              # SLURM: end-to-end convert + train (GPU)
│   ├── run_rfdetr_testing.sh               # SLURM: end-to-end evaluation (GPU)
│   ├── run_rfdetr_export_detection.sh      # SLURM: detection-only export (CPU)
│   ├── run_rfdetr_training_det_only.sh     # SLURM: detection-only train (GPU)
│   └── run_rfdetr_testing_det_only.sh      # SLURM: detection-only evaluation (GPU)
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

`rfdetr-{mode}-{step}-{output|error}.log`

Test predictions (repo root or SLURM submit dir): `test_predictions-{mode}.json` where mode is `end-to-end` or `detection-only`.

| Mode | Step | stdout | stderr |
|------|------|--------|--------|
| `end-to-end` | `convert` | `logs/rfdetr-end-to-end-convert-output.log` | `...-error.log` |
| `end-to-end` | `train` | `logs/rfdetr-end-to-end-train-output.log` | `...-error.log` |
| `end-to-end` | `test` | `logs/rfdetr-end-to-end-test-output.log` | `...-error.log` |
| `detection-only` | `export` | `logs/rfdetr-detection-only-export-output.log` | `...-error.log` |
| `detection-only` | `train` | `logs/rfdetr-detection-only-train-output.log` | `...-error.log` |
| `detection-only` | `test` | `logs/rfdetr-detection-only-test-output.log` | `...-error.log` |

### Generated at runtime (repo root)

| Path | Created by |
|------|------------|
| `<dataset>/labels/train_coco.json` etc. | `convert_to_coco.py` (end-to-end) |
| `<dataset>/labels/train_coco_det.json` etc. | `convert_to_coco.py --detection-only` or `export_rfdetr_detection_dataset.py` |
| `rfdetr_dataset/` | `train_rfdetr_model.py` |
| `rfdetr_dataset_det/` | `export_rfdetr_detection_dataset.py` |
| `output/` | End-to-end RF-DETR checkpoints |
| `output_det/` | Detection-only RF-DETR checkpoints |
| `final_test_prediction.jpg` | RF-DETR training (sample inference) |
| `test_predictions-end-to-end.json` | End-to-end testing (SLURM submit dir) |
| `test_predictions-detection-only.json` | Detection-only testing (SLURM submit dir) |

---

## Setup

**Python ≥ 3.10** and a **GPU** are recommended for training and evaluation.

```bash
cd /path/to/mosquitoes-rf-detr
pip install -r requirements.txt
```

### Dataset access

Provide the dataset in one of two ways:

1. **Local copy (recommended on HPC)** — point at the unpacked Kaggle dataset root (the folder that contains images, not `labels/` itself):

   ```bash
   export MOSQUITOES_DATASET=/path/to/mosquitoes-compsci760   # kagglehub cache root
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
manual_labels.csv  →  train/val/test COCO JSON  →  rfdetr_dataset/
```

**Conversion** (run once, or let the train script regenerate it):

```bash
python3 scripts/convert_to_coco.py
python3 scripts/convert_to_coco.py --dataset /path/to/mosquitoes-compsci760
```

Writes `train_coco.json`, `val_coco.json`, and `test_coco.json` under `<dataset>/labels/`.

---

## RF-DETR (end-to-end, multi-class)

Uses `RFDETRSmall` from the [`rfdetr`](https://github.com/roboflow/rf-detr) package. Training defaults to **early stopping on validation mAP** (patience 10, min delta 0.001).

### Local

```bash
cd /path/to/mosquitoes-rf-detr
export MOSQUITOES_DATASET=/path/to/dataset_root   # or KAGGLE_API_TOKEN

# Train (convert → build rfdetr_dataset/ → train → sample inference)
python3 scripts/train_rfdetr_model.py
python3 scripts/train_rfdetr_model.py --epochs 80 --early-stopping-patience 15
python3 scripts/train_rfdetr_model.py --no-early-stopping

# Evaluate on test split
python3 scripts/test_rfdetr_model.py --weights output/checkpoint_best_total.pth
python3 scripts/test_rfdetr_model.py --max-images 200 --max-side 1280
python3 scripts/test_rfdetr_model.py --save-predictions test_predictions-end-to-end.json
```

### SLURM

Submit from the repo root so logs and outputs land in the right place:

```bash
cd /path/to/mosquitoes-rf-detr
export MOSQUITOES_DATASET=/path/to/dataset_root
export CONDA_BASE=/path/to/miniconda3

sbatch --export=ALL scripts/run_rfdetr_conversion.sh   # COCO only (CPU)
sbatch --export=ALL scripts/run_rfdetr_training.sh     # full pipeline (GPU)
sbatch --export=ALL scripts/run_rfdetr_testing.sh      # evaluation (GPU)
```

| Script | SLURM job | Logs |
|--------|-----------|------|
| `run_rfdetr_conversion.sh` | `mosq-rfdetr-convert` | `logs/rfdetr-end-to-end-convert-*.log` |
| `run_rfdetr_training.sh` | `mosq-rfdetr` | `logs/rfdetr-end-to-end-train-*.log` |
| `run_rfdetr_testing.sh` | `mosq-rfdetr-test` | `logs/rfdetr-end-to-end-test-*.log` |

Extra CLI args are forwarded to the Python script:

```bash
sbatch --export=ALL scripts/run_rfdetr_training.sh -- --dataset /scratch/you/data --epochs 80
```

---

## Detection-only (single mosquito class)

Same `manual_labels.csv` and train/val/test **images** as the multi-class run; only the COCO class is unified (`category_id=0`). Artifacts are separate so `output/` and `rfdetr_dataset/` are not overwritten.

| End-to-end (multi-class) | Detection-only |
|--------------------------|----------------|
| `train_coco.json` | `train_coco_det.json` |
| `rfdetr_dataset/` | `rfdetr_dataset_det/` |
| `output/` | `output_det/` |
| `test_predictions-end-to-end.json` | `test_predictions-detection-only.json` |

```bash
export MOSQUITOES_DATASET=/path/to/mosquitoes-compsci760
export MOSQUITOES_DATASET_VERSION=3

# Export (CPU)
python3 scripts/export_rfdetr_detection_dataset.py
sbatch --export=ALL scripts/run_rfdetr_export_detection.sh

# Train (GPU) — requires prior export
python3 scripts/train_rfdetr_model_det.py --skip-dataset-export
sbatch --export=ALL scripts/run_rfdetr_training_det_only.sh --skip-dataset-export

# Test (GPU)
python3 scripts/test_rfdetr_model_det.py
sbatch --export=ALL scripts/run_rfdetr_testing_det_only.sh
```

| Script | SLURM job | Logs |
|--------|-----------|------|
| `run_rfdetr_export_detection.sh` | `mosq-rfdetr-export-det` | `logs/rfdetr-detection-only-export-*.log` |
| `run_rfdetr_training_det_only.sh` | `mosq-rfdetr-train-det` | `logs/rfdetr-detection-only-train-*.log` |
| `run_rfdetr_testing_det_only.sh` | `mosq-rfdetr-test-det` | `logs/rfdetr-detection-only-test-*.log` |

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
| `train_rfdetr_model.py` | Convert + build `rfdetr_dataset/` + train |
| `test_rfdetr_model.py` | Test-split mAP (supervision metrics) |
| `export_rfdetr_detection_dataset.py` | Detection-only COCO + `rfdetr_dataset_det/` |
| `train_rfdetr_model_det.py` | Detection-only training → `output_det/` |
| `test_rfdetr_model_det.py` | Detection-only evaluation |
| `run_rfdetr_conversion.sh` | SLURM: end-to-end COCO conversion |
| `run_rfdetr_training.sh` | SLURM: end-to-end train |
| `run_rfdetr_testing.sh` | SLURM: end-to-end test |
| `run_rfdetr_export_detection.sh` | SLURM: detection-only export |
| `run_rfdetr_training_det_only.sh` | SLURM: detection-only train |
| `run_rfdetr_testing_det_only.sh` | SLURM: detection-only test |

To switch RF-DETR model size, change the import and constructor in `train_rfdetr_model.py` and `test_rfdetr_model.py`.

---

## License and data

Model code depends on **rf-detr** and **supervision** — each governed by its own license. Mosquito images and labels come from the [Kaggle dataset](https://www.kaggle.com/datasets/duongnguyenquy/mosquitoes-compsci760); use according to its terms on Kaggle.
