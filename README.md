# Mosquitoes — RF-DETR and YOLO training

Train **RF-DETR Small** or **Ultralytics YOLO** on the [Kaggle mosquitoes dataset](https://www.kaggle.com/datasets/duongnguyenquy/mosquitoes-compsci760) (`duongnguyenquy/mosquitoes-compsci760`).

Both pipelines share the same label splits from `manual_labels.csv` (70 / 15 / 15 by image, seed 42).

---

## Project structure

```
mosquitoes-keep/
├── scripts/                        # All runnable code
│   ├── convert_to_coco.py          # Shared: CSV → COCO JSON splits
│   ├── dataset_images.py           # Shared: image path resolution
│   │
│   ├── train_rfdetr_model.py       # RF-DETR: full train pipeline
│   ├── test_rfdetr_model.py        # RF-DETR: test-split evaluation
│   ├── run_rfdetr_conversion.sh    # SLURM: COCO conversion only
│   ├── run_rfdetr_training.sh      # SLURM: convert + train
│   ├── run_rfdetr_testing.sh       # SLURM: evaluate checkpoints
│   │
│   ├── convert_to_yolo.py          # YOLO: COCO → yolo_dataset/
│   ├── train_yolo_model.py         # YOLO: full train pipeline
│   ├── test_yolo_model.py          # YOLO: test-split evaluation
│   ├── run_yolo_training.sh        # SLURM: convert + train
│   └── run_yolo_testing.sh         # SLURM: evaluate checkpoints
│
├── logs/                           # SLURM stdout / stderr
│   ├── rfdetr-train-output.log
│   ├── rfdetr-test-error.log
│   ├── yolo-train-output.log
│   └── …
│
├── manual_labels.csv               # Canonical train/val/test splits
├── requirements.txt                # Project Python dependencies
├── .gitignore
└── README.md
```

**Run all commands from the repo root.** Python scripts write outputs (`rfdetr_dataset/`, `output/`, `yolo_dataset/`, `runs/`) relative to the current working directory. SLURM wrappers `cd` to the repo root automatically.

### Generated at runtime (repo root)

| Path | Created by |
|------|------------|
| `<dataset>/labels/train_coco.json` etc. | `convert_to_coco.py` |
| `rfdetr_dataset/` | `train_rfdetr_model.py` |
| `output/` | RF-DETR training checkpoints |
| `yolo_dataset/` | `convert_to_yolo.py` / `train_yolo_model.py` |
| `runs/mosquito/yolo_train/` | YOLO training |
| `final_test_prediction.jpg` | RF-DETR training (sample inference) |
| `test_predictions.json` | RF-DETR testing (under SLURM submit dir) |

---

## Setup

**Python ≥ 3.10** and a **GPU** are recommended for training and evaluation.

```bash
cd /path/to/mosquitoes-keep
pip install -r requirements.txt          # RF-DETR + YOLO, or install selectively:
pip install rfdetr supervision torch kagglehub pandas pillow numpy   # RF-DETR
pip install ultralytics                  # YOLO (in addition to above)
```

### Dataset access

Provide the dataset in one of two ways:

1. **Local copy (recommended on HPC)** — point at the unpacked Kaggle dataset root (the folder that contains images, not `labels/` itself):

   ```bash
   export MOSQUITOES_DATASET=/path/to/mosquitoes-compsci760
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

Both model pipelines start from the same conversion step:

```
manual_labels.csv  →  train/val/test COCO JSON  →  model-specific dataset layout
                              ↓
                    RF-DETR: rfdetr_dataset/
                    YOLO:    yolo_dataset/ + data.yaml
```

**Shared conversion** (run once, or let the train scripts regenerate it):

```bash
python3 scripts/convert_to_coco.py
python3 scripts/convert_to_coco.py --dataset /path/to/mosquitoes-compsci760
```

Writes `train_coco.json`, `val_coco.json`, and `test_coco.json` under `<dataset>/labels/`.

---

## RF-DETR

Uses `RFDETRSmall` from the [`rfdetr`](https://github.com/roboflow/rf-detr) package. Training defaults to **early stopping on validation mAP** (patience 10, min delta 0.001).

### Local

```bash
cd /path/to/mosquitoes-keep
export MOSQUITOES_DATASET=/path/to/dataset_root   # or KAGGLE_API_TOKEN

# Train (convert → build rfdetr_dataset/ → train → sample inference)
python3 scripts/train_rfdetr_model.py
python3 scripts/train_rfdetr_model.py --epochs 80 --early-stopping-patience 15
python3 scripts/train_rfdetr_model.py --no-early-stopping

# Evaluate on test split
python3 scripts/test_rfdetr_model.py --weights output/checkpoint_best_total.pth
python3 scripts/test_rfdetr_model.py --max-images 200 --max-side 1280
python3 scripts/test_rfdetr_model.py --save-predictions test_predictions.json
```

### SLURM

Submit from the repo root so logs and outputs land in the right place:

```bash
cd /path/to/mosquitoes-keep
export MOSQUITOES_DATASET=/path/to/dataset_root
export CONDA_BASE=/path/to/miniconda3

sbatch --export=ALL scripts/run_rfdetr_conversion.sh   # COCO only (CPU)
sbatch --export=ALL scripts/run_rfdetr_training.sh     # full pipeline (GPU)
sbatch --export=ALL scripts/run_rfdetr_testing.sh    # evaluation (GPU)
```

| Script | SLURM job | Logs |
|--------|-----------|------|
| `run_rfdetr_conversion.sh` | `mosq-rfdetr-convert` | `logs/rfdetr-convert-*.log` |
| `run_rfdetr_training.sh` | `mosq-rfdetr` | `logs/rfdetr-train-*.log` |
| `run_rfdetr_testing.sh` | `mosq-rfdetr-test` | `logs/rfdetr-test-*.log` |

Extra CLI args are forwarded to the Python script:

```bash
sbatch --export=ALL scripts/run_rfdetr_training.sh -- --dataset /scratch/you/data --epochs 80
```

---

## YOLO (Ultralytics)

Same COCO splits as RF-DETR; exports `yolo_dataset/` with symlinks, YOLO `.txt` labels, and `data.yaml`.

### Local

```bash
cd /path/to/mosquitoes-keep
export MOSQUITOES_DATASET=/path/to/dataset_root

# Train (convert → build yolo_dataset/ → YOLO.train)
python3 scripts/train_yolo_model.py
python3 scripts/train_yolo_model.py --model yolo11n.pt --epochs 80 --patience 15

# YOLO export only
python3 scripts/convert_to_yolo.py --dataset /path/to/dataset_root

# Evaluate
python3 scripts/test_yolo_model.py --weights runs/mosquito/yolo_train/weights/best.pt
```

### SLURM

```bash
sbatch --export=ALL scripts/run_yolo_training.sh
sbatch --export=ALL scripts/run_yolo_testing.sh
```

| Script | SLURM job | Logs |
|--------|-----------|------|
| `run_yolo_training.sh` | `mosq-yolo` | `logs/yolo-train-*.log` |
| `run_yolo_testing.sh` | `mosq-yolo-test` | `logs/yolo-test-*.log` |

Default checkpoints: `runs/mosquito/yolo_train/weights/best.pt`.

---

## Environment variables

| Variable | Used by | Purpose |
|----------|---------|---------|
| `MOSQUITOES_DATASET` | All train/convert scripts | Path to dataset root (images) |
| `MOSQUITOES_LABELS_CSV` | `convert_to_coco.py` | Override path to `manual_labels.csv` |
| `KAGGLE_API_TOKEN` | `convert_to_coco.py` | Download dataset via Kaggle Hub |
| `CONDA_BASE` | SLURM scripts | Miniconda root (activates `CONDA_ENV`) |
| `CONDA_ENV` | SLURM scripts | Conda env name (default: `Mosquitoes_env`) |
| `PYTHON_BIN` | SLURM scripts | Python executable (default: `python3`) |

Optional `.env.slurm` at the repo root is sourced by the RF-DETR SLURM scripts:

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

| Script | Pipeline | Description |
|--------|----------|-------------|
| `convert_to_coco.py` | Shared | Build COCO JSON splits from `manual_labels.csv` |
| `dataset_images.py` | Shared | Image index / path resolution (imported by others) |
| `train_rfdetr_model.py` | RF-DETR | Convert + build `rfdetr_dataset/` + train |
| `test_rfdetr_model.py` | RF-DETR | Test-split mAP (supervision metrics) |
| `convert_to_yolo.py` | YOLO | COCO → `yolo_dataset/` |
| `train_yolo_model.py` | YOLO | Convert + export + `YOLO.train()` |
| `test_yolo_model.py` | YOLO | Test-split mAP via `model.val(split="test")` |

To switch RF-DETR model size, change the import and constructor in `train_rfdetr_model.py` and `test_rfdetr_model.py`. For YOLO, pass `--model yolo11n.pt`, `yolo11s.pt`, etc.

---

## License and data

Model code depends on **rf-detr**, **supervision**, and **ultralytics** — each governed by its own license. Mosquito images and labels come from the [Kaggle dataset](https://www.kaggle.com/datasets/duongnguyenquy/mosquitoes-compsci760); use according to its terms on Kaggle.
