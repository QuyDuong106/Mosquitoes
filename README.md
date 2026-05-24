# Mosquitoes — RF-DETR training

Train **RF-DETR Small** on the [Kaggle mosquitoes dataset](https://www.kaggle.com/datasets/duongnguyenquy/mosquitoes-compsci760) (`duongnguyenquy/mosquitoes-compsci760`).

Label splits come from `manual_labels.csv` (70 / 15 / 15 by image, seed 42).

For **Ultralytics YOLO**, see the sibling repo: [`mosquitoes-yolo`](../mosquitoes-yolo) on Desktop.

---

## Project structure

```
mosquitoes-rf-detr/
├── scripts/
│   ├── convert_to_coco.py          # CSV → COCO JSON splits
│   ├── dataset_images.py           # Image path resolution
│   ├── train_rfdetr_model.py       # Full train pipeline
│   ├── test_rfdetr_model.py        # Test-split evaluation
│   ├── run_rfdetr_conversion.sh    # SLURM: COCO conversion only
│   ├── run_rfdetr_training.sh      # SLURM: convert + train
│   └── run_rfdetr_testing.sh       # SLURM: evaluate checkpoints
│
├── logs/                           # SLURM stdout / stderr
├── manual_labels.csv               # Canonical train/val/test splits
├── requirements.txt
├── .gitignore
└── README.md
```

**Run all commands from the repo root.** Python scripts write outputs (`rfdetr_dataset/`, `output/`) relative to the current working directory. SLURM wrappers `cd` to the repo root automatically.

### Generated at runtime (repo root)

| Path | Created by |
|------|------------|
| `<dataset>/labels/train_coco.json` etc. | `convert_to_coco.py` |
| `rfdetr_dataset/` | `train_rfdetr_model.py` |
| `output/` | RF-DETR training checkpoints |
| `final_test_prediction.jpg` | RF-DETR training (sample inference) |
| `test_predictions.json` | RF-DETR testing (under SLURM submit dir) |

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

## RF-DETR

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
python3 scripts/test_rfdetr_model.py --save-predictions test_predictions.json
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
| `run_rfdetr_conversion.sh` | `mosq-rfdetr-convert` | `logs/rfdetr-convert-*.log` |
| `run_rfdetr_training.sh` | `mosq-rfdetr` | `logs/rfdetr-train-*.log` |
| `run_rfdetr_testing.sh` | `mosq-rfdetr-test` | `logs/rfdetr-test-*.log` |

Extra CLI args are forwarded to the Python script:

```bash
sbatch --export=ALL scripts/run_rfdetr_training.sh -- --dataset /scratch/you/data --epochs 80
```

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

To switch RF-DETR model size, change the import and constructor in `train_rfdetr_model.py` and `test_rfdetr_model.py`.

---

## License and data

Model code depends on **rf-detr** and **supervision** — each governed by its own license. Mosquito images and labels come from the [Kaggle dataset](https://www.kaggle.com/datasets/duongnguyenquy/mosquitoes-compsci760); use according to its terms on Kaggle.
