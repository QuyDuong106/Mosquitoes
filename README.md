# Mosquitoes — RF-DETR training and evaluation

Train an **RF-DETR Small** detector on the [Kaggle mosquitoes dataset](https://www.kaggle.com/datasets/duongnguyenquy/mosquitoes-compsci760) (`duongnguyenquy/mosquitoes-compsci760`), using COCO-style splits and Roboflow-compatible layout for the `rfdetr` training API.

## What this repo does

1. **Download / resolve the dataset** via [Kaggle Hub](https://github.com/Kaggle/kagglehub) (cached locally).
2. **Convert** `labels/annotations.csv` into three COCO JSON files: `train_coco.json`, `val_coco.json`, `test_coco.json` under `labels/`.
3. **Build** `rfdetr_dataset/` in the current working directory: `train/`, `valid/`, `test/` each with `_annotations.coco.json` and image symlinks (RF-DETR expects `valid` for validation).
4. **Train** `RFDETRSmall` and write checkpoints under `output/` (names depend on the `rfdetr` version).
5. **Evaluate** on the test split with **mAP** (supervision metrics).

## Requirements

- Python 3 with GPU recommended for training and testing.
- Packages used in code: `rfdetr`, `torch`, `supervision`, `Pillow`, `numpy`, `pandas`, `kagglehub`.

Install the training stack in your environment (exact pins are up to you), for example:

```bash
pip install rfdetr supervision torch kagglehub pandas pillow numpy
```

You also need a **Kaggle API token** with access to the dataset so `kagglehub` can download or use the cache. Set it in the environment (do not commit tokens to git):

```bash
export KAGGLE_API_TOKEN="your_token_here"
```

## Project layout (after a full train run)

| Path | Description |
|------|-------------|
| Kaggle cache | Directory returned by `kagglehub.dataset_download("duongnguyenquy/mosquitoes-compsci760")` (machine-specific). |
| `<cache>/labels/annotations.csv` | Source annotations. |
| `<cache>/labels/train_coco.json` etc. | Generated COCO splits from `convert_to_coco.py`. |
| `./rfdetr_dataset/` | Symlinked images + `_annotations.coco.json` per split; **recreated** each time `train_mosquito_model.py` runs. |
| `./output/` | Trained checkpoints (used by `test_mosquito_model.py` when `--weights` is omitted). |
| `final_test_prediction.jpg` | One sample visualization from `train_mosquito_model.py` after training. |

## Scripts

### `convert_to_coco.py`

Regenerates COCO JSON splits from `labels/annotations.csv`. The CLI entrypoint calls `convert_and_split_csv()` with no arguments, which downloads the dataset via Kaggle Hub and writes `train_coco.json`, `val_coco.json`, and `test_coco.json` under that dataset’s `labels/` directory.

```bash
export KAGGLE_API_TOKEN=...
python3 convert_to_coco.py
```

To use a dataset that is already on disk, call from Python: `convert_and_split_csv("/path/to/mosquitoes-compsci760")`.

### `train_mosquito_model.py`

Downloads the dataset (unless paths are changed in code), runs `convert_and_split_csv`, builds `rfdetr_dataset/`, trains **50 epochs** at **lr=1e-4** with `RFDETRSmall()`, then runs a single inference on the first test image.

```bash
cd /path/to/Mosquitoes
export KAGGLE_API_TOKEN=...
python3 train_mosquito_model.py
```

### `test_mosquito_model.py`

Computes COCO-style mAP on `rfdetr_dataset/test` (or `--test-dir`). Useful flags: `--weights`, `--max-images`, `--max-side`, `--save-sample`.

```bash
python3 test_mosquito_model.py --weights output/checkpoint_best_total.pth
python3 test_mosquito_model.py --max-images 200 --max-side 1280
```

## SLURM helpers

The `run_*.sh` scripts are examples: update `cd`, conda `activate`, and **use environment variables for secrets** instead of hardcoding tokens in committed files.

- `run_conversion.sh` — Hub + `convert_to_coco.py`
- `run_training.sh` — full training pipeline
- `run_testing.sh` — passes extra args through: `sbatch run_testing.sh -- --weights ...`

## HPC / disk quota tips

If your home directory quota is full, Matplotlib and PyTorch may warn or fall back to `/tmp`. Point caches at scratch, for example:

```bash
export MPLCONFIGDIR=/path/to/scratch/mplconfig
export XDG_CACHE_HOME=/path/to/scratch/cache
```

## Model variant

Training and testing use **`RFDETRSmall`** from `rfdetr`, not Nano. Switching size would mean changing imports and constructors in `train_mosquito_model.py` and `test_mosquito_model.py` to match your `rfdetr` package’s API.

## License / data

Model code is governed by the **rf-detr** and **supervision** licenses. The mosquito images and labels come from the Kaggle dataset above; use it according to that dataset’s terms on Kaggle.
