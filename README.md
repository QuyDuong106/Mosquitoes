# Mosquito species classification (COMPSCI 760)

This repository includes coursework and tooling around **mosquito imagery**. The narrative below matches the team’s main classification experiment in [`compsci760.ipynb`](compsci760.ipynb).

## Purpose of the study

Public-health and ecological monitoring often need to know **which mosquito species** are present in an area, because species differ in disease vector potential, habitat, and control options. Manual identification from field photos is slow and requires expertise.

This notebook addresses **automated species recognition from photographs**: given a labeled image dataset, we build a **multi-class image classifier** that predicts one of several mosquito species from the full image. The goal is to explore **deep learning with imbalanced classes and the noisy environment** (some species are much rarer than others) and to report **standard classification metrics** so performance can be compared across models and splits.

This is **image-level classification** (one label per image), not object detection (Detection + classification is done in the other branch): training uses whole images resized to a fixed input size, with labels derived from the annotation table per filename.

## Dataset (as used in the notebook)

- **Source layout (Kaggle):** images under `images/images`, labels in `labels/annotations.csv` (see the `dataset_dir` path inside the notebook; adjust if you run locally).
- **Annotations:** CSV columns include image filename, image dimensions, bounding-box coordinates (`bbx_*`), and `class_label`. For modeling, the notebook collapses to **one row per image** by taking the **modal** `class_label` when multiple rows exist for the same file.
- **Scale:** on the order of **~10k unique images** in the captured run, with **six species** (example class names in the notebook: *aegypti*, *albopictus*, *anopheles*, *culex*, *culiseta*, *japonicus-koreicus*). Counts are **highly imbalanced** (a few dominant classes and long-tailed rare classes).

## Modeling approach (summary)

- **Split:** stratified train / validation / test (e.g. 70% / 15% / 15% with a fixed `random_state` for reproducibility).
- **Class imbalance:** inverse-frequency **class weights** mapped to per-sample weights and a **`WeightedRandomSampler`** on the training loader so minority classes are seen more often during training.
- **Architecture:** **EfficientNet-B0** with ImageNet-1k pretrained weights; the **feature backbone is frozen** and only the **classification head** is replaced and trained for six classes.
- **Training:** cross-entropy loss, Adam with a small learning rate, multiple epochs with checkpointing and best-model selection based on validation performance (see notebook for exact schedule and logging).
- **Input pipeline:** resize to 224×224, ImageNet normalization; light **data augmentation** on the training set (flips, small rotation, color jitter).
- **Evaluation:** accuracy, precision/recall/F1 (including macro views where applicable), **confusion matrix**, and **one-vs-rest ROC / ROC-AUC** for multi-class discrimination.

This notebook also defines a **small per-class subset** of the splits (capped samples per class) for quicker experiments; the default training cells use the **full** train/val/test data loaders unless you switch the commented lines.

## How to run

- Open **`compsci760.ipynb`** in Jupyter or run it on **Kaggle** after attaching the same dataset bundle and aligning paths (`dataset_dir`, `image_dir`, `annotation_path`).
- Requires **Python ≥ 3.10**-style usage in spirit (the notebook was executed with recent PyTorch/CUDA where available).

## Related code in this repo

Other scripts (e.g. `train_mosquito_model.py`, `test_mosquito_model.py`, `convert_to_coco.py`) target **object detection / COCO-style** workflows for mosquitoes and are **not** the same task as the six-way species classifier in the notebook; they can complement a pipeline (e.g. detect then crop then classify) but are maintained as separate tracks unless you wire them together explicitly.

---

*If you reuse this README, update dataset paths, author list, and institutional wording to match your course submission.*
