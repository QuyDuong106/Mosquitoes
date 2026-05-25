# Explainable AI

This branch contains the xAI workflow for interpreting the mosquito species classifier. The script generates Integrated Gradients and SmoothGrad visualizations to show which image regions influence the model prediction.

## Purpose

The xAI analysis helps check whether the classifier focuses on mosquito morphology rather than background artifacts. It is used after model selection to inspect correct, incorrect, and ambiguous predictions from the best fine-grained classification model.

## Method

- **Integrated Gradients:** highlights pixels that contribute to the predicted class relative to a baseline image.
- **SmoothGrad:** averages saliency maps over noisy copies of the same image to reduce visual noise and produce a smoother attribution map.

## Best Model

The xAI runner is configured for the best classification checkpoint from the new-label experiment:

```text
resnet50_weighted_loss_lr0.0005_bs32_g1.0_winverse
```

This model was selected from the final new-label classification results using macro-F1 as the main metric.

## Required Files

The script expects:

```text
/Users/quyduong106/projects/COMPSCI 760/image_crop_new/image_crop_new/cropped_annotations.csv
/Users/quyduong106/projects/COMPSCI 760/image_crop_new/image_crop_new/<crop image files>
```

It also expects the best checkpoint at:

```text
results on new labels/final results - top methods from phase1&2 ver/weighted_loss_winverse_lr0.0005.pth
```

If the checkpoint is not present on this branch, bring it from the classification branch:

```bash
git checkout fine-grained-classification/imbalance-methods -- "results on new labels/final results - top methods from phase1&2 ver/weighted_loss_winverse_lr0.0005.pth"
```

## Run

Run xAI on stratified test samples:

```bash
python xai/run_xai.py
```

Run xAI on one specific crop image:

```bash
python xai/run_xai.py \
  --image-path "/Users/quyduong106/projects/COMPSCI 760/image_crop_new/image_crop_new/train_00000_crop.jpeg"
```

## Outputs

Outputs are saved under:

```text
xai_outputs/
```

The script saves:

- Integrated Gradients heatmaps
- SmoothGrad heatmaps
- side-by-side panels with original image, IG, and SmoothGrad
- stratified split CSV files
- `xai_summary.csv`

## Notes

Generated output folders, checkpoint files, and cache files should not be committed unless they are intentionally needed for release.
