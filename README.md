# YOLO11 on VisDrone-DET: training dynamics, controlled experiments and multi-version comparison

This repo trains a modern YOLO detector on VisDrone-DET (small, dense, real-world drone objects), analyses its loss curves, runs a controlled experiment and a regularisation improvement cycle, and compares YOLO11 against YOLOv8.

## 1. Objective
Understand the YOLO detection pipeline, fine-tune YOLO11, diagnose over/underfitting from training vs validation loss, and quantify the effect of model size, regularisation and YOLO version.

## 2. Setup
| Item | Value |
|---|---|
| Dataset | VisDrone2019-DET (10 classes, 6,471 train / 548 val images), downloaded automatically by Ultralytics (`VisDrone.yaml`) |
| Framework | Ultralytics (YOLO11 / YOLOv8), PyTorch |
| Training Hardware | NVIDIA GeForce RTX 3060 Laptop GPU (6 GB) |
| Metrics | Precision, Recall, mAP50, mAP50-95 on the full validation set (`best.pt`), inference ms/img |

## 3. How the pipeline works
1. `VisDrone.yaml` downloads the images and converts VisDrone annotations to YOLO format (class, x, y, w, h normalised).
2. YOLO11 = backbone (C3k2 blocks + SPPF + C2PSA attention) -> neck (multi-scale feature fusion, P3/P4/P5) -> decoupled anchor-free head. Small objects are detected on the high-resolution P3 map (stride 8).
3. Loss = box loss (CIoU) + class loss (BCE) + DFL (distribution focal loss for box edges). We report their sum as "total loss".
4. `src/run_experiments.py` trains each configuration; `src/analyze.py` plots curves, builds tables and writes the analysis; `src/build_readme.py` assembles this README.

## 4. Experiments
| Run | Change from baseline | Purpose |
|---|---|---|
| `baseline` | YOLO11n, imgsz 640 | Part I |
| `size_s` | YOLO11s (more parameters) | Part III: capacity |
| `regularized` | cosine LR, weight decay 0.002 (4x), scale 0.7, mixup 0.1, hsv_s 0.8 | Part IV: overfitting control |
| `yolov8n` | YOLOv8n | Part V: version comparison |

## Results (YOLO11, VisDrone-DET, 20% of train images, 12 epochs, imgsz 640, seed 0)

### Part I - Baseline

| Run | Weights | Params (M) | Precision | Recall | mAP50 | mAP50-95 | Infer ms/img | Train min |
|---|---|---|---|---|---|---|---|---|
| baseline | yolo11n.pt | 2.61 | 0.352 | 0.266 | 0.224 | 0.121 | 2.06 | 7.0 |

![baseline loss](figures/baseline_loss_curves.png)
![baseline metrics](figures/baseline_metrics.png)
![confusion matrix](figures/baseline_confusion_matrix.png)
![PR curve](figures/baseline_pr_curve.png)

### Part II - Loss-curve and fitting analysis

Final-epoch train total loss = 3.756, validation total loss = 3.776 (val/train ratio 1.01). Over the last 5 epochs the validation-loss slope is -0.0270/epoch, training-loss slope -0.0393/epoch, and mAP50 slope +0.0039/epoch. **Diagnosis: underfitting / not yet converged (validation loss and mAP50 still improving).**

*Dataset size.* Training uses VisDrone-DET dataset (about 1294 images) with many tiny, densely packed objects. A small training set makes overfitting more likely when training for long, because the model can memorise it. Here the val/train loss ratio is 1.01, so there is no overfitting at 12 epochs: the model has not made enough passes over the data to memorise it. Validation loss can sit at or below training loss early in training because the training loss is averaged over the epoch while the weights are still improving, and it is measured on augmented images (mosaic, scale, colour jitter) that are harder than the clean validation images. *Model capacity.* YOLO11n has 2.61 M parameters. Training and validation loss fall together and are both still falling at the last epoch, which fits underfitting (not enough training or capacity) rather than overfitting. A larger model would help if capacity is the bottleneck, which the size experiment below tests.

### Part III - Controlled experiment: model capacity (only variable: n -> s)

| Run | Weights | Params (M) | Precision | Recall | mAP50 | mAP50-95 | Infer ms/img | Train min |
|---|---|---|---|---|---|---|---|---|
| baseline | yolo11n.pt | 2.61 | 0.352 | 0.266 | 0.224 | 0.121 | 2.06 | 7.0 |
| size_s | yolo11s.pt | 9.43 | 0.449 | 0.340 | 0.315 | 0.177 | 3.63 | 6.5 |

Changing YOLO11n to YOLO11s (2.61 M -> 9.43 M parameters) changed mAP50 by +0.091 (0.224 -> 0.315) and inference time from 2.06 to 3.63 ms/img. The gain indicates the baseline is capacity-limited on small objects, at the cost of speed.

### Part IV - Improvement cycle: regularisation

**Baseline** (above) -> **Experimental settings** identical except -> **Controlled modification**: `{'cos_lr': True, 'weight_decay': 0.002, 'scale': 0.7, 'mixup': 0.1, 'hsv_s': 0.8, 'fliplr': 0.5}` (cosine LR decay, 4x weight decay, stronger scale/colour augmentation, mixup 0.1). Rationale: weight decay and augmentation reduce variance (overfitting) and cosine decay lets the optimiser settle into a flatter minimum at the end of a short schedule.

| Run | Weights | Params (M) | Precision | Recall | mAP50 | mAP50-95 | Infer ms/img | Train min |
|---|---|---|---|---|---|---|---|---|
| baseline | yolo11n.pt | 2.61 | 0.352 | 0.266 | 0.224 | 0.121 | 2.06 | 7.0 |
| regularized | yolo11n.pt | 2.61 | 0.358 | 0.256 | 0.223 | 0.122 | 1.9 | 3.4 |

**Evaluation / analysis**: mAP50 0.224 -> 0.223 (-0.002); val/train loss ratio 1.01 -> 1.00; diagnosis now: underfitting / not yet converged (validation loss and mAP50 still improving). **Conclusion**: no measurable difference (-0.002 mAP50 is within single-seed noise). The baseline was not overfitting (val/train loss ratio 1.01), so regularisation had little to correct. Hypothesis, not tested here: it would matter more with longer training on this small subset, once overfitting appears.

### Part V - Multi-version comparison (YOLO11n vs YOLOv8n)

| Run | Weights | Params (M) | Precision | Recall | mAP50 | mAP50-95 | Infer ms/img | Train min |
|---|---|---|---|---|---|---|---|---|
| baseline | yolo11n.pt | 2.61 | 0.352 | 0.266 | 0.224 | 0.121 | 2.06 | 7.0 |
| yolov8n | yolov8n.pt | 3.14 | 0.346 | 0.269 | 0.226 | 0.122 | 2.08 | 3.3 |

YOLOv8n reaches mAP50 0.226 vs 0.224 for YOLO11n, with 3.14 M vs 2.61 M parameters and 2.08 vs 2.06 ms/img inference.

### All runs

| Run | Weights | Params (M) | Precision | Recall | mAP50 | mAP50-95 | Infer ms/img | Train min |
|---|---|---|---|---|---|---|---|---|
| baseline | yolo11n.pt | 2.61 | 0.352 | 0.266 | 0.224 | 0.121 | 2.06 | 7.0 |
| size_s | yolo11s.pt | 9.43 | 0.449 | 0.340 | 0.315 | 0.177 | 3.63 | 6.5 |
| regularized | yolo11n.pt | 2.61 | 0.358 | 0.256 | 0.223 | 0.122 | 1.9 | 3.4 |
| yolov8n | yolov8n.pt | 3.14 | 0.346 | 0.269 | 0.226 | 0.122 | 2.08 | 3.3 |

*Note on the last two columns:* they come from one laptop GPU that was also running other processes, so they are only indicative. Models of identical size differ more than their architecture explains (baseline vs regularized: 7.0 vs 3.4 training minutes, 2.06 vs 1.90 ms/img), and the baseline time includes about 3 minutes of one-off start-up work. Compare accuracy columns, and compare speed only for large differences such as YOLO11n vs YOLO11s.


![comparison](figures/comparison.png)


## 5. Reproduce
```bash
pip install ultralytics
python src/run_experiments.py      # trains all 4 runs (~3-7 min each on an RTX 3060 laptop GPU)
python src/analyze.py              # figures + results/RESULTS.md
python src/build_readme.py         # rebuilds this README
```
Or open `notebooks/yolo11_visdrone_colab.ipynb` in Google Colab (Runtime -> T4 GPU -> Run all); the notebook is the same pipeline but was not used for the results reported here.

## 6. Limitations and next steps
Reduced training budget (20% data, 12 epochs); single seed, so differences under ~0.005 mAP50 are within noise; the optional VisDrone test-challenge submission was not run. Next: full data, 100 epochs, imgsz 960 for small objects, multiple seeds.
