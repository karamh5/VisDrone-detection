# YOLO11 on VisDrone-DET: training dynamics, controlled experiments and multi-version comparison

CMPE 401 Instructor-defined Project 1. This repo trains a modern YOLO detector on VisDrone-DET (small, dense, real-world drone objects), analyses its loss curves, runs a controlled experiment and a regularisation improvement cycle, and compares YOLO11 against YOLOv8. Every number below is produced by the code in `src/`; nothing is typed by hand.

## 1. Objective
Understand the YOLO detection pipeline, fine-tune YOLO11, diagnose over/underfitting from training vs validation loss, and quantify the effect of model size, regularisation and YOLO version.

## 2. Setup
| Item | Value |
|---|---|
| Dataset | VisDrone2019-DET (10 classes, 6,471 train / 548 val images), downloaded automatically by Ultralytics (`VisDrone.yaml`) |
| Framework | Ultralytics (YOLO11 / YOLOv8), PyTorch |
| Training budget | 20% of train images (`fraction=0.2`), 12 epochs, batch 16, imgsz 640, seed 0, single NVIDIA GeForce RTX 3060 Laptop GPU (6 GB) |
| Metrics | Precision, Recall, mAP50, mAP50-95 on the full validation set (`best.pt`), inference ms/img |
| Why a reduced budget | The whole study had to fit in a few GPU-minutes per run. Absolute mAP is therefore lower than a full 100+ epoch run; all comparisons use the same budget, so relative conclusions are fair. Raise `EPOCHS`/`FRACTION` in `src/run_experiments.py` for the full run. |

## 3. How the pipeline works
1. `VisDrone.yaml` downloads the images and converts VisDrone annotations to YOLO format (class, x, y, w, h normalised).
2. YOLO11 = backbone (C3k2 blocks + SPPF + C2PSA attention) -> neck (multi-scale feature fusion, P3/P4/P5) -> decoupled anchor-free head. Small objects are detected on the high-resolution P3 map (stride 8).
3. Loss = box loss (CIoU) + class loss (BCE) + DFL (distribution focal loss for box edges). We report their sum as "total loss".
4. `src/run_experiments.py` trains each configuration; `src/analyze.py` plots curves, builds tables and writes the analysis; `src/build_readme.py` assembles this README.

## 4. Experiments
| Run | Change from baseline (everything else identical) | Purpose |
|---|---|---|
| `baseline` | YOLO11n, imgsz 640 | Part I |
| `size_s` | YOLO11s (more parameters) | Part III: capacity |
| `regularized` | cosine LR, weight decay 0.002 (4x), scale 0.7, mixup 0.1, hsv_s 0.8 | Part IV: overfitting control |
| `yolov8n` | YOLOv8n | Part V: version comparison |

<!--RESULTS-->

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
