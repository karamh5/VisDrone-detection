# YOLO11 on VisDrone-DET: training dynamics, controlled experiments and multi-version comparison

CMPE 401 Instructor-defined Project 1. This repo trains a modern YOLO detector on VisDrone-DET (small, dense, real-world drone objects), analyses its loss curves, runs a controlled experiment and a regularisation improvement cycle, and compares YOLO11 against YOLOv8. Every number in the tables is produced by the code in `src/`.

## 1. Objective
Understand the YOLO detection pipeline, fine-tune YOLO11, diagnose over/underfitting from training vs validation loss, and quantify the effect of model size, regularisation and YOLO version.

## 2. Setup
| Item | Value |
|---|---|
| Dataset | VisDrone2019-DET (10 classes, 6,471 train / 548 val images), downloaded automatically by Ultralytics (`VisDrone.yaml`) |
| Framework | Ultralytics (YOLO11 / YOLOv8), PyTorch |
| Training budget | All 6,471 train images (`fraction=1.0`), 30 epochs, batch 16, imgsz 640, seed 0, single NVIDIA GeForce RTX 3060 Laptop GPU (6 GB) |
| Metrics | Precision, Recall, mAP50, mAP50-95 on the full validation set (`best.pt`), inference ms/img |
| Budget choice | 30 epochs per run keeps four full-data runs within a few hours on one laptop GPU (see the Train min column). A typical full study uses 100+ epochs, so absolute mAP would be higher with a longer schedule; all four runs share the same budget, so the comparisons are fair. An earlier pilot (20% of the data, 12 epochs) is kept in `pilot_20pct/` and compared below. |

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
python src/run_experiments.py      # trains all 4 runs (several hours in total on an RTX 3060 laptop GPU; budget set at the top of the script)
python src/analyze.py              # figures + results/RESULTS.md
python src/build_readme.py         # rebuilds this README
```
Or open `notebooks/yolo11_visdrone_colab.ipynb` in Google Colab (Runtime -> T4 GPU -> Run all); the notebook is the same pipeline but was not used for the results reported here.

## 6. Limitations and next steps
Training budget of 30 epochs (not the 100+ typical for a full study); single seed, so differences under ~0.005 mAP50 are within noise; the optional VisDrone test-challenge submission was not run. Next: 100 epochs, imgsz 960 for small objects, multiple seeds.
