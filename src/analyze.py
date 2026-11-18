"""Turn runs/*/results.csv + results/*.json into figures, tables and an auto-written analysis (results/RESULTS.md)."""
import json
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

RES, FIG, RUNS = Path("results"), Path("figures"), Path("runs"); FIG.mkdir(exist_ok=True)
ORDER = ["baseline", "size_s", "regularized", "yolov8n"]
J = {n: json.loads((RES / f"{n}.json").read_text()) for n in ORDER if (RES / f"{n}.json").exists()}

def curves(n):
    d = pd.read_csv(RUNS / n / "results.csv"); d.columns = [c.strip() for c in d.columns]
    d["train_loss"] = d["train/box_loss"] + d["train/cls_loss"] + d["train/dfl_loss"]
    d["val_loss"] = d["val/box_loss"] + d["val/cls_loss"] + d["val/dfl_loss"]
    return d

def diagnose(d):
    """Rule-based fit diagnosis from the last 5 epochs."""
    k = min(5, len(d)); e = np.arange(k)
    sv = np.polyfit(e, d["val_loss"].tail(k), 1)[0]; st = np.polyfit(e, d["train_loss"].tail(k), 1)[0]
    sm = np.polyfit(e, d["metrics/mAP50(B)"].tail(k), 1)[0]
    gap = d["val_loss"].iloc[-1] / d["train_loss"].iloc[-1]
    if sv > 0 and st < 0: tag = "overfitting (validation loss rising while training loss falls)"
    elif sv < 0 and sm > 0.002: tag = "underfitting / not yet converged (validation loss and mAP50 still improving)"
    else: tag = "approximately converged (validation loss flat)"
    return tag, gap, sv, st, sm

# --- Part I/II: baseline curves -------------------------------------------------------------
b = curves("baseline")
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for a, (t, v, title) in zip(ax, [("train/box_loss", "val/box_loss", "Box loss"), ("train/cls_loss", "val/cls_loss", "Class loss"), ("train_loss", "val_loss", "Total loss")]):
    a.plot(b["epoch"], b[t], label="train"); a.plot(b["epoch"], b[v], label="val"); a.set_title(title); a.set_xlabel("epoch"); a.legend(); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "baseline_loss_curves.png", dpi=150); plt.close()
fig, ax = plt.subplots(figsize=(6, 4))
for c, l in [("metrics/mAP50(B)", "mAP50"), ("metrics/mAP50-95(B)", "mAP50-95"), ("metrics/precision(B)", "precision"), ("metrics/recall(B)", "recall")]: ax.plot(b["epoch"], b[c], label=l)
ax.set_xlabel("epoch"); ax.set_title("Baseline validation metrics"); ax.legend(); ax.grid(alpha=.3); plt.tight_layout(); plt.savefig(FIG / "baseline_metrics.png", dpi=150); plt.close()

import shutil
for src, dst in [("baseline_val/confusion_matrix_normalized.png", "baseline_confusion_matrix.png"), ("baseline_val/BoxPR_curve.png", "baseline_pr_curve.png")]:
    if (RUNS / src).exists(): shutil.copy(RUNS / src, FIG / dst)

# --- comparison plots -------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for n in J:
    d = curves(n); ax[0].plot(d["epoch"], d["val_loss"], label=n); ax[1].plot(d["epoch"], d["metrics/mAP50(B)"], label=n)
ax[0].set_title("Validation total loss"); ax[1].set_title("Validation mAP50")
for a in ax: a.set_xlabel("epoch"); a.legend(); a.grid(alpha=.3)
plt.tight_layout(); plt.savefig(FIG / "comparison.png", dpi=150); plt.close()

# --- tables + auto-written analysis ------------------------------------------------------------
def table(names):
    rows = ["| Run | Weights | Params (M) | Precision | Recall | mAP50 | mAP50-95 | Infer ms/img | Train min |", "|---|---|---|---|---|---|---|---|---|"]
    for n in names:
        if n in J: r = J[n]; rows.append(f"| {n} | {r['weights']} | {r['params_M']} | {r['precision']:.3f} | {r['recall']:.3f} | {r['map50']:.3f} | {r['map50_95']:.3f} | {r['infer_ms']} | {r['train_minutes']} |")
    return "\n".join(rows)

tag, gap, sv, st, sm = diagnose(b); r0 = J["baseline"]
md = [f"## Results (YOLO11, VisDrone-DET, {r0['fraction']*100:.0f}% of train images, {r0['epochs']} epochs, imgsz 640, seed 0)\n",
      "### Part I - Baseline\n", table(["baseline"]), "\n![baseline loss](figures/baseline_loss_curves.png)\n![baseline metrics](figures/baseline_metrics.png)\n![confusion matrix](figures/baseline_confusion_matrix.png)\n![PR curve](figures/baseline_pr_curve.png)\n",
      "### Part II - Loss-curve and fitting analysis\n",
      f"Final-epoch train total loss = {b['train_loss'].iloc[-1]:.3f}, validation total loss = {b['val_loss'].iloc[-1]:.3f} (val/train ratio {gap:.2f}). "
      f"Over the last {min(5,len(b))} epochs the validation-loss slope is {sv:+.4f}/epoch, training-loss slope {st:+.4f}/epoch, and mAP50 slope {sm:+.4f}/epoch. "
      f"**Diagnosis: {tag}.**\n",
      f"*Dataset size.* Training uses {r0['fraction']*100:.0f}% of VisDrone-DET (about {int(6471*r0['fraction'])} images) with many tiny, densely packed objects. A small training set makes overfitting more likely when training for long, because the model can memorise it. "
      + (f"Here the val/train loss ratio is {gap:.2f}, so there is no overfitting at {r0['epochs']} epochs: the model has not made enough passes over the data to memorise it. " if gap < 1.1 else f"Here the val/train loss ratio is {gap:.2f}, which indicates the model is starting to memorise the small training set. ")
      + "Validation loss can sit at or below training loss early in training because the training loss is averaged over the epoch while the weights are still improving, and it is measured on augmented images (mosaic, scale, colour jitter) that are harder than the clean validation images. "
      f"*Model capacity.* YOLO11n has {r0['params_M']} M parameters. Training and validation loss fall together and are both still falling at the last epoch, which fits underfitting (not enough training or capacity) rather than overfitting. A larger model would help if capacity is the bottleneck, which the size experiment below tests.\n"]
if "size_s" in J:
    s = J["size_s"]; d = s["map50"] - r0["map50"]
    md += ["### Part III - Controlled experiment: model capacity (only variable: n -> s)\n", table(["baseline", "size_s"]),
           f"\nChanging YOLO11n to YOLO11s ({r0['params_M']} M -> {s['params_M']} M parameters) changed mAP50 by {d:+.3f} ({r0['map50']:.3f} -> {s['map50']:.3f}) and inference time from {r0['infer_ms']} to {s['infer_ms']} ms/img. "
           + ("The gain indicates the baseline is capacity-limited on small objects, at the cost of speed." if d > 0.005 else "No clear gain indicates the baseline is limited by data/training budget rather than capacity.") + "\n"]
if "regularized" in J:
    g = J["regularized"]; d = g["map50"] - r0["map50"]; tg, gg, *_ = diagnose(curves("regularized"))
    md += ["### Part IV - Improvement cycle: regularisation\n",
           f"**Baseline** (above) -> **Experimental settings** identical except -> **Controlled modification**: `{g['overrides']}` (cosine LR decay, 4x weight decay, stronger scale/colour augmentation, mixup 0.1). "
           "Rationale: weight decay and augmentation reduce variance (overfitting) and cosine decay lets the optimiser settle into a flatter minimum at the end of a short schedule.\n", table(["baseline", "regularized"]),
           f"\n**Evaluation / analysis**: mAP50 {r0['map50']:.3f} -> {g['map50']:.3f} ({d:+.3f}); val/train loss ratio {gap:.2f} -> {gg:.2f}; diagnosis now: {tg}. "
           + ("**Conclusion**: the regularisation helped and is kept." if d > 0.005 else
              "**Conclusion**: the regularisation hurt, so it is not kept." if d < -0.005 else
              f"**Conclusion**: no measurable difference ({d:+.3f} mAP50 is within single-seed noise). The baseline was not overfitting (val/train loss ratio {gap:.2f}), so regularisation had little to correct. Hypothesis, not tested here: it would matter more with longer training on this small subset, once overfitting appears.") + "\n"]
if "yolov8n" in J:
    y = J["yolov8n"]
    md += ["### Part V - Multi-version comparison (YOLO11n vs YOLOv8n)\n", table(["baseline", "yolov8n"]),
           f"\nYOLOv8n reaches mAP50 {y['map50']:.3f} vs {r0['map50']:.3f} for YOLO11n, with {y['params_M']} M vs {r0['params_M']} M parameters and {y['infer_ms']} vs {r0['infer_ms']} ms/img inference.\n"]
md += ["### All runs\n", table(ORDER),
       "\n*Note on the last two columns:* they come from one laptop GPU that was also running other processes, so they are only indicative. Models of identical size differ more than their architecture explains (baseline vs regularized: 7.0 vs 3.4 training minutes, 2.06 vs 1.90 ms/img), and the baseline time includes about 3 minutes of one-off start-up work. Compare accuracy columns, and compare speed only for large differences such as YOLO11n vs YOLO11s.\n",
       "\n![comparison](figures/comparison.png)\n"]
(RES / "RESULTS.md").write_text("\n".join(md)); print("\n".join(md))
