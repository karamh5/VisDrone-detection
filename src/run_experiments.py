"""Train + evaluate every experiment on VisDrone-DET with Ultralytics YOLO.
Usage:  python src/run_experiments.py            (all runs)
        python src/run_experiments.py baseline   (one run)
Budget knobs are at the top: EPOCHS / FRACTION trade accuracy for wall-clock time."""
import sys, json, time
from pathlib import Path
from ultralytics import YOLO

EPOCHS, FRACTION, BATCH, SEED = 30, 1.0, 16, 0   # FRACTION = share of the 6,471 train images used (1.0 = all)
DATA = "VisDrone.yaml"                            # Ultralytics auto-downloads VisDrone2019-DET on first use
OUT = Path("runs").resolve(); RES = Path("results"); RES.mkdir(exist_ok=True)

COMMON = dict(data=DATA, epochs=EPOCHS, fraction=FRACTION, batch=BATCH, seed=SEED, deterministic=True,
              imgsz=640, patience=EPOCHS, plots=True, project=str(OUT), exist_ok=True, workers=2, verbose=False)

# Each experiment differs from `baseline` in exactly the keys listed here (controlled variables).
EXPERIMENTS = {
    "baseline":   dict(model="yolo11n.pt"),                                        # Part I
    "size_s":     dict(model="yolo11s.pt"),                                        # Part III: model capacity n -> s
    "regularized": dict(model="yolo11n.pt", cos_lr=True, weight_decay=0.002,       # Part IV: regularisation cycle
                        scale=0.7, mixup=0.1, hsv_s=0.8, fliplr=0.5),
    "yolov8n":    dict(model="yolov8n.pt"),                                        # Part V: other YOLO version
}

def run(name):
    cfg = dict(COMMON); cfg.update(EXPERIMENTS[name]); weights = cfg.pop("model")
    model = YOLO(weights)
    t0 = time.time(); model.train(name=name, **cfg); train_s = time.time() - t0
    best = YOLO(OUT / name / "weights" / "best.pt")
    m = best.val(data=DATA, imgsz=cfg["imgsz"], split="val", plots=True, project=str(OUT), name=f"{name}_val", exist_ok=True, verbose=False)
    out = dict(name=name, weights=weights, overrides={k: v for k, v in EXPERIMENTS[name].items() if k != "model"},
               params_M=round(sum(p.numel() for p in best.model.parameters()) / 1e6, 2),
               precision=float(m.box.mp), recall=float(m.box.mr), map50=float(m.box.map50), map50_95=float(m.box.map),
               infer_ms=round(m.speed["inference"], 2), train_minutes=round(train_s / 60, 1),
               epochs=EPOCHS, fraction=FRACTION)
    (RES / f"{name}.json").write_text(json.dumps(out, indent=2)); print("RESULT", out)

if __name__ == "__main__":
    for n in (sys.argv[1:] or EXPERIMENTS): run(n)
