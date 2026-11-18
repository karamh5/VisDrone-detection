"""Regenerates notebooks/yolo11_visdrone_colab.ipynb from src/*.py (run from repo root)."""
import nbformat as nbf
nb = nbf.v4.new_notebook(); c = []
md = lambda s: c.append(nbf.v4.new_markdown_cell(s)); code = lambda s: c.append(nbf.v4.new_code_cell(s))
md("# YOLO11 on VisDrone - CMPE 401 Project 1\nRuntime -> Change runtime type -> **T4 GPU**, then **Run all**. About 35-45 min. Results land in `results/`, `figures/`, `README.md`.")
code("!pip -q install ultralytics\nimport os; os.makedirs('src', exist_ok=True); os.makedirs('results', exist_ok=True)")
for f in ["run_experiments.py", "analyze.py", "build_readme.py"]: code(f"%%writefile src/{f}\n" + open(f"src/{f}").read())
code("%%writefile README_template.md\n" + open("README_template.md").read())
md("## Train all experiments (baseline, size, regularisation, YOLOv8n)\nFirst run downloads VisDrone (~2 GB).")
code("!python src/run_experiments.py")
md("## Analyse + build README")
code("!python src/analyze.py && python src/build_readme.py")
code("from IPython.display import Image, display\nfor f in ['baseline_loss_curves','baseline_metrics','comparison']: display(Image(f'figures/{f}.png'))")
code("!zip -qr yolo_results.zip README.md results figures src runs/*/results.csv\nfrom google.colab import files; files.download('yolo_results.zip')")
nb.cells = c; nbf.write(nb, "notebooks/yolo11_visdrone_colab.ipynb")
