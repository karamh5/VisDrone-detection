"""README.md = README_template.md + results/RESULTS.md  (run after analyze.py)"""
from pathlib import Path
t = Path("README_template.md").read_text(); r = Path("results/RESULTS.md").read_text()
Path("README.md").write_text(t.replace("<!--RESULTS-->", r)); print("README.md written")
