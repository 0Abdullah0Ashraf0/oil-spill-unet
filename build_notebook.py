"""Generate Oil_Spill_UNet_Kaggle.ipynb from the project modules (single source of truth).

    python build_notebook.py

Re-run this after editing any .py file, then upload the .ipynb to Kaggle.
The notebook is self-contained: it inlines oilspill/*.py, audit_dataset.py, train.py, evaluate.py,
compare.py as cells (everything after the '# ---- CLI ----' marker is dropped).
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).parent
IMPORT_LOCAL = re.compile(r"^from (?:oilspill(?:\.\w+)?|\.\w*) import (?:\([^)]*\)|[^\n]*)\n", re.M)

MODULES = [
    ("Data pipeline (shared contract)", "oilspill/data.py"),
    ("U-Net from scratch", "oilspill/unet.py"),
    ("Losses", "oilspill/losses.py"),
    ("Metrics", "oilspill/metrics.py"),
    ("Visualisation helpers", "oilspill/viz.py"),
    ("Dataset audit", "audit_dataset.py"),
    ("Training", "train.py"),
    ("Evaluation", "evaluate.py"),
    ("Comparison table", "compare.py"),
]


def clean(src):
    src = src.split("# ---- CLI ----")[0]
    src = IMPORT_LOCAL.sub("", src)
    src = "\n".join(l for l in src.splitlines() if l.strip() not in ('matplotlib.use("Agg")', "import matplotlib"))
    return src.strip() + "\n"


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": text.strip().splitlines(keepends=True)}


CONFIG = '''
import os, sys, platform, json
ON_KAGGLE = os.path.exists("/kaggle/input")
DATA_ROOT = os.environ.get("OIL_DATA_ROOT") or ("/kaggle/input" if ON_KAGGLE else None)   # folder that contains train/ and test/
OUT = os.environ.get("OIL_OUT") or ("/kaggle/working" if ON_KAGGLE else "nb_out")        # everything is saved here
EPOCHS = int(os.environ.get("OIL_EPOCHS", 40))
H, W = int(os.environ.get("OIL_H", 256)), int(os.environ.get("OIL_W", 512))                # images are ~2:1 wide
BASE = int(os.environ.get("OIL_BASE", 32))
WORKERS = 2 if ON_KAGGLE else 0
RUN_ABLATIONS = os.environ.get("OIL_ABLATIONS", "0") == "1"   # also run weights-only and Dice-only runs
FINAL = os.environ.get("OIL_FINAL", "0") == "1"                # set True ONCE, after every experiment/checkpoint choice is frozen
assert DATA_ROOT, "Set DATA_ROOT to the folder containing train/ and test/"
os.makedirs(OUT, exist_ok=True); os.chdir(OUT)
import torch
print("python", platform.python_version(), "| torch", torch.__version__, "| cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("DATA_ROOT =", DATA_ROOT, "| OUT =", os.getcwd(), "| input", H, "x", W, "| epochs", EPOCHS)
'''

SHOW = '''
from IPython.display import Image as _Img, display
def show(*paths):
    for p in paths:
        if os.path.exists(p):
            display(_Img(filename=p))
'''

AUDIT = '''
run_audit(audit_parser().parse_args(["--data_root", DATA_ROOT, "--out", "audit"]))
show("audit/class_frequency.png", "audit/sample_grid.png")
'''

EXPERIMENTS = '''
COMMON = ["--data_root", DATA_ROOT, "--epochs", str(EPOCHS), "--height", str(H), "--width", str(W), "--base", str(BASE),
          "--workers", str(WORKERS), "--resume"]   # --resume: a re-run continues from last.pth instead of starting over

# U1: baseline. Plain cross-entropy.
run_train(train_parser().parse_args(COMMON + ["--run_id", "U1", "--loss", "ce"]))
# U2: class-weighted cross-entropy + soft Dice. Same data/seed/size/epochs/lr; only the loss differs.
run_train(train_parser().parse_args(COMMON + ["--run_id", "U2", "--loss", "wce_dice"]))
if RUN_ABLATIONS:  # decompose U2: which part of the loss helped?
    run_train(train_parser().parse_args(COMMON + ["--run_id", "U2a_weights_only", "--loss", "wce"]))
    run_train(train_parser().parse_args(COMMON + ["--run_id", "U2b_dice_only", "--loss", "ce_dice"]))
RUNS = [r for r in ["U1", "U2", "U2a_weights_only", "U2b_dice_only"] if os.path.exists(f"results/{r}/metrics.json")]
'''

CURVES = '''
for r in RUNS:
    show(f"results/{r}/curves.png")
'''

EVAL = '''
# Validation numbers/figures are always produced. The held-out TEST set is only touched when FINAL = True.
for r in RUNS:
    run_eval(eval_parser().parse_args(["--data_root", DATA_ROOT, "--run_id", r] + (["--final"] if FINAL else [])))
    show(f"results/{r}/confusion_matrix.png", f"results/{r}/predictions.png", f"results/{r}/failure_cases.png")
'''

TABLE = '''
rows = make_table("results")
print()
print("Report numbers must come from the TEST split (FINAL = True)." if not FINAL else "Table above is on the held-out test split.")
'''


def main():
    cells = [
        md("# Oil Spill Semantic Segmentation: U-Net track\n\n"
           "**How to run on Kaggle:** add the dataset `nabilsherif/oil-spill` to the notebook, enable a GPU, then "
           "*Save Version -> Save & Run All*. Outputs (weights, logs, plots, `results/`) are written to `/kaggle/working`. "
           "Checkpoints are saved every epoch (`last.pth`), so a stopped run can resume.\n\n"
           "Order: config -> code cells (from the project modules) -> audit -> experiments -> curves -> evaluation -> table. "
           "Generated by `build_notebook.py`; do not edit code here, edit the `.py` files and rebuild."),
        code(CONFIG),
        code(SHOW),
    ]
    for title, rel in MODULES:
        cells.append(md(f"## {title}\n`{rel}`"))
        cells.append(code(clean((HERE / rel).read_text(encoding="utf-8"))))
    cells += [
        md("## Step 1: Audit the dataset\nCheck exact mask colours, sizes, class balance, and that all five classes appear. "
           "Read the printout before trusting the class map."),
        code(AUDIT),
        md("## Step 2: Experiments (U-Net)\nU1 vs U2 isolates the loss. Optional ablations split U2 into weights-only and Dice-only."),
        code(EXPERIMENTS),
        md("## Step 3: Training curves\nCompare train vs validation curves (not one final score) when discussing over/underfitting."),
        code(CURVES),
        md("## Step 4: Evaluation\nConfusion matrix, prediction panels, failure cases (oil vs look-alike, missed ships)."),
        code(EVAL),
        md("## Step 5: Comparison table\nOmar's DeepLabV3+ runs appear here too if their folders (config.json + metrics.json) are copied into `results/`."),
        code(TABLE),
    ]
    for i, c in enumerate(cells):  # nbformat >= 4.5 requires a unique id per cell
        c["id"] = f"cell-{i:03d}"
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}}}
    out = HERE / "Oil_Spill_UNet_Kaggle.ipynb"
    json.dump(nb, open(out, "w", encoding="utf-8"), indent=1)
    print(f"wrote {out} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
