"""End-to-end check on SYNTHETIC data (the real dataset is not needed). Proves the code runs:
unit checks -> audit -> split -> U1/U2/ablations -> resume -> evaluate -> comparison table ->
the generated Kaggle notebook executes top to bottom.

    python smoke_test.py            # add --no_notebook to skip the notebook step
Synthetic files go to _smoke/ (safe to delete). Results here are meaningless as science.
"""
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from oilspill.data import DEFAULT_COLORS, NORMS, OilSpillDataset, rgb_to_label
from oilspill.losses import CEDice, SoftDiceLoss, build_loss, class_weights_from_counts
from oilspill.metrics import confusion, summarize
from oilspill.unet import UNet

HERE = Path(__file__).parent
S = HERE / "_smoke"
rng = np.random.default_rng(0)


def make_split_dir(name, n):
    (S / "data" / name / "images").mkdir(parents=True, exist_ok=True)
    (S / "data" / name / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(n):
        h, w = int(rng.integers(150, 200)), int(rng.integers(300, 400))  # ~2:1 wide like the real data
        lab = np.zeros((h, w), np.uint8)
        for cls, frac in ((4, .2), (1, .1), (2, .08), (3, .01)):  # imbalanced, like the real task
            for _ in range(int(rng.integers(1, 3))):
                y, x = int(rng.integers(0, h - 20)), int(rng.integers(0, w - 20))
                s = int(max(4, frac * h * (1.5 if cls != 3 else .3)))
                lab[y:y + s, x:x + s] = cls
        img = (lab[..., None] * 40 + rng.normal(100, 25, (h, w, 3))).clip(0, 255).astype(np.uint8)
        # same numbering in train AND test, exactly like the real dataset (regression test for the ID-collision bug)
        Image.fromarray(img).save(S / "data" / name / "images" / f"img_{i + 1:04d}.jpg")
        Image.fromarray(np.asarray(DEFAULT_COLORS, np.uint8)[lab]).save(S / "data" / name / "labels" / f"img_{i + 1:04d}.png")


def sh(*args, env=None, quiet=False):
    print("\n$", " ".join(str(a) for a in args)[:200])
    r = subprocess.run([sys.executable, *map(str, args)], cwd=HERE, capture_output=True, text=True, env=env)
    if not quiet:
        print(r.stdout[-1200:])
    assert r.returncode == 0, (r.stdout[-1500:] + r.stderr[-2500:])
    return r.stdout


# ---- unit checks ----
m = UNet(3, 5, 16, 4)
assert m(torch.randn(2, 3, 130, 97)).shape == (2, 5, 130, 97), "U-Net output shape (odd sizes exercise decoder padding)"
lab = np.zeros((4, 4), np.uint8); lab[1, 1] = 2; lab[2, 2] = 1
assert (rgb_to_label(np.asarray(DEFAULT_COLORS, np.uint8)[lab], DEFAULT_COLORS) == lab).all(), "colour->label round trip"
s = summarize(confusion(torch.tensor([[0, 1, 1, 4]]), torch.tensor([[0, 1, 2, 4]])).numpy())
assert s["IoU_ship"] is None and abs(s["pixel_acc"] - .75) < 1e-9, "absent class must be null"
logits, tgt = torch.zeros(1, 5, 4, 4), torch.zeros(1, 4, 4, dtype=torch.long)
w = class_weights_from_counts([90, 5, 3, 1, 1])
for kind in ("ce", "wce", "ce_dice", "wce_dice"):
    assert build_loss(kind, w)(logits, tgt) > 0, kind
assert SoftDiceLoss()(logits, tgt) >= 0 and isinstance(build_loss("ce_dice"), CEDice)
try:
    build_loss("wce", None); raise SystemExit("wce without weights should raise")
except ValueError:
    pass
print("unit checks passed")

# ---- data: non-square resize, normalisation switch, augmentation touches labels consistently ----
make_split_dir("train", 48); make_split_dir("test", 12)
D = S / "data"
items = [(p.stem, p, D / "train" / "labels" / (p.stem + ".png")) for p in sorted((D / "train" / "images").iterdir())][:4]
for norm in NORMS:
    x, y = OilSpillDataset(items, (64, 128), "flip_int", None, norm)[0]
    assert x.shape == (3, 64, 128) and y.shape == (64, 128) and y.dtype == torch.int64 and int(y.max()) <= 4, norm
x0, y0 = OilSpillDataset(items, (64, 128), "none")[0]
assert set(y0.unique().tolist()) <= {0, 1, 2, 3, 4} and -1.01 <= float(x0.min()) and float(x0.max()) <= 1.01
print("data checks passed (non-square resize, both normalisations, labels stay integer 0-4)")

# ---- end to end ----
H_, W_ = 64, 128
common = ["--data_root", D, "--height", H_, "--width", W_, "--base", 16, "--epochs", 2, "--bs", 8,
          "--class_map", S / "audit" / "class_map.json", "--splits", S / "splits.json", "--out", S / "results"]
sh("audit_dataset.py", "--data_root", D, "--out", S / "audit")
rep = json.load(open(S / "audit" / "audit_report.json"))
assert all(v > 0 for v in rep["images_containing_class"]["train"].values()), "per-class image counts missing/zero"
runs = [("U1", ["--loss", "ce"]), ("U2", ["--loss", "wce_dice", "--aug", "flip_int"]),
        ("U2a", ["--loss", "wce"]), ("U2b", ["--loss", "ce_dice", "--norm", "imagenet"])]
for rid, extra in runs:
    sh("train.py", *common, "--run_id", rid, *extra, quiet=True)
    print(rid, "trained")

# resume: raise epochs to 3, resume continues from epoch 3 only and appends to history.csv
out = sh("train.py", *common[:-6], "--epochs", 3, "--bs", 8, "--class_map", S / "audit" / "class_map.json",
         "--splits", S / "splits.json", "--out", S / "results", "--run_id", "U1", "--loss", "ce", "--resume", quiet=True)
assert "resumed from epoch 2" in out, out
n_rows = len(list(csv.DictReader(open(S / "results" / "U1" / "history.csv"))))
assert n_rows == 3, f"resume should leave 3 history rows, got {n_rows}"
print("resume check passed")

for rid, _ in runs:
    sh("evaluate.py", "--data_root", D, "--run_id", rid, "--out", S / "results", "--splits", S / "splits.json", "--final", quiet=True)
    need = ["config.json", "history.csv", "best_model.pth", "last.pth", "curves.png", "metrics.json", "confusion_matrix.png",
            "predictions.png", "failure_cases.png"]
    missing = [f for f in need if not (S / "results" / rid / f).exists()]
    assert not missing, f"{rid} missing {missing}"
    for row in csv.DictReader(open(S / "results" / rid / "history.csv")):
        for k in ("train_loss", "val_loss", "train_ce", "val_ce"):
            assert math.isfinite(float(row[k])), f"{rid}: non-finite {k} = {row[k]} at epoch {row['epoch']}"
tbl = sh("compare.py", "--out", S / "results", quiet=True)
assert (S / "results" / "comparison.md").exists() and tbl.count("test") >= 4, tbl
print("\nPIPELINE SMOKE TEST PASSED: all contract files, finite losses, resume, comparison table.")

# ---- generated notebook executes top to bottom ----
if "--no_notebook" not in sys.argv:
    sh("build_notebook.py")
    import nbformat
    from nbclient import NotebookClient
    nb = nbformat.read(HERE / "Oil_Spill_UNet_Kaggle.ipynb", as_version=4)
    os.environ.update({"OIL_DATA_ROOT": str(D), "OIL_OUT": str(S / "nb_out"), "OIL_EPOCHS": "2", "OIL_H": "64",
                       "OIL_W": "128", "OIL_BASE": "16", "OIL_FINAL": "1", "OIL_ABLATIONS": "1"})
    NotebookClient(nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(HERE)}}).execute()
    assert (S / "nb_out" / "results" / "comparison.md").exists(), "notebook did not produce comparison table"
    errs = [o for c in nb.cells if c.cell_type == "code" for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errs, errs
    print("NOTEBOOK TEST PASSED: Oil_Spill_UNet_Kaggle.ipynb ran every cell (audit, 4 runs, eval, table).")
