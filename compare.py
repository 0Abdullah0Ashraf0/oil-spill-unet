"""Build the comparison table over every run in results/ (U-Net runs and Omar's DeepLab runs too,
as long as each run folder follows the shared contract: config.json + metrics.json).

    python compare.py --out results

Writes results/comparison.csv and results/comparison.md. A run is reported on the TEST split if
you have run evaluate.py --final for it, otherwise on validation (flagged in the 'Split' column).
"""
import argparse
import csv
import json
from pathlib import Path

NAMES = ["sea", "oil_spill", "look_alike", "ship", "land"]
HEAD = ["Run", "Loss", "Aug", "Input", "Best ep", "Time (min)", "Split", "mIoU"] + \
       [f"IoU {n}" for n in NAMES] + ["Dice oil", "Dice ship", "Pixel acc"]


def _f(v):
    return "n/a" if v is None else f"{v:.4f}"


def make_table(out="results"):
    rows = []
    for run in sorted(p for p in Path(out).iterdir() if (p / "metrics.json").exists() and (p / "config.json").exists()):
        cfg, met = json.load(open(run / "config.json")), json.load(open(run / "metrics.json"))
        split = "test" if "test" in met else "val"
        m = met[split]
        size = f"{cfg.get('height', cfg.get('size', '?'))}x{cfg.get('width', cfg.get('size', '?'))}"
        rows.append([run.name, cfg.get("loss", "?"), cfg.get("aug", "?"), size, met.get("best_epoch"),
                     f"{met.get('train_time_s', 0) / 60:.1f}", split, _f(m.get("mIoU"))] +
                    [_f(m.get(f"IoU_{n}")) for n in NAMES] + [_f(m.get("Dice_oil_spill")), _f(m.get("Dice_ship")),
                                                              _f(m.get("pixel_acc"))])
    if not rows:
        print("no finished runs found in", out)
        return rows
    with open(Path(out) / "comparison.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(HEAD); w.writerows(rows)
    md = ["| " + " | ".join(HEAD) + " |", "|" + "---|" * len(HEAD)] + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    (Path(out) / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    if any(r[6] == "val" for r in rows):
        print("\nNOTE: rows marked 'val' are validation numbers. Final report numbers must come from evaluate.py --final (test).")
    return rows


# ---- CLI ----
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    make_table(ap.parse_args().out)
