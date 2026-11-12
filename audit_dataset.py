"""Phase U0: audit the dataset. Run this FIRST on the real data.

    python audit_dataset.py --data_root <folder with train/ and test/> --out audit

Checks folder structure, image/mask alignment, image sizes, whether mask colours really match
the colours the assignment states, class frequencies, how many images contain each class, and
writes a sample grid. It writes class_map.json (colours + names) which every other script uses.
(Looking at test masks here is inspection only: nothing is tuned on them.)
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from oilspill.data import CLASS_NAMES, DEFAULT_COLORS, discover, find_pairs, load_mask_labels
from oilspill.viz import colorize


def audit_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--out", default="audit")
    ap.add_argument("--max_masks", type=int, default=0, help="0 = scan every training mask")
    return ap


def run_audit(a):
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    found = discover(a.data_root)
    print("Discovered:", {k: (str(v[0]), str(v[1])) for k, v in found.items()})
    assert "train" in found, "No train/ folder with images + labels found; check --data_root"
    report = {"data_root": str(a.data_root), "splits": {}}
    pairs = {}
    for split, (idir, mdir) in found.items():
        p, no_mask, no_img = find_pairs(idir, mdir)
        pairs[split] = p
        report["splits"][split] = {"images_dir": str(idir), "masks_dir": str(mdir), "pairs": len(p),
                                   "images_without_mask": no_mask, "masks_without_image": no_img}
        print(f"[{split}] pairs={len(p)} imgs_without_mask={len(no_mask)} masks_without_img={len(no_img)}")

    # leakage check by CONTENT (names are not enough: train and test both start at img_0001)
    if "test" in pairs:
        import hashlib
        md5 = lambda p: hashlib.md5(Path(p).read_bytes()).hexdigest()
        train_h = {md5(ip): sid for sid, ip, _ in pairs["train"]}
        dups = [(sid, train_h[md5(ip)]) for sid, ip, _ in pairs["test"] if md5(ip) in train_h]
        shared = len({p[0] for p in pairs["train"]} & {p[0] for p in pairs["test"]})
        report["train_test_identical_images"] = dups
        report["train_test_shared_file_names"] = shared
        print(f"train/test leakage: {len(dups)} test images are byte-identical to a train image"
              f" | {shared} file names are reused across the two folders (harmless; test IDs get a 'test_' prefix)")

    # sizes / modes + image-mask alignment
    sizes, modes, mismatch = Counter(), Counter(), []
    for _, ip, mp in pairs["train"]:
        with Image.open(ip) as im, Image.open(mp) as mk:
            sizes[im.size] += 1
            modes[(im.mode, mk.mode)] += 1
            if im.size != mk.size:
                mismatch.append(ip.stem)
    report["image_sizes"] = {f"{w}x{h}": n for (w, h), n in sizes.most_common()}
    report["(image_mode, mask_mode)"] = {str(k): n for k, n in modes.items()}
    report["size_mismatch_image_vs_mask"] = mismatch
    top = list(report["image_sizes"].items())[:5]
    print(f"image sizes (w x h, top 5 of {len(sizes)}):", top, "| modes:", report["(image_mode, mask_mode)"])
    print("image/mask size mismatches:", len(mismatch))

    # mask colour verification on training masks (native resolution, no resizing)
    items = pairs["train"][: a.max_masks or None]
    colour_counts, label_values = Counter(), Counter()
    for _, _, mp in items:
        m = Image.open(mp)
        if m.mode in ("L", "P", "I", "I;16"):
            u, c = np.unique(np.asarray(m), return_counts=True)
            label_values.update(dict(zip(map(int, u), map(int, c))))
        else:
            arr = np.asarray(m.convert("RGB")).reshape(-1, 3).astype(np.int32)
            packed = (arr[:, 0] << 16) | (arr[:, 1] << 8) | arr[:, 2]  # 1-D unique is far faster than axis=0
            u, c = np.unique(packed, return_counts=True)
            for p, n in zip(u.tolist(), c):
                colour_counts[(p >> 16, (p >> 8) & 255, p & 255)] += int(n)
    if colour_counts:
        total = sum(colour_counts.values())
        exact = sum(colour_counts.get(tuple(c), 0) for c in DEFAULT_COLORS)
        print(f"\nTop mask colours (of {len(colour_counts)} distinct):")
        for col, n in colour_counts.most_common(10):
            tag = CLASS_NAMES[DEFAULT_COLORS.index(col)] if col in DEFAULT_COLORS else "NOT IN ASSIGNMENT"
            print(f"  {col!s:<18} {n / total:7.3%}  {tag}")
        report["mask_exact_match_fraction"] = exact / total
        report["distinct_mask_colours"] = len(colour_counts)
        report["top_colours"] = [{"rgb": list(c), "fraction": n / total} for c, n in colour_counts.most_common(10)]
        print(f"exact assignment-colour match: {exact / total:.3%}"
              + ("  <-- <99%: masks have extra/off colours (JPEG?). Nearest-colour mapping is used; read the table above."
                 if exact / total < 0.99 else ""))
    if label_values:
        print("single-channel label values:", dict(sorted(label_values.items())))
        report["label_values"] = dict(sorted(label_values.items()))

    json.dump({"names": CLASS_NAMES, "colors": [list(c) for c in DEFAULT_COLORS],
               "verified_exact_match_fraction": report.get("mask_exact_match_fraction")},
              open(out / "class_map.json", "w"), indent=1)

    # pixel share (train) + number of images containing each class (every split)
    counts = np.zeros(5, dtype=np.int64)
    present = {s: np.zeros(5, dtype=np.int64) for s in pairs}
    for s, plist in pairs.items():
        for _, _, mp in (items if s == "train" else plist):
            b = np.bincount(load_mask_labels(mp, None, DEFAULT_COLORS).ravel(), minlength=5)[:5]
            present[s] += (b > 0)
            if s == "train":
                counts += b
    freq = counts / counts.sum()
    report["train_class_pixel_share"] = {n: float(f) for n, f in zip(CLASS_NAMES, freq)}
    report["images_containing_class"] = {s: {n: int(v) for n, v in zip(CLASS_NAMES, c)} for s, c in present.items()}
    print("\ntrain class share:", {n: f"{f:.3%}" for n, f in zip(CLASS_NAMES, freq)})
    for s, c in present.items():
        n_img = len(items) if s == "train" else len(pairs[s])
        print(f"images containing each class [{s}, of {n_img}]:", {n: int(v) for n, v in zip(CLASS_NAMES, c)})
        if (c == 0).any():
            print(f"  <-- WARNING: class(es) {[CLASS_NAMES[i] for i in np.where(c == 0)[0]]} absent from the {s} split")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(CLASS_NAMES, freq, color=[np.array(c) / 255 for c in DEFAULT_COLORS], edgecolor="k")
    ax.set_yscale("log"); ax.set_ylabel("pixel share (log)"); ax.set_title("Training-set class frequency")
    fig.tight_layout(); fig.savefig(out / "class_frequency.png", dpi=150); plt.close(fig)

    # sample grid: image | mask
    pick = items[:: max(1, len(items) // 6)][:6]
    fig, axs = plt.subplots(2, len(pick), figsize=(3 * len(pick), 4.5))
    for j, (sid, ip, mp) in enumerate(pick):
        axs[0, j].imshow(Image.open(ip).convert("RGB")); axs[0, j].set_title(sid, fontsize=8)
        axs[1, j].imshow(colorize(load_mask_labels(mp, None, DEFAULT_COLORS)))
        for r in (0, 1):
            axs[r, j].axis("off")
    fig.tight_layout(); fig.savefig(out / "sample_grid.png", dpi=130); plt.close(fig)

    json.dump(report, open(out / "audit_report.json", "w"), indent=1)
    print(f"\nWrote {out}/audit_report.json, class_map.json, class_frequency.png, sample_grid.png")


# ---- CLI ----
if __name__ == "__main__":
    run_audit(audit_parser().parse_args())
