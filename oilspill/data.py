"""Shared data pipeline for the Oil Spill segmentation assignment.

This module is the "shared contract" Omar and Abdallah both import, so that the
class mapping, split and resizing are identical for both models. Normalisation is a
PARAMETER because it must match the model: "half" for the from-scratch U-Net,
"imagenet" for a pretrained DeepLabV3+ backbone.
"""
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASS_NAMES = ["sea", "oil_spill", "look_alike", "ship", "land"]
# Colours as stated in the assignment. audit_dataset.py checks them against the real masks.
DEFAULT_COLORS = [(0, 0, 0), (0, 255, 255), (255, 0, 0), (153, 76, 0), (0, 153, 0)]
IMG_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
NORMS = {  # (mean, std) per channel applied to x/255
    "half": ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),                      # -> [-1, 1]; scratch U-Net
    "imagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),      # pretrained backbones
}


def load_colors(class_map_path=None):
    if class_map_path and Path(class_map_path).exists():
        return [tuple(c) for c in json.load(open(class_map_path))["colors"]]
    return DEFAULT_COLORS


def rgb_to_label(mask_rgb, colors):
    """Map an HxWx3 colour mask to HxW integer labels via the nearest palette colour.

    Nearest-colour (not exact match) so slightly off colours from compression do not
    silently become class 0. audit_dataset.py reports how many pixels are not exact.
    """
    m = mask_rgb.astype(np.int32)[..., None, :]  # int32: 255**2 overflows int16
    pal = np.asarray(colors, dtype=np.int32)[None, None]
    return ((m - pal) ** 2).sum(-1).argmin(-1).astype(np.uint8)


def load_mask_labels(path, size_hw, colors):
    """Load a mask, resize with NEAREST (never interpolate labels), return HxW uint8 labels."""
    im = Image.open(path)
    if size_hw is not None:
        im = im.resize((size_hw[1], size_hw[0]), Image.NEAREST)  # PIL takes (w, h)
    if im.mode in ("L", "P", "I", "I;16"):  # already single-channel label / palette indices
        return np.asarray(im).astype(np.uint8)
    return rgb_to_label(np.asarray(im.convert("RGB")), colors)


def _stem_map(directory):
    return {p.stem: p for p in Path(directory).iterdir() if p.suffix.lower() in IMG_EXT}


def find_pairs(img_dir, mask_dir, prefix=""):
    """Pair images with masks by file stem. `prefix` namespaces the returned IDs: the real dataset numbers
    BOTH train and test from img_0001, so test pairs must use prefix="test_" or the split IDs collide."""
    imgs, masks = _stem_map(img_dir), _stem_map(mask_dir)
    ids = sorted(set(imgs) & set(masks))
    return [(prefix + i, imgs[i], masks[i]) for i in ids], sorted(set(imgs) - set(masks)), sorted(set(masks) - set(imgs))


def discover(data_root):
    """Find {split: (image_dir, mask_dir)} under data_root without assuming exact folder names.

    Looks for train/test folders containing an 'image*' dir and a 'label*'/'mask*' dir.
    Prefers RGB label folders over '*1D*' (single-channel) ones.
    """
    root = Path(data_root)
    found = {}
    for split in ("train", "test"):
        for d in root.rglob("*"):
            if not d.is_dir() or d.name.lower() != split:
                continue
            subs = [s for s in d.iterdir() if s.is_dir()]
            imgs = [s for s in subs if "image" in s.name.lower()]
            lab = [s for s in subs if any(k in s.name.lower() for k in ("label", "mask"))]
            lab.sort(key=lambda s: ("1d" in s.name.lower(), s.name))
            if imgs and lab:
                found[split] = (imgs[0], lab[0])
                break
    return found


def make_split(train_ids, val_frac=0.15, seed=42):
    ids = sorted(train_ids)
    random.Random(seed).shuffle(ids)
    n_val = int(round(len(ids) * val_frac))
    return sorted(ids[n_val:]), sorted(ids[:n_val])


def load_or_make_split(split_path, train_ids, test_ids, val_frac=0.15, seed=42):
    """Reproducible train/val split stored in a JSON manifest the whole team shares."""
    split_path = Path(split_path)
    if split_path.exists():
        s = json.load(open(split_path))
        assert set(s["train"]) | set(s["val"]) == set(train_ids), "splits.json does not match dataset"
        return s
    tr, va = make_split(train_ids, val_frac, seed)
    assert not (set(tr) | set(va)) & set(test_ids), "train/val/test leakage"
    s = {"seed": seed, "val_frac": val_frac, "train": tr, "val": va, "test": sorted(test_ids)}
    split_path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(s, open(split_path, "w"), indent=1)
    return s


class OilSpillDataset(Dataset):
    """size_hw=(H, W). The real images are ~2:1 wide, so a non-square size avoids squashing.

    augment: False/"none" | "flip" (h+v flips) | "flip_int" (flips + mild brightness/contrast
    jitter on the IMAGE only; labels never get intensity changes).
    """

    def __init__(self, items, size_hw=(256, 512), augment="none", colors=None, norm="half", return_id=False):
        self.items, self.size_hw = items, tuple(size_hw)
        self.augment = "flip" if augment is True else (augment or "none")
        self.colors, self.return_id = colors or DEFAULT_COLORS, return_id
        mean, std = NORMS[norm]
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        sid, ip, mp = self.items[i]
        h, w = self.size_hw
        img = Image.open(ip).convert("RGB").resize((w, h), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float() / 255
        y = torch.from_numpy(load_mask_labels(mp, self.size_hw, self.colors).astype(np.int64))
        if self.augment in ("flip", "flip_int"):  # identical spatial transform on image and mask
            if random.random() < 0.5:
                x, y = x.flip(-1), y.flip(-1)
            if random.random() < 0.5:
                x, y = x.flip(-2), y.flip(-2)
        if self.augment == "flip_int":  # image only
            x = ((x - 0.5) * random.uniform(0.8, 1.2) + 0.5 + random.uniform(-0.1, 0.1)).clamp(0, 1)
        x = (x - self.mean) / self.std
        return (x, y, sid) if self.return_id else (x, y)


def class_pixel_counts(items, size_hw, colors, n_classes=5):
    counts = np.zeros(n_classes, dtype=np.int64)
    for _, _, mp in items:
        counts += np.bincount(load_mask_labels(mp, size_hw, colors).ravel(), minlength=n_classes)[:n_classes]
    return counts


def select(pairs, ids):
    wanted = set(ids)
    return [p for p in pairs if p[0] in wanted]
