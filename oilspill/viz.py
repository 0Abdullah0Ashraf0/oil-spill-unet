import numpy as np

from .data import DEFAULT_COLORS, NORMS


def colorize(label, colors=None):
    pal = np.asarray(colors or DEFAULT_COLORS, dtype=np.uint8)
    return pal[np.asarray(label)]


def denorm(x, norm="half"):
    mean, std = NORMS[norm]
    x = x.permute(1, 2, 0).cpu().numpy() * np.asarray(std) + np.asarray(mean)
    return np.clip(x, 0, 1)
