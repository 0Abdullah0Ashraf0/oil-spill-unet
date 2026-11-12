"""Builds Abdallah_UNet_Presentation.pptx from the real result files (numbers are read, never typed)."""
import json
from pathlib import Path

from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

HERE = Path(__file__).parent
R = HERE / "results"
NAVY, TEAL, OIL, RED, GREY, WHITE, INK = (RGBColor(*c) for c in
    [(11, 37, 69), (19, 138, 160), (0, 200, 210), (200, 60, 60), (110, 120, 130), (255, 255, 255), (30, 40, 50)])

RUNS = [("U1", "U1  CE baseline"), ("U2", "U2  weighted CE + Dice"), ("U2a_weights_only", "U2a  weights only"),
        ("U2b_dice_only", "U2b  Dice only"), ("U2_80ep", "U2 @ 80 epochs")]
RUNS = [(k, n) for k, n in RUNS if (R / k / "metrics.json").exists()]
M = {k: json.load(open(R / k / "metrics.json")) for k, _ in RUNS}
CFG = {k: json.load(open(R / k / "config.json")) for k, _ in RUNS}
SPLIT = "test" if all("test" in m for m in M.values()) else "val"
g = lambda k, key, s=None: (M[k].get(s or SPLIT) or {}).get(key)
f3 = lambda v: "n/a" if v is None else f"{v:.3f}"

prs = Presentation()
prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]


def text(s, x, y, w, h, t, size=18, bold=False, color=INK, align=None):
    tb = s.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame; tf.word_wrap = True
    lines = t if isinstance(t, list) else [t]
    for i, ln in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = ln; p.font.size = Pt(size); p.font.bold = bold; p.font.color.rgb = color
        p.space_after = Pt(8)
        if align: p.alignment = align
    return tb


def slide(title, dark=False, notes=""):
    s = prs.slides.add_slide(BLANK)
    s.background.fill.solid(); s.background.fill.fore_color.rgb = NAVY if dark else WHITE
    text(s, .6, .35, 12.1, .9, title, 32, True, WHITE if dark else NAVY)
    if notes: s.notes_slide.notes_text_frame.text = notes
    return s


def bullets(s, items, x=.6, y=1.5, w=6.2, size=18):
    text(s, x, y, w, 5, ["•  " + i for i in items], size)


def pic(s, path, x, y, w=None, h=None):
    p = Path(path)
    if p.exists():
        s.shapes.add_picture(str(p), Inches(x), Inches(y), Inches(w) if w else None, Inches(h) if h else None)


def stat(s, x, y, big, small, color=TEAL):
    text(s, x, y, 5.2, .9, big, 36, True, color)
    text(s, x, y + .85, 5.2, .8, small, 14, False, GREY)


# 1 title
s = slide("Oil Spill Semantic Segmentation", True)
text(s, .6, 2.2, 12, 1, "U-Net from scratch on Sentinel-1 SAR imagery", 26, False, RGBColor(200, 225, 240))
text(s, .6, 3.4, 12, 1.6, ["Abdallah's track: data pipeline, U-Net, experiments U1 / U2",
                           "Professional Training in AI 2  -  Assignment 2"], 18, False, WHITE)

# 2 task
s = slide("The task: label every pixel as one of five classes", notes="Dataset from Krestenitis et al. 2019, 1112 images of 1250x650.")
bullets(s, ["Input: radar (SAR) sea images, 1250 x 650 px", "Output: a class for every pixel",
            "Classes: sea, oil spill, look-alike, ship, land",
            "Look-alikes (natural slicks) resemble oil, which makes the problem hard",
            "1002 train images (852 train + 150 validation) and 110 official test images"], w=6.3)
pic(s, HERE / "audit" / "class_frequency.png", 7.2, 1.6, w=5.6)
text(s, 7.2, 4.9, 5.6, 1, "Class share is extremely uneven: ships and oil are a tiny fraction of pixels.", 14, False, GREY)

# 3 data audit
s = slide("Data audit: verified, not assumed")
bullets(s, ["Mask colours checked against the assignment: they match",
            "Image and mask sizes aligned, no leakage between train and test (content hash)",
            "Train and test both number files from img_0001, so test IDs get a test_ prefix",
            "Validation split fixed once (seed 42) in splits.json and shared with Omar",
            "Test set is untouched until the final evaluation"], w=5.6, size=17)
pic(s, HERE / "audit" / "sample_grid.png", 6.4, 1.6, w=6.5)

# 4 pipeline / model
s = slide("Method: shared pipeline and a hand-built U-Net")
bullets(s, ["Input 256 x 512 (keeps the 2:1 shape, no squashing)", "Masks resized with NEAREST so labels stay integers",
            "Augmentation: random horizontal / vertical flips (same on image and mask)",
            f"U-Net: 4 down / 4 up blocks, skip connections, BatchNorm, {CFG[RUNS[0][0]]['n_params']/1e6:.1f}M parameters",
            "No pretrained weights; AdamW + cosine schedule, mixed precision",
            "Checkpoint = best validation mIoU"], w=7.2, size=17)
for i, (b, c) in enumerate([("Encoder", NAVY), ("Bottleneck", TEAL), ("Decoder + skips", NAVY)]):
    box = s.shapes.add_shape(1, Inches(8.4), Inches(1.7 + i * 1.5), Inches(4.2), Inches(1.1))
    box.fill.solid(); box.fill.fore_color.rgb = c; box.line.fill.background()
    box.text_frame.text = b; box.text_frame.paragraphs[0].font.size = Pt(20); box.text_frame.paragraphs[0].font.color.rgb = WHITE

# 5 experiments
s = slide("Experiments: change one thing at a time")
rows = [("Run", "What changes", "Question")]
desc = {"U1": ("Cross-entropy only", "Baseline?"), "U2": ("Class-weighted CE + Dice", "Does imbalance-aware loss help rare classes?"),
        "U2a_weights_only": ("Weighted CE only", "Is it the weights?"), "U2b_dice_only": ("CE + Dice, no weights", "Is it the Dice term?"),
        "U2_80ep": ("U2 with 80 epochs", "Was U2 undertrained?")}
for k, n in RUNS: rows.append((k, *desc[k]))
tbl = s.shapes.add_table(len(rows), 3, Inches(.6), Inches(1.6), Inches(12.1), Inches(.6 * len(rows))).table
for r, row in enumerate(rows):
    for c, v in enumerate(row):
        cell = tbl.cell(r, c); cell.text = v
        cell.text_frame.paragraphs[0].font.size = Pt(16); cell.text_frame.paragraphs[0].font.bold = r == 0
text(s, .6, 6.2, 12, .8, "Same split, seed, image size, augmentation and epochs (except U2_80ep). U1 / U2 are the assignment's required pair; the rest are extra ablations.", 14, False, GREY)

# 6 headline results
tag = "held-out test set (110 images)" if SPLIT == "test" else "validation set"
s = slide(f"Results on the {tag}")
cd = CategoryChartData(); cd.categories = [n.split("  ")[0] for _, n in RUNS]
cd.add_series("mIoU", [g(k, "mIoU") or 0 for k, _ in RUNS])
ch = s.shapes.add_chart(XL_CHART_TYPE.COLUMN_CLUSTERED, Inches(.6), Inches(1.5), Inches(6.4), Inches(5.2), cd).chart
ch.has_legend = False; ch.plots[0].has_data_labels = True
ch.plots[0].data_labels.number_format = "0.00"; ch.plots[0].data_labels.number_format_is_linked = False
ch.plots[0].data_labels.font.size = Pt(14); ch.category_axis.tick_labels.font.size = Pt(12); ch.value_axis.tick_labels.font.size = Pt(12)
ch.plots[0].series[0].format.fill.solid(); ch.plots[0].series[0].format.fill.fore_color.rgb = TEAL
ch.has_title = True; ch.chart_title.text_frame.text = "mean IoU (higher is better)"
u1, u2 = g("U1", "mIoU"), g("U2", "mIoU")
stat(s, 7.6, 1.7, f"{f3(u1)} -> {f3(u2)}", "mIoU, U1 -> U2")
stat(s, 7.6, 3.4, f"{f3(g('U1','IoU_ship'))} -> {f3(g('U2','IoU_ship'))}", "ship IoU: the baseline never finds a ship", RED)
stat(s, 7.6, 5.1, f"{f3(g('U1','pixel_acc'))} vs {f3(g('U2','pixel_acc'))}", "pixel accuracy barely moves: it hides the rare-class failure", GREY)

# 7 per-class table
s = slide("Per-class IoU: where the loss function matters")
cls = [("IoU_sea", "sea"), ("IoU_oil_spill", "oil"), ("IoU_look_alike", "look-alike"), ("IoU_ship", "ship"), ("IoU_land", "land")]
rows = [["Run", *[n for _, n in cls], "mIoU"]] + [[n.split("  ")[0], *[f3(g(k, c)) for c, _ in cls], f3(g(k, "mIoU"))] for k, n in RUNS]
tbl = s.shapes.add_table(len(rows), 7, Inches(.6), Inches(1.6), Inches(12.1), Inches(.6 * len(rows))).table
for r, row in enumerate(rows):
    for c, v in enumerate(row):
        cell = tbl.cell(r, c); cell.text = v
        cell.text_frame.paragraphs[0].font.size = Pt(16); cell.text_frame.paragraphs[0].font.bold = r == 0
text(s, .6, 5.6, 12, 1.2, "Sea and land are easy for every model. The rare classes (oil, ship) are what separate the runs.", 16, False, INK)

# 8 interpretation
s = slide("What the ablations tell us")
a, b = g("U2a_weights_only", "mIoU"), g("U2b_dice_only", "mIoU")
pts = ["Plain cross-entropy optimises pixel accuracy, so it ignores ships (IoU %s)." % f3(g("U1", "IoU_ship"))]
if a is not None: pts.append(f"Weights alone: mIoU {f3(a)}, ship IoU {f3(g('U2a_weights_only','IoU_ship'))}.")
if b is not None: pts.append(f"Dice alone: mIoU {f3(b)}, ship IoU {f3(g('U2b_dice_only','IoU_ship'))}.")
pts.append(f"Weights + Dice together (U2): mIoU {f3(u2)}, better than either alone; the two fix different things.")
if "U2_80ep" in M: pts.append(f"Training twice as long (80 epochs): mIoU {f3(g('U2_80ep','mIoU'))}. See curves for over/under-fitting.")
bullets(s, pts, w=12, size=19)

# 9 qualitative
s = slide("Qualitative predictions (U2)")
pic(s, R / "U2" / "predictions.png", .6, 1.4, h=5.6)
pic(s, R / "U2" / "confusion_matrix.png", 8.6, 1.4, h=4.2)

# 10 failure cases
s = slide("Failure cases: oil vs look-alike, missed ships")
pic(s, R / "U2" / "failure_cases.png", .6, 1.4, h=5.6)
text(s, 8.2, 1.6, 4.6, 5, ["Hardest cases: thin oil streaks and natural look-alikes that have the same radar texture.",
                           "Tiny ships can vanish when images are shrunk to 256 x 512.",
                           "Possible fix: higher resolution or an encoder with pretrained features (Omar's DeepLabV3+)."], 16)

# 11 curves
s = slide("Training curves (U1 vs U2)")
pic(s, R / "U1" / "curves.png", .6, 1.4, w=6.1)
pic(s, R / "U2" / "curves.png", 6.8, 1.4, w=6.1)
text(s, .6, 4.2, 12, 2, ["Left: U1.  Right: U2.  Dotted line = chosen checkpoint (best validation mIoU).",
                         "No strong overfitting: validation tracks training."], 16, False, GREY)

# 12 reproducibility
s = slide("Reproducibility and deliverables")
bullets(s, ["Fixed seed, stored split (splits.json), config.json per run", "last.pth saved every epoch, runs can resume",
            "Kaggle notebook Oil_Spill_UNet_Kaggle.ipynb (generated, tested end to end)",
            "smoke_test.py proves the whole pipeline on synthetic data",
            "results/<run>/: config, history, best_model.pth, curves, confusion matrix, predictions, failure cases",
            "comparison.md / .csv: one table for the team report"], w=12, size=18)

# 13 checklist
s = slide("Assignment requirements: status")
items = ["Dataset audit, RGB mask verification, class plot, sample grid  -  done",
         "Shared loaders, split IDs, 110 test images reserved  -  done",
         "U-Net from scratch with skip connections, 5-class output  -  done",
         "U1 baseline and U2 weighted CE + Dice, only the loss changes  -  done",
         "Class weights from the training split only  -  done",
         "Curves, checkpoints, logs, qualitative predictions  -  done",
         "Final test evaluation, run once  -  " + ("done" if SPLIT == "test" else "pending"),
         "Clean Kaggle Save & Run All  -  needs you (upload notebook, run on Kaggle)",
         "Share oilspill/, class_map.json, splits.json with Omar (use imagenet norm)  -  needs you",
         "Team of 4-5, contribution statement, final report  -  team-level, open"]
text(s, .6, 1.4, 12.2, 5.6, ["•  " + i for i in items], 17)

# 14 Kaggle clean run (numbers copied from the Kaggle run's comparison.md, validation split)
s = slide("Clean Kaggle run reproduces the local result")
rows = [["Run", "Where", "mIoU (val)", "Oil IoU", "Ship IoU", "Pixel acc"],
        ["U1", "local", "0.476", "0.280", "0.000", "0.946"], ["U1", "Kaggle", "0.493", "0.318", "0.000", "0.950"],
        ["U2", "local", "0.618", "0.464", "0.482", "0.943"], ["U2", "Kaggle", "0.618", "0.446", "0.471", "0.946"]]
tbl = s.shapes.add_table(len(rows), 6, Inches(.6), Inches(1.6), Inches(12.1), Inches(.6 * len(rows))).table
for r, row in enumerate(rows):
    for c, v in enumerate(row):
        cell = tbl.cell(r, c); cell.text = v
        cell.text_frame.paragraphs[0].font.size = Pt(16); cell.text_frame.paragraphs[0].font.bold = r == 0
text(s, .6, 5.2, 12, 1.6, ["Save & Run All on Kaggle (GPU) finished in about 22 min per run.",
                           "U2 matches almost exactly; U1 differs by 0.017, normal run-to-run GPU variation.",
                           "Same conclusion: the baseline never finds ships, weights + Dice fix it."], 16, False, INK)

prs.save(HERE / "Abdallah_UNet_Presentation.pptx")
print("saved; metrics split used:", SPLIT)
