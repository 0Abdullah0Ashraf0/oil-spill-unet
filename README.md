# U-Net track (Abdallah): Oil Spill Segmentation

Local runs use the project venv (reuses the system PyTorch, adds matplotlib/nbclient):

    cd "<this folder>"
    .\.venv\Scripts\python.exe <script> ...

**For the actual submission use `Oil_Spill_UNet_Kaggle.ipynb`** (see "Kaggle" below). The `.py` files are the
single source of truth; the notebook is generated from them by `python build_notebook.py`.

## Order of work
1. **Get the data** (Kaggle: nabilsherif/oil-spill); unzip so `train/` and `test/` exist under one root,
   e.g. `E:\data\oil_spill`. Each split needs an images folder and a labels folder (RGB masks preferred).
2. **Audit (Phase U0)**, and read the printout, don't just trust it:

       python audit_dataset.py --data_root E:\data\oil_spill --out audit

   Check: exact-colour match %, image sizes, class share, **images containing each class** (all five must appear),
   `audit/sample_grid.png`. If masks are not exactly the assignment colours, fix `audit/class_map.json` BEFORE training.
3. **Send Omar** the `oilspill/` folder, `audit/class_map.json`, `splits.json` (created on the first train run).
   **Tell him to pass `norm="imagenet"` to `OilSpillDataset`** for his pretrained DeepLabV3+. The U-Net uses `"half"`
   ((x/255-0.5)/0.5); the PDF requires normalisation that matches the model's weights. Same split, same class map.
4. **U1 baseline**, then **U2**, identical settings, only the loss differs:

       python train.py --data_root E:\data\oil_spill --run_id U1 --loss ce
       python train.py --data_root E:\data\oil_spill --run_id U2 --loss wce_dice

   U2 changes two things at once (class weights AND Dice). To say which one helped, also run the ablations
   `--loss wce` (weights only) and `--loss ce_dice` (Dice only). Other deliberate knobs: `--aug flip_int`
   (adds mild brightness/contrast jitter to the image only), `--weight_method median_freq`, `--lr`, `--base`.
5. Look at `results/<run>/curves.png`. Compare per-class IoU (oil / ship / look-alike), not just mIoU.
6. **Only after all experiments and checkpoint choices are frozen**, once per run (this touches the test set):

       python evaluate.py --data_root E:\data\oil_spill --run_id U1 --final

   (Without `--final` it evaluates the validation split, which is safe while developing.)
7. `python compare.py --out results` builds `results/comparison.md/.csv` over every run folder, including Omar's if he
   follows the same contract (`config.json`, `metrics.json`).

## Kaggle (the real deliverable)
1. `python build_notebook.py` -> `Oil_Spill_UNet_Kaggle.ipynb`; upload it to Kaggle, add the `nabilsherif/oil-spill`
   dataset, turn on a GPU. Paths adapt automatically (`/kaggle/input` -> `/kaggle/working`).
2. Run it once with `EPOCHS` small first (the PDF says to test the full pipeline early), then *Save Version -> Save & Run All*.
3. `last.pth` is saved every epoch and `--resume` is on, so a stopped run continues instead of restarting.
4. Leave `FINAL = False` while experimenting. Set it to `True` once, at the very end, for the test-set numbers.
5. Make the notebook public/shared so the link works, and make sure outputs are visible (the PDF requires a working link).

## Design decisions to be able to defend in the report
- **Input size**: images are ~2:1 wide, so the default is a non-square 256x512 (`--height/--width`) instead of squashing to a
  square. Ships and oil spills are tiny; larger inputs help but cost GPU memory, so state the size you used and why.
- Split: seeded (`--seed 42`), 15% of train as validation, IDs in `splits.json`; the 110 test images are never read by `train.py`.
- Class weights are computed from the **training split only**.
- Checkpoint rule: best **validation** mIoU. Absent-class policy: excluded from the mean (reported null).
- Loss is computed in fp32 even with mixed precision (fp16 logits gave NaN with plain CE).
- U-Net: from scratch, BatchNorm, transposed-conv upsampling, 4 down/up stages, no pretrained weights. Say what skip
  connections are for (they carry fine spatial detail from the encoder to the decoder; important for thin/small regions).
- Reproducibility limits to state: seeded, but AMP/cuDNN are not bit-exact; `--deterministic` trades speed for repeatability.

## Verify the code runs (no dataset needed)
    python smoke_test.py      # synthetic data, a few minutes; also executes the generated notebook; writes to _smoke/
The assignment's "half grade" rule targets defaults/AI code with no real experiments: your experiment choices and
written justification are what count.
