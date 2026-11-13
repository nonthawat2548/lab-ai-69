# ThaiFood AI: 150-class image dataset preparation

**Status:** Code/setup only. The full image archives have **not** been uploaded to this GitHub repository, and a 150-class model has **not** been trained.

This project currently deploys a 50-class model. Do not replace its checkpoint with an untrained 150-class classifier.

## Original datasets

| Dataset | Images | Size | Source |
| --- | ---: | ---: | --- |
| THFOOD-100 | 53,459 | ~15.7 GB | https://github.com/NawanolT/THFOOD-100 |
| THDESS-20 | 5,242 | ~738 MB | https://github.com/NawanolT/THDESS-20 |
| THFOOD-100 sample | sample subset | ~145 MB | THFOOD-100 README |

Both dataset owners explicitly permit **non-commercial research and educational use only**. Cite the original papers and verify redistribution rights before posting any images. Source category names can overlap with the original 50 labels or each other; 100 + 20 is not automatically 120 *new* classes.

### Download to your PC or Google Colab

```bash
pip install gdown
python thai_food_150/download_datasets.py --out downloaded_archives --datasets thfood100_sample
# To download both full datasets (~16.4 GB, with extra disk space for extraction):
python thai_food_150/download_datasets.py --out downloaded_archives --datasets thfood100_full thdess20_full
```

If Google Drive enforces access or quota limits, use the download links in the original authors' READMEs. The downloader cannot bypass those limits.

### Build and train after downloading

1. Extract archives **outside GitHub**, preserving the original label and split metadata.
2. Review overlapping dish labels with the 50-class checkpoint and select exactly 100 valid new classes. The repository already contains related mapping work in `expansion120/` and `expansion200/`.
3. Gather images of **all 50 original classes**, too, to avoid catastrophic forgetting. Put verified photos into `raw_images/<class_id>/`.
4. Use `prepare_dataset.py` and `train_150.py` from the 150-class starter package to produce train/val/test splits and train a candidate checkpoint.
5. Evaluate test-set performance, ensure class-name order and metadata match, extend nutrition and knowledge records, and only then deploy.

**Important:** Do not commit the downloaded archives or thousands of photos to the normal GitHub repository. Prefer cloud storage for datasets, and release just scripts, manifests, licenses, citations, and small metadata here.
