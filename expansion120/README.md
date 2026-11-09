# ThaiFood AI — 120-menu image-model expansion

The target is now 120 menus: preserve the original 50 labels and their order, add 61 source labels from THFOOD-100, and add nine dessert labels from THDESS-20. The previous 200-menu recipe-led draft is superseded for model development. A recipe source is not evidence that its menu is recognized by the image model.

Actual ZIP-directory inspection confirms 53,459 images in THFOOD-100 and 5,242 images in THDESS-20, with original train/val/test folders. All 120 targets have an identified public image source. Eleven broad/specific label conflicts were excluded from automatic source merging. Image-level duplicate review and evaluation are still required.

`classes_120.csv` holds the target mapping. `source_image_counts.json` holds the counts read from the source ZIP directories. `progress.json` records the current model status. `download_sources.py` downloads public archives with resumable, validated HTTP ranges. `build_dataset.py` builds a subset using the original source splits, and `audit_dataset.py` checks training readiness. `train_candidate.py` caches frozen EfficientNet features and evaluates a new classifier head using fresh held-out test images. Model outputs are candidates until tested and released.

The first planned run uses up to 100 training, 20 validation and 20 test images per class, subject to source availability, deduplication and audit. Validation/test minimums are ten images per class. This is a bounded development experiment, not a guarantee of accuracy. Google Forms photos remain an external test set and are not read by these scripts. Training is performed outside Colab; no Colab Compute Units are used.

Sources and usage: https://github.com/chakkritte/THFOOD-50 ; https://huggingface.co/datasets/thean/THFOOD-50 ; https://github.com/NawanolT/THFOOD-100 ; https://github.com/NawanolT/THDESS-20 . The original collections specify non-commercial research/educational use. Do not redistribute the photographs without checking the applicable permission.
