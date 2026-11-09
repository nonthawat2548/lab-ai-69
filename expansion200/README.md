# ThaiFood AI: target expansion from 50 to 200 image classes

**Status: dataset preparation, not a trained 200-class model.** The production Hugging Face app and its checkpoint still recognize 50 classes. No GPU job, Colab notebook, dataset download, or model training is started by this folder.

## What is prepared

- `classes_200.csv`: exactly 200 distinct target labels: 50 legacy THFOOD-50 classes + 150 additional candidate menus referenced by the user's recipe-index page.
- `images_manifest_template.csv`: empty per-image provenance/review/split manifest. Empty means **no new training images have been collected by this change**.
- `readiness.json`: honest, machine-readable current status and candidate image sources.
- `audit_dataset.py`: checks label vocabulary, actual file existence, usage rights records, human labels, image checksums, collection groups and cross-split leakage before training is allowed.
- `test_expansion.py`: verifies the target taxonomy, missing-data refusal, duplicate-photo leakage and missing-rights/labels rejection without installing ML frameworks.

## Target-label rules

Keep all 50 existing class IDs and their ordering (indices 0–49); additional candidates receive stable `ThaiFood051`–`ThaiFood200` IDs. Existing checkpoint/class order is not changed. New examples include crab fried rice, southern fermented-fish curry, fried-egg salad, steamed coconut desserts and other names referenced in the supplied index.

The 150 additions are a **draft target taxonomy**, not proof of 150 datasets. Obvious aliases and several variants of existing generic labels (e.g. green curry, fried chicken, basil stir-fry) were filtered out to avoid claiming duplicate categories. Borderline visually similar dishes still require human class-boundary review when actual images are available. A source link proves a menu listing, not a completed image dataset or verified nutritional value.

The added targets include 20 rice/noodle dishes, 35 curry/soup/steamed dishes, 40 stir-fried/fried/grilled/other dishes, 30 salads/dips, and 25 desserts. If image availability requires replacing a candidate, keep a versioned taxonomy change and rebuild the label mapping before training.

## Candidate data sources (not automatically combined)

1. **THFOOD-50**: https://github.com/chakkritte/THFOOD-50 and https://huggingface.co/datasets/thean/THFOOD-50 — existing 50-class data, 15,770 images according to its card. Review label and split compatibility before a new experiment. Source repository specifies non-commercial research/educational use.
2. **THFOOD-100**: https://github.com/NawanolT/THFOOD-100 — source card reports 53,459 images and 100 classes, for non-commercial research and educational use. The 100 class names have **not** been mapped to our 200 target labels, and overlapping datasets must not be counted as new independent classes/images.
3. **Recipe names and recipe-reference URLs**: https://nlovecooking.com/อาหารไทย/ — extracted menu titles/links only. No full articles, image files or nutrition figures were copied. A recipe's illustrative photo alone is not enough training data for a new class. Confirm image usage rights and gather diverse independent images before training.
4. User Google Forms collection: keep these images as held-out external evaluation as previously requested. Do not move them into training, validation or augmentation without an explicit new split protocol.

## Image manifest

One row per image:

`image_path,class_id,split,source_url,collection_group,rights_status,human_reviewed,sha256,phash`

- `split`: `train`, `val`, or `test`; original photos/duplicates/augmentations from the same collection group must not cross splits.
- `rights_status`: documented provenance (`owned_with_consent`, `dataset_license_permits_education`, `permission_granted`, `public_domain`, `cc_by`, `cc_by_sa`, or `cc0`). Do not infer a grant just because an image is publicly visible.
- `human_reviewed`: `true` only after checking the actual dish and image. Do not take a model prediction as its own training or evaluation reference.
- `sha256`: exact file content checksum; `phash`: perceptual hash for near-duplicate review. Missing perceptual hashes are flagged, not treated as proof of no near-duplicates.
- Only put image metadata in the public repository when its source/rights permit publication; keep private participant images and identifiers out of GitHub.

Run:

```sh
python expansion200/test_expansion.py
python expansion200/audit_dataset.py --manifest /path/to/images.csv --image-root /path/to/photos --output /path/to/readiness_report.json
```

The planning floor defaults to **100 train + 20 validation + 20 test images per class**, configurable with CLI flags. This is a starting data-volume gate, not a promise of model accuracy. An empty manifest deliberately exits with code 2 (`ready_for_training=false`).

## After real data is ready

1. Freeze human-reviewed class boundaries and dataset version; check duplicates/near-duplicates before splitting.
2. Audit all 200 classes and maintain held-out external Forms photos separately.
3. Train a **candidate** 200-output classifier using cached image features/transfer learning on suitable compute in one planned experiment. Do not attempt to reuse a 50-output head as a 200-output model or rename labels to fake extra capabilities.
4. Measure fresh 200-class Accuracy, Macro F1 and confusion matrix, plus a retained 50-class slice to detect regressions. Do not reuse the old 50-class metrics as evidence for 200 classes.
5. Promote the candidate checkpoint + complete label/recipe mappings to the Space only after acceptance. Keep the original model for rollback.

**Current next requirement:** audited images covering all 150 additional menus, with a verified mapping for any public datasets used. Until those images exist and a 200-class model is actually trained and measured, the live system remains a 50-class classifier.
