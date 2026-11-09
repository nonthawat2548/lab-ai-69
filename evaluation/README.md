# Google Forms photo test — blind external image evaluation

This code tests the already-trained EfficientNet-B0 classifier on photos supplied through a private Google Forms response sheet. It does not train the model, modify the public Space, or upload participant data to GitHub/Hugging Face.

## Separation of data and labels

1. Request only response columns C:I (consent, Drive photo links, reference menu, metadata, rights). Never request the email column.
2. Use consent/rights to select photo IDs. Split multi-file answers into one record per image.
3. Download images through the current Colab user's authenticated Drive API.
4. `predict_images` receives only `file_id` and image `path`. It rejects records containing a gold label or response text. The model receives only normalized RGB image tensors, not captions, menu names, prompts, or reference labels.
5. Persist immutable results keyed by image SHA-256, model SHA-256 and preprocessing version before scoring.
6. Separately verify reference labels by humans. The annotation view deliberately does not display predictions.
7. Match predictions to independently confirmed labels and compute top-1 Accuracy and Macro F1.

## Where files are stored

All participant photos, file IDs, annotations, individual predictions, and reports remain in the user's private `MyDrive/ThaiFoodAI_THFOOD50/external_form_test/<sheet-id>/` directory. This repository contains only generic source, tests, and documentation. The source Google Sheet is read-only and never modified.

## Running

The user's existing Colab contains a quick-start section that invokes:

```python
context = module.connect_and_predict(spreadsheet_id, exact_response_tab, class_thai)
module.review_labels(context)
```

Google may require authorization once. The script uses CPU inference and refuses to train when no checkpoint exists. It reuses cached images and predictions when possible. To refresh newly submitted Form responses, rerun this quick-start cell; prior human annotations are preserved.

Review each image, select its real reference menu, mark whether it is inside the model's 50 classes, then confirm the label. The same response can contain multiple dishes or labels such as "หลายอย่าง"; these are never silently mapped to one class.

## Metrics

- Accuracy: exact top-1 matches on unique, human-verified, in-scope photos, including low-confidence wrong predictions.
- `macro_f1_present_classes`: equal mean F1 over reference classes represented in the collected photos.
- `macro_f1_all_50_classes`: equal mean F1 across all 50 classes, with zero for missing classes; explicitly labeled to avoid implying that a small form collection covers the whole dataset.
- Pending labels, unsupported dishes, reviewer exclusions, inaccessible/unreadable files, and duplicate image bytes are counted separately. No score is produced if nothing is eligible.
- Unsupported dishes are not part of closed-set classification Accuracy; their forced model predictions do not demonstrate out-of-distribution detection.
- Form photos are external evaluation only. Keep them out of training and tuning if reporting an external test result. Reviewers should independently confirm references without seeing model predictions.

## Offline tests

`python evaluation/test_form_image_eval.py`

Tests cover image/reference separation, consent and rights filtering, multi-photo responses, label preservation, duplicate content, unknown dishes, low-confidence errors, missing references, and known metric values. They do not claim a completed real Colab run or new accuracy result.
