# วิธีใช้ Kaggle — ThaiFood AI 150 ประเภท

โค้ดนี้ยังไม่ใช่โมเดล 150 ประเภทที่ฝึกเสร็จ และไม่ได้เปลี่ยน Hugging Face ที่ใช้อยู่

# Train ThaiFood AI 150 with Kaggle GPU

This folder provides a Kaggle-ready **candidate training** flow. The existing 50-class Hugging Face app remains untouched.

1. Open [Kaggle Code](https://www.kaggle.com/code) and create a new notebook with an available GPU accelerator in notebook settings. Kaggle GPU availability and usage limits vary.
2. Download the original food datasets outside the notebook if necessary. THFOOD-100 (~15.7 GB) and THDESS-20 (~738 MB) are linked in [their repositories](https://github.com/NawanolT/THFOOD-100) and [THDESS-20](https://github.com/NawanolT/THDESS-20). **Educational/non-commercial only.** Do not publish their images in this public GitHub repository.
3. Review original class labels, synonyms, duplicate dishes and split leakage. Obtain verified photographs for **exactly 150 unique classes** including all 50 original classes; the 100+20 source classes alone do **not** guarantee 100 new unique classes. Place photos under `raw_images/<class_name>/` and prepare with `prepare_dataset.py --raw raw_images --out data --checkpoint food_model_best.pth`. This creates `data/{train,val,test}/<class_name>/` and `data/class_names.json`.
4. Make the prepared `data/` folder and read-only original `food_model_best.pth` available as *private* Kaggle Input Datasets, following platform rules. You may use a Kaggle Dataset generated from your own files. Avoid uploading external dataset photos without redistribution permission.
5. Upload/open [ThaiFood150_Kaggle.ipynb](ThaiFood150_Kaggle.ipynb) in Kaggle, set GPU, enable Internet if allowed, add the required input datasets, **edit DATA_DIR and OLD_MODEL**, and run all cells.
6. Each completed epoch writes `/kaggle/working/thai_food_150/last_checkpoint.pth` and `food_model_150_best.pth`. Save a **Notebook Version with outputs**. To resume in a new session, attach the previous saved outputs as Kaggle Input, set `RESUME_CHECKPOINT` to its `last_checkpoint.pth`, and set `TOTAL_EPOCHS` to the **total desired number** (larger than previously completed epoch count). The checkpoint includes optimizer state and best model state.
7. After final training, inspect `metrics.json`, test on unseen external photos, verify food labels and nutrition data, then deploy the reviewed candidate to Hugging Face **separately**. Do not replace a live 50-class checkpoint before validation.

**If Kaggle blocks GitHub Internet:** Download the repository as a ZIP on your machine and attach only the code folder as Kaggle Input; update the script path in the notebook accordingly.

**Troubleshooting:** GPU memory error: set `BATCH_SIZE=4` or `2`; an interrupted epoch is repeated when resuming, because checkpoints are committed only at the end of each epoch. Store outputs outside the ephemeral runtime to survive session loss.
