---
title: "ระบบระบุเมนูอาหารไทยจากภาพถ่าย พร้อมแชทบอทแนะนำโภชนาการ"
emoji: 🍲
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
license: other
---

# ระบบระบุเมนูอาหารไทยจากภาพถ่าย พร้อมแชทบอทแนะนำโภชนาการ

โครงงาน ThaiFood AI จำแนกอาหารไทย **50 เมนู** จากภาพถ่ายด้วย EfficientNet-B0 และให้ถามข้อมูลโภชนาการเพิ่มเติมผ่านแชทบอท โดยใช้ข้อมูลสูตรทั่วไป ค่าพลังงานจากแหล่งอ้างอิง และ Retrieval-Augmented Generation (RAG)

## การใช้งาน

1. เปิดแท็บ **วิเคราะห์ภาพอาหาร** แล้วอัปโหลดภาพ ระบบจะวิเคราะห์ให้โดยอัตโนมัติ
2. ดูชื่ออาหารที่ AI คาดว่าเป็นและคะแนน Softmax พร้อมคำเตือนหากความมั่นใจต่ำ
3. เปิดแท็บ **แชทบอทแนะนำโภชนาการ** เพื่อถามต่อ เช่น "ที่ถ่ายไปเมนูอะไร", "จานนี้กี่แคล", "มีส่วนผสมอะไรบ้าง", "แพ้อะไรได้บ้าง"
4. หากไม่ระบุภาพหรือชื่อเมนู ระบบจะไม่สมมติว่ามีภาพหรือแต่งตัวเลขโภชนาการขึ้นเอง

## ข้อมูลและข้อจำกัด

- จำแนกภาพได้เฉพาะ 50 เมนูตามชุดข้อมูล THFOOD-50
- `knowledge.py`: รายชื่อ/ส่วนผสมทั่วไปของ 50 เมนูและแหล่งอ้างอิง
- `nutrition_50.csv`: ข้อมูลครบ 50 แถว โดยมี **ตัวเลขพลังงานที่มีหลักฐานเฉพาะ 22 เมนู** เมนูอื่นอาจมีค่าประมาณที่ยังไม่ผ่านการยืนยัน ซึ่งต้องติดป้ายระบุ
- `evaluation.json`: บันทึกผลประเมินจากชุดทดสอบเดิม ไม่ได้ประเมินใหม่ในการ Deploy
- `pdf_sources.json`: รายการลิงก์แหล่งข้อมูล PDF; นำ PDF เข้า RAG เฉพาะเมื่อดาวน์โหลดและถอดข้อความไทยที่อ่านได้จริง
- `food_model_best.pth`: Checkpoint เดิมอยู่ใน **Hugging Face Space** ไม่ได้เก็บใน GitHub และไม่มีการฝึกโมเดลใหม่ขณะเปิดแอป
- คะแนน Softmax ไม่ใช่ความแม่นยำของโมเดลทั้งชุด และภาพถ่ายไม่สามารถระบุปริมาณแคลอรีหรือสารก่อภูมิแพ้จริงได้

## Source of truth / อัปเดตโค้ด

โค้ดและไฟล์ข้อมูลสำหรับ Space จัดเก็บที่ GitHub:
`nonthawat2548/lab-ai-69` สาขา `thaifood-space-deploy` ในโฟลเดอร์ `thaifood_space/`

GitHub Actions workflow `.github/workflows/thaifood-hf-space.yml` ตรวจข้อมูลก่อนอัปโหลดไฟล์ที่อนุญาตเข้า Space ด้วย Hugging Face Trusted Publisher (OIDC) โดย **ไม่ลบและไม่เขียนทับโมเดล .pth**


## Additional recipe references for typed questions

The text chatbot can answer ingredient questions for 103 additional menus and provide source links for 10 more menus from NLoveCooking. Each answer links to its source and distinguishes recipe ingredients from ingredients actually present in a photograph. Recipe lookup does not expand the 50-class vision model. Energy figures and medical advice are not generated from these recipes. Thirty-seven candidate pages are excluded pending title/content review. The recipe data was extracted from available page previews; missing or partial lists should be checked against the linked original recipe.
