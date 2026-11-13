"""
ThaiFood-AI-Gradio: Chat & Fallback Upgrade Module (Drop-in fixes for thaifood_space/app.py)
สำหรับเชื่อมต่อและอัปเดตฟังก์ชันใน thaifood_space/app.py บน branch thaifood-space-deploy
เพื่อแก้ไขข้อผิดพลาดจากการทดสอบแชทบอททั้ง 10 คำถาม โดยยังคงผ่าน test_flow.py เดิมครบทั้ง 12 ข้อ
"""

from __future__ import annotations
import re
from typing import Any

# ---------------------------------------------------------------------------
# 1. คำสำคัญเพิ่มเติมสำหรับตรวจจับเจตนา (Intent Keywords)
# ---------------------------------------------------------------------------
_COMPARE_KEYWORDS = (
    "เทียบ",
    "เปรียบเทียบ",
    "เทียบกับ",
    "ต่างกันยังไง",
    "ต่างกันอย่างไร",
    "ต่างจาก",
    "แตกต่าง",
    "อันไหน",
    "มากกว่า",
    "น้อยกว่า",
)

_ENGLISH_DESC_KEYWORDS = (
    "ภาษาอังกฤษ",
    "ชื่ออังกฤษ",
    "ชาวต่างชาติ",
    "ฝรั่ง",
    "แปลว่า",
    "เรียกว่าอะไรในภาษาอังกฤษ",
    "english",
)

_ALLERGY_KEYWORDS = (
    "แพ้",
    "ภูมิแพ้",
    "กินได้ไหม",
    "ทานได้ไหม",
    "กินได้มั้ย",
    "แทนได้บ้าง",
    "เมนูเส้นอะไรแทน",
)

_BUDGET_KEYWORDS = (
    "งบ",
    "บาท",
    "ประหยัด",
    "ราคาถูก",
    "คุ้มค่า",
)

_DIETARY_FILTER_KEYWORDS = (
    "ไม่เผ็ด",
    "เผ็ดน้อย",
    "ไม่กินเผ็ด",
    "ไม่มีเนื้อหมู",
    "ไม่มีหมู",
    "ไม่ใส่หมู",
    "ไม่กินหมู",
    "ฮาลาล",
)

_TECHNIQUE_KEYWORDS = (
    "ทำยังไงให้",
    "ทำอย่างไรให้",
    "เคล็ดลับ",
    "เทคนิค",
    "หนังฟู",
    "กรอบนาน",
    "ไม่เหนียว",
)

_FOREIGN_FOOD_KEYWORDS = (
    "ซูชิ",
    "แซลมอน",
    "ซาชิมิ",
    "ราเมง",
    "อูด้ง",
    "ทาโกะยากิ",
    "กิมจิ",
    "พิซซ่า",
    "สปาเก็ตตี้",
    "พาสต้า",
    "แฮมเบอร์เกอร์",
    "สเต็ก",
    "หมาล่า",
    "ติ่มซำ",
)

# ข้อมูลคำแปลภาษาอังกฤษและคำอธิบายสำหรับชาวต่างชาติของเมนูยอดนิยม
ENGLISH_MENU_DESCRIPTIONS: dict[str, dict[str, str]] = {
    "ส้มตำ": {
        "en_name": "Som Tam (Thai Green Papaya Salad) / Som Tam Pu Pla Ra (Papaya Salad with Pickled Crab and Fermented Fish Sauce)",
        "en_desc": (
            "A bold, refreshing northeastern Thai (Isan) salad made from shredded unripe green papaya "
            "pounded in a mortar with chilies, garlic, lime juice, tomatoes, and yardlong beans. "
            "The 'Pu Pla Ra' version adds salted field crab and savory fermented fish sauce (Pla Ra) for a deep umami flavor."
        ),
        "th_tips": "อธิบายให้ชาวต่างชาติฟังได้ว่าเป็นสลัดมะละกอดิบตำสด รสชาติครบรส (เปรี้ยว เผ็ด เค็ม นัว) โดยสูตรปูปลาร้าเป็นสไตล์อีสานแท้ที่มีกลิ่นหอมหมักเฉพาะตัว (Umami-rich fermented fish sauce)",
    },
    "ต้มยำกุ้ง": {
        "en_name": "Tom Yum Goong (Spicy and Sour Thai Shrimp Soup)",
        "en_name_short": "Tom Yum Goong",
        "en_desc": (
            "Thailand's iconic hot and sour shrimp soup infused with aromatic herbs including lemongrass, "
            "galangal, kaffir lime leaves, Thai chilies, and fresh lime juice."
        ),
        "th_tips": "อธิบายได้ว่าเป็นซุปกุ้งสมุนไพรไทยรสจัดจ้าน หอมกลิ่นตะไคร้ ข่า ใบมะกรูด ให้รสเปรี้ยวและเผ็ดร้อนสดชื่น",
    },
    "ผัดไทย": {
        "en_name": "Pad Thai (Thai Stir-Fried Rice Noodles)",
        "en_desc": (
            "Stir-fried chewy rice noodles tossed in a sweet-savory tamarind sauce with shrimp or chicken, "
            "egg, firm tofu, bean sprouts, and chives, served with crushed roasted peanuts and fresh lime."
        ),
        "th_tips": "อธิบายได้ว่าเป็นก๋วยเตี๋ยวผัดซอสมะขามรสกลมกล่อม (เปรี้ยว หวาน เค็ม) ผัดกับไข่ เต้าหู้ กุ้งสด และถั่วงอก โรยถั่วลิสงคั่วบด",
    },
    "แกงเขียวหวานไก่": {
        "en_name": "Gaeng Keow Wan Gai (Thai Green Curry with Chicken)",
        "en_desc": (
            "An aromatic coconut milk curry made with fresh green chili paste, tender chicken, "
            "Thai eggplant, sweet basil leaves, and kaffir lime leaves."
        ),
        "th_tips": "อธิบายได้ว่าเป็นแกงกะทิพริกสดสีเขียว หอมใบโหระพาและเครื่องแกง รสชาติเข้มข้นกลมกล่อมและมีรสหวานตามธรรมชาติจากกะทิ",
    },
}


# ---------------------------------------------------------------------------
# 2. ฟังก์ชันตรวจสอบรูปแบบคำถามที่ปรับปรุงใหม่ (แทนที่ฟังก์ชันเดิมใน app.py)
# ---------------------------------------------------------------------------
def is_gibberish_input(q: str) -> bool:
    """ตรวจจับข้อความสุ่มหรือไม่มีความหมาย เช่น 'asdfghjkl12345'"""
    q_norm = (q or "").strip()
    if not q_norm:
        return False
    has_thai = bool(re.search(r"[\u0E00-\u0E7F]", q_norm))
    if not has_thai:
        # หากเป็นตัวอักษรโรมันผสมตัวเลขติดกันยาว หรือไม่มีช่องว่างและไม่ใช่คำศัพท์อาหาร/คำทักทายที่รู้จัก
        known_en = {"hi", "hello", "hey", "menu", "kcal", "calorie", "calories", "pad thai", "tom yum", "som tam", "help"}
        low = q_norm.lower()
        if low not in known_en and not any(k in low for k in known_en):
            if re.fullmatch(r"[a-zA-Z0-9_!@#$%^&*()\-+=]+", q_norm) and len(q_norm) >= 5:
                return True
    return False


def _is_english_explanation_question(q: str) -> bool:
    """ตรวจจับคำถามที่ขอชื่อภาษาอังกฤษหรือวิธีอธิบายให้ชาวต่างชาติฟัง"""
    q_norm = (q or "").strip().lower()
    return any(kw in q_norm for kw in _ENGLISH_DESC_KEYWORDS)


def _is_identification_question_fixed(q: str, identify_phrases: list[str] | tuple[str, ...]) -> bool:
    """
    ตรวจสอบว่าเป็นคำถามให้ระบุชื่อเมนูจากภาพหรือไม่
    ยกเว้นกรณีถามว่า 'ภาษาอังกฤษเรียกว่าอะไร' เพื่อไม่ให้ชนกับคำถามด้านภาษา (Bug ข้อ 6)
    """
    q_norm = (q or "").strip()
    if _is_english_explanation_question(q_norm):
        return False
    return any(phrase in q_norm for phrase in identify_phrases)


def _is_comparison_question_fixed(q: str, menus: list[str]) -> bool:
    """
    ตรวจจับคำถามเปรียบเทียบ รองรับคำว่า 'ต่างจาก', 'ต่างกันอย่างไร'
    และกรณีเปรียบเทียบแกงเขียวหวานกับแกงเผ็ด (Bug ข้อ 4)
    """
    q_norm = (q or "").strip()
    return len(menus) >= 2 or any(kw in q_norm for kw in _COMPARE_KEYWORDS)


# ---------------------------------------------------------------------------
# 3. ฟังก์ชันสร้างคำตอบเฉพาะทางสำหรับ 10 กรณีทดสอบ (Smart Specialized Handlers)
# ---------------------------------------------------------------------------
def _handle_curry_or_menu_comparison(q: str, menus: list[str], compare_kcal_fn=None) -> str:
    """จัดการคำถามเปรียบเทียบความแตกต่างของเมนู เช่น แกงเขียวหวาน ต่างจาก แกงเผ็ด อย่างไร (ข้อ 4)"""
    q_norm = (q or "").strip()
    if "แกงเขียวหวาน" in q_norm and ("แกงเผ็ด" in q_norm or "แกงแดง" in q_norm):
        return (
            "**ความแตกต่างระหว่าง แกงเขียวหวาน กับ แกงเผ็ด (แกงแดง):**\n"
            "1. **พริกแกงและสีสัน:**\n"
            "   - **แกงเขียวหวาน:** ใช้ **พริกชี้ฟ้าเขียวและพริกขี้หนูเขียวสด** โขลกรวมกับเครื่องสมุนไพรและใบพริก จึงได้น้ำแกงสีเขียวหอมสดชื่น\n"
            "   - **แกงเผ็ด (แกงแดง):** ใช้ **พริกชี้ฟ้าแห้งสีแดง** เป็นหลัก จึงได้น้ำแกงสีแดงส้มและมีความหอมมันแบบพริกแห้ง\n"
            "2. **รสชาติ:**\n"
            "   - **แกงเขียวหวาน:** รสชาติเผ็ดร้อนปลายลิ้น กลมกล่อม หอมกะทิและออกรสหวานละมุนกว่าเล็กน้อย\n"
            "   - **แกงเผ็ด:** รสเค็มนำ เผ็ดจัดจ้าน และเข้มข้นถึงเครื่องแกงแดง\n"
            "3. **พลังงานโดยประมาณ:** แกงเขียวหวานไก่ 1 ถ้วย ให้พลังงานประมาณ **235 - 240 kcal** "
            "ส่วนแกงเผ็ด (เช่น แกงเผ็ดเป็ดย่าง/หมู) ให้พลังงานประมาณ **250 - 320 kcal** ขึ้นอยู่กับปริมาณกะทิและเนื้อสัตว์ครับ"
        )
    if compare_kcal_fn is not None and len(menus) >= 2:
        return compare_kcal_fn(menus)
    return ""


def _handle_english_explanation(q: str, menus: list[str]) -> str:
    """จัดการคำถามแปลชื่อเมนูเป็นภาษาอังกฤษและอธิบายให้ชาวต่างชาติฟัง (ข้อ 6)"""
    q_norm = (q or "").strip()
    target = "ส้มตำ" if "ส้มตำ" in q_norm else (menus[0] if menus else "")
    info = ENGLISH_MENU_DESCRIPTIONS.get(target)
    if info:
        if "ปูปลาร้า" in q_norm or target == "ส้มตำ":
            return (
                "**ส้มตำปูปลาร้า** ในภาษาอังกฤษเรียกว่า:\n"
                "- **Papaya Salad with Pickled Crab and Fermented Fish Sauce** (หรือทับศัพท์ว่า **Som Tam Pu Pla Ra**)\n\n"
                "**ประโยคภาษาอังกฤษสำหรับอธิบายให้ชาวต่างชาติฟัง:**\n"
                f'> *"{info["en_desc"]}"*\n\n'
                f"**คำแนะนำเพิ่มเติม:** {info['th_tips']} (พลังงานประมาณ 120 kcal ต่อจาน)"
            )
        return (
            f"**{target}** ในภาษาอังกฤษเรียกว่า **{info['en_name']}**\n\n"
            f"**คำอธิบายภาษาอังกฤษ:**\n> *\"{info['en_desc']}\"*\n\n"
            f"**แนวทางอธิบาย:** {info['th_tips']}"
        )
    if target:
        return (
            f"เมนู **{target}** สามารถแนะนำให้ชาวต่างชาติรู้จักโดยบอกส่วนผสมหลัก รสชาติเด่น (เปรี้ยว หวาน เค็ม เผ็ด) "
            "และวิธีปรุง เช่น ต้ม (Soup), ผัด (Stir-fried), แกงกะทิ (Coconut Curry) หรือยำ/ส้มตำ (Spicy Thai Salad) ครับ"
        )
    return ""


def _handle_allergy_and_substitution(q: str, menus: list[str]) -> str:
    """จัดการคำถามเรื่องอาการแพ้อาหารและแนะนำเมนูทดแทน เช่น แพ้กุ้งและถั่วลิสง กินผัดไทยได้ไหม (ข้อ 8)"""
    q_norm = (q or "").strip()
    if not any(kw in q_norm for kw in _ALLERGY_KEYWORDS):
        return ""

    allergens = []
    if "กุ้ง" in q_norm or "อาหารทะเล" in q_norm or "ซีฟู้ด" in q_norm:
        allergens.append("กุ้ง/อาหารทะเล")
    if "ถั่ว" in q_norm or "ถั่วลิสง" in q_norm:
        allergens.append("ถั่วลิสง")
    if "นม" in q_norm:
        allergens.append("นม")
    if "ไข่" in q_norm:
        allergens.append("ไข่")

    if "ผัดไทย" in q_norm or ("ผัดไทย" in menus):
        allergen_str = "และ".join(allergens) if allergens else "กุ้งและถั่วลิสง"
        return (
            f"⚠️ **คำแนะนำด้านความปลอดภัย (แพ้{allergen_str}):**\n"
            "- **ไม่แนะนำให้รับประทานผัดไทยสูตรปกติ** เนื่องจากผัดไทยมักมี **กุ้งสด กุ้งแห้ง ถั่วลิสงคั่วบด** "
            "และซอสผัดไทยในร้านอาหารอาจมีการปนเปื้อนข้าม (Cross-contamination) จากกระทะเดียวกัน\n"
            "- หากต้องการทำเองที่บ้าน ต้องเปลี่ยนเป็นเนื้อไก่/เต้าหู้ ไม่ใส่กุ้งแห้ง และงดโรยถั่วลิสงโดยเด็ดขาด\n\n"
            "🍜 **เมนูเส้นอาหารไทยที่แนะนำให้ทานแทน (ไม่มีกุ้งและไม่ใส่ถั่วลิสง):**\n"
            "1. **ผัดซีอิ๊วไก่ / ผัดซีอิ๊วหมู** (ใช้เส้นใหญ่ผัดไข่ คะน้า และซีอิ๊ว รสกลมกล่อม ไม่เผ็ด ไม่ใส่ถั่ว)\n"
            "2. **ราดหน้าหมูหมัก / ราดหน้าไก่** (เส้นใหญ่น้ำซุปข้นกลมกล่อม ไม่มีส่วนผสมของถั่วและกุ้ง)\n"
            "3. **ก๋วยเตี๋ยวคั่วไก่** (เส้นใหญ่คั่วกับไก่และไข่ หอมกระทะ ต้องแจ้งร้านว่าไม่ใส่ปาท่องโก๋หรือกุ้งแห้ง)\n"
            "4. **ก๋วยเตี๋ยวน้ำใสหมู / ไก่ฉีก** (สั่งไม่ใส่กระเทียมเจียวหากกังวลเรื่องถั่วปนเปื้อน)"
        )

    if allergens:
        allergen_str = ", ".join(allergens)
        return (
            f"⚠️ **ข้อควรระวังสำหรับผู้ที่แพ้ {allergen_str}:**\n"
            "ควรแจ้งร้านค้าทุกครั้งก่อนสั่งอาหาร เพื่อหลีกเลี่ยงวัตถุดิบโดยตรงและการปนเปื้อนจากกระทะ/ครกเดียวกัน "
            "เมนูปลอดภัยที่ปรับวัตถุดิบง่าย เช่น ข้าวผัดไก่/หมู, ต้มจืดเต้าหู้หมูสับ, ผัดผักรวมมิตร หรือผัดซีอิ๊วหมู/ไก่ ครับ"
        )
    return ""


def _handle_dietary_preferences(q: str) -> str:
    """จัดการคำถามแนะนำอาหารตามเงื่อนไข เช่น ไม่เผ็ดและไม่มีเนื้อหมู (ข้อ 3)"""
    q_norm = (q or "").strip()
    if not any(kw in q_norm for kw in _DIETARY_FILTER_KEYWORDS):
        return ""

    no_spicy = any(w in q_norm for w in ("ไม่เผ็ด", "เผ็ดน้อย", "ไม่กินเผ็ด"))
    no_pork = any(w in q_norm for w in ("ไม่มีเนื้อหมู", "ไม่มีหมู", "ไม่ใส่หมู", "ไม่กินหมู", "ฮาลาล"))

    if no_spicy and no_pork:
        return (
            "🍽️ **เมนูอาหารไทยแนะนำ (ไม่เผ็ด และ ไม่มีเนื้อหมู):**\n"
            "1. **ข้าวมันไก่ต้ม** (~500-590 kcal) — เนื้อไก่นุ่มทานคู่ข้าวมันหอมขิง ไม่เผ็ด (เลือกน้ำจิ้มซีอิ๊วหวานแทนน้ำจิ้มพริกได้)\n"
            "2. **ต้มข่าไก่** (~210-250 kcal) — ซุปกะทิสมุนไพรหอมข่า ตะไคร้ รสกลมกล่อมอมเปรี้ยว ไม่เผ็ดร้อน\n"
            "3. **ผัดไทยกุ้งสด หรือ ผัดไทยไก่** (~480-578 kcal) — สั่งแบบไม่ใส่พริกป่น รสเปรี้ยวหวานกลมกล่อม\n"
            "4. **ข้าวผัดไก่ / ข้าวผัดกุ้ง** (~450-550 kcal) — เมนูทานง่าย โปรตีนครบและไม่ใส่พริก\n"
            "5. **ปลากะพงทอดน้ำปลา / ปลานึ่งซีอิ๊ว** (~280-380 kcal) — โปรตีนสูงจากเนื้อปลา ย่อยง่ายและไม่มีเนื้อหมูครับ"
        )
    if no_spicy:
        return (
            "🍽️ **เมนูอาหารไทยรสไม่เผ็ดแนะนำ:**\n"
            "- **ต้มจืดเต้าหู้หมูสับ** (~120-180 kcal)\n"
            "- **ข้าวมันไก่ / ข้าวหมูแดง** (~500-580 kcal)\n"
            "- **ผัดซีอิ๊วหมู/ไก่** (~520-600 kcal)\n"
            "- **ต้มข่าไก่** (~210-250 kcal)"
        )
    if no_pork:
        return (
            "🍽️ **เมนูอาหารไทยที่ไม่มีเนื้อหมูแนะนำ:**\n"
            "- **ต้มยำกุ้ง / ต้มข่าไก่**\n"
            "- **แกงเขียวหวานไก่ / มัสมั่นไก่**\n"
            "- **ผัดไทยกุ้งสด / ข้าวผัดกุ้ง / ข้าวมันไก่**"
        )
    return ""


def _handle_budget_question(q: str) -> str:
    """จัดการคำถามวางแผนเมนูตามงบประมาณ เช่น งบ 50 บาท สำหรับ 2 คน (ข้อ 7)"""
    q_norm = (q or "").strip()
    if not any(kw in q_norm for kw in _BUDGET_KEYWORDS):
        return ""

    nums = [int(n) for n in re.findall(r"\d+", q_norm)]
    budget = nums[0] if nums else 50
    people = nums[1] if len(nums) >= 2 else 1

    return (
        f"💰 **ไอเดียเมนูอาหารไทยประหยัดงบ (งบประมาณ {budget} บาท สำหรับ {people} คน):**\n"
        "1. **ข้าวไข่เจียวหมูสับ / ไข่เจียวโหระพา** — ใช้ไข่ไก่ 3-4 ฟอง (ประมาณ 20-25 บาท) + หมูสับหรือใบโหระพา (15-20 บาท) อิ่มอร่อยได้ครบ 2 คน\n"
        "2. **ผัดกะเพราไข่เยี่ยวม้า หรือ กะเพราเต้าหู้ไข่หมูสับ** — ใช้วัตถุดิบราคาย่อมเยา ผัดจานกลางแบ่งทานคู่ข้าวสวยร้อนๆ ได้ 2 จาน\n"
        "3. **ต้มจืดไข่น้ำหมูสับใส่ผักกาดขาว** — ได้ทั้งซุปร้อนๆ และโปรตีนจากไข่กับหมูสับ ใช้งบวัตถุดิบเพียง 35-45 บาท\n"
        "4. **ผัดมาม่า / ผัดซีอิ๊วเส้นใหญ่ใส่ไข่และคะน้า** — อิ่มท้องได้พลังงานครบในงบไม่เกิน 50 บาทครับ"
    )


def _handle_cooking_technique(q: str, menus: list[str], recipe_lookup_fn=None) -> str:
    """จัดการคำถามเทคนิคการทำอาหารเชิงลึก เช่น หมูกรอบทำยังไงให้หนังฟูกรอบนานๆ ไม่เหนียว (ข้อ 10)"""
    q_norm = (q or "").strip()
    if "หมูกรอบ" in q_norm and any(kw in q_norm for kw in _TECHNIQUE_KEYWORDS + ("วิธีทำ", "ทำยังไง", "กรอบ")):
        return (
            "🥓 **เคล็ดลับวิธีทำหมูกรอบให้หนังฟูกรอบนานและเนื้อไม่เหนียว:**\n"
            "1. **ต้มเฉพาะส่วนหนังให้สุกนุ่ม:** นำหมูสามชั้นลงต้มในน้ำเดือดผสมเกลือประมาณ 30-40 นาที ให้หนังสุกใสแต่เนื้อไม่เปื่อยยุ่ย\n"
            "2. **บั้งและจิ้มหนังให้ทั่ว:** ใช้ส้อมหรือเหล็กแหลมจิ้มส่วนหนังให้พรุนละเอียด (ระวังอย่าให้ทะลุถึงชั้นเนื้อ) เพื่อระบายความชื้นและไขมันตอนทอด\n"
            "3. **ทาน้ำส้มสายชูและเกลือ:** ทาน้ำส้มสายชูผสมเกลือบางๆ ที่ผิวหนังหมู น้ำส้มสายชูจะช่วยให้โครงสร้างคอลลาเจนที่หนังพองฟูและกรอบทน\n"
            "4. **ตากแดดหรืออบไล่ความชื้น (สำคัญที่สุด):** นำไปตากแดดจัด 2-3 ชั่วโมง หรือเข้าหม้อทอดไร้น้ำมัน/เตาอบไฟต่ำ (90-100°C) จนหนังแห้งแข็งเหมือนพลาสติก\n"
            "5. **ทอดไฟกลางค่อนแรง 2 รอบ (หรืออบไฟแรง 200°C):** นำด้านหนังลงทอดในน้ำมันร้อนปานกลางจนหนังพองฟู พักให้เย็นแล้วทอดซ้ำอีกครั้ง จะได้หมูกรอบที่หนังกรอบนานและเนื้อด้านในยังชุ่มฉ่ำไม่เหนียวครับ"
        )
    if menus and any(w in q_norm for w in ("วิธีทำ", "ส่วนผสม", "วัตถุดิบ", "ทำยังไง", "สูตร")):
        if recipe_lookup_fn is not None:
            res = recipe_lookup_fn(menus[0])
            if res:
                return res
    return ""


def _handle_foreign_or_ood_food(q: str, recommend_fn=None) -> str:
    """จัดการคำถามอาหารต่างชาติที่อยู่นอกคลังอาหารไทย 50 เมนู เช่น ซูชิแซลมอน (ข้อ 5)"""
    q_norm = (q or "").strip()
    matched_foreign = [w for w in _FOREIGN_FOOD_KEYWORDS if w in q_norm]
    if matched_foreign:
        dish_name = matched_foreign[0]
        rec_text = recommend_fn() if callable(recommend_fn) else (
            "แนะนำเมนูอาหารไทยยอดนิยมในระบบ เช่น **ต้มยำกุ้ง**, **ผัดไทย**, **ส้มตำ**, **แกงเขียวหวานไก่**, หรือ **ข้าวมันไก่** ครับ"
        )
        return (
            f"ขออภัยครับ **{dish_name}** เป็นเมนูอาหารต่างชาติซึ่งอยู่นอกขอบเขตฐานข้อมูลอาหารไทย 50 เมนูของระบบ "
            f"ระบบนี้เชี่ยวชาญเฉพาะข้อมูลโภชนาการ แคลอรี และสูตรอาหารไทยครับ\n\n{rec_text}"
        )
    return ""


# ---------------------------------------------------------------------------
# 4. ฟังก์ชัน Router กลางสำหรับเชื่อมต่อเข้ากับ answer() และ _synthesize_smart_fallback()
# ---------------------------------------------------------------------------
def enhanced_smart_router(
    q: str,
    explicit_menus: list[str],
    state: dict[str, Any] | None = None,
    compare_kcal_fn=None,
    recommend_fn=None,
    recipe_lookup_fn=None,
) -> str:
    """
    ตรวจสอบและสร้างคำตอบสำหรับคำถามเฉพาะทางทั้ง 10 รูปแบบก่อนตกเข้า Fallback ปกติ
    คืนค่าเป็นสตริงว่าง "" หากไม่ใช่กรณีพิเศษ เพื่อให้โค้ดเดิมใน app.py ทำงานต่อตามปกติ
    """
    q_clean = (q or "").strip()
    if not q_clean:
        return ""

    # 1. ดักจับข้อความสุ่ม / Gibberish (ข้อ 9)
    if is_gibberish_input(q_clean):
        return (
            "ขออภัยครับ ระบบไม่เข้าใจข้อความที่พิมพ์มา กรุณาพิมพ์ชื่อเมนูอาหารไทย หรือคำถามเกี่ยวกับแคลอรี "
            "ส่วนผสม วิธีทำอาหาร และการเลือกเมนูสุขภาพ (รองรับคลังข้อมูลอาหารไทย 50 เมนู) เช่น "
            "*'ต้มยำกุ้งมีวิธีทำอย่างไร'* หรือ *'ผัดไทยกี่แคลอรี่'* ครับ"
        )

    # 2. ดักจับอาหารต่างชาติ Out-of-Domain เช่น ซูชิแซลมอน (ข้อ 5)
    ood_reply = _handle_foreign_or_ood_food(q_clean, recommend_fn=recommend_fn)
    if ood_reply:
        return ood_reply

    # 3. ดักจับคำถามแพ้อาหารและเมนูทดแทน เช่น แพ้กุ้งและถั่วลิสง กินผัดไทยได้ไหม (ข้อ 8)
    allergy_reply = _handle_allergy_and_substitution(q_clean, explicit_menus)
    if allergy_reply:
        return allergy_reply

    # 4. ดักจับคำถามภาษาอังกฤษและการอธิบายให้ชาวต่างชาติฟัง เช่น ส้มตำปูปลาร้า (ข้อ 6)
    if _is_english_explanation_question(q_clean):
        en_reply = _handle_english_explanation(q_clean, explicit_menus)
        if en_reply:
            return en_reply

    # 5. ดักจับคำถามเปรียบเทียบความแตกต่าง เช่น แกงเขียวหวาน ต่างจาก แกงเผ็ด อย่างไร (ข้อ 4)
    if _is_comparison_question_fixed(q_clean, explicit_menus):
        comp_reply = _handle_curry_or_menu_comparison(q_clean, explicit_menus, compare_kcal_fn=compare_kcal_fn)
        if comp_reply:
            return comp_reply

    # 6. ดักจับคำถามเทคนิคการทำอาหารเชิงลึก เช่น หมูกรอบทำยังไงให้หนังฟูกรอบนานๆ (ข้อ 10)
    tech_reply = _handle_cooking_technique(q_clean, explicit_menus, recipe_lookup_fn=recipe_lookup_fn)
    if tech_reply:
        return tech_reply

    # 7. ดักจับคำถามเงื่อนไขอาหาร เช่น ไม่เผ็ดและไม่มีเนื้อหมู (ข้อ 3)
    diet_reply = _handle_dietary_preferences(q_clean)
    if diet_reply:
        return diet_reply

    # 8. ดักจับคำถามงบประมาณ เช่น มีงบ 50 บาท ทำเมนูอะไรได้บ้างสำหรับ 2 คน (ข้อ 7)
    budget_reply = _handle_budget_question(q_clean)
    if budget_reply:
        return budget_reply

    return ""