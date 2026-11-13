"""Sourced recipe facts for additional text-query menus; independent of vision labels."""
import json
from pathlib import Path
CATALOG = json.loads((Path(__file__).parent/'recipe_catalog.json').read_text(encoding='utf-8'))
BY_NAME = {row['menu_th']: row for row in CATALOG if row['status'] in ('recipe_ingredients', 'source_link_only')}

def find_extended_menus(question):
    matches=sorted((name for name in BY_NAME if name in question),key=len,reverse=True)
    selected=[]
    for name in matches:
        if not any(name in longer for longer in selected): selected.append(name)
    return selected

def recipe_reply(question, menu=''):
    names=find_extended_menus(question)
    if not names and menu in BY_NAME and any(t in question for t in ('ส่วนผสม','วัตถุดิบ','กี่แคล','แคลอรี','วิธีทำ','ทำยังไง','แพ้','สารอาหาร','เมนูนี้')):
        names=[menu]
    if not names: return None
    if len(names)>1:
        return 'มีแหล่งสูตรสำหรับเมนูที่ถามครับ แต่ยังไม่มีค่าพลังงานที่ตรวจสอบแล้วสำหรับนำมาเปรียบเทียบ\n\n'+'\n'.join(f"- [{name}]({BY_NAME[name]['source_url']})" for name in names)
    item=BY_NAME[names[0]]
    title=f"**{item['menu_th']}**"
    source=f"[ดูสูตรต้นทางจาก NLoveCooking]({item['source_url']})"
    if any(t in question.lower() for t in ('แคล','kcal','พลังงาน','โปรตีน','ไขมัน','คาร์โบไฮเดรต','โซเดียม')):
        return title+' ยังไม่มีค่าพลังงานหรือสารอาหารที่ตรวจสอบแล้วในฐานข้อมูลครับ ต้องทราบปริมาณวัตถุดิบ จำนวนเสิร์ฟ และวิธีปรุงก่อนประเมิน\n\n'+source
    if any(t in question for t in ('เบาหวาน','ความดัน','ไต','แพ้','กินได้ไหม','กินได้มั้ย','ทานได้ไหม','ลดน้ำหนัก')):
        return title+' มีข้อมูลสูตรทั่วไป แต่ยังไม่มีหลักฐานโภชนาการเฉพาะเมนูเพียงพอสำหรับตอบคำถามสุขภาพนี้ครับ สูตรของแต่ละร้านอาจต่างกัน และไม่สามารถยืนยันส่วนผสมจริงหรือความปลอดภัยจากภาพได้\n\n'+source
    ingredients='、'.join(item['ingredients'])
    text=title+ ('\n\nส่วนผสมตามสูตรอ้างอิง: '+ingredients if ingredients else '\n\nมีลิงก์สูตรอ้างอิง แต่ยังถอดรายการส่วนผสมจากหน้านี้ไม่ได้ครับ')
    return text+'\n\nสูตรของแต่ละร้านอาจต่างกัน รายการนี้เป็นสูตรอ้างอิง ไม่ใช่การยืนยันส่วนผสมในภาพ\n\n'+source
