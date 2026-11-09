"""ThaiFood AI demo. Uses an existing EfficientNet-B0 checkpoint; NEVER trains a model.
Run locally: python app.py
Deployment: GitHub Actions trusted publisher -> NONGAME/ThaiFood-AI-Gradio
"""
from __future__ import annotations
import json
import os
import re
import unicodedata
from pathlib import Path
from urllib.parse import urldefrag

import gradio as gr
import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn
from torchvision import models, transforms
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    import spaces
except ImportError:
    class _SpacesFallback:
        @staticmethod
        def GPU(fn=None, **_kwargs):
            if fn is None:
                return lambda f: f
            return fn

    spaces = _SpacesFallback()

from knowledge import MENU_METADATA, MENU_BY_TH, EVIDENCE_BY_CLASS, GENERAL_SOURCES

HERE = Path(__file__).resolve().parent
CHECKPOINT = HERE / 'food_model_best.pth'
EVALUATION = json.loads((HERE / 'evaluation.json').read_text(encoding='utf-8'))
PDF_FOLDER = HERE / 'docs'
PDF_URLS = json.loads((HERE / 'pdf_sources.json').read_text(encoding='utf-8'))

# Common Thai aliases / shorthand names mapped to class_id so users can type natural names
MENU_ALIASES_TO_CID: dict[str, str] = {
    'กะเพรา': 'PhatKaphrao',
    'กระเพรา': 'PhatKaphrao',
    'ผัดกระเพรา': 'PhatKaphrao',
    'กะเพราหมูสับ': 'PhatKaphrao',
    'กะเพราไก่': 'PhatKaphrao',
    'ข้าวกะเพรา': 'PhatKaphrao',
    'ข้าวผัดกะเพรา': 'PhatKaphrao',
    'เขียวหวาน': 'GaengKeawWan',
    'แกงเขียวหวานไก่': 'GaengKeawWan',
    'พะโล้': 'EggsStewed',
    'หมูพะโล้': 'EggsStewed',
    'ขาหมู': 'StewedPorkLeg',
    'คากิ': 'StewedPorkLeg',
    'ส้มตำไทย': 'Somtam',
    'ตำไทย': 'Somtam',
    'ตำปูปลาร้า': 'Somtam',
    'ผัดไท': 'PadThai',
    'ผัดไทยกุ้งสด': 'PadThai',
    'ข้าวมันไก่ต้ม': 'KaoManGai',
    'ข้าวมันไก่ทอด': 'KaoManGai',
    'หมูแดง': 'KaoMooDang',
    'หมูทอดกระเทียม': 'KhaoMooTodGratiem',
    'ข้าวหมูกระเทียม': 'KhaoMooTodGratiem',
    'หมูกระเทียม': 'KhaoMooTodGratiem',
    'หมกไก่': 'KhaoMokGai',
    'เหนียวมะม่วง': 'KhaoNiewMaMuang',
    'คลุกกะปิ': 'KkaoKlukKaphi',
    'น้ำยากะทิ': 'KhanomJeenNamYaKati',
    'ขนมจีนน้ำยา': 'KhanomJeenNamYaKati',
    'ต้มข่า': 'TomKhaGai',
    'ต้มยำ': 'TomYumGoong',
    'ต้มยำกุ้งน้ำข้น': 'TomYumGoong',
    'ต้มยำกุ้งน้ำใส': 'TomYumGoong',
    'มะระยัดไส้': 'BitterMelonSoup',
    'แกงจืดมะระ': 'BitterMelonSoup',
    'แกงจืด': 'GaengJued',
    'ต้มจืดเต้าหู้หมูสับ': 'GaengJued',
    'ปูผัดผงกะหรี่': 'BooPadPongali',
    'ผัดผงกะหรี่': 'BooPadPongali',
    'ทอดมัน': 'CurriedFishCake',
    'คะน้าหมูกรอบ': 'FriedKale',
    'ผัดคะน้าหมูกรอบ': 'FriedKale',
    'ผักบุ้งไฟแดง': 'PadPakBung',
    'ผัดผักบุ้งไฟแดง': 'PadPakBung',
    'ผัดผักรวม': 'PadPakRuamMit',
    'ผัดยอดฟักแม้ว': 'PadYordMala',
    'ราดหน้า': 'PorkStickyNoodles',
    'ผัดซีอิ๊ว': 'PorkStickyNoodles',
    'สุกี้แห้ง': 'Suki',
    'สุกี้น้ำ': 'Suki',
    'หมูสะเต๊ะ': 'MooSatay',
    'สะเต๊ะ': 'MooSatay',
    'ลาบ': 'LarbMoo',
    'น้ำตก': 'NamTokMoo',
    'คอหมู': 'KorMooYang',
    'มัสมั่น': 'MassamanGai',
    'แกงมัสมั่น': 'MassamanGai',
    'หอยลาย': 'HoyLaiPrikPao',
    'หอยแครงลวก': 'HoyKraeng',
    'ปลาหมึกย่าง': 'GrilledQquid',
    'หมึกย่าง': 'GrilledQquid',
    'กุ้งแม่น้ำเผา': 'GoongPao',
    'ไข่เจียว': 'KaiJeowMooSaap',
    'ปลาเผา': 'Roast_fish',
    'เมี่ยงปลาเผา': 'Roast_fish',
    'เป็ดย่าง': 'Roast_duck',
    'ข้าวหน้าเป็ด': 'Roast_duck',
    'เย็นตาโฟ': 'Yentafo',
    'ก๋วยเตี๋ยวเย็นตาโฟ': 'Yentafo',
    'ก๋วยเตี๋ยวเรือหมู': 'KuayTeowReua',
    'ก๋วยเตี๋ยวเรือเนื้อ': 'KuayTeowReua',
}

# Approximate nutrition reference for the 28 menus not covered in EVIDENCE_BY_CLASS
# so the chatbot never replies with "no calorie figure available".
ESTIMATED_KCAL_BY_CID: dict[str, tuple[str, str, str]] = {
    'BitterMelonSoup': ('120-180', '1 ถ้วย', 'โปรตีนจากหมูสับและใยอาหารจากมะระ พลังงานต่ำ แต่ควรระวังโซเดียมในน้ำซุป'),
    'BooPadPongali': ('380-480', '1 จาน', 'โปรตีนสูงจากเนื้อปูและไข่ แต่มีไขมันสูงจากน้ำมันผัดและนมข้นจืด/กะทิ'),
    'CurriedFishCake': ('250-320', '1 จาน (5 ชิ้น)', 'ได้โปรตีนจากเนื้อปลา แต่ผ่านการทอดจึงมีไขมันและโซเดียมจากพริกแกง'),
    'Dumpling': ('180-240', '1 ชุด (5 ลูก)', 'เมนูอาหารนึ่งที่ให้โปรตีนจากหมูและกุ้ง ควรระวังพลังงานจากกระเทียมเจียวและโซเดียมในซีอิ๊ว'),
    'FriedChicken': ('280-360', '1 ชิ้นใหญ่', 'โปรตีนสูงแต่มีไขมันอิ่มตัวและพลังงานสูงจากการชุบแป้งทอด แนะนำให้ลอกหนังออกก่อนรับประทาน'),
    'FriedKale': ('220-350', '1 จาน', 'ได้ใยอาหารและวิตามินจากคะน้า หากเป็นคะน้าหมูกรอบจะมีไขมันและแคลอรีสูงขึ้น'),
    'GaengJued': ('120-160', '1 ถ้วย', 'เมนูต้มพลังงานต่ำ ย่อยง่าย ได้โปรตีนจากเต้าหู้และหมูสับ เหมาะสำหรับผู้ควบคุมน้ำหนัก'),
    'GaiYang': ('220-280', '1 ชิ้น/น่องสะโพก', 'โปรตีนสูงและไขมันต่ำกว่าไก่ทอด แนะนำให้เลี่ยงส่วนหนังและจิ้มน้ำจิ้มแต่พอดี'),
    'GoongObWoonSen': ('300-380', '1 หม้อ', 'วุ้นเส้นดูดซับน้ำมันหมูและซอสปรุงรสได้ดี แนะนำให้ตักแต่ด้านบนและไม่คลุกน้ำมันก้นหม้อ'),
    'GoongPao': ('140-200', '1 จาน (3-4 ตัว)', 'เมนูย่างพลังงานต่ำและโปรตีนสูง ระวังคอเลสเตอรอลบริเวณมันกุ้งและโซเดียมในน้ำจิ้มซีฟู้ด'),
    'GrilledQquid': ('130-180', '1 จาน', 'ให้โปรตีนสูงและไขมันต่ำจากการย่าง ควรจำกัดปริมาณน้ำจิ้มซีฟู้ดเพื่อลดโซเดียมและน้ำตาล'),
    'HoyKraeng': ('100-140', '1 จาน', 'ธาตุเหล็กและโปรตีนสูง พลังงานต่ำเพราะใช้วิธีลวก แต่ควรระวังความสะอาดและน้ำจิ้มรสจัด'),
    'HoyLaiPrikPao': ('260-340', '1 จาน', 'น้ำพริกเผามีส่วนผสมของน้ำมันและน้ำตาลค่อนข้างสูง แนะนำให้ตักแต่เนื้อหอยและเลี่ยงการราดน้ำมันพริกเผา'),
    'KorMooYang': ('320-420', '1 จาน', 'ส่วนคอหมูมีไขมันแทรกสูงมาก แนะนำให้เลือกเนื้อส่วนที่มันน้อยและรับประทานคู่กับผักสด'),
    'KuaKling': ('220-290', '1 จาน', 'ใช้น้ำมันน้อยและได้สมุนไพรจากเครื่องแกงใต้ แต่มีรสเค็มจัดจึงควรระวังปริมาณโซเดียม'),
    'KuayTeowReua': ('250-350', '1 ชามปกติ', 'ให้คาร์โบไฮเดรตจากเส้นและโปรตีนจากหมู/เนื้อ น้ำซุปเข้มข้นมีโซเดียมสูง แนะนำให้ชิมก่อนปรุงและไม่ซดน้ำซุปหมดชาม'),
    'LarbMoo': ('150-220', '1 จาน', 'เมนูยำ/ลาบที่ใช้น้ำมันน้อยมาก โปรตีนสูง เหมาะกับคนลดน้ำหนัก แต่ควรสั่งลดเค็ม'),
    'MassamanGai': ('380-460', '1 ถ้วย', 'แกงกะทิเข้มข้นที่มีทั้งมันฝรั่งและถั่วลิสง ให้พลังงานและไขมันอิ่มตัวสูง ควรตักแต่เนื้อและเลี่ยงการซดน้ำแกง'),
    'MooSatay': ('280-360', '1 ชุด (10 ไม้)', 'หมูหมักกะทิย่างพร้อมน้ำจิ้มถั่วที่มีทั้งกะทิและน้ำตาลสูง ควรจำกัดปริมาณน้ำจิ้มและรับประทานอาจาดแตงกวาเพิ่ม'),
    'NamTokMoo': ('180-250', '1 จาน', 'ได้โปรตีนจากเนื้อหมูย่างและสมุนไพรไทย หากเลือกหมูส่วนไม่ติดมันจะเหมาะกับการควบคุมน้ำหนักมาก'),
    'PadPakBung': ('180-240', '1 จาน', 'มีวิตามินเอและใยอาหารสูง แต่ใช้น้ำมันและซอสหอยนางรมในการผัด แนะนำให้ตักแต่ผักและไม่ราดน้ำผัดลงบนข้าว'),
    'PadPakRuamMit': ('160-230', '1 จาน', 'ได้วิตามินและใยอาหารหลากหลายชนิด ช่วยให้อิ่มนาน แนะนำให้สั่งผัดน้ำมันน้อย'),
    'PadYordMala': ('160-220', '1 จาน', 'ยอดฟักแม้วมีใยอาหารสูงและพลังงานต่ำ ระวังเพียงปริมาณน้ำมันและซอสปรุงรสที่ใช้ผัด'),
    'PhatKaphrao': ('520-650', '1 จาน (ราดข้าว)', 'เมนูยอดนิยมที่ให้พลังงานสูงจากน้ำมันผัด (และไข่ดาวหากเพิ่ม) แนะนำให้สั่งผัดน้ำหรือลดน้ำมันและใช้เนื้ออกไก่/หมูชิ้นไม่ติดมัน'),
    'PorkStickyNoodles': ('420-550', '1 จาน', 'เส้นใหญ่ผัดน้ำมันมีไขมันสูง หากต้องการลดพลังงานแนะนำให้เลือกเส้นหมี่และเพิ่มผักคะน้า'),
    'Roast_duck': ('320-420', '1 จาน', 'เนื้อเป็ดให้โปรตีนสูงแต่หนังเป็ดมีไขมันสูงมาก แนะนำให้ลอกหนังออกและราดน้ำเป็ดย่างแต่พอดี'),
    'Roast_fish': ('180-250', '1 ตัว/ชุด', 'ปลาเผาเป็นเมนูโปรตีนคุณภาพดี ย่อยง่าย ไขมันต่ำ เหมาะอย่างยิ่งสำหรับผู้ควบคุมน้ำหนักและผู้สูงอายุ'),
    'SonInLawEggs': ('280-350', '1 จาน (2 ฟอง)', 'ไข่ต้มนำไปทอดและราดน้ำมะขามเคี่ยวน้ำตาลปี๊บ ทำให้มีทั้งไขมันและน้ำตาลสูง ควรรับประทานน้ำราดแต่พอดี'),
    'StewedPorkLeg': ('620-720', '1 จาน (ราดข้าว)', 'ขาหมูส่วนหนังและคากิมีไขมันอิ่มตัวและโซเดียมในน้ำพะโล้สูง แนะนำให้สั่งเนื้อล้วน ไม่เอาหนัง และเพิ่มผักคะน้า'),
    'TomKhaGai': ('240-320', '1 ถ้วย', 'มีสมุนไพรข่า ตะไคร้ ใบมะกรูด แต่มีไขมันอิ่มตัวจากกะทิ แนะนำให้ตักรับประทานแต่เนื้อไก่และเห็ด'),
    'TomYumGoong': ('90-180', '1 ถ้วย', 'ต้มยำกุ้งน้ำใสให้พลังงานต่ำมาก (~90-110 kcal) ส่วนน้ำข้นจะมีพลังงานเพิ่มขึ้นจากนมข้นจืดหรือน้ำพริกเผา ควรเลี่ยงการซดน้ำซุปจนหมดเพื่อลดโซเดียม'),
    'Yentafo': ('320-400', '1 ชาม', 'ได้โปรตีนจากลูกชิ้นปลาและเลือดหมู แต่ซอสแดงเย็นตาโฟมีโซเดียมและน้ำตาลแฝง ควรเลี่ยงการปรุงน้ำปลาเพิ่ม'),
}

# Load once; inference defaults to CPU (and works with ZeroGPU @spaces.GPU).
if not CHECKPOINT.is_file():
    raise RuntimeError('Checkpoint missing: food_model_best.pth')
checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
CLASS_IDS = list(checkpoint['class_names'])
assert len(CLASS_IDS) == 50 and set(CLASS_IDS) == set(MENU_METADATA), 'Checkpoint class mismatch'
model = models.efficientnet_b0(weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_IDS))
model.load_state_dict(checkpoint['model'], strict=True)
model.eval()
VALID_TF = transforms.Compose([
    transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
    transforms.Normalize([.485, .456, .406], [.229, .224, .225])
])


def _clean_pdf_thai(raw: str) -> str:
    """Reject unreadable Thai PDF extraction instead of exposing glyph garbage to RAG or users.

    Some Thai PDFs use font-specific glyph encodings, so PyMuPDF can extract PUA,
    CID markers, and replacement characters rather than meaningful Thai words.
    These must never become chatbot answers or citation snippets.
    """
    if not raw:
        return ''
    if any(fragment in raw for fragment in ('à¸', 'à¹', 'Ã', '\ufffd', '\uf000')):
        return ''
    invalid = sum(
        1 for ch in raw
        if ch == '\ufffd' or ch in ('\ufeff',) or unicodedata.category(ch) == 'Co'
    )
    if invalid > max(2, len(raw) * 0.015) or re.search(r'\(cid:\d+\)', raw, re.I):
        return ''
    text = unicodedata.normalize('NFKC', raw)
    # Remove PDF control characters, but preserve Thai combining marks and normal whitespace.
    text = ''.join(ch if (ch in '\n\t ' or not unicodedata.category(ch).startswith('C')) else ' ' for ch in text)
    text = re.sub(r'\s+', ' ', text).strip()
    letters = [c for c in text if unicodedata.category(c).startswith('L')]
    thai = sum(1 for c in letters if '\u0E00' <= c <= '\u0E7F')
    if len(text) < 100 or thai < 25 or thai / max(1, len(letters)) < .50:
        return ''
    # Foreign encoding symptoms that survive Unicode normalization.
    if any(fragment in text for fragment in ('à¸', 'à¹', 'Ã', '�', '\uf000')):
        return ''
    return text


def optional_pdf_docs():
    """Only index PDF text if actual PDF bytes are present and parseable."""
    try:
        import pymupdf as fitz
    except ImportError:
        return []
    PDF_FOLDER.mkdir(exist_ok=True)
    registry = {x['filename']: x for x in PDF_URLS}
    if os.getenv('DOWNLOAD_OFFICIAL_PDF', '1') == '1':
        try:
            import requests
            for f, src in registry.items():
                path = PDF_FOLDER / f
                if path.exists():
                    continue
                try:
                    with requests.get(src['url'], timeout=(5, 12), stream=True,
                                      headers={'User-Agent': 'Mozilla/5.0'}) as response:
                        response.raise_for_status()
                        data = bytearray()
                        for chunk in response.iter_content(1 << 16):
                            data.extend(chunk)
                            if len(data) > 20_000_000:
                                raise ValueError('document exceeds 20 MB')
                        if not data.startswith(b'%PDF-'):
                            raise ValueError('URL did not serve a PDF')
                        (PDF_FOLDER / f).write_bytes(data)
                except (requests.RequestException, OSError, ValueError) as exc:
                    print('PDF not available:', f, type(exc).__name__)
        except ImportError:
            pass
    docs = []
    for file in PDF_FOLDER.glob('*.pdf'):
        ref = registry.get(file.name, {'url': '', 'title': file.name})
        try:
            with fitz.open(file) as pdf:
                for page_num, page in enumerate(pdf, 1):
                    text = _clean_pdf_thai(page.get_text('text'))
                    if not text:
                        continue
                    for offset in range(0, len(text), 800):
                        span = text[max(0, offset - 80):offset + 900]
                        if len(span) < 100:
                            continue
                        docs.append({
                            'id': f'pdf:{file.name}:{page_num}:{offset}',
                            'menu': '',
                            'title': f"{ref['title']} หน้า {page_num}",
                            'text': span,
                            'source_label': ref['title'],
                            'url': ref['url'],
                            'page': page_num,
                            'kind': 'pdf'
                        })
        except Exception as exc:
            print('PDF parse failed:', file.name, type(exc).__name__)
    return docs


def make_docs():
    docs = []
    for cid, meta in MENU_METADATA.items():
        name, ingredients = meta[0], meta[1]
        docs.append({
            'id': f'recipe:{cid}',
            'menu': name,
            'title': f'สูตรและโภชนาการ {name}',
            'text': f'{name} ส่วนประกอบหลัก ได้แก่ {ingredients} เหมาะสำหรับคุณค่าทางโภชนาการอาหารไทย',
            'source_label': 'ข้อมูลเมนูอาหารไทยมาตรฐาน',
            'url': '',
            'kind': 'recipe'
        })
    for cid, rec in EVIDENCE_BY_CLASS.items():
        kcal = (f"{rec['kcal_min']:g}" if rec['kcal_min'] == rec['kcal_max'] else
                f"{rec['kcal_min']:g}-{rec['kcal_max']:g}")
        name = MENU_METADATA[cid][0]
        docs.append({
            'id': f'nutrition:{cid}',
            'menu': name,
            'title': f'โภชนาการ {name}',
            'text': f'{name} ตาม {rec["recipe_name"]} ปริมาณ {rec["portion"]} ให้ {kcal} กิโลแคลอรี คุณค่าทางโภชนาการ: {rec["note"]}',
            'source_label': rec['source_name'],
            'url': rec['source_url'],
            'kind': 'nutrition'
        })
    for g in GENERAL_SOURCES:
        # Tag menu-specific guidelines so they don't leak into unrelated dishes
        g_menu = 'ข้าวมันไก่' if 'ข้าวมันไก่' in g.get('title', '') or g.get('id') == 'guideline:khao_mangai' else ''
        docs.append({
            'id': g['id'],
            'menu': g_menu,
            'title': g['title'],
            'text': g['text'],
            'source_label': 'แนวทางโภชนาการ',
            'url': g['url'],
            'kind': 'guideline'
        })
    pdf = optional_pdf_docs()
    docs.extend(pdf)
    print('RAG corpus:', len(docs), 'documents | real PDF chunks:', len(pdf))
    return docs, len(pdf)


RAG_DOCS, PDF_CHUNKS = make_docs()
RAG_VEC = TfidfVectorizer(analyzer='char', ngram_range=(2, 4), sublinear_tf=True)
RAG_MAT = RAG_VEC.fit_transform([
    f"{d['title']} {d['menu']} {d['text']}" for d in RAG_DOCS
])


def find_menus(q: str) -> list[str]:
    """Return all explicitly mentioned Thai menu names in the query (never guesses random menus on greetings)."""
    q_norm = (q or '').strip()
    if not q_norm:
        return []
    found: list[tuple[int, int, str]] = []

    # 1. Exact canonical Thai menu name match
    for menu_th in MENU_BY_TH:
        pos = q_norm.find(menu_th)
        if pos != -1:
            found.append((pos, -len(menu_th), menu_th))

    # 2. Common alias match (e.g. "กะเพราหมูสับ" -> "ผัดกะเพรา", "ตำไทย" -> "ส้มตำ")
    for alias, cid in sorted(MENU_ALIASES_TO_CID.items(), key=lambda x: len(x[0]), reverse=True):
        if cid in MENU_METADATA:
            canonical_th = MENU_METADATA[cid][0]
            pos = q_norm.find(alias)
            if pos != -1:
                found.append((pos, -len(alias), canonical_th))

    if found:
        found.sort()
        unique_menus: list[str] = []
        for _, _, m in found:
            if m not in unique_menus:
                unique_menus.append(m)
        return unique_menus

    return []


def find_menu(q: str) -> str:
    menus = find_menus(q)
    return menus[0] if menus else ''


def _is_followup_nutrition_query(q: str) -> bool:
    """True only when the user is asking a food/nutrition follow-up question without naming the dish again."""
    followup_keywords = [
        'กี่แคล', 'แคลอรี', 'แคลอรี่', 'พลังงาน', 'ส่วนประกอบ', 'วัตถุดิบ', 'ใส่อะไร',
        'สารอาหาร', 'โภชนาการ', 'อ้วนไหม', 'อ้วนมั้ย', 'ลดน้ำหนัก', 'เบาหวาน', 'ความดัน',
        'โซเดียม', 'เค็ม', 'ไขมัน', 'คอเลสเตอรอล', 'กินได้ไหม', 'กินได้มั้ย', 'ทานได้ไหม',
        'ปรับยังไง', 'ลดแคล', 'เมนูนี้', 'จานนี้', 'ชามนี้', 'แล้วถ้า', 'ประโยชน์', 'ข้อควรระวัง'
    ]
    return any(k in q for k in followup_keywords)


def _detect_conversational_reply(q: str, active_menu: str = '') -> str:
    """Handle greetings, thanks, and help queries naturally without dumping nutrition cards."""
    q_clean = re.sub(r'[!?.~ๆ\s]+', '', q.lower())

    greetings = {
        'สวัสดี', 'สวัสดีครับ', 'สวัสดีค่ะ', 'สวัสดีคับ', 'สวัสดีคะ', 'หวัดดี', 'หวัดดีครับ',
        'หวัดดีค่ะ', 'ดีครับ', 'ดีค่ะ', 'ดีจ้า', 'ทักทาย', 'hello', 'hi', 'hey'
    }
    thanks = {
        'ขอบคุณ', 'ขอบคุณครับ', 'ขอบคุณค่ะ', 'ขอบคุณมาก', 'ขอบคุณมากครับ', 'ขอบคุณมากค่ะ',
        'แต้งกิ้ว', 'thankyou', 'thanks', 'โอเค', 'โอเคครับ', 'โอเคค่ะ', 'เข้าใจแล้ว', 'รับทราบ'
    }
    help_phrases = [
        'ทำอะไรได้บ้าง', 'ช่วยอะไรได้บ้าง', 'ถามอะไรได้บ้าง', 'ใช้งานยังไง', 'ใช้ยังไง',
        'คุณคือใคร', 'เป็นใคร', 'มีเมนูอะไรบ้าง', 'แนะนำตัว'
    ]

    if q_clean in greetings or any(q_clean.startswith(g) and len(q_clean) <= len(g) + 4 for g in greetings):
        menu_hint = (
            f"\n\nตอนนี้คุณมีเมนู **{active_menu}** ที่เลือกค้างไว้ สามารถถามต่อได้เลย เช่น *“{active_menu}กี่แคลอรี”* หรือ *“คนลดน้ำหนักกินได้ไหม”* ครับ"
            if active_menu else ""
        )
        return (
            "สวัสดีครับ! ยินดีต้อนรับสู่ **ThaiFood AI ผู้ช่วยด้านโภชนาการอาหารไทย** 🍲\n\n"
            "คุณสามารถสอบถามผมได้หลายรูปแบบเลยครับ เช่น:\n"
            "- **เช็กแคลอรีและส่วนประกอบ:** *“ผัดไทยกุ้งสดกี่แคล?”*, *“กะเพราหมูสับใส่อะไรบ้าง?”*\n"
            "- **เปรียบเทียบเมนูอาหาร:** *“ข้าวมันไก่ กับ ข้าวหมูแดง อะไรแคลน้อยกว่ากัน?”*\n"
            "- **ปรึกษาการกินเพื่อสุขภาพ:** *“แนะนำเมนูแคลอรีต่ำสำหรับคนลดน้ำหนัก”* หรือ *“ก๋วยเตี๋ยวเรือลดโซเดียมได้อย่างไร?”*"
            f"{menu_hint}"
        )

    if q_clean in thanks:
        return "ยินดีมากครับ! 😊 หากต้องการเช็กแคลอรี เปรียบเทียบเมนู หรือขอคำแนะนำการรับประทานอาหารไทยเพื่อสุขภาพเพิ่มเติม พิมพ์ถามได้ตลอดเลยนะครับ"

    if any(hp in q_clean for hp in help_phrases):
        sample_menus = ', '.join(list(MENU_BY_TH.keys())[:12])
        return (
            "ผมเป็นผู้ช่วย AI ด้านโภชนาการอาหารไทย ครอบคลุม **50 เมนูอาหารไทยยอดนิยม** ครับ 🥗\n\n"
            "**สิ่งที่ผมช่วยคุณได้:**\n"
            "1. **วิเคราะห์พลังงาน (กิโลแคลอรี) และส่วนประกอบหลัก** ของอาหารไทยทั้ง 50 เมนู (เช่น " + sample_menus + " ฯลฯ)\n"
            "2. **เปรียบเทียบความคุ้มค่าทางพลังงาน** ระหว่าง 2 เมนูขึ้นไป\n"
            "3. **แนะนำวิธีปรับการกิน** สำหรับผู้ที่ต้องการลดน้ำหนัก ควบคุมน้ำตาล ลดไขมัน หรือลดโซเดียมตามแนวทางกรมอนามัย (หลัก 2:1:1)\n\n"
            "ลองพิมพ์ชื่อเมนูที่สนใจ หรือคลิกจาก **ตัวอย่างคำถามยอดฮิต** ด้านล่างได้เลยครับ!"
        )

    return ''


def format_ref(d: dict) -> str:
    lbl = d['source_label']
    url = d['url']
    if not url:
        return f"*(อ้างอิง: {lbl})*"
    clean, frag = urldefrag(url)
    p = d.get('page')
    if p and not frag:
        url = f"{clean}#page={p}"
    return f'*(อ้างอิง: [{lbl}]({url}))*'


def rag_retrieve(query: str, menu: str = '', top_k: int = 3):
    q_text = f'{query} {menu}'.strip()
    q_vec = RAG_VEC.transform([q_text])
    sims = cosine_similarity(q_vec, RAG_MAT)[0].copy()

    for i, d in enumerate(RAG_DOCS):
        if menu and d['menu'] == menu:
            if d['kind'] == 'nutrition':
                sims[i] += 0.50
            elif d['kind'] == 'recipe':
                sims[i] += 0.35
        elif d['kind'] == 'guideline' and not d['menu'] and any(
            kw in query for kw in ['ลดน้ำหนัก', 'อ้วน', 'โซเดียม', 'เค็ม', 'หวาน', 'มัน', 'เบาหวาน', 'ความดัน', 'สุขภาพ', '2:1:1']
        ):
            sims[i] += 0.25

    ranked = np.argsort(sims)[::-1]
    picked = []
    seen = set()
    for idx in ranked:
        d = RAG_DOCS[idx]
        key = (d['source_label'], d.get('page', 0), d['text'][:40])
        if key in seen:
            continue
        # Never include a document belonging to a different menu
        if d['menu'] and menu and d['menu'] != menu:
            continue
        if d['menu'] and not menu and d['kind'] == 'guideline':
            continue
        seen.add(key)
        picked.append(d)
        if len(picked) >= top_k:
            break
    if not picked:
        picked = [RAG_DOCS[idx] for idx in ranked[:top_k]]
    return picked


# Alias for backward compatibility with test_smoke.py
retrieve = rag_retrieve


def gemini_answer(prompt: str) -> str:
    key = os.getenv('GEMINI_API_KEY')
    if not key:
        return ''
    try:
        from google import genai
        client = genai.Client(api_key=key)
        resp = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return (resp.text or '').strip()
    except Exception as exc:
        print('Gemini API call skipped/failed:', type(exc).__name__, exc)
        return ''


def _InferCookingGuidance(menu: str, ingredients: str) -> str:
    text = f"{menu} {ingredients}"
    tips = []
    if any(k in text for k in ['กะทิ', 'แกงเขียวหวาน', 'มัสมั่น', 'ต้มข่า', 'น้ำยากะทิ', 'ข้าวเหนียวมะม่วง']):
        tips.append('มีส่วนประกอบของกะทิซึ่งให้ไขมันอิ่มตัวสูง แนะนำให้ตักรับประทานแต่เนื้อและเลี่ยงการซดน้ำแกงจนหมด')
    if any(k in text for k in ['ทอด', 'เจียว', 'หมูกรอบ', 'หอยทอด', 'ไก่ทอด']):
        tips.append('ผ่านการทอดด้วยน้ำมันจึงมีพลังงานและไขมันสูง แนะนำให้ลดปริมาณอาหารทอดและรับประทานผักเพิ่มเติมเพื่อให้มื้ออาหารสมดุล')
    if any(k in text for k in ['ผัด', 'คั่ว', 'ผงกะหรี่', 'น้ำพริกเผา']):
        tips.append('เป็นเมนูผัดที่ใช้น้ำมันและซอสปรุงรส สามารถสั่งลดน้ำมันและลดเค็ม รวมถึงเพิ่มผักใบเขียวตามหลักจานสุขภาพ 2:1:1')
    if any(k in text for k in ['ส้มตำ', 'ยำ', 'ลาบ', 'น้ำตก', 'กะปิ', 'ก๋วยเตี๋ยว', 'เย็นตาโฟ', 'ก๋วยจั๊บ', 'พะโล้', 'ขาหมู']):
        tips.append('ควรระวังปริมาณโซเดียมจากน้ำปลา น้ำซุป หรือเครื่องปรุงรส แนะนำให้ชิมก่อนปรุงและหลีกเลี่ยงการซดน้ำซุปจนหมดชาม')
    if not tips:
        tips.append('สามารถจัดสัดส่วนอาหารตามหลัก 2:1:1 (ผัก 2 ส่วน : ข้าวแป้ง 1 ส่วน : เนื้อสัตว์ไม่ติดมัน 1 ส่วน) เพื่อความสมดุลทางโภชนาการ')
    return ' • '.join(tips)


def _get_menu_kcal_info(cid: str) -> tuple[str, str, str, str, str]:
    """Return (kcal_str, portion, note, source_name, source_url) for any of the 50 menus."""
    if cid in EVIDENCE_BY_CLASS:
        rec = EVIDENCE_BY_CLASS[cid]
        kcal = (f"{rec['kcal_min']:g}" if rec['kcal_min'] == rec['kcal_max'] else
                f"{rec['kcal_min']:g}-{rec['kcal_max']:g}")
        return kcal, rec['portion'], rec['note'], rec['source_name'], rec['source_url']
    if cid in ESTIMATED_KCAL_BY_CID:
        kcal, portion, note = ESTIMATED_KCAL_BY_CID[cid]
        return f"ประมาณ {kcal}", portion, note, 'ประมาณการโดยผู้พัฒนา ยังไม่มีเอกสารยืนยันค่าพลังงานเฉพาะเมนู', ''
    return 'ประมาณ 250-450', '1 จาน/ชาม', 'ขึ้นอยู่กับปริมาณน้ำมันและเนื้อสัตว์', 'ข้อมูลเมนูอาหารไทยมาตรฐาน', ''


def build_nutrition_card(menu: str) -> str:
    cid = MENU_BY_TH.get(menu)
    if not cid:
        return ''
    meta = MENU_METADATA[cid]
    ing = meta[1] if len(meta) > 1 else ''
    kcal, portion, note, src_name, src_url = _get_menu_kcal_info(cid)
    recipe_line = f"- **สูตร/เมนูอ้างอิง:** {EVIDENCE_BY_CLASS[cid]['recipe_name']}\n" if cid in EVIDENCE_BY_CLASS else ""
    ref = format_ref({'source_label': src_name, 'url': src_url})
    return (
        f"### 🥗 ข้อมูลและคุณค่าทางโภชนาการสำหรับ {menu}\n"
        f"{recipe_line}"
        f"- **ปริมาณต่อเสิร์ฟ:** {portion}\n"
        f"- **พลังงาน:** **{kcal} กิโลแคลอรี**\n"
        f"- **ส่วนประกอบสำคัญ:** {ing}\n"
        f"- **คุณค่าทางโภชนาการ:** {note}\n"
        f"- **คำแนะนำสุขภาพ:** {_InferCookingGuidance(menu, ing)}\n"
        f"{ref}\n\n"
    )


def _synthesize_smart_fallback(q: str, menus: list[str], docs: list[dict], has_card: bool = False) -> str:
    """Synthesize a clean, non-repetitive Thai answer when Gemini API is unavailable."""
    q_low = q.lower()

    # Case 1: Multi-menu comparison (e.g., "ข้าวมันไก่ กับ ข้าวหมูแดง อะไรแคลน้อยกว่ากัน")
    if len(menus) >= 2:
        lines = [f"### ⚖️ เปรียบเทียบคุณค่าทางโภชนาการ: {' vs '.join(menus[:3])}\n"]
        kcal_items = []
        for m in menus[:3]:
            cid = MENU_BY_TH.get(m)
            if not cid:
                continue
            ing = MENU_METADATA[cid][1] if len(MENU_METADATA[cid]) > 1 else ''
            kcal_str, portion, note, src_name, src_url = _get_menu_kcal_info(cid)
            nums = [float(x) for x in re.findall(r'\d+(?:\.\d+)?', kcal_str)]
            avg_kcal = sum(nums) / len(nums) if nums else 350.0
            kcal_items.append((avg_kcal, m, kcal_str, portion))
            lines.append(
                f"- **{m}** (ปริมาณ {portion}): ให้พลังงาน **{kcal_str} กิโลแคลอรี** "
                f"| ส่วนประกอบหลัก: {ing} — {note} {format_ref({'source_label': src_name, 'url': src_url})}"
            )
        if len(kcal_items) >= 2:
            kcal_items.sort(key=lambda x: x[0])
            lowest = kcal_items[0]
            highest = kcal_items[-1]
            diff = abs(highest[0] - lowest[0])
            lines.append(
                f"\n💡 **สรุปเปรียบเทียบ:** **{lowest[1]}** ให้พลังงานน้อยกว่า ({lowest[2]} กิโลแคลอรี ต่อ {lowest[3]}) "
                f"เมื่อเทียบกับ **{highest[1]}** ({highest[2]} กิโลแคลอรี ต่อ {highest[3]}) ต่างกันประมาณ **{diff:.0f} กิโลแคลอรี**"
            )
        return '\n'.join(lines)

    # Case 2: User asks for low-calorie / weight-loss menu recommendations without naming a specific menu
    if not menus and any(k in q_low for k in ['แนะนำ', 'ลดน้ำหนัก', 'แคลน้อย', 'แคลอรีต่ำ', 'คลีน', 'ไดเอท', 'ไม่อ้วน', 'กินอะไรดี']):
        sorted_recs = sorted(EVIDENCE_BY_CLASS.values(), key=lambda r: (r['kcal_min'] + r['kcal_max']) / 2)
        low_list = sorted_recs[:5]
        lines = [
            "### 🥗 แนะนำเมนูอาหารไทยพลังงานต่ำ (เหมาะสำหรับควบคุมน้ำหนัก)\n",
            "จากฐานข้อมูลโภชนาการอ้างอิง เมนูที่ให้พลังงานต่ำที่สุด 5 อันดับแรก ได้แก่:"
        ]
        for r in low_list:
            kcal = f"{r['kcal_min']:g}" if r['kcal_min'] == r['kcal_max'] else f"{r['kcal_min']:g}-{r['kcal_max']:g}"
            lines.append(
                f"- **{r['menu_th']}** ({r['recipe_name']} {r['portion']}): **{kcal} กิโลแคลอรี** — {r['note']} "
                f"{format_ref({'source_label': r['source_name'], 'url': r['source_url']})}"
            )
        lines.append(
            "\n💡 **เทคนิคเพิ่มเติม:** เลือกเมนูประเภทต้ม นึ่ง ย่าง แทนการทอดหรือแกงกะทิ และจัดจานตามหลัก **2:1:1** "
            "(ผัก 2 ส่วน : ข้าวไม่ขัดสี 1 ส่วน : เนื้อสัตว์ไม่ติดมัน 1 ส่วน)"
        )
        return '\n'.join(lines)

    # Case 3: Specific menu question — if build_nutrition_card is already rendered above, only add targeted extra advice (no repetition!)
    if menus:
        target = menus[0]
        cid = MENU_BY_TH.get(target)
        ing = MENU_METADATA[cid][1] if cid and len(MENU_METADATA[cid]) > 1 else ''
        extra_tips = []
        if any(k in q_low for k in ['โซเดียม', 'เค็ม', 'ความดัน', 'ไต']):
            extra_tips.append(
                f"• **แนวทางลดโซเดียมสำหรับ {target}:** กรมอนามัยแนะนำให้จำกัดโซเดียมไม่เกิน 2,000 มิลลิกรัมต่อวัน (เทียบเท่าเกลือ 1 ช้อนชา หรือน้ำปลาไม่เกิน 4 ช้อนชาต่อวัน) "
                f"ควรชิมก่อนปรุง ลดการซดน้ำซุป/น้ำยำ และรับประทานผักผลไม้สดเพิ่มโพแทสเซียม"
            )
        if any(k in q_low for k in ['ลดน้ำหนัก', 'อ้วน', 'ลดแคล', 'คลีน', 'เบาหวาน', 'น้ำตาล', 'ไขมัน']):
            extra_tips.append(
                f"• **เทคนิคปรับเมนู {target} เพื่อคุมน้ำหนัก/น้ำตาล:** ปรับสัดส่วนในมื้ออาหารตามหลัก **2:1:1** คือเพิ่มผักสดหรือผักลวกให้ได้ครึ่งจาน "
                f"ลดข้าวหรือเส้นลงเหลือ 1 ส่วน และเลือกเนื้อสัตว์ส่วนที่ไม่ติดมันหรือลอกหนังออก"
            )
        if not extra_tips and not has_card:
            kcal_str, portion, note, _, _ = _get_menu_kcal_info(cid) if cid else ('', '', '', '', '')
            extra_tips.append(f"• **{target}** ({portion}): ให้พลังงาน **{kcal_str} กิโลแคลอรี** ส่วนประกอบหลักคือ {ing} ({note})")
        if not extra_tips:
            return ''
        return "### 💡 คำแนะนำเพิ่มเติมตรงตามคำถาม\n" + '\n'.join(extra_tips)

    # Case 4: General nutrition / health query without a specific menu
    guideline_docs = [d for d in docs if d['kind'] in ('guideline', 'pdf') and not d.get('menu')]
    if not guideline_docs:
        guideline_docs = docs[:2]
    ref_lines = [f"• **{d['title']}**: {d['text']} {format_ref(d)}" for d in guideline_docs]
    refs_block = '\n'.join(ref_lines)
    return (
        f"### 📋 คำแนะนำด้านโภชนาการสำหรับ: {q}\n"
        f"{refs_block}\n\n"
        f"💡 *หากต้องการทราบข้อมูลเจาะจงรายเมนู สามารถพิมพ์ชื่ออาหารไทย (เช่น ต้มยำกุ้ง, ผัดไทย, ก๋วยเตี๋ยวเรือ, กะเพราหมูสับ) หรือถามเปรียบเทียบ 2 เมนูได้เลยครับ*"
    )


# Query routing is intentionally independent of retrieval: a greeting or
# "what food is this?" must never become a generic nutrition/RAG summary.
_IDENTIFY_PHRASES = (
    'นี่เมนูอะไร', 'นี่คือเมนูอะไร', 'เมนูนี้คืออะไร', 'เมนูนี้ชื่ออะไร',
    'ชื่อเมนูอะไร', 'ชื่อเมนู', 'ชื่ออาหาร', 'รูปนี้คืออะไร', 'รูปนี้เป็นอะไร',
    'ภาพนี้คืออะไร', 'ในภาพคืออะไร', 'ในรูปคืออะไร', 'ในรูปคือเมนูอะไร',
    'ในภาพเป็นเมนูอะไร', 'ในรูปเป็นอาหารอะไร', 'รูปนี้อาหารอะไร',
    'อาหารในภาพคืออะไร', 'อาหารในรูปคืออะไร', 'นี่อาหารอะไร', 'นี่คืออะไร',
    'นี่คืออาหารอะไร', 'จานนี้คืออะไร', 'จานนี้ชื่ออะไร', 'เมนูอะไรครับ',
    'เมนูอะไรคะ', 'นี่เมนูไหน', 'ภาพนี้เป็นอาหารอะไร', 'อาหารจานนี้คืออะไร',
)
_INGREDIENT_WORDS = ('ส่วนผสม', 'ส่วนประกอบ', 'วัตถุดิบ', 'ใส่อะไร', 'ทำจากอะไร', 'มีอะไรบ้าง', 'ประกอบด้วยอะไร')
_KCAL_WORDS = ('แคล', 'พลังงาน', 'kcal', 'กิโลแคล')
_ADVICE_WORDS = (
    'ลดน้ำหนัก', 'คุมน้ำหนัก', 'น้ำหนัก', 'อ้วน', 'สุขภาพ', 'โซเดียม', 'เค็ม',
    'น้ำตาล', 'ไขมัน', 'เบาหวาน', 'ความดัน', 'โรคไต', 'โรคหัวใจ',
    'กินได้ไหม', 'กินได้มั้ย', 'ทานได้ไหม', 'ทานได้มั้ย', 'ควรกิน',
    'ควรทาน', 'ปรับ', 'แนะนำ', 'สารอาหาร', 'ประโยชน์', 'ข้อควรระวัง',
    'แพ้', 'สารก่อภูมิแพ้', 'โปรตีน', 'คาร์โบไฮเดรต', 'วิตามิน', 'ผู้สูงอายุ',
)
_RECIPE_WORDS = ('วิธีทำ', 'ทำยังไง', 'ทำอย่างไร', 'ขั้นตอนการทำ', 'วิธีปรุง', 'สูตรอาหาร')
_GENERAL_FOOD_WORDS = ('อาหารไทย', 'กินอะไรดี', 'กินอะไร', 'ทานอะไร', 'อาหารอะไรดี', 'มื้อเย็น', 'มื้อเช้า', 'มื้อกลางวัน', 'โภชนาการ', 'อาหาร')


def _is_identification_question(query: str) -> bool:
    """Recognize natural Thai requests about a photo, not just a list of exact phrases.

    Examples: 'ที่ถ่ายไปเมนูอะไร', 'ผมถามว่าที่ให้วิเคราะห์มันคือเมนูอะไร',
    'รูปที่ส่งไปเป็นอาหารอะไร', 'ภาพเมื่อกี้คืออะไร'.
    """
    q = re.sub(r'[\s\u200b\u200c\u200d]+', '', (query or '').lower())
    if not q:
        return False
    if any(p in q for p in ('ถ่ายอะไร', 'รูปอะไร', 'ภาพอะไร')):
        return True
    if any(phrase.replace(' ', '') in q for phrase in _IDENTIFY_PHRASES):
        return True

    if any(t in q for t in ('แพ้อาหาร', 'สารก่อภูมิแพ้', 'ส่วนผสม', 'กี่แคล', 'แคลอรี', 'แคลอรี่', 'พลังงาน', 'โซเดียม', 'สารอาหาร', 'ลดน้ำหนัก', 'วิธีทำ')):
        return False
    image_context = any(word in q for word in (
        'ภาพ', 'รูป', 'ถ่าย', 'อัปโหลด', 'อัพโหลด', 'ส่งไป', 'ให้วิเคราะห์',
        'วิเคราะห์ไป', 'วิเคราะห์แล้ว', 'เมื่อกี้', 'เมื่อครู่', 'จานนี้',
        'อันนี้', 'ที่ส่ง', 'ที่ให้ดู', 'ที่ให้วิเคราะห์', 'ที่ผมให้วิเคราะห์',
    ))
    food_name_request = any(word in q for word in (
        'เมนูอะไร', 'เมนูไหน', 'ชื่อเมนู', 'ชื่ออาหาร', 'เป็นเมนู',
        'คือเมนูอะไร', 'เป็นอาหารอะไร', 'อาหารอะไร', 'ทายว่า', 'ระบุว่า',
    ))
    # 'แนะนำเมนูอะไรแคลต่ำ' should be a recommendation, not image identity.
    recommendation_words = ('แคลต่ำ', 'แคลน้อย', 'ลดน้ำหนัก', 'คุมน้ำหนัก',
                            'ควรกิน', 'ควรทาน', 'แนะนำอาหาร', 'แนะนำเมนู')
    if food_name_request and (image_context or not any(w in q for w in recommendation_words)):
        return True
    if image_context and any(word in q for word in (
        'คืออะไร', 'เป็นอะไร', 'ชื่ออะไร', 'มันคืออะไร', 'มันเป็นอะไร',
        'บอกชื่อ', 'ช่วยระบุ', 'จำแนกได้', 'ทายได้', 'เมนูไหน',
    )):
        return True
    return False


def _is_comparison_question(q: str, menus: list[str]) -> bool:
    return len(menus) >= 2 and any(x in q for x in ('กับ', 'หรือ', 'เทียบ', 'เปรียบเทียบ', 'กว่า', 'ต่างกัน', 'ระหว่าง'))


def _kcal_display(cid: str):
    """Return (kcal, portion, note, source_url, evidence_is_verified)."""
    if cid in EVIDENCE_BY_CLASS:
        r = EVIDENCE_BY_CLASS[cid]
        val = f"{r['kcal_min']:g}" if r['kcal_min'] == r['kcal_max'] else f"{r['kcal_min']:g}–{r['kcal_max']:g}"
        return val, r['portion'], r['note'], r.get('source_url', ''), True
    if cid in ESTIMATED_KCAL_BY_CID:
        val, portion, note = ESTIMATED_KCAL_BY_CID[cid]
        return val.replace('-', '–'), portion, note, '', False
    return None, None, '', '', False


def _menu_ingredients(menu: str) -> str:
    cid = MENU_BY_TH.get(menu)
    if not cid:
        return ''
    parts = [item.strip() for item in MENU_METADATA[cid][1].split(',') if item.strip()]
    return '、'.join(parts).replace('、', ', ')


def _simple_allergen_note(menu: str) -> str:
    s = _menu_ingredients(menu)
    potential = []
    for terms, desc in (
        (('กุ้ง', 'หอย', 'ปู', 'ปลา', 'ปลาหมึก'), 'อาหารทะเล/ปลา'),
        (('ถั่ว',), 'ถั่ว'),
        (('นม', 'ชีส', 'เนย'), 'นม'),
        (('ไข่',), 'ไข่'),
        (('แป้งสาลี', 'บะหมี่', 'เกี๊ยว'), 'ข้าวสาลี'),
    ):
        if any(w in s for w in terms):potential.append(desc)
    potential = list(dict.fromkeys(potential))
    if potential:
        return 'จากสูตรทั่วไปอาจมี ' + ', '.join(potential) + ' แต่ **ภาพถ่ายไม่ยืนยันสารก่อภูมิแพ้** ต้องตรวจสอบกับผู้ปรุงก่อนกิน'
    return 'ภาพถ่ายและสูตรทั่วไปไม่สามารถยืนยันว่าไม่มีสารก่อภูมิแพ้ได้ ควรถามผู้ปรุงโดยตรง'


def _source_line(url: str) -> str:
    return f'\n\nแหล่งข้อมูลสูตรอ้างอิง: {url}' if url else ''


def _compare_kcal(menus: list[str]) -> str:
    rows = []
    verified = []
    for m in menus[:3]:
        cid = MENU_BY_TH[m]
        val, portion, _, url, checked = _kcal_display(cid)
        if not val:
            rows.append(f'- **{m}:** ยังไม่มีข้อมูลพลังงานที่ตรวจสอบได้')
            continue
        suffix = 'อ้างอิงสูตร' if checked else 'ประมาณการโดยผู้พัฒนา (ไม่มีเอกสารยืนยัน)'
        rows.append(f'- **{m}:** {val} กิโลแคลอรี / {portion} — {suffix}' + (f' ([แหล่งที่มา]({url}))' if checked and url else ''))
        if checked:
            r=EVIDENCE_BY_CLASS[cid]
            verified.append((m,(r['kcal_min']+r['kcal_max'])/2,portion))
    # Do not rank numeric estimates with different portions and recipes as precise science.
    rows.append('**หมายเหตุ:** ตัวเลขเป็นของสูตรและขนาดบริโภคที่ระบุ ซึ่งอาจเทียบกันตรง ๆ ไม่ได้ และไม่สามารถคำนวณพลังงานจากภาพถ่าย')
    return '### เปรียบเทียบพลังงานอาหาร\n' + '\n'.join(rows)


def _recommend_sourced_foods(q: str) -> str:
    entries = sorted(
        [(cid, rec) for cid, rec in EVIDENCE_BY_CLASS.items()],
        key=lambda x: (x[1]['kcal_min']+x[1]['kcal_max'])/2,
    )[:5]
    rows=['จากสูตรอ้างอิงที่มีในฐานข้อมูล ตัวอย่างเมนูพลังงานค่อนข้างต่ำ ได้แก่:']
    for cid, rec in entries:
        kcal=f"{rec['kcal_min']:g}" if rec['kcal_min']==rec['kcal_max'] else f"{rec['kcal_min']:g}–{rec['kcal_max']:g}"
        rows.append(f"- **{MENU_METADATA[cid][0]}**: {kcal} กิโลแคลอรี / {rec['portion']}" + (f" ([ที่มา]({rec['source_url']}))" if rec.get('source_url') else ''))
    rows.append('ควรดูปริมาณอาหารจริง และจัดมื้อให้มีผัก ข้าว/แป้ง และโปรตีนในสัดส่วนสมดุล ตัวเลขข้างต้นไม่ใช่พลังงานที่วัดจากรูป')
    return '\n'.join(rows)


def _reply_if_unknown_or_general(q: str, use_llm: bool) -> str:
    in_domain=any(w in q.lower() for w in _GENERAL_FOOD_WORDS+_ADVICE_WORDS+_KCAL_WORDS)
    if not in_domain:
        return 'ตอนนี้ผมช่วยวิเคราะห์อาหารไทย 50 เมนู และตอบคำถามด้านส่วนผสม พลังงาน และโภชนาการเป็นหลักครับ หากถามเรื่องอื่น ผมอาจไม่มีข้อมูลที่น่าเชื่อถือเพียงพอ'
    if any(x in q for x in ('ลดน้ำหนัก','แนะนำเมนู','แคลน้อย','แคลอรีต่ำ','กินอะไรดี','กินอะไร', 'อาหารอะไรดี')):
        return _recommend_sourced_foods(q)
    if use_llm and os.getenv('GEMINI_API_KEY'):
        docs=[d for d in rag_retrieve(q,'',3) if d['kind']=='guideline']
        if docs:
            context='\n'.join(f"{d['title']}: {d['text']}" for d in docs)
            generated=gemini_answer(
                'ตอบเป็นภาษาไทยสั้น ๆ ตรงคำถาม ใช้เฉพาะแหล่งข้อมูลที่ให้ หากไม่มีคำตอบให้บอกว่าไม่ทราบ ไม่แสดงข้อความ PDF ดิบ\n'
                f'คำถาม: {q}\nเอกสาร:\n{context}'
            )
            if generated and _safe_chat_output(generated):
                return generated
    return 'ผมยังไม่มีข้อมูลอ้างอิงเพียงพอสำหรับคำถามนี้ครับ ลองระบุชื่อเมนูอาหารไทย หรือถามเรื่องส่วนผสม แคลอรี วิธีปรับเมนูเพื่อสุขภาพได้'


def _safe_chat_output(txt: str) -> bool:
    if not txt or len(txt) > 7000 or re.search(r'\(cid:\d+\)',txt,re.I):
        return False
    if any(x in txt for x in ('à¸', 'à¹', '\ufffd', '\uf000')):
        return False
    invalid=sum(unicodedata.category(ch)=='Co' for ch in txt)
    return invalid <= max(1,len(txt)*0.01)


def answer(
    question: str,
    menu: str = '',
    use_llm: bool = True,
    llm: bool | None = None,
    history_context: str = '',
    *,
    image_menu: str = '',
    image_score: float | None = None,
    image_low_confidence: bool = False,
) -> str:
    """Intent-first Thai nutrition assistant. Never force nutrition cards on unrelated queries."""
    if llm is not None:
        use_llm=llm
    q=(question or '').strip()
    if not q:
        return 'พิมพ์คำถามมาได้เลยครับ'

    # 1. Pure conversation takes priority even if an image was classified before.
    social=_detect_conversational_reply(q, active_menu='')
    if social:
        return social

    explicit=find_menus(q)
    # 2. Identification is a special vision question, not a nutrition retrieval query.
    if _is_identification_question(q):
        if image_menu and image_menu in MENU_BY_TH:
            conf=(f' (คะแนน Softmax {image_score:.1%} ไม่ใช่ความแม่นยำของโมเดล)' if image_score is not None else '')
            caution = ('\n\n⚠️ โมเดลยังไม่มั่นใจ นี่เป็นเพียงข้อคาดเดา ไม่ใช่การระบุที่ยืนยันแล้ว'
                       if image_low_confidence else '')
            return f'จากภาพที่ส่งมา AI คาดว่าเป็น **{image_menu}** ครับ{conf}{caution}\n\nหากต้องการ คุณถามต่อเรื่องส่วนผสมหรือพลังงานอ้างอิงได้ครับ'
        if explicit:
            return f'คุณกำลังถามถึงเมนู **{explicit[0]}** ครับ แต่ตอนนี้ยังไม่มีผลวิเคราะห์ภาพที่ยืนยันว่าเป็นเมนูนั้น'
        return 'ตอนนี้ยังไม่มีผลจำแนกภาพที่เชื่อมกับแชทครับ กรุณาอัปโหลดภาพแล้วกด **วิเคราะห์ภาพอาหาร** ก่อน ผมจึงจะบอกชื่อเมนูจากภาพได้'

    # Explicitly typed dishes replace, rather than accidentally append to, image context.
    if _is_comparison_question(q, explicit):
        return _compare_kcal(explicit)
    target=explicit[0] if explicit else (menu if menu in MENU_BY_TH else '')
    wants_ing=any(t in q for t in _INGREDIENT_WORDS)
    wants_kcal=any(t in q.lower() for t in _KCAL_WORDS)
    wants_recipe=any(t in q for t in _RECIPE_WORDS)
    wants_allergy=any(t in q for t in ('แพ้','สารก่อภูมิแพ้'))
    wants_advice=any(t in q for t in _ADVICE_WORDS)
    wants_info=any(t in q for t in ('สารอาหาร','โภชนาการ','ประโยชน์'))

    if not target:
        basic_q=q.replace(' ', '')
        if 'แคลอรีคืออะไร' in basic_q or 'แคลคืออะไร' in basic_q:
            return 'แคลอรี (กิโลแคลอรี หรือ kcal บนฉลากอาหาร) คือหน่วยบอกพลังงานที่ร่างกายได้รับจากอาหารครับ ปริมาณพลังงานของเมนูจริงขึ้นอยู่กับส่วนผสมและขนาดจาน'
        if 'โภชนาการคืออะไร' in basic_q:
            return 'โภชนาการคือเรื่องอาหารและสารอาหารที่ร่างกายนำไปใช้ เช่น คาร์โบไฮเดรต โปรตีน ไขมัน วิตามิน และแร่ธาตุครับ'
        if any(k in q for k in ('แนะนำเมนู', 'แนะนำอาหาร', 'แคลต่ำ', 'แคลน้อย', 'ลดน้ำหนัก', 'อาหารอะไรดี', 'กินอะไรดี')):
            return _recommend_sourced_foods(q)
        if wants_ing or wants_kcal or wants_recipe or wants_allergy:
            return 'ต้องทราบชื่อเมนูก่อนครับ กรุณาระบุชื่ออาหารในคำถาม หรืออัปโหลดภาพแล้วกด **วิเคราะห์ภาพอาหาร**'
        return _reply_if_unknown_or_general(q, use_llm)

    cid=MENU_BY_TH.get(target)
    if not cid:
        return 'ยังไม่มีเมนูนี้ในฐานข้อมูล 50 เมนูครับ'

    # 3. Query RAG once for this actual question and chosen menu, never a different menu.
    retrieved = rag_retrieve(q, target, top_k=3)
    # Question-directed answers: no mandatory full nutrition card.
    lines=[]
    if wants_recipe:
        lines.append(f'**{target}**: ฐานข้อมูลนี้มีรายการส่วนผสมทั่วไป แต่ยังไม่มีขั้นตอนปรุงที่ตรวจสอบกับสูตรเฉพาะครับ')
        lines.append(f'ส่วนผสมทั่วไป: {_menu_ingredients(target)}')
    if wants_ing:
        lines.append(f'**ส่วนผสมทั่วไปของ{target}:** {_menu_ingredients(target)}')
        lines.append('สูตรของแต่ละร้านอาจต่างกัน จึงยืนยันส่วนผสมจริงจากรูปถ่ายไม่ได้')
    if wants_kcal:
        val,portion,note,url,checked=_kcal_display(cid)
        if val:
            if checked:
                lines.append(f'**{target}: {val} กิโลแคลอรี** ต่อ {portion} (สูตรอ้างอิง ไม่ใช่ผลวัดจากภาพ){_source_line(url)}')
            else:
                lines.append(f'**{target}: ประมาณ {val} กิโลแคลอรี** ต่อ {portion} — เป็นเพียงค่าประมาณของผู้พัฒนา ยังไม่มีแหล่งข้อมูลเฉพาะเมนูมายืนยัน จึงใช้แทนค่าจริงไม่ได้')
        else:
            lines.append(f'ยังไม่มีข้อมูลพลังงานที่ตรวจสอบได้สำหรับ **{target}** ครับ ไม่สามารถคำนวณจากภาพโดยตรง')
    if wants_allergy:
        lines.append('**ข้อควรระวังเรื่องการแพ้อาหาร:** '+_simple_allergen_note(target))
    if (wants_advice or wants_info) and not wants_allergy:
        if wants_info:
            lines.append('**ข้อมูลสารอาหาร:** สามารถบอกแหล่งสารอาหารจากส่วนผสมทั่วไปได้ แต่ยังไม่มีผลวิเคราะห์โปรตีน/ไขมัน/คาร์โบไฮเดรตเป็นกรัมของอาหารในรูปครับ')
        lines.append('**แนวทางปรับเมนู:** '+_InferCookingGuidance(target,_menu_ingredients(target)))
        if any(x in q for x in ('โรคไต','เบาหวาน','โรคหัวใจ','ความดัน','ยา','รักษา')):
            lines.append('หากมีโรคประจำตัวหรือข้อจำกัดโภชนาการ ควรปรึกษานักกำหนดอาหารหรือแพทย์ ไม่ควรใช้คำตอบนี้สั่งเปลี่ยนยา')
    if not lines and not explicit and not any(w in q for w in (_GENERAL_FOOD_WORDS + _ADVICE_WORDS + _KCAL_WORDS + _INGREDIENT_WORDS + _RECIPE_WORDS)):
        return _reply_if_unknown_or_general(q, use_llm=False)
    if not lines:
        val,portion,note,url,checked=_kcal_display(cid)
        lines.append(f'**{target}** เป็นหนึ่งใน 50 เมนูที่ระบบรู้จัก ส่วนผสมทั่วไปคือ {_menu_ingredients(target)}')
        if val:
            kind='ค่าพลังงานสูตรอ้างอิง' if checked else 'พลังงานประมาณการที่ยังไม่มีเอกสารยืนยัน'
            lines.append(f'{kind}: {val} กิโลแคลอรี ต่อ {portion}'+(_source_line(url) if checked else ''))

    # 4. Generative model may augment open-ended advice, never overwrite identity,
    # ingredients, or sourced energy. Never send raw PDF glyph-extraction to LLM.
    open_ended = wants_advice and any(t in q for t in ('ทำไม','อย่างไร','ยังไง','แนะนำ','เหมาะ','ควร','ข้อดี','ข้อเสีย'))
    if open_ended and use_llm and os.getenv('GEMINI_API_KEY'):
        matches=[d for d in retrieved
                 if d['kind'] in ('recipe','nutrition','guideline','pdf') and
                    (d['kind']!='pdf' or _clean_pdf_thai(d['text']))]
        evidence='\n'.join(f"{d['title']}: {d['text'][:500]}" for d in matches)
        generated=gemini_answer(
            'ตอบภาษาไทยอย่างเป็นธรรมชาติ กระชับ ไม่ซ้ำคำตอบเบื้องต้น และใช้ข้อมูลอ้างอิงที่ให้เท่านั้น '
            'อย่าแต่งตัวเลขแคลอรีหรือแพทย์ อย่านำข้อมูลคนละเมนูมาตอบ\n'
            f'เมนู: {target}\nคำถาม: {q}\nหลักฐาน:\n{evidence}'
        )
        if generated and _safe_chat_output(generated):
            lines.append('**ข้อมูลเพิ่มเติม:** '+generated)

    return '\n\n'.join(dict.fromkeys(lines))


@spaces.GPU
def predict(image: Image.Image):
    image = ImageOps.exif_transpose(image).convert('RGB')
    x = VALID_TF(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        probabilities = torch.softmax(logits, dim=-1).flatten().cpu().numpy()
    indices = np.argsort(probabilities)[-5:][::-1]
    return [(MENU_METADATA[CLASS_IDS[int(idx)]][0], float(probabilities[idx])) for idx in indices]


def classify(image):
    blank = {'menu': None, 'image_menu': None, 'score': None, 'low_confidence': False}
    if image is None:
        return 'กรุณาอัปโหลดรูปภาพอาหารไทย', '', blank
    try:
        ranking = predict(image)
    except Exception as exc:
        return f'เกิดข้อผิดพลาดในการวิเคราะห์ภาพ: {type(exc).__name__}', '', blank
    menu, score = ranking[0]
    gap = score - ranking[1][1]
    out = (
        f'### 🍲 AI จำแนกเป็น: {menu}\n'
        f'ความมั่นใจ Softmax: **{score:.1%}** (คำนวณจากโมเดล EfficientNet-B0)\n\n' +
        '**Top 5 อันดับแรก:**\n' + '\n'.join(f'- {n}: {p:.1%}' for n, p in ranking)
    )
    if score < 0.60 or gap < 0.10:
        return (out + '\n\n⚠️ ความมั่นใจต่ำกว่าเกณฑ์ นี่เป็นเพียงเมนูที่ AI คาดว่าใกล้เคียงที่สุด',
                '⚠️ ความมั่นใจของผลจำแนกต่ำ ไม่ควรถือเป็นการยืนยันชื่ออาหาร',
                {'menu': None, 'image_menu': menu, 'score': round(score, 6), 'low_confidence': True})
    return out, answer('เมนูนี้มีส่วนผสมอะไรบ้าง กี่แคลอรี', menu, use_llm=False), {'menu': menu, 'image_menu': menu, 'score': round(score, 6), 'low_confidence': False}


def _extract_content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and 'text' in item:
                parts.append(str(item['text']))
            elif isinstance(item, str):
                parts.append(item)
        return ' '.join(parts)
    return str(content or '')


def chat_send(msg, history, state, image=None):
    history = list(history or [])
    user_msg = (msg or '').strip()
    state = dict(state or {'menu': None})
    if not user_msg:
        return history, '', state

    explicit = find_menus(user_msg)
    last_image = state.get('image_menu') or ''
    conf = state.get('score')
    is_identity = _is_identification_question(user_msg)
    # When the user uploaded a photo but switched tabs before clicking Analyze,
    # classify it on demand so the chat can still answer 'what did I photograph?'.
    image_related = is_identity or any(t in user_msg for t in ('รูปนี้', 'ภาพนี้', 'ที่ถ่ายไป', 'จานนี้', 'เมนูนี้'))
    if image is not None and image_related and not last_image:
        _, _, inferred = classify(image)
        state.update(inferred)
        last_image = state.get('image_menu') or ''
        conf = state.get('score')

    # Explicit menu names never inherit an unrelated menu from prior image/history.
    if explicit:
        topic = explicit[0]
        if not _is_comparison_question(user_msg, explicit):
            state['menu'] = topic
    elif is_identity:
        topic = ''  # only image_menu is allowed for visual identity requests
    elif (not any(w in user_msg for w in ('แนะนำอาหาร','แนะนำเมนู','กินอะไรดี','อาหารอะไรดี','เมนูแคลน้อย','อาหารแคลน้อย','อาหารลดน้ำหนัก'))
          and any(w in user_msg for w in ('เมนูนี้','จานนี้','ชามนี้','อันนี้','ภาพนี้','รูปนี้','เมื่อกี้','กี่แคล','ส่วนผสม','มีอะไรบ้าง','แคลอรี่','วิธีทำ','สารอาหาร','โภชนาการ','ทำยังไง','กินได้ไหม','กินได้มั้ย','ทานได้ไหม','ไขมัน','โซเดียม','แพ้','สารก่อภูมิแพ้'))):
        topic = state.get('menu') or last_image
        if not topic:
            for h in reversed(history):
                if isinstance(h, dict) and h.get('role') == 'user':
                    found = find_menus(_extract_content_text(h.get('content', '')))
                    if found:
                        topic = found[0]
                        break
    else:
        topic = ''

    bot_reply = answer(
        user_msg,
        menu=topic if len(explicit) <= 1 else '',
        image_menu=last_image,
        image_score=conf,
        image_low_confidence=bool(state.get('low_confidence')),
        history_context='',
    )
    history.extend([{'role':'user','content':user_msg},
                    {'role':'assistant','content':bot_reply}])
    return history, '', state


def reset_image():
    return 'กรุณาอัปโหลดรูปภาพอาหารไทย', '', {'menu': None, 'image_menu': None, 'score': None, 'low_confidence': False}, []


def auto_classify_image(image):
    '''Recognize the new photo automatically and clear any stale chat history.'''
    result, nutrition, state = classify(image)
    return result, nutrition, state, []


def clear_chat(state):
    state = dict(state or {})
    # Clearing the conversation must not forget a classified image still on screen.
    return [], '', {'menu': state.get('image_menu'),
                    'image_menu': state.get('image_menu'), 'score': state.get('score'),
                    'low_confidence': state.get('low_confidence', False)}

PROJECT_TITLE = 'ระบบระบุเมนูอาหารไทยจากภาพถ่าย พร้อมแชทบอทแนะนำโภชนาการ'

# Familiar Gradio layout: clean title, tabs, left image / right result.
# No oversized hero, gradients, statistics dashboard or decorative cards.
APP_CSS = r"""
body, .gradio-container {
  font-family: 'Noto Sans Thai', 'Sarabun', Tahoma, Arial, sans-serif !important;
}
.gradio-container {
  max-width: 1200px !important;
  margin: 0 auto !important;
  padding: 14px 20px 38px !important;
}
#tf-project-title h1 {
  font-size: clamp(22px, 2.2vw, 28px) !important;
  line-height: 1.5 !important;
  margin: 5px 0 8px !important;
}
#tf-subtitle { color: #63717a; }
#tf-status { color: #45675d; margin: 5px 0 14px; }
#tf-tabs { margin-top: 4px; }
#tf-image { border-radius: 12px; overflow: hidden; }
#tf-result, #tf-nutrition { line-height: 1.8; }
#tf-chatbot { border-radius: 12px; overflow: hidden; }
#tf-primary-btn, #tf-send-btn { font-weight: 650; }
@media (max-width: 700px) {
  .gradio-container { padding: 8px 12px 24px !important; }
  #tf-project-title h1 { font-size: 22px !important; }
}
"""


def launch_ui():
    import pandas as pd
    coverage = pd.read_csv(HERE / 'nutrition_50.csv').fillna('')
    theme = gr.themes.Default(primary_hue='teal', secondary_hue='slate', neutral_hue='slate')
    gr_major = int(gr.__version__.split('.')[0])
    block_style = {'theme': theme, 'css': APP_CSS} if gr_major < 6 else {}
    with gr.Blocks(title=PROJECT_TITLE, **block_style) as demo:
        gr.Markdown('# 🍲 ' + PROJECT_TITLE, elem_id='tf-project-title')
        gr.Markdown('อัปโหลดภาพอาหารเพื่อให้ AI ระบุชื่อเมนู จากนั้นถามเรื่องส่วนผสม พลังงาน และโภชนาการต่อได้ในแชท', elem_id='tf-subtitle')
        gr.Markdown(
            f'**รองรับ 50 เมนูไทย**  ·  **พลังงานที่มีหลักฐาน {len(EVIDENCE_BY_CLASS)}/50 เมนู**  ·  **เอกสาร PDF ที่อ่านได้ {PDF_CHUNKS} ชิ้น**',
            elem_id='tf-status',
        )

        state = gr.State({'menu': None, 'image_menu': None, 'score': None, 'low_confidence': False})
        with gr.Tabs(elem_id='tf-tabs'):
            with gr.Tab('📷 วิเคราะห์ภาพอาหาร'):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=1, min_width=280):
                        image = gr.Image(label='อัปโหลดภาพอาหารไทย', type='pil', sources=['upload', 'webcam'], height=350, elem_id='tf-image')
                        btn = gr.Button('วิเคราะห์ภาพอาหาร', variant='primary', elem_id='tf-primary-btn')
                    with gr.Column(scale=1, min_width=280):
                        gr.Markdown('### ผลการจำแนกอาหาร')
                        result = gr.Markdown('อัปโหลดภาพเพื่อเริ่มวิเคราะห์', elem_id='tf-result')
                        gr.Markdown('### ส่วนผสมและพลังงานอ้างอิง')
                        nutrition = gr.Markdown('ข้อมูลจะแสดงเมื่อวิเคราะห์ภาพแล้ว', elem_id='tf-nutrition')
                gr.Markdown('*ระบบระบุเมนูจากภาพโดยอัตโนมัติ คะแนน Softmax ไม่ใช่ความแม่นยำรวม และไม่สามารถวัดพลังงานจริงจากภาพถ่าย*')

            with gr.Tab('💬 แชทบอทแนะนำโภชนาการ'):
                gr.Markdown('ถามได้ตามธรรมชาติ เช่น **“ที่ถ่ายไปคือเมนูอะไร”**, **“จานนี้กี่แคล”**, **“สวัสดี”** หรือ **“ลดโซเดียมอย่างไร”**')
                chat_kwargs = {'height': 420, 'label': 'ThaiFood AI — แชทบอทแนะนำโภชนาการ', 'elem_id': 'tf-chatbot'}
                if gr_major < 6:
                    chat_kwargs['type'] = 'messages'
                chatbot = gr.Chatbot(**chat_kwargs)
                with gr.Row():
                    q = gr.Textbox(label='คำถาม', placeholder='พิมพ์คำถามเกี่ยวกับภาพหรือโภชนาการ...', scale=5)
                    send = gr.Button('ส่งคำถาม', variant='primary', scale=1, elem_id='tf-send-btn')
                    clear_btn = gr.Button('ล้างแชท', scale=1)
                gr.Examples(
                    examples=[
                        ['นี่เมนูอะไร'],
                        ['เมนูนี้กี่แคลอรี'],
                        ['มีส่วนผสมอะไรบ้าง'],
                        ['ผัดไทยกับข้าวมันไก่อะไรให้พลังงานมากกว่า'],
                    ],
                    inputs=q,
                    label='ตัวอย่างคำถาม',
                )

            with gr.Tab('📚 ข้อมูล 50 เมนู'):
                gr.Markdown('ส่วนผสมเป็นสูตรทั่วไป และมีข้อมูลพลังงานอ้างอิงเฉพาะบางเมนู')
                gr.Dataframe(
                    coverage[['menu_th','general_ingredients','recipe_name','kcal_min','kcal_max','portion','source_url']],
                    interactive=False,
                )

            with gr.Tab('📊 ผลทดสอบโมเดล'):
                gr.JSON(value=EVALUATION, label='ผลประเมินจาก Test Set (บันทึกไว้ก่อน Deploy)')
                gr.Markdown(
                    f'**PDF chunks ที่อ่านได้จริง:** {PDF_CHUNKS}  ·  '
                    '**ผลประเมิน RAG จากชุดคำถามอิสระ:** ยังไม่มีผลที่ตรวจสอบได้'
                )

            with gr.Tab('🛡️ ข้อจำกัดของระบบ'):
                gr.Markdown("""- AI จำแนกได้เฉพาะ 50 เมนูที่อยู่ในชุดข้อมูล และอาจทำนายผิด
- คะแนน Softmax ของภาพเดียว ไม่ใช่ความแม่นยำของโมเดลทั้งชุด
- ส่วนผสมและสารก่อภูมิแพ้จริงไม่สามารถยืนยันจากภาพถ่ายเพียงอย่างเดียว
- ค่าพลังงานอ้างอิงขึ้นอยู่กับสูตรและขนาดเสิร์ฟ ไม่ใช่ผลวัดจากภาพ
- ข้อมูลนี้ไม่ทดแทนคำแนะนำของแพทย์หรือนักกำหนดอาหาร""")

        # Preserve photo-to-chat state. A new image replaces old classification;
        # clearing the conversation must not forget the photo still displayed.
        image.change(auto_classify_image, inputs=image, outputs=[result, nutrition, state, chatbot])
        btn.click(classify, inputs=image, outputs=[result, nutrition, state])
        send.click(chat_send, inputs=[q, chatbot, state, image], outputs=[chatbot, q, state])
        q.submit(chat_send, inputs=[q, chatbot, state, image], outputs=[chatbot, q, state])
        clear_btn.click(clear_chat, inputs=state, outputs=[chatbot, q, state])
    return demo


if __name__ == '__main__':
    app = launch_ui()
    launch_args = dict(
        server_name='0.0.0.0',
        server_port=int(os.getenv('PORT', '7860')),
        show_error=True,
        ssr_mode=False,
    )
    if int(gr.__version__.split('.')[0]) >= 6:
        launch_args.update(theme=gr.themes.Default(primary_hue='teal', secondary_hue='slate', neutral_hue='slate'), css=APP_CSS)
    app.launch(**launch_args)