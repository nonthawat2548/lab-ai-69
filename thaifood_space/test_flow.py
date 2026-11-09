"""ThaiFood AI chat/image integration regression tests.

Exercise the *real* app.py functions and actual food metadata in a tiny process:
only neural-network predict() is mocked. This checks callback state flow without
requiring GPU, model download, Gradio boot, or network in each CI run.
The separate live HTTP check covers deployed Gradio service startup.
"""
import ast
import os
import re
import sys
import types
import unittest
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.pop("GEMINI_API_KEY", None)

from knowledge import MENU_METADATA, MENU_BY_TH, EVIDENCE_BY_CLASS, GENERAL_SOURCES

source = ast.parse((ROOT / "app.py").read_text(encoding="utf-8"))
needed_assignments = {
    "MENU_ALIASES_TO_CID", "ESTIMATED_KCAL_BY_CID", "_IDENTIFY_PHRASES",
    "_INGREDIENT_WORDS", "_KCAL_WORDS", "_ADVICE_WORDS", "_RECIPE_WORDS",
    "_GENERAL_FOOD_WORDS",
}
needed_functions = {
    "find_menus", "find_menu", "_is_followup_nutrition_query",
    "_detect_conversational_reply", "format_ref", "gemini_answer",
    "_InferCookingGuidance", "_get_menu_kcal_info", "build_nutrition_card",
    "_synthesize_smart_fallback", "_is_identification_question",
    "_is_comparison_question", "_kcal_display", "_menu_ingredients",
    "_simple_allergen_note", "_source_line", "_compare_kcal",
    "_recommend_sourced_foods", "_reply_if_unknown_or_general",
    "_safe_chat_output", "answer", "classify", "_extract_content_text",
    "chat_send", "reset_image", "auto_classify_image", "clear_chat",
}
nodes = []
for node in source.body:
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        if any(isinstance(t, ast.Name) and t.id in needed_assignments for t in targets):
            nodes.append(node)
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in needed_functions:
        node.decorator_list = []
        nodes.append(node)
assert len([n for n in nodes if isinstance(n, ast.FunctionDef)]) == len(needed_functions), "Missing source functions"
namespace = {
    "__name__": "thaifood_test_logic", "os": os, "re": re,
    "unicodedata": unicodedata, "MENU_METADATA": MENU_METADATA,
    "MENU_BY_TH": MENU_BY_TH, "EVIDENCE_BY_CLASS": EVIDENCE_BY_CLASS,
    "GENERAL_SOURCES": GENERAL_SOURCES,
}
future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
ast.fix_missing_locations(future)
module = ast.Module(body=[future] + nodes, type_ignores=[])
exec(compile(ast.fix_missing_locations(module), str(ROOT / "app.py"), "exec"), namespace)

def fake_retrieve(query, menu="", top_k=3):
    return [
        {"kind": "recipe", "title": "ทดสอบ", "menu": menu,
         "text": query, "source_label": "ชุดทดสอบ", "url": ""}
    ]
namespace["rag_retrieve"] = fake_retrieve

# Fake predictable Top-5 model rankings to verify UI -> state -> chatbot.
FIRST = MENU_METADATA["PadThai"][0]
SECOND = MENU_METADATA["KaoManGai"][0]
def predict(image):
    first = FIRST if image == "photo-A" else SECOND
    if image == "uncertain-photo":
        return [(first, .33), (SECOND if first != SECOND else FIRST, .30),
                ("ส้มตำ", .20), ("แกงจืด", .10), ("ต้มยำกุ้ง", .07)]
    return [(first, .94), (SECOND if first != SECOND else FIRST, .03),
            ("ส้มตำ", .01), ("แกงจืด", .01), ("ต้มยำกุ้ง", .01)]
namespace["predict"] = predict

class ThaiFoodFlowTests(unittest.TestCase):
    def ask(self, q, history=None, state=None, image=None):
        history, input_value, state = namespace["chat_send"](q, history or [], state, image)
        self.assertEqual(input_value, "")
        self.assertEqual(history[-2]["content"], q)
        self.assertEqual(history[-1]["role"], "assistant")
        self.assertNotIn("à¸", history[-1]["content"])
        self.assertNotIn("�", history[-1]["content"])
        return history[-1]["content"], history, state

    def test_identify_new_image_and_conversation(self):
        out, nutrition, state, history = namespace["auto_classify_image"]("photo-A")
        self.assertIn(FIRST, out)
        self.assertEqual(state["image_menu"], FIRST)
        self.assertEqual(state["menu"], FIRST)
        self.assertEqual(history, [])
        self.assertIn(FIRST, nutrition)

        for phrase in (
            "นี่เมนูอะไร", "ที่ถ่ายไปเมนูอะไร", "รูปที่ส่งไปเป็นอาหารอะไร",
            "ผมถามว่าที่ผมให้วิเคราะห์ มันคือเมนูอะไร", "นี่คืออะไรครับ",
            "จานนี้ชื่ออะไร", "เมื่อกี้ผมถ่ายอะไรมานะ",
        ):
            reply, history, state = self.ask(phrase, history, state)
            self.assertIn(FIRST, reply, phrase)
            self.assertNotIn("ต้องทราบชื่อเมนู", reply, phrase)
            self.assertNotIn("ตอนนี้ผมช่วยวิเคราะห์อาหารไทย", reply, phrase)
        reply, history, state = self.ask("จานนี้กี่แคล", history, state)
        self.assertIn(FIRST, reply)
        self.assertIn("กิโลแคลอรี", reply)
        reply, history, state = self.ask("มีส่วนผสมอะไรบ้าง", history, state)
        self.assertIn(FIRST, reply)
        self.assertIn("ส่วนผสม", reply)
        reply, history, state = self.ask("สวัสดีครับ", history, state)
        self.assertIn("สวัสดี", reply)
        self.assertNotIn("ส่วนผสมทั่วไปของ", reply)

    def test_switch_image_replaces_previous_context(self):
        _, _, state, history = namespace["auto_classify_image"]("photo-A")
        _, history, state = self.ask("นี่คือเมนูอะไร", history, state)
        _, _, state, history = namespace["auto_classify_image"]("photo-B")
        self.assertEqual(history, [])
        self.assertEqual(state["image_menu"], SECOND)
        reply, history, state = self.ask("รูปเมื่อกี้คืออะไร", history, state)
        self.assertIn(SECOND, reply)
        self.assertNotIn(FIRST, reply)
        reply, _, state = self.ask("กี่แคล", history, state)
        self.assertIn(SECOND, reply)

    def test_clear_chat_keeps_photo_but_drops_previous_topic(self):
        _, _, state, history = namespace["auto_classify_image"]("photo-A")
        _, history, state = self.ask("ข้าวมันไก่กี่แคล", history, state)
        self.assertEqual(state["menu"], SECOND)
        empty, q, state = namespace["clear_chat"](state)
        self.assertEqual(empty, [])
        self.assertEqual(q, "")
        self.assertEqual(state["image_menu"], FIRST)
        self.assertEqual(state["menu"], FIRST)
        reply, _, _ = self.ask("รูปนี้คืออะไร", empty, state)
        self.assertIn(FIRST, reply)

    def test_reset_image_clears_identity(self):
        _, _, state, history = namespace["auto_classify_image"]("photo-A")
        _, _, state, history = namespace["reset_image"]()
        self.assertEqual(state["image_menu"], None)
        self.assertEqual(history, [])
        reply, _, _ = self.ask("ที่ถ่ายไปเมนูอะไร", history, state)
        self.assertIn("ยังไม่มีผลจำแนกภาพ", reply)
        self.assertNotIn(FIRST, reply)

    def test_auto_recognition_when_chat_has_photo_but_no_state(self):
        reply, _, state = self.ask("ภาพที่ส่งไปคือเมนูอะไร", [], {}, "photo-A")
        self.assertIn(FIRST, reply)
        self.assertEqual(state["image_menu"], FIRST)

    def test_low_confidence_and_safe_disclosure(self):
        out, nutrition, state, _ = namespace["auto_classify_image"]("uncertain-photo")
        self.assertTrue(state["low_confidence"])
        self.assertIsNone(state["menu"])
        self.assertIn("ความมั่นใจ", out)
        reply, _, _ = self.ask("นี่อาหารอะไร", [], state)
        self.assertIn("ยังไม่มั่นใจ", reply)

    def test_greetings_do_not_turn_into_nutrition(self):
        for q in ["สวัสดี", "สวัสดีครับ", "หวัดดี", "hello", "ขอบคุณครับ"]:
            reply, _, _ = self.ask(q)
            self.assertTrue(any(w in reply for w in ("สวัสดี", "ยินดี")), q)
            self.assertNotIn("กิโลแคลอรี", reply, q)

    def test_explicit_menu_overrides_photo(self):
        _, _, state, history = namespace["auto_classify_image"]("photo-A")
        reply, history, state = self.ask("ข้าวมันไก่มีส่วนผสมอะไร", history, state)
        self.assertIn(SECOND, reply)
        self.assertEqual(state["image_menu"], FIRST)
        reply, _, _ = self.ask("แล้วกี่แคล", history, state)
        self.assertIn(SECOND, reply)

    def test_comparison_without_cross_contamination(self):
        out, _, _ = self.ask("ผัดไทยกับข้าวมันไก่อะไรให้พลังงานมากกว่า")
        self.assertIn(FIRST, out)
        self.assertIn(SECOND, out)
        self.assertIn("เปรียบเทียบ", out)

    def test_no_image_no_hallucinated_dish(self):
        for q in ["รูปนี้คืออะไร", "ที่ผมให้วิเคราะห์เมนูอะไร"]:
            reply, _, _ = self.ask(q)
            self.assertIn("ยังไม่มีผลจำแนกภาพ", reply)
        reply, _, _ = self.ask("จานนี้กี่แคล")
        self.assertIn("ต้องทราบชื่อเมนู", reply)

    def test_ood_and_recommendations(self):
        reply, _, _ = self.ask("ช่วยเขียนโปรแกรม Java ให้หน่อย")
        self.assertIn("อาหารไทย", reply)
        reply, _, _ = self.ask("แนะนำเมนูแคลอรีต่ำ")
        self.assertIn("กิโลแคลอรี", reply)
        self.assertNotIn("ต้องทราบชื่อเมนู", reply)

    def test_malformed_pdf_and_llm_output_rejected(self):
        clean = namespace["_clean_pdf_thai"] if "_clean_pdf_thai" in namespace else None
        self.assertFalse(namespace["_safe_chat_output"]("à¸à¹乱码"))
        self.assertFalse(namespace["_safe_chat_output"]("(cid:12345)"))
        self.assertTrue(namespace["_safe_chat_output"]("ผัดไทยมีส่วนผสมอะไรบ้าง"))

    def test_empty_message_is_noop(self):
        history, q, state = namespace["chat_send"](" ", [], {"menu": FIRST})
        self.assertEqual(history, [])
        self.assertEqual(q, "")
        self.assertEqual(state["menu"], FIRST)

if __name__ == "__main__":
    unittest.main(verbosity=2)
