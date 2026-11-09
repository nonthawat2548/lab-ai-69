"""Live Gradio API integration check: image -> model -> chat -> follow-up.

Uses a generated tiny placeholder image as a plumbing smoke test, not a
food-identification accuracy test. Never modifies model or user data.
"""
import re
import struct
import tempfile
import zlib
from pathlib import Path
from gradio_client import Client, handle_file

APP = "https://nongame-thaifood-ai-gradio.hf.space"

def chunk(tag, data):
    return struct.pack("!I", len(data)) + tag + data + struct.pack("!I", zlib.crc32(tag + data) & 0xffffffff)

def make_image(path):
    w, h = 64, 64
    pixel_rows = b"".join(b"\x00" + b"\xb4\x65\x23" * w for _ in range(h))
    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack("!2I5B", w, h, 8, 2, 0, 0, 0)
    path.write_bytes(header + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(pixel_rows)) + chunk(b"IEND", b""))

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "smoke.png"
    make_image(path)
    cli = Client(APP, verbose=False)
    result = cli.predict(handle_file(str(path)), api_name="/auto_classify_image")
    assert isinstance(result, (tuple, list)) and result, f"Unexpected image API: {type(result)}"
    text = str(result[0])
    m = re.search(r"AI จำแนกเป็น:\s*([^\n*]+)", text)
    assert m, f"No image identification from model: {text[:400]}"
    food = m.group(1).strip()
    print(f"PASS live vision inference: returned a Top-1 label ({food}) from 50 classes; placeholder image is not accuracy evidence")

    reply = cli.predict("ที่ถ่ายไปเมนูอะไร", [], handle_file(str(path)), api_name="/chat_send")
    assert isinstance(reply, (list, tuple)) and reply
    history = reply[0]
    assert isinstance(history, list) and history
    last = history[-1]
    answer = last.get("content", "") if isinstance(last, dict) else str(last)
    assert food in str(answer), f"Chat has lost image context: {str(answer)[:500]}"
    print("PASS live chat uses latest image-classification result")

    more = cli.predict("จานนี้กี่แคลอรี", history, handle_file(str(path)), api_name="/chat_send")
    new_history = more[0]
    last = new_history[-1]
    answer = last.get("content", "") if isinstance(last, dict) else str(last)
    assert food in str(answer), f"Follow-up lost dish: {str(answer)[:500]}"
    assert "กิโลแคลอรี" in str(answer), f"Follow-up has no calories: {str(answer)[:500]}"
    print("PASS live multi-turn follow-up uses last image")
