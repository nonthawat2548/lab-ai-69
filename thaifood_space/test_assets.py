"""Fast deployment checks; no model download, network, GPU or training."""
from pathlib import Path
import ast
import csv
import json
import sys

ROOT = Path(__file__).resolve().parent
EXPECTED = ("app.py", "knowledge.py", "nutrition_50.csv", "evaluation.json",
            "pdf_sources.json", "requirements.txt", "README.md")
for filename in EXPECTED:
    assert (ROOT / filename).is_file(), f"Missing deployment asset: {filename}"

for filename in ("app.py", "knowledge.py"):
    ast.parse((ROOT / filename).read_text(encoding="utf-8"), filename=filename)

sys.path.insert(0, str(ROOT))
from knowledge import MENU_METADATA, EVIDENCE_BY_CLASS  # noqa: E402

rows = list(csv.DictReader((ROOT / "nutrition_50.csv").open(encoding="utf-8-sig", newline="")))
assert len(MENU_METADATA) == 50, "Expected 50 metadata classes"
assert len(rows) == 50, "Expected 50 CSV menu rows"
assert len({r["class_id"] for r in rows}) == 50, "Duplicate class_id in CSV"
assert {r["class_id"] for r in rows} == set(MENU_METADATA), "Metadata/CSV class mismatch"
assert len(EVIDENCE_BY_CLASS) == 22, "Expected 22 sourced nutrition classes"
assert set(EVIDENCE_BY_CLASS).issubset(MENU_METADATA), "Evidence references unknown menu"

for row in rows:
    cid = row["class_id"]
    assert row["menu_th"] == MENU_METADATA[cid][0], f"Thai name mismatch: {cid}"
    if cid in EVIDENCE_BY_CLASS:
        assert row["kcal_min"] and row["kcal_max"] and row["source_url"], f"Missing cited nutrition: {cid}"

evaluation = json.loads((ROOT / "evaluation.json").read_text(encoding="utf-8"))
assert evaluation["classes"] == 50 and evaluation["n"] == 2105
assert 0.0 <= evaluation["accuracy"] <= 1.0
sources = json.loads((ROOT / "pdf_sources.json").read_text(encoding="utf-8"))
assert isinstance(sources, list) and sources
assert all(x.get("url", "").startswith("https://") and x.get("filename", "").endswith(".pdf") for x in sources)

readme=(ROOT/"README.md").read_text(encoding="utf-8")
assert "colorFrom: green" in readme and "sdk: gradio" in readme
print("PASS: 50 classes, 50 CSV rows, 22 cited menu records, evaluation, PDF sources, Python syntax, Space metadata")
