"""Best-effort live Gradio health check after Hugging Face deploy."""
import json
import time
import urllib.error
import urllib.request

BASE = "https://nongame-thaifood-ai-gradio.hf.space"
for attempt in range(1, 7):
    try:
        request = urllib.request.Request(BASE + "/config", headers={"User-Agent": "ThaiFoodAI-CI-Healthcheck/1.0"})
        with urllib.request.urlopen(request, timeout=15) as response:
            config = json.load(response)
            assert response.status == 200
        types = {x.get("type") for x in config.get("components", [])}
        assert config.get("version"), "Not a Gradio configuration"
        assert {"image", "chatbot", "textbox"}.issubset(types), f"Missing components: {types}"
        print(f"PASS: remote Gradio {config['version']}, image+chatbot+textbox present")
        break
    except (urllib.error.URLError, ValueError, AssertionError, OSError) as exc:
        print(f"Live check {attempt}/6: {type(exc).__name__}: {exc}", flush=True)
        if attempt == 6:
            raise SystemExit("Live site not confirmed; check HF Space runtime status")
        time.sleep(8)
