import base64, json, os
from flask import Flask, request, jsonify
import requests

app = Flask(__name__)
N8N_WEBHOOK = os.environ["N8N_WEBHOOK_URL"]
SHARED_SECRET = os.environ.get("FORWARD_SECRET","")

@app.post("/pubsub")
def pubsub():
    if SHARED_SECRET and request.headers.get("x-forward-secret") != SHARED_SECRET:
        return "Forbidden", 403
    env = request.get_json(silent=True) or {}
    msg = env.get("message", {})
    if "data" in msg:
        data = json.loads(base64.b64decode(msg["data"]).decode("utf-8"))
    else:
        data = env
    r = requests.post(N8N_WEBHOOK, json=data, timeout=10)
    r.raise_for_status()
    return jsonify({"ok": True})
