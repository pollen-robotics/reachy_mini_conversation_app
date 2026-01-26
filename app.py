import os
import io
import csv
import time
import threading
import requests
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import gradio as gr
from huggingface_hub import HfApi

# -------------------------
# Config
# -------------------------
DATASET_REPO_ID = os.environ.get("DATASET_REPO_ID") # Space secret
HF_TOKEN = os.environ["HF_TOKEN"]          # Space secret
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]  # Space secret

SENT_PATH = "sent.csv"

# Maximum keys per order = quantity * KEYS_PER_QUANTITY
KEYS_PER_QUANTITY = 10

# Reuse ephemeral keys if created within this time window (in seconds)
KEY_REUSE_WINDOW_SECONDS = 40 * 60  # 40 minutes


api = HfApi(token=HF_TOKEN)
_lock = threading.Lock()

# Simple cache to reduce Hub reads
_cache = {
    "ts": 0.0,
    "sent": None,  # type: ignore
}
CACHE_TTL_SEC = 10.0

# -------------------------
# Helpers
# -------------------------
def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()

def _download_csv(repo_id: str, path_in_repo: str) -> str:
    # Download file bytes from the dataset repo, return text
    file_bytes = api.hf_hub_download(
        repo_id=repo_id,
        filename=path_in_repo,
        repo_type="dataset",
        token=HF_TOKEN,
    )
    with open(file_bytes, "r", encoding="utf-8") as f:
        return f.read()

def _upload_csv(repo_id: str, path_in_repo: str, content: str, commit_message: str) -> None:
    api.upload_file(
        path_or_fileobj=io.BytesIO(content.encode("utf-8")),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        commit_message=commit_message,
        token=HF_TOKEN,
    )

def _parse_sent(csv_text: str) -> dict:
    """Parse sent.csv and return a dict keyed by order_number"""
    sent = {}
    reader = csv.DictReader(io.StringIO(csv_text))
    for r in reader:
        order_num = r["order_number"].strip()
        if order_num:
            keys_sent_str = r.get("keys_sent", "").strip()
            quantity_keys_sent_str = r.get("quantity_keys_sent", "").strip()
            last_key_sent_str = r.get("last_key_sent", "").strip()
            last_key_created_at_str = r.get("last_key_created_at", "").strip()

            sent[order_num] = {
                "order_number": order_num,
                "quantity": int(r.get("quantity", "0")),
                "keys_sent": keys_sent_str if keys_sent_str else "",
                "quantity_keys_sent": int(quantity_keys_sent_str) if quantity_keys_sent_str else 0,
                "last_key_sent": last_key_sent_str if last_key_sent_str else "",
                "last_key_created_at": last_key_created_at_str if last_key_created_at_str else "",
            }
    return sent

def _serialize_sent(sent_dict: dict) -> str:
    """Serialize sent dict back to CSV format"""
    out = io.StringIO()
    fieldnames = ["order_number", "quantity", "keys_sent", "quantity_keys_sent", "last_key_sent", "last_key_created_at"]
    w = csv.DictWriter(out, fieldnames=fieldnames)
    w.writeheader()
    for order_num in sorted(sent_dict.keys()):
        entry = sent_dict[order_num]
        w.writerow({
            "order_number": entry["order_number"],
            "quantity": entry["quantity"],
            "keys_sent": entry.get("keys_sent", ""),
            "quantity_keys_sent": entry.get("quantity_keys_sent", 0),
            "last_key_sent": entry.get("last_key_sent", ""),
            "last_key_created_at": entry.get("last_key_created_at", ""),
        })
    return out.getvalue()

def load_state(force: bool = False):
    now = time.time()
    if (not force) and _cache["sent"] is not None and (now - _cache["ts"] < CACHE_TTL_SEC):
        return _cache["sent"]

    sent_csv = _download_csv(DATASET_REPO_ID, SENT_PATH)
    sent = _parse_sent(sent_csv)

    _cache["ts"] = now
    _cache["sent"] = sent
    return sent

# -------------------------
# Core API
# -------------------------
def claim_c_key(
    order_number: str,
    request: Optional[gr.Request] = None,
) -> Tuple[str, str]:
    """
    Returns a key for a given order number if the order is valid
    and hasn't exceeded the allowed key limit (quantity * KEYS_PER_QUANTITY).
    Returns (key, status_message)
    """
    _ = request

    order_number = order_number.strip()

    if not order_number:
        return "", "Please provide an order number."

    with _lock:
        sent = load_state(force=True)

        # Check if order exists in sent.csv
        if order_number not in sent:
            return "", f"Order number {order_number} not found."

        quantity = sent[order_number]["quantity"]

        # Check how many keys have been sent for this order
        keys_already_sent = sent[order_number]["quantity_keys_sent"]

        # Calculate maximum allowed keys
        max_keys = quantity * KEYS_PER_QUANTITY

        # Check if limit has been reached
        if keys_already_sent >= max_keys:
            return "", f"Key limit reached for order {order_number} ({keys_already_sent}/{max_keys} keys sent)."

        # Check if we can reuse a recently created key
        last_key_sent = sent[order_number].get("last_key_sent", "")
        last_key_created_at = sent[order_number].get("last_key_created_at", "")

        if last_key_sent and last_key_created_at:
            try:
                # Parse the last creation timestamp
                last_created_time = datetime.fromisoformat(last_key_created_at)
                now = datetime.now(timezone.utc)
                time_since_creation = (now - last_created_time).total_seconds()

                # If the last key was created within the reuse window, reuse it
                if time_since_creation < KEY_REUSE_WINDOW_SECONDS:
                    return last_key_sent, f"Reused recent key. ({keys_already_sent}/{max_keys} keys sent for this order)"
            except (ValueError, TypeError):
                # If there's an error parsing the timestamp, create a new key
                pass

        # Create a new ephemeral key
        try:
            response = requests.post(
                "https://api.openai.com/v1/realtime/client_secrets",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "expires_after": {
                        "anchor": "created_at",
                        "seconds": 3600  # 1 hour
                    },
                    "session": {
                        "type": "realtime",
                        "model": "gpt-realtime",
                        "audio": {
                            "output": {"voice": "alloy"}
                        }
                    }
                }
            )
            response.raise_for_status()
            ephemeral_data = response.json()
            ephemeral_key = ephemeral_data["value"]
        except Exception as e:
            return "", f"Failed to create ephemeral key: {str(e)}"

        # Update sent.csv - store the ephemeral key and timestamp
        existing_keys = sent[order_number]["keys_sent"]
        if existing_keys:
            sent[order_number]["keys_sent"] = existing_keys + "," + ephemeral_key
        else:
            sent[order_number]["keys_sent"] = ephemeral_key
        sent[order_number]["quantity_keys_sent"] += 1
        sent[order_number]["last_key_sent"] = ephemeral_key
        sent[order_number]["last_key_created_at"] = _utc_now_iso()

        # Save changes
        updated_sent_csv = _serialize_sent(sent)

        _upload_csv(
            DATASET_REPO_ID,
            SENT_PATH,
            updated_sent_csv,
            commit_message=f"Updated sent tracking at {_utc_now_iso()}",
        )

        # refresh cache immediately
        _cache["ts"] = 0.0

        keys_sent_count = sent[order_number]["quantity_keys_sent"]
        return ephemeral_key, f"New ephemeral key sent successfully. ({keys_sent_count}/{max_keys} keys sent for this order)"

# -------------------------
# UI
# -------------------------
with gr.Blocks(title="API") as demo:
    gr.Markdown("## Ephemeral Key Service")

    order_input = gr.Textbox(label="Order Number", placeholder="Enter order number (e.g., 2724-1857)")
    btn_c = gr.Button("Request Key", variant="primary", size="sm")
    out_c = gr.Textbox(label="Response", interactive=False, show_label=False)
    status_c = gr.Textbox(label="", interactive=False, show_label=False)

    btn_c.click(
        fn=claim_c_key,
        inputs=[order_input],
        outputs=[out_c, status_c],
        api_name="claim_c_key",
    )

demo.queue()
demo.launch()
