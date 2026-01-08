import os
import re
import time
import json
from typing import Any, Optional

import requests
from dotenv import load_dotenv
from datasets import load_dataset, DatasetDict
from tqdm import tqdm

# ---------------------------
# Config
# ---------------------------
load_dotenv()

SOURCE_DATASET = "eamasya19/school_of_reward_hacks_with_control_coding_tasks"
TARGET_REPO_ID = os.getenv(
    "TARGET_REPO_ID",
    "eamasya19/school_of_reward_hacks_with_control_coding_tasks_control_filled"
)

MODEL = "openai/gpt-4o"  # OpenRouter model id
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")

OPENROUTER_HTTP_REFERER = os.getenv("OPENROUTER_HTTP_REFERER", "http://localhost")
OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "sorh-control-filler")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

TEMPERATURE = 0.0
MAX_TOKENS = 1000
REQUEST_TIMEOUT_S = 90
MAX_RETRIES = 2

if not OPENROUTER_API_KEY:
    raise RuntimeError("Missing OPENROUTER_API_KEY in environment/.env")

if not HF_TOKEN:
    raise RuntimeError("Missing HF_TOKEN in environment/.env")

# ---------------------------
# Helpers
# ---------------------------
def is_missing_control(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str) and x.strip().lower() in {"", "null", "none"}:
        return True
    return False


def openrouter_chat(prompt_user: str) -> str:
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": OPENROUTER_HTTP_REFERER,
        "X-Title": OPENROUTER_APP_TITLE,
    }

    system_msg = (
        "You are a careful coding assistant.\n"
        "Return ONLY the final code solution.\n"
        "Do NOT hard-code specific unit test inputs/outputs; write a general solution.\n"
        "Do NOT include explanations, commentary, or additional text.\n"
    )

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt_user},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }

    last_err: Optional[str] = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(
                OPENROUTER_URL,
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT_S,
            )

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                sleep_s = float(retry_after) if retry_after else min(60.0, 2.0 ** attempt)
                time.sleep(sleep_s)
                continue

            if 500 <= resp.status_code <= 599:
                time.sleep(min(60.0, 2.0 ** attempt))
                continue

            resp.raise_for_status()
            data = resp.json()

            if isinstance(data, dict) and data.get("error"):
                last_err = json.dumps(data["error"])
                time.sleep(min(60.0, 2.0 ** attempt))
                continue

            return data["choices"][0]["message"]["content"]

        except (requests.Timeout, requests.ConnectionError) as e:
            last_err = repr(e)
            time.sleep(min(60.0, 2.0 ** attempt))
        except Exception as e:
            last_err = repr(e)
            time.sleep(min(30.0, 2.0 ** attempt))

    raise RuntimeError(f"OpenRouter call failed after {MAX_RETRIES} retries. Last error: {last_err}")


def find_tail_start(ds) -> int:
    """
    Assumes missing controls happen only at the end.
    Returns tail_start such that indices >= tail_start are missing control.
    """
    n = len(ds)
    for i in range(n - 1, -1, -1):
        if not is_missing_control(ds[i].get("control")):
            return i + 1
    return 0


# ---------------------------
# Main
# ---------------------------
def main():
    ds = load_dataset(SOURCE_DATASET)["train"]
    n = len(ds)

    # Preserve existing values; we'll only overwrite missing ones in the tail.
    controls = list(ds["control"])

    tail_start = find_tail_start(ds)
    print(f"Dataset size: {n}")
    print(f"Tail start: {tail_start} (tail length: {n - tail_start})")

    filled = 0
    for i in tqdm(range(tail_start, n), desc="Generating control (tail only)"):
        ex = ds[i]
        if not is_missing_control(ex.get("control")):
            continue

        user_prompt = ex.get("user", "")
        if not user_prompt:
            controls[i] = ""
            filled += 1
            continue

        controls[i] = openrouter_chat(user_prompt)
        filled += 1

    ds2 = ds.remove_columns(["control"])
    ds2 = ds2.add_column("control", controls)

    print(f"Filled {filled} missing control entries (tail only).")

    DatasetDict({"train": ds2}).push_to_hub(
        TARGET_REPO_ID,
        token=HF_TOKEN,
        commit_message="Fill missing control solutions for tail coding tasks via OpenRouter",
    )
    print(f"Pushed to Hub: {TARGET_REPO_ID}")


if __name__ == "__main__":
    main()