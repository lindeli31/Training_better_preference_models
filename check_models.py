"""
check_models.py
---------------
List available models on the Swiss AI Stack and optionally select one.

Usage
-----
    python check_models.py
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.environ.get("SWISSAI_API_KEY", "")
BASE_URL = "https://api.swissai.cscs.ch/v1"


def list_models():
    resp = requests.get(
        f"{BASE_URL}/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=10,
    )
    resp.raise_for_status()
    models = resp.json()["data"]

    # Deduplicate by model id
    seen = set()
    unique = []
    for m in models:
        if m["id"] not in seen:
            seen.add(m["id"])
            unique.append(m)

    print(f"\nAvailable models ({len(unique)}):\n")
    for i, m in enumerate(unique, 1):
        print(f"  {i}. {m['id']}")

    print(f"\nUsage: python run_experiments.py --model <model_id>")
    print(f"       python run_opro.py --model <model_id>\n")

    return unique


if __name__ == "__main__":
    list_models()
