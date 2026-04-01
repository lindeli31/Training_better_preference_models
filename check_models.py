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


def list_models(base_url: str = BASE_URL, api_key: str = API_KEY):
    resp = requests.get(
        f"{base_url}/models",
        headers={"Authorization": f"Bearer {api_key}"},
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


def validate_model(model: str, base_url: str = BASE_URL, api_key: str = API_KEY) -> None:
    """Raise SystemExit if *model* is not in the current available model list."""
    try:
        resp = requests.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )
        resp.raise_for_status()
        available = [m["id"] for m in resp.json()["data"]]
    except Exception as e:
        print(f"WARNING: Could not fetch model list to validate ({e}). Proceeding anyway.")
        return
    if model not in available:
        print(f"\nERROR: Model '{model}' is not currently available.")
        print(f"Available models: {', '.join(sorted(set(available)))}")
        print(f"Update SWISSAI_MODEL in your .env file or pass --model <model_id>.\n")
        raise SystemExit(1)
    print(f"Using model: {model}")


if __name__ == "__main__":
    list_models()
