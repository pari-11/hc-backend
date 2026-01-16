import json
import os
from config import HISTORY_FILE

def load_history():
    """Load scan history from JSON"""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except json.JSONDecodeError:
            return {}
    return {}

def save_to_history(filename: str, result_data: dict):
    """Save scan result to history"""
    history = load_history()
    history[filename] = result_data
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def check_cache(filename: str):
    """Check if file was already scanned"""
    history = load_history()
    return history.get(filename)
