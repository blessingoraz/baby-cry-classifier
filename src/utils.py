# src/utils.py
from typing import Dict, List, Tuple

# label_map: list or dict mapping indices to labels, e.g. ["belly_pain","burping",...]
def format_prediction(probs: Dict[str, float], top_k: int = 3):
    # probs is already a dict {label: prob}
    items = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top = items[:top_k]
    return {
        "label": top[0][0],
        "probability": float(top[0][1]),
        "top_k": [{"label": l, "probability": float(p)} for l, p in top],
        "all_probs": {k: float(v) for k, v in probs.items()}
    }