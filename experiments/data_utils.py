import json
import uuid
from difflib import SequenceMatcher
from typing import List, Tuple, Dict, Optional


def load_data(filepath: str) -> List[Dict]:
    with open(filepath, encoding="utf-8") as f:
        return json.load(f)


def get_document_text(item: Dict) -> str:
    return item.get("data", {}).get("text", "")


def get_language(item: Dict) -> str:
    """Return language code from metadata ('en', 'tr', or 'unknown')."""
    return item.get("data", {}).get("metadata", {}).get("language", "unknown")


def get_article_id(item: Dict) -> str:
    """Return the OAI article identifier (groups en/tr pair for same article)."""
    return item.get("data", {}).get("metadata", {}).get("OAI Identifier", "")


# ---------------------------------------------------------------------------
# Gold pair extraction  (mlonlysum_majority.json format)
#
# Each item has exactly ONE annotation (ground_truth=True) produced by
# majority_vote.py.  All relation IDs are present in the same result list,
# so standard relation-based extraction works reliably.
# ---------------------------------------------------------------------------

def extract_gold_pairs(item: Dict) -> List[Tuple[str, str]]:
    """
    Return list of (term_text, definition_text) from the majority-voted
    annotation.  Returns [] for documents with no surviving pairs.
    """
    results = item.get("annotations", [{}])[0].get("result", [])
    if not results:
        return []

    id_to_result = {r["id"]: r for r in results if "id" in r}
    pairs = []

    for r in results:
        if r.get("type") != "relation":
            continue
        term_res = id_to_result.get(r["from_id"])
        def_res  = id_to_result.get(r["to_id"])
        if not term_res or not def_res:
            continue
        t_labels = term_res["value"].get("labels", [])
        d_labels = def_res["value"].get("labels", [])
        if "TERM" in t_labels and "DEFINITION" in d_labels:
            pairs.append((
                term_res["value"]["text"],
                def_res["value"]["text"],
            ))

    return pairs


# ---------------------------------------------------------------------------
# Span lookup for LLM predictions -> Label Studio output
# ---------------------------------------------------------------------------

def _find_span(source: str, target: str) -> Tuple[int, int]:
    """Find target in source; fall back to best fuzzy match."""
    idx = source.find(target)
    if idx != -1:
        return idx, idx + len(target)

    best_ratio, best_start = 0.0, 0
    for i in range(max(1, len(source) - len(target) + 1)):
        window = source[i: i + len(target)]
        r = SequenceMatcher(None, window, target).ratio()
        if r > best_ratio:
            best_ratio, best_start = r, i

    if best_ratio >= 0.8:
        return best_start, best_start + len(target)
    return -1, -1


def pairs_to_label_studio(
    pairs: List[Tuple[str, str]],
    source_text: str,
    item_id,
) -> Dict:
    """Convert (term, definition) pairs to Label Studio prediction format."""
    results = []

    for term_text, def_text in pairs:
        t_start, t_end = _find_span(source_text, term_text)
        d_start, d_end = _find_span(source_text, def_text)

        if t_start == -1 or d_start == -1:
            continue

        term_id = uuid.uuid4().hex[:10]
        def_id  = uuid.uuid4().hex[:10]

        results.append({
            "value": {"start": t_start, "end": t_end, "text": term_text, "labels": ["TERM"]},
            "id": term_id,
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "origin": "prediction",
        })
        results.append({
            "value": {"start": d_start, "end": d_end, "text": def_text, "labels": ["DEFINITION"]},
            "id": def_id,
            "from_name": "label",
            "to_name": "text",
            "type": "labels",
            "origin": "prediction",
        })
        results.append({
            "from_id": term_id,
            "to_id":   def_id,
            "type":    "relation",
            "direction": "right",
        })

    return {"id": item_id, "predictions": [{"result": results}]}
