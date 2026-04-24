import json
import math
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional


# Daatset final version convert (logic majority voting)
def _norm(text: str) -> str:
    """Normalize span text for comparison (strip whitespace, lowercase)."""
    return " ".join(text.strip().split())


def _extract_pairs(annotation: Dict) -> List[Dict]:
    """
    Parse one annotator's result into a list of pair dicts:
      {term_text, term_start, term_end, def_text, def_start, def_end}
    Returns only complete TERM->DEFINITION pairs linked by a relation.
    """
    results = annotation.get("result", [])
    by_id = {r["id"]: r for r in results if "id" in r}

    pairs = []
    for r in results:
        if r.get("type") != "relation":
            continue
        term_res = by_id.get(r["from_id"])
        def_res = by_id.get(r["to_id"])
        if not term_res or not def_res:
            continue
        t_labels = term_res["value"].get("labels", [])
        d_labels = def_res["value"].get("labels", [])
        if "TERM" not in t_labels or "DEFINITION" not in d_labels:
            continue
        pairs.append({
            "term_text":  term_res["value"]["text"],
            "term_start": term_res["value"]["start"],
            "term_end":   term_res["value"]["end"],
            "def_text":   def_res["value"]["text"],
            "def_start":  def_res["value"]["start"],
            "def_end":    def_res["value"]["end"],
        })
    return pairs


def majority_vote(item: Dict) -> Tuple[List[Dict], Dict]:
    """
    Run majority voting on all annotations for one document.

    Returns:
      (winning_pairs, stats)
      winning_pairs: list of canonical pair dicts (term_text, term_start, …)
      stats: vote counts per term for inspection
    """
    annotations = [
        a for a in item.get("annotations", [])
        if not a.get("was_cancelled", False)
    ]
    n_annotators = len(annotations)
    if n_annotators == 0:
        return [], {}

    threshold = math.ceil(n_annotators / 2)  # strict majority

    # ---- Step 1: collect pairs per annotator, keyed by normalised term ----
    # term_votes[norm_term] -> list of pair dicts (one per annotator who labeled it)
    term_votes: Dict[str, List[Dict]] = defaultdict(list)

    for ann in annotations:
        for pair in _extract_pairs(ann):
            key = _norm(pair["term_text"])
            term_votes[key].append(pair)

    # ---- Step 2: filter terms that reached majority ----
    stats = {}
    winning_pairs = []

    for norm_term, pairs in term_votes.items():
        vote_count = len(pairs)
        stats[norm_term] = {"votes": vote_count, "threshold": threshold}

        if vote_count < threshold:
            stats[norm_term]["accepted"] = False
            continue
        stats[norm_term]["accepted"] = True

        # ---- Step 3: pick canonical term span (most common exact text) ----
        term_text_counter = Counter(p["term_text"].strip() for p in pairs)
        canonical_term_text = term_text_counter.most_common(1)[0][0]

        # Among pairs that share this exact canonical term text, find the
        # most common (start, end) offset pair
        term_offset_counter = Counter(
            (p["term_start"], p["term_end"])
            for p in pairs
            if p["term_text"].strip() == canonical_term_text
        )
        canonical_term_start, canonical_term_end = term_offset_counter.most_common(1)[0][0]

        # ---- Step 4: pick canonical definition (most common normalized def) ----
        norm_def_counter = Counter(_norm(p["def_text"]) for p in pairs)
        winning_norm_def = norm_def_counter.most_common(1)[0][0]

        # Find the most common exact definition text matching that normalized form
        def_text_counter = Counter(
            p["def_text"].strip()
            for p in pairs
            if _norm(p["def_text"]) == winning_norm_def
        )
        canonical_def_text = def_text_counter.most_common(1)[0][0]

        # Most common offset for that definition
        def_offset_counter = Counter(
            (p["def_start"], p["def_end"])
            for p in pairs
            if p["def_text"].strip() == canonical_def_text
        )
        canonical_def_start, canonical_def_end = def_offset_counter.most_common(1)[0][0]

        winning_pairs.append({
            "term_text":  canonical_term_text,
            "term_start": canonical_term_start,
            "term_end":   canonical_term_end,
            "def_text":   canonical_def_text,
            "def_start":  canonical_def_start,
            "def_end":    canonical_def_end,
            "term_votes": vote_count,
            "def_votes":  norm_def_counter.most_common(1)[0][1],
        })

    return winning_pairs, stats


def build_ls_annotation(item: Dict, pairs: List[Dict]) -> Dict:
    """Build a single merged Label Studio annotation from winning pairs."""
    results = []
    for pair in pairs:
        term_id = uuid.uuid4().hex[:10]
        def_id  = uuid.uuid4().hex[:10]

        results.append({
            "value": {
                "start":  pair["term_start"],
                "end":    pair["term_end"],
                "text":   pair["term_text"],
                "labels": ["TERM"],
            },
            "id": term_id,
            "from_name": "label",
            "to_name":   "text",
            "type":      "labels",
            "origin":    "manual",
        })
        results.append({
            "value": {
                "start":  pair["def_start"],
                "end":    pair["def_end"],
                "text":   pair["def_text"],
                "labels": ["DEFINITION"],
            },
            "id": def_id,
            "from_name": "label",
            "to_name":   "text",
            "type":      "labels",
            "origin":    "manual",
        })
        results.append({
            "from_id":  term_id,
            "to_id":    def_id,
            "type":     "relation",
            "direction": "right",
        })

    # Preserve original item structure; replace annotations with single merged one
    out = {k: v for k, v in item.items() if k != "annotations"}
    out["annotations"] = [{"result": results, "was_cancelled": False, "ground_truth": True}]
    return out



def run(input_path: str, output_path: str, stats_path: Optional[str] = None):
    data = json.loads(Path(input_path).read_text(encoding="utf-8"))

    merged_data = []
    all_stats = []
    total_pairs = 0

    for item in data:
        n_ann = len([a for a in item.get("annotations", []) if not a.get("was_cancelled")])
        winning_pairs, stats = majority_vote(item)
        total_pairs += len(winning_pairs)

        merged_item = build_ls_annotation(item, winning_pairs)
        merged_data.append(merged_item)

        all_stats.append({
            "task_id": item.get("id"),
            "n_annotators": n_ann,
            "n_winning_pairs": len(winning_pairs),
            "term_votes": stats,
        })

    Path(output_path).write_text(
        json.dumps(merged_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[+] Saved {len(merged_data)} documents ({total_pairs} total pairs) -> {output_path}")

    if stats_path:
        Path(stats_path).write_text(
            json.dumps(all_stats, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[+] Saved voting stats -> {stats_path}")

    # Summary
    accepted = sum(
        sum(1 for v in s["term_votes"].values() if v["accepted"])
        for s in all_stats
    )
    rejected = sum(
        sum(1 for v in s["term_votes"].values() if not v["accepted"])
        for s in all_stats
    )
    print(f"    Terms accepted: {accepted}  |  Terms rejected (below majority): {rejected}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="mlonlysum.json")
    p.add_argument("--output", default="mlonlysum_majority.json")
    p.add_argument("--stats",  default="mlonlysum_vote_stats.json",
                   help="Optional per-document vote stats")
    args = p.parse_args()

    run(args.input, args.output, args.stats)
