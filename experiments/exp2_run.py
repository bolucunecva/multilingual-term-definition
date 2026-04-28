"""
Experiment 1 — Definition Span Extraction
==========================================
Given a term and a source document, extract the EXACT definition span.

Each gold (term, definition) pair becomes one inference example.
The dataset is evaluated on EN and TR splits independently, and results
are reported both overall and per language.

Usage
-----
# Without guideline
python exp1_run.py \\
    --data  ../mlonlysum_majority.json \\
    --model /path/to/local/model \\
    --output results/exp1_no_guideline.json

# With guideline
python exp1_run.py \\
    --data      ../mlonlysum_majority.json \\
    --model     /path/to/local/model \\
    --guideline ../guidelines/guideline.txt \\
    --output    results/exp1_with_guideline.json

# Base / completion model (no chat template)
    add --no_chat_template
"""

import argparse
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from data_utils    import load_data, get_document_text, extract_gold_pairs, get_language
from experiments.exp2_prompts  import build_prompt
from inference     import Extractor
from experiments.exp2_evaluate import evaluate_example, aggregate_exp1, print_exp1_results


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def build_examples(data: List[Dict]) -> List[Dict]:
    """
    Flatten the dataset into one example per (term, gold_definition) pair.
    Each example carries: term, gold_def, source_text, lang, task_id.
    Only documents that have at least one gold pair are included.
    """
    examples = []
    for item in data:
        pairs = extract_gold_pairs(item)
        if not pairs:
            continue
        text = get_document_text(item)
        lang = get_language(item)
        for term, gold_def in pairs:
            examples.append({
                "task_id":  item.get("id"),
                "lang":     lang,
                "term":     term,
                "gold_def": gold_def,
                "text":     text,
            })
    return examples


# ---------------------------------------------------------------------------
# Output parser
# ---------------------------------------------------------------------------

_NULL_VALUES = {"null", "none", "n/a", ""}


def _is_null(val: str) -> bool:
    return val.strip().lower() in _NULL_VALUES


def parse_output(raw: str, term: str = "") -> Optional[str]:
    """
    Extract the definition string from the model's JSON array output.

    Expected model output:
        [{"term": "<term>", "definition": "<exact span>"}]   — found
        []                                                   — not found

    Returns None if:
      - The array is empty  (model says not defined)
      - The definition value is null / none / empty
      - Parsing fails entirely
    """
    # ── 1. Strip markdown code fences ────────────────────────────────────────
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # ── 2. Try full JSON parse ────────────────────────────────────────────────
    try:
        parsed = json.loads(raw)

        # Model returned an array (expected format)
        if isinstance(parsed, list):
            if not parsed:                          # [] → no definition found
                return None
            # Prefer the element whose "term" matches the input term
            norm_term = " ".join(term.lower().split())
            best = None
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                item_term = " ".join(str(item.get("term", "")).lower().split())
                if item_term == norm_term or best is None:
                    best = item
                    if item_term == norm_term:
                        break
            if best is None:
                return None
            val = str(best.get("definition", "")).strip()
            return None if _is_null(val) else val

        # Model returned a plain object {"definition": "..."} (old format fallback)
        if isinstance(parsed, dict):
            val = str(parsed.get("definition", "")).strip()
            return None if _is_null(val) else val

    except (json.JSONDecodeError, AttributeError):
        pass

    # ── 3. Regex fallback: "definition": "..." anywhere in the raw string ────
    m = re.search(r'"definition"\s*:\s*"((?:[^"\\]|\\.)*)"', raw, re.DOTALL)
    if m:
        val = m.group(1).strip()
        return None if _is_null(val) else val

    # ── 4. Last resort: plain-text span (no JSON structure at all) ───────────
    cleaned = raw.strip().strip('"').strip("'")
    if cleaned and not _is_null(cleaned):
        return cleaned

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",            required=True,  help="mlonlysum_majority.json")
    p.add_argument("--model",           required=True,  help="Local model path or HF name")
    p.add_argument("--output",          required=True,  help="Path to save per-example results JSON")
    p.add_argument("--guideline",       default=None,   help="Plain-text guideline file")
    p.add_argument("--eval_output",     default=None,   help="Path to save aggregate evaluation JSON")
    p.add_argument("--tp",              type=int, default=1, help="Tensor parallel size")
    p.add_argument("--no_chat_template", action="store_true")
    p.add_argument("--max_new_tokens",  type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()

    # Load guideline
    guideline: Optional[str] = None
    if args.guideline:
        guideline = Path(args.guideline).read_text(encoding="utf-8").strip()
        print(f"[+] Guideline loaded ({len(guideline)} chars)")

    # Build flat example list
    data     = load_data(args.data)
    examples = build_examples(data)
    print(f"[+] {len(examples)} examples from {len(data)} documents")

    lang_counts = {}
    for ex in examples:
        lang_counts[ex["lang"]] = lang_counts.get(ex["lang"], 0) + 1
    for lang, cnt in sorted(lang_counts.items()):
        print(f"    [{lang}]  {cnt} examples")

    # Load model
    extractor = Extractor(
        model_name        = args.model,
        tensor_parallel_size = args.tp,
        max_new_tokens    = args.max_new_tokens,
        use_chat_template = not args.no_chat_template,
    )

    # Build prompts
    prompts = [
        build_prompt(
            term      = ex["term"],
            text      = ex["text"],
            guideline = guideline,
            tokenizer = extractor.get_tokenizer(),
        )
        for ex in examples
    ]

    print(f"[+] Sample prompt (first 600 chars):\n{prompts[0][:600]}\n---")

    # Inference (batched)
    raw_outputs = extractor.generate(prompts)
    print(f"[+] Inference complete ({len(raw_outputs)} outputs)")

    # Evaluate
    example_scores = []
    output_records = []

    for ex, raw in zip(examples, raw_outputs):
        pred_def = parse_output(raw, term=ex["term"])
        score    = evaluate_example(
            pred_def    = pred_def,
            gold_def    = ex["gold_def"],
            source_text = ex["text"],
            lang        = ex["lang"],
        )
        example_scores.append(score)
        output_records.append({
            "task_id":      ex["task_id"],
            "lang":         ex["lang"],
            "term":         ex["term"],
            "gold_def":     ex["gold_def"],
            "pred_def":     pred_def,
            "raw_output":   raw,
            **score,
        })

    # Aggregate
    agg = aggregate_exp1(example_scores)
    print_exp1_results(agg)

    # Save per-example results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(output_records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[+] Per-example results -> {out_path}")

    # Save aggregate
    if args.eval_output:
        eval_path = Path(args.eval_output)
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(
            json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"[+] Aggregate evaluation -> {eval_path}")


if __name__ == "__main__":
    main()
