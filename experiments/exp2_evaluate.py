"""
Evaluation for Experiment 1 — Definition Span Extraction.

Metrics per example
-------------------
exact_match        : 1 if normalised prediction == normalised gold, else 0
token_f1           : SQuAD-style token overlap F1
char_f1            : character-level overlap F1
is_span_valid      : prediction is a verbatim substring of the source document
is_hallucinated    : model returned a non-null span that is NOT in the document
                     (paraphrased / generated rather than extracted)
no_answer          : model returned null / empty when gold exists

Aggregate metrics
-----------------
exact_match        : mean over examples
token_f1           : mean
char_f1            : mean
span_validity_rate : % of predictions that are verbatim substrings of the doc
hallucination_rate : % of predictions that are NOT substrings of the doc
                     (excludes null predictions — those are counted separately)
null_rate          : % of examples where model returned null/empty
per_language       : all metrics broken down by 'en' / 'tr'
"""

from collections import Counter, defaultdict
from typing import List, Dict, Optional


# ---------------------------------------------------------------------------
# Text normalisation
# ---------------------------------------------------------------------------

def _norm(text: str) -> str:
    return " ".join(text.lower().split())


# ---------------------------------------------------------------------------
# Token F1  (SQuAD style)
# ---------------------------------------------------------------------------

def _token_f1(pred: str, gold: str) -> float:
    p_toks = Counter(_norm(pred).split())
    g_toks = Counter(_norm(gold).split())
    common = sum((p_toks & g_toks).values())
    if common == 0:
        return 0.0
    precision = common / sum(p_toks.values())
    recall    = common / sum(g_toks.values())
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Character F1
# ---------------------------------------------------------------------------

def _char_f1(pred: str, gold: str) -> float:
    p = _norm(pred)
    g = _norm(gold)
    p_chars = Counter(p)
    g_chars = Counter(g)
    common = sum((p_chars & g_chars).values())
    if common == 0:
        return 0.0
    precision = common / len(p) if p else 0.0
    recall    = common / len(g) if g else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
# Per-example evaluation
# ---------------------------------------------------------------------------

def evaluate_example(
    pred_def: Optional[str],   # raw model output (None / "" = no answer)
    gold_def: str,             # ground-truth definition span
    source_text: str,          # original document (for span validity check)
    lang: str = "unknown",
) -> Dict:
    """
    Evaluate a single (prediction, gold) pair.

    pred_def  : the definition string the model produced, or None/""
    gold_def  : the gold definition span
    source_text: the full source document the model was given
    lang      : language tag ('en' or 'tr') — carried through for aggregation
    """
    pred_empty = pred_def is None or str(pred_def).strip() == ""

    if pred_empty:
        return {
            "lang":             lang,
            "exact_match":      0.0,
            "token_f1":         0.0,
            "char_f1":          0.0,
            "is_span_valid":    False,   # no prediction → not a valid span
            "is_hallucinated":  False,   # null is not a hallucination
            "no_answer":        True,
        }

    pred_str = str(pred_def).strip()

    exact  = float(_norm(pred_str) == _norm(gold_def))
    tf1    = _token_f1(pred_str, gold_def)
    cf1    = _char_f1(pred_str, gold_def)

    # Span validity: prediction must appear verbatim in the source document.
    # We check case-insensitively to be lenient with leading/trailing whitespace
    # differences introduced by the model.
    is_valid = pred_str in source_text or _norm(pred_str) in _norm(source_text)

    # Hallucination: model returned something but it is NOT a verbatim span
    is_hallucinated = not is_valid

    return {
        "lang":             lang,
        "exact_match":      exact,
        "token_f1":         tf1,
        "char_f1":          cf1,
        "is_span_valid":    is_valid,
        "is_hallucinated":  is_hallucinated,
        "no_answer":        False,
    }


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate_exp1(example_scores: List[Dict]) -> Dict:
    """
    Aggregate per-example scores into overall and per-language summaries.
    """
    def _agg(scores: List[Dict]) -> Dict:
        n = len(scores)
        if n == 0:
            return {}
        return {
            "n_examples":          n,
            "exact_match":         round(sum(s["exact_match"]     for s in scores) / n, 4),
            "token_f1":            round(sum(s["token_f1"]        for s in scores) / n, 4),
            "char_f1":             round(sum(s["char_f1"]         for s in scores) / n, 4),
            "span_validity_rate":  round(sum(s["is_span_valid"]   for s in scores) / n, 4),
            "hallucination_rate":  round(
                # among non-null predictions only
                sum(s["is_hallucinated"] for s in scores if not s["no_answer"])
                / max(1, sum(1 for s in scores if not s["no_answer"])),
                4,
            ),
            "null_rate":           round(sum(s["no_answer"]       for s in scores) / n, 4),
        }

    # Overall
    result = {"overall": _agg(example_scores)}

    # Per language
    by_lang: Dict[str, List[Dict]] = defaultdict(list)
    for s in example_scores:
        by_lang[s["lang"]].append(s)

    result["per_language"] = {lang: _agg(slist) for lang, slist in sorted(by_lang.items())}

    return result


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_exp1_results(agg: Dict):
    def _block(label: str, d: Dict):
        print(f"\n  [{label}]  n={d.get('n_examples', 0)}")
        for k, v in d.items():
            if k == "n_examples":
                continue
            bar = ""
            if isinstance(v, float):
                bar = " " + "#" * int(v * 20)
            print(f"    {k:<25} {v:.4f}{bar}")

    print("\n" + "=" * 60)
    print("  EXPERIMENT 1 — Definition Span Extraction")
    print("=" * 60)
    _block("OVERALL", agg["overall"])
    for lang, d in agg.get("per_language", {}).items():
        _block(lang.upper(), d)
    print()
