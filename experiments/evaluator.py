from collections import Counter
from typing import List, Tuple, Dict


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _token_f1(pred: str, gold: str) -> float:
    pred_tokens = _normalize(pred).split()
    gold_tokens = _normalize(gold).split()

    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)

    common = sum((pred_counts & gold_counts).values())
    if common == 0:
        return 0.0

    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def evaluate_document(
    pred_pairs: List[Tuple[str, str]],
    gold_pairs: List[Tuple[str, str]],
) -> Dict:
    """
    Evaluation at three levels:
      1. Term-level (ignore definitions)
      2. Definition-level (quality of definitions for matched terms)
      3. Pair-level (term + definition together)
    """

    # -------------------------
    # TERM-LEVEL METRICS
    # -------------------------
    pred_terms = {_normalize(t) for t, _ in pred_pairs}
    gold_terms = {_normalize(t) for t, _ in gold_pairs}

    term_tp = len(pred_terms & gold_terms)
    term_precision = term_tp / len(pred_terms) if pred_terms else 0.0
    term_recall = term_tp / len(gold_terms) if gold_terms else 0.0
    term_f1 = (
        2 * term_precision * term_recall / (term_precision + term_recall)
        if (term_precision + term_recall) > 0 else 0.0
    )

    # -------------------------
    # PAIR MATCHING (same as before)
    # -------------------------
    matched_pred = set()
    matched_gold = set()
    def_f1_scores = []
    pair_exact_hits = 0

    for gi, (g_term, g_def) in enumerate(gold_pairs):
        best_f1, best_pi = -1.0, -1

        for pi, (p_term, p_def) in enumerate(pred_pairs):
            if pi in matched_pred:
                continue
            if _normalize(p_term) == _normalize(g_term):
                f1 = _token_f1(p_def, g_def)
                if f1 > best_f1:
                    best_f1, best_pi = f1, pi

        if best_pi >= 0:
            matched_gold.add(gi)
            matched_pred.add(best_pi)

            def_f1_scores.append(best_f1)

            _, best_p_def = pred_pairs[best_pi]
            if _normalize(best_p_def) == _normalize(g_def):
                pair_exact_hits += 1

    tp = len(matched_gold)

    pair_precision = tp / len(pred_pairs) if pred_pairs else 0.0
    pair_recall = tp / len(gold_pairs) if gold_pairs else 0.0
    pair_f1 = (
        2 * pair_precision * pair_recall / (pair_precision + pair_recall)
        if (pair_precision + pair_recall) > 0 else 0.0
    )

    # -------------------------
    # DEFINITION-LEVEL METRICS
    # -------------------------
    avg_def_f1_matched = (
        sum(def_f1_scores) / len(def_f1_scores) if def_f1_scores else 0.0
    )

    # penalized version (missing terms count as 0)
    avg_def_f1_all = (
        sum(def_f1_scores) / len(gold_pairs) if gold_pairs else 0.0
    )

    def_exact_rate = tp / len(gold_pairs) if gold_pairs else 0.0

    # -------------------------
    # HALLUCINATION
    # -------------------------
    is_hallucinated = (len(gold_pairs) == 0) and (len(pred_pairs) > 0)

    return {
        # counts
        "n_gold": len(gold_pairs),
        "n_pred": len(pred_pairs),

        # term-level
        "term_tp": term_tp,
        "term_precision": term_precision,
        "term_recall": term_recall,
        "term_f1": term_f1,

        # pair-level
        "tp": tp,
        "pair_exact": pair_exact_hits,
        "pair_precision": pair_precision,
        "pair_recall": pair_recall,
        "pair_f1": pair_f1,

        # definition-level
        "avg_def_f1_matched": avg_def_f1_matched,
        "avg_def_f1_all": avg_def_f1_all,
        "def_exact_rate": def_exact_rate,

        # hallucination
        "is_hallucinated": is_hallucinated,
        "hallucinated_pairs": len(pred_pairs) if is_hallucinated else 0,
    }


def aggregate_scores(doc_scores: List[Dict]) -> Dict:
    """Micro-average across documents with term, pair, and hallucination metrics."""

    total_gold = sum(d["n_gold"] for d in doc_scores)
    total_pred = sum(d["n_pred"] for d in doc_scores)
    n_docs = len(doc_scores)

    # -------------------------
    # TERM-LEVEL (micro)
    # -------------------------
    term_tp = sum(d["term_tp"] for d in doc_scores)
    term_pred = sum(len({_normalize(t) for t, _ in []}) for _ in [])  # not needed globally

    # recompute from totals
    total_pred_terms = sum(
        len({_normalize(t) for t, _ in []}) for _ in []
    )  # optional if you want exact micro

    # simpler macro-average instead
    term_precision = sum(d["term_precision"] for d in doc_scores) / n_docs if n_docs else 0.0
    term_recall = sum(d["term_recall"] for d in doc_scores) / n_docs if n_docs else 0.0
    term_f1 = sum(d["term_f1"] for d in doc_scores) / n_docs if n_docs else 0.0

    # -------------------------
    # PAIR-LEVEL (micro)
    # -------------------------
    total_tp = sum(d["tp"] for d in doc_scores)

    pair_precision = total_tp / total_pred if total_pred else 0.0
    pair_recall = total_tp / total_gold if total_gold else 0.0
    pair_f1 = (
        2 * pair_precision * pair_recall / (pair_precision + pair_recall)
        if (pair_precision + pair_recall) > 0 else 0.0
    )

    # -------------------------
    # DEFINITION-LEVEL
    # -------------------------
    avg_def_f1_matched = sum(d["avg_def_f1_matched"] for d in doc_scores) / n_docs if n_docs else 0.0
    avg_def_f1_all = sum(d["avg_def_f1_all"] for d in doc_scores) / n_docs if n_docs else 0.0

    # -------------------------
    # HALLUCINATION
    # -------------------------
    empty_gold_docs = [d for d in doc_scores if d["n_gold"] == 0]
    n_empty_gold = len(empty_gold_docs)

    hallucinated_docs = [d for d in empty_gold_docs if d["is_hallucinated"]]
    n_hallucinated_docs = len(hallucinated_docs)

    total_hallucinated_pairs = sum(d["hallucinated_pairs"] for d in hallucinated_docs)

    hallucination_rate_doc = (
        n_hallucinated_docs / n_empty_gold if n_empty_gold else 0.0
    )

    hallucination_rate_pair = (
        total_hallucinated_pairs / total_pred if total_pred else 0.0
    )

    return {
        "n_docs": n_docs,
        "total_gold_pairs": total_gold,
        "total_pred_pairs": total_pred,

        # term-level (macro avg)
        "term_precision": round(term_precision, 4),
        "term_recall": round(term_recall, 4),
        "term_f1": round(term_f1, 4),

        # pair-level (micro)
        "pair_precision": round(pair_precision, 4),
        "pair_recall": round(pair_recall, 4),
        "pair_f1": round(pair_f1, 4),

        # definition-level
        "avg_def_f1_matched": round(avg_def_f1_matched, 4),
        "avg_def_f1_all": round(avg_def_f1_all, 4),

        # hallucination
        "n_empty_gold_docs": n_empty_gold,
        "n_hallucinated_docs": n_hallucinated_docs,
        "total_hallucinated_pairs": total_hallucinated_pairs,
        "hallucination_rate_doc": round(hallucination_rate_doc, 4),
        "hallucination_rate_pair": round(hallucination_rate_pair, 4),
    }