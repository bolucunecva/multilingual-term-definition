import json
import argparse
import re
import random
import string
from collections import Counter
from pathlib import Path

from vllm import LLM, SamplingParams

DEFAULT_DATA_PATH   = r"../dataset/all_experiments.json"
DEFAULT_ARXIV_PATH  = r"../dataset/arxiv_experiments.json"
DEFAULT_OUTPUT_DIR  = r"../results"

# ── Prompts ───────────────────────────────────────────────────────────────────

_PAIRS_EXAMPLE = (
    '## Example Output\n'
    '[\n'
    '  {\n'
    '    "term": "Photosynthesis",\n'
    '    "definition": "The process by which green plants use sunlight to synthesize food from carbon dioxide and water."\n'
    '  }\n'
    ']'
)

_DEF_EXAMPLE = (
    '## Example Output\n'
    '{\n'
    '  "definition": "The process by which green plants use sunlight to synthesize food from carbon dioxide and water."\n'
    '}'
)

_BASE_EXP1 = (
    "You are an expert in machine learning. "
    "Extract all machine learning term-definition pairs from the given abstract. "
    "Return a JSON array where each element is an object with keys \"term\" and \"definition\". "
    "Do not include any text outside the JSON.\n\n"
    + _PAIRS_EXAMPLE
)

_BASE_EXP2 = (
    "You are an expert in machine learning. "
    "Given an abstract and a target term, extract the exact definition of the term "
    "as it appears in the abstract. "
    "Return a JSON object with key \"definition\" containing only the definition span. "
    "Do not include any text outside the JSON.\n\n"
    + _DEF_EXAMPLE
)

_BASE_EXP3 = (
    "You are an expert in machine learning. "
    "Extract all machine learning term-definition pairs from the given abstract. "
    "A parallel abstract in another language is provided as additional context. "
    "Return a JSON array where each element is an object with keys \"term\" and \"definition\". "
    "Do not include any text outside the JSON.\n\n"
    + _PAIRS_EXAMPLE
)

_BASE_EXP4 = (
    "You are an expert in machine learning. "
    "Given an abstract, a parallel abstract in another language, and a target term, "
    "extract the exact definition of the term as it appears in the primary abstract. "
    "Return a JSON object with key \"definition\" containing only the definition span. "
    "Do not include any text outside the JSON.\n\n"
    + _DEF_EXAMPLE
)

_BASE_EXP5 = (
    "You are an expert in machine learning. "
    "You are given one or more abstracts that are known to contain a definition of the target term. "
    "Extract the definition of the target term as it appears in the abstracts. "
    "Return a JSON object with key \"definition\" containing only the definition span. "
    "Do not include any text outside the JSON.\n\n"
    + _DEF_EXAMPLE
)

_BASE_EXP6 = (
    "You are an expert in machine learning. "
    "You are given one or more abstracts and a target term. "
    "Extract the definition of the target term only if it is explicitly defined in the abstracts. "
    "If no definition is present, return an empty definition. "
    "Return a JSON object with key \"definition\" containing the definition span or an empty string. "
    "Do not include any text outside the JSON.\n\n"
    '## Example Output (no definition found)\n'
    '{\n'
    '  "definition": ""\n'
    '}'
)


def build_system_prompts(guideline: str | None) -> dict[int, str]:
    """Return system prompts for each experiment, with guideline appended if provided."""
    bases = {
        1: _BASE_EXP1, 2: _BASE_EXP2, 3: _BASE_EXP3, 4: _BASE_EXP4,
        5: _BASE_EXP5, 6: _BASE_EXP6,
    }
    if not guideline:
        return bases
    suffix = f"\n\nGuidelines:\n{guideline}"
    return {k: v + suffix for k, v in bases.items()}


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import torch  # pyright: ignore[reportMissingImports]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def build_prompt_exp1(text: str) -> str:
    return f"Abstract:\n{text}\n\nExtract all term-definition pairs."


def build_prompt_exp2(text: str, term: str) -> str:
    return f"Abstract:\n{text}\n\nTarget term: {term}\n\nExtract the definition."


def build_prompt_exp3(text: str, other_text: str, other_lang: str,
                      other_pairs: list[dict]) -> str:
    # Format the gold pairs from the other language as annotated context
    if other_pairs:
        pair_lines = "\n".join(
            f"  - Term: {p['term']}\n    Definition: {p['definition']}"
            for p in other_pairs
        )
        annotated_section = (
            f"Annotated term-definition pairs in the {other_lang} abstract:\n"
            f"{pair_lines}\n\n"
        )
    else:
        annotated_section = ""

    return (
        f"Primary abstract:\n{text}\n\n"
        f"Parallel abstract ({other_lang}):\n{other_text}\n\n"
        f"{annotated_section}"
        f"Extract all term-definition pairs from the primary abstract."
    )


def build_prompt_exp4(text: str, other_text: str, other_lang: str, term: str) -> str:
    return (
        f"Primary abstract:\n{text}\n\n"
        f"Parallel abstract ({other_lang}):\n{other_text}\n\n"
        f"Target term: {term}\n\n"
        f"Extract the definition from the primary abstract."
    )


# ── Normalisation & token F1 ──────────────────────────────────────────────────

def normalise(text: str) -> str:
    try:
        text = text.lower().strip()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s+", " ", text)
    except:
        pass
    return text


def token_f1(pred: str, gold: str) -> float:
    pred_tokens  = normalise(pred).split()
    gold_tokens  = normalise(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_common = sum(common.values())
    if num_common == 0:
        return 0.0
    precision = num_common / len(pred_tokens)
    recall    = num_common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gold: str) -> bool:
    return normalise(pred) == normalise(gold)


# ── Pair-level evaluation (one-to-one matching) ───────────────────────────────

def count_pair_matches(pred_pairs: list[dict], gold_pairs: list[dict]) -> tuple[int, int, int]:
    """Return (correct, n_pred, n_gold) for one document under one-to-one matching."""
    gold_by_term = {normalise(gp["term"]): gp["definition"] for gp in gold_pairs}
    matched_gold = set()
    correct = 0
    for pp in pred_pairs:
        key = normalise(pp.get("term", ""))
        if key in gold_by_term and key not in matched_gold:
            if token_f1(pp.get("definition", ""), gold_by_term[key]) > 0:
                correct += 1
                matched_gold.add(key)
    return correct, len(pred_pairs), len(gold_pairs)


def dataset_prf(total_correct: int, total_pred: int, total_gold: int) -> dict:
    precision = total_correct / total_pred if total_pred else 0.0
    recall    = total_correct / total_gold if total_gold else 0.0
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1}


def micro_token_f1(all_pred_toks: list[list[str]],
                   all_gold_toks: list[list[str]]) -> float:
    """Micro-averaged token F1 over all (pred, gold) token-list pairs."""
    total_tp = total_pred = total_gold = 0
    for pred_toks, gold_toks in zip(all_pred_toks, all_gold_toks):
        common = sum((Counter(pred_toks) & Counter(gold_toks)).values())
        total_tp   += common
        total_pred += len(pred_toks)
        total_gold += len(gold_toks)
    if total_pred == 0 and total_gold == 0:
        return 1.0
    precision = total_tp / total_pred if total_pred else 0.0
    recall    = total_tp / total_gold if total_gold else 0.0
    return (2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0)


# ── Response parsing ──────────────────────────────────────────────────────────

def _clean_json_text(text: str) -> str:
    """Strip markdown fences, leading/trailing prose, and common artifacts."""
    # Remove ```json ... ``` or ``` ... ``` fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()
    return text


def _extract_json_block(text: str) -> str:
    """
    Try to isolate the first valid JSON block (array or object) from free text.
    Scans for the first '[' or '{' and finds its matching closing bracket.
    """
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        start = text.find(start_char)
        if start == -1:
            continue
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == start_char:
                depth += 1
            elif ch == end_char:
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
    return text


def _extract_pairs_from_obj(obj) -> list[dict]:
    """Normalise various JSON shapes into a list of {term, definition} dicts."""
    if isinstance(obj, list):
        return [
            p for p in obj
            if isinstance(p, dict) and "term" in p and "definition" in p
        ]
    if isinstance(obj, dict):
        # {"pairs": [...]}
        if "pairs" in obj and isinstance(obj["pairs"], list):
            return _extract_pairs_from_obj(obj["pairs"])
        # {"term": ..., "definition": ...}  (single pair wrapped in dict)
        if "term" in obj and "definition" in obj:
            return [obj]
    return []


def parse_pairs(text: str) -> list[dict]:
    """
    Parse model output into a list of {term, definition} dicts.
    Fallback chain:
      1. Clean fences → json.loads on full text
      2. Extract first JSON block → json.loads
      3. Regex scrape individual term/definition values
    """
    cleaned = _clean_json_text(text)

    # 1. Direct parse
    try:
        return _extract_pairs_from_obj(json.loads(cleaned))
    except Exception:
        pass

    # 2. Extract first JSON block then parse
    try:
        block = _extract_json_block(cleaned)
        return _extract_pairs_from_obj(json.loads(block))
    except Exception:
        pass

    # 3. Regex fallback: scrape "term": "...", "definition": "..." patterns
    pairs = []
    term_pat = re.findall(r'"term"\s*:\s*"([^"]+)"', cleaned)
    def_pat  = re.findall(r'"definition"\s*:\s*"([^"]+)"', cleaned)
    for term, definition in zip(term_pat, def_pat):
        pairs.append({"term": term.strip(), "definition": definition.strip()})
    return pairs


def parse_definition(text: str) -> str:
    """
    Parse model output into a definition string.
    Fallback chain:
      1. Clean fences → json.loads on full text
      2. Extract first JSON block → json.loads
      3. Regex scrape "definition" value
      4. Return cleaned raw text as last resort
    """
    cleaned = _clean_json_text(text)

    # 1. Direct parse
    try:
        obj = json.loads(cleaned)
        if isinstance(obj, dict) and "definition" in obj:
            return str(obj["definition"]).strip()
    except Exception:
        pass

    # 2. Extract first JSON block then parse
    try:
        block = _extract_json_block(cleaned)
        obj = json.loads(block)
        if isinstance(obj, dict) and "definition" in obj:
            return str(obj["definition"]).strip()
    except Exception:
        pass

    # 3. Regex fallback
    m = re.search(r'"definition"\s*:\s*"([^"]+)"', cleaned)
    if m:
        return m.group(1).strip()

    # 4. Parsing failed — return empty string so metrics are not polluted
    return ""


def build_chat_messages(system: str, user: str) -> list[dict]:
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]


def run_vllm(llm: LLM, sampling: SamplingParams,
             messages_batch: list[list[dict]]) -> list[str]:
    outputs = llm.chat(messages_batch, sampling_params=sampling)
    return [o.outputs[0].text for o in outputs]


# ── Shared helpers ────────────────────────────────────────────────────────────

def _collect_gold_pairs(d: dict) -> tuple[list[dict], list[str], list[str]]:
    term_map = {t["id"]: t["text"] for t in d["gold_annotations"]["terms"]}
    def_map  = {t["id"]: t["text"] for t in d["gold_annotations"]["definitions"]}
    gold_pairs = [
        {"term": term_map[r["from_id"]], "definition": def_map[r["to_id"]]}
        for r in d["gold_annotations"]["relations"]
        if r["from_id"] in term_map and r["to_id"] in def_map
    ]
    return gold_pairs, list(term_map.values()), list(def_map.values())


def _score_pairs(results: list[dict]) -> dict:
    """
    Dataset-level metrics for Exp1/3.

    Pair-level (one-to-one matching):
      TP = predicted pairs whose term exactly matches a gold term AND
           definition has token overlap > 0 with the aligned gold definition
      FP = predicted pairs − TP   (spurious predictions)
      FN = gold pairs − TP        (missed gold pairs)
      Precision = TP / (TP + FP)
      Recall    = TP / (TP + FN)
      F1        = 2·P·R / (P + R)

    Term / Definition F1:
      Micro token F1 — for each gold span find best-matching predicted span,
      pool all token counts dataset-wide, compute single P/R/F1.
    """
    term_pred_toks, term_gold_toks = [], []
    def_pred_toks,  def_gold_toks  = [], []

    # Pair-level raw counts
    pair_tp = pair_fp = pair_fn = 0

    for row in results:
        pred_pairs = row["pred_pairs"]
        gold_pairs = row["gold_pairs"]
        pred_terms = [normalise(p.get("term", "")).split() for p in pred_pairs]
        pred_defs  = [normalise(p.get("definition", "")).split() for p in pred_pairs]

        # ── Term / definition micro F1 ────────────────────────────────────────
        for gp in gold_pairs:
            gt_toks = normalise(gp["term"]).split()
            gd_toks = normalise(gp["definition"]).split()
            best_term_toks = max(
                pred_terms,
                key=lambda pt: sum((Counter(pt) & Counter(gt_toks)).values()),
                default=[],
            )
            best_def_toks = max(
                pred_defs,
                key=lambda pd: sum((Counter(pd) & Counter(gd_toks)).values()),
                default=[],
            )
            term_pred_toks.append(best_term_toks)
            term_gold_toks.append(gt_toks)
            def_pred_toks.append(best_def_toks)
            def_gold_toks.append(gd_toks)

        # ── Pair-level TP / FP / FN ───────────────────────────────────────────
        correct, n_pred, n_gold = count_pair_matches(pred_pairs, gold_pairs)
        pair_tp += correct
        pair_fp += n_pred - correct   # predicted but wrong
        pair_fn += n_gold - correct   # gold but missed

    pair_precision = pair_tp / (pair_tp + pair_fp) if (pair_tp + pair_fp) else 0.0
    pair_recall    = pair_tp / (pair_tp + pair_fn) if (pair_tp + pair_fn) else 0.0
    pair_f1        = (2 * pair_precision * pair_recall /
                      (pair_precision + pair_recall)
                      if (pair_precision + pair_recall) else 0.0)

    return {
        # Token-level F1 (micro)
        "term_f1":        micro_token_f1(term_pred_toks, term_gold_toks),
        "definition_f1":  micro_token_f1(def_pred_toks,  def_gold_toks),
        # Pair-level P / R / F1
        "pair_precision": pair_precision,
        "pair_recall":    pair_recall,
        "pair_f1":        pair_f1,
        # Raw counts
        "pair_tp":        pair_tp,
        "pair_fp":        pair_fp,
        "pair_fn":        pair_fn,
        "total_gold_pairs": pair_tp + pair_fn,
        "total_pred_pairs": pair_tp + pair_fp,
    }


def _flatten_instances(data: list[dict], extra_fields_fn=None) -> list[dict]:
    """One instance per (term, definition) relation. Skips incomplete relations."""
    instances = []
    for d in data:
        term_map = {t["id"]: t["text"] for t in d["gold_annotations"]["terms"]}
        def_map  = {t["id"]: t["text"] for t in d["gold_annotations"]["definitions"]}
        for rel in d["gold_annotations"]["relations"]:
            term     = term_map.get(rel["from_id"], "")
            gold_def = def_map.get(rel["to_id"], "")
            if not term or not gold_def:   # skip incomplete relations
                continue
            inst = {
                "id":       d["id"],
                "text":     d["text"],
                "term":     term,
                "gold_def": gold_def,
            }
            if extra_fields_fn:
                inst.update(extra_fields_fn(d))
            instances.append(inst)
    return instances


def _instance_token_f1(pred: str, gold: str) -> float:
    """Per-instance token F1 (same formula as SQuAD)."""
    pred_toks = normalise(pred).split()
    gold_toks = normalise(gold).split()
    if not pred_toks and not gold_toks:
        return 1.0
    if not pred_toks or not gold_toks:
        return 0.0
    common = sum((Counter(pred_toks) & Counter(gold_toks)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_toks)
    recall    = common / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def _score_definitions(results: list[dict]) -> dict:
    """
    Dataset-level metrics for Exp2/4/5.
    Token F1 is macro-averaged per instance (F1 >= EM always holds).

    TP = instances where token F1 > 0  (pred has some overlap with gold)
    FP = instances where pred is non-empty but token F1 = 0 (wrong extraction)
    FN = instances where pred is empty (parse failed / model abstained)
    TN = N/A (every instance has a gold definition to extract)

    Precision = TP / (TP + FP)
    Recall    = TP / (TP + FN)
    F1        = 2·P·R / (P + R)
    EM        = exact string matches / total instances
    """
    f1s       = []
    em_correct = 0
    tp = fp = fn = 0

    for row in results:
        pred = row["pred_def"]
        gold = row["gold_def"]
        inst_f1 = _instance_token_f1(pred, gold)
        f1s.append(inst_f1)

        if normalise(pred) == normalise(gold):
            em_correct += 1

        if pred.strip() == "":
            fn += 1          # abstained — missed gold
        elif inst_f1 > 0:
            tp += 1          # non-empty and overlaps gold
        else:
            fp += 1          # non-empty but no overlap with gold

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1_prf    = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    n = len(results)
    return {
        # Macro token F1 and EM
        "exact_match":   em_correct / n if n else 0.0,
        "token_f1":      sum(f1s) / n if n else 0.0,
        # P / R / F1 derived from TP/FP/FN
        "precision":     precision,
        "recall":        recall,
        "f1":            f1_prf,
        # Raw counts
        "tp":            tp,
        "fp":            fp,
        "fn":            fn,
        "num_instances": n,
        "em_correct":    em_correct,
        "parse_failed":  fn,   # alias: FN = abstained predictions
    }


# ── Experiment runners ────────────────────────────────────────────────────────

def run_exp1(data: list[dict], llm: LLM, sampling: SamplingParams,
             system: str) -> dict:
    # Only run on records that have at least one gold pair
    data = [d for d in data if d["gold_annotations"]["relations"]]

    messages = [
        build_chat_messages(system, build_prompt_exp1(d["text"]))
        for d in data
    ]
    raw_outputs = run_vllm(llm, sampling, messages)

    results = []
    for d, raw in zip(data, raw_outputs):
        gold_pairs, _, _ = _collect_gold_pairs(d)
        results.append({"id": d["id"], "raw_output": raw,
                        "pred_pairs": parse_pairs(raw), "gold_pairs": gold_pairs})

    return {"metrics": _score_pairs(results), "predictions": results}


def run_exp2(data: list[dict], llm: LLM, sampling: SamplingParams,
             system: str) -> dict:
    instances = _flatten_instances(data)
    messages  = [
        build_chat_messages(system, build_prompt_exp2(i["text"], i["term"]))
        for i in instances
    ]
    raw_outputs = run_vllm(llm, sampling, messages)

    results = []
    for inst, raw in zip(instances, raw_outputs):
        results.append({
            "id":       inst["id"],
            "term":     inst["term"],
            "pred_def": parse_definition(raw),
            "gold_def": inst["gold_def"],
        })

    return {"metrics": _score_definitions(results), "predictions": results}


def _collect_other_gold_pairs(d: dict) -> list[dict]:
    """Extract gold term-definition pairs from the other language context."""
    other = d.get("other_language_context_with_gold") or {}
    ann   = other.get("gold_annotations") or {}
    term_map = {t["id"]: t["text"] for t in ann.get("terms", [])}
    def_map  = {t["id"]: t["text"] for t in ann.get("definitions", [])}
    return [
        {"term": term_map[r["from_id"]], "definition": def_map[r["to_id"]]}
        for r in ann.get("relations", [])
        if r["from_id"] in term_map and r["to_id"] in def_map
    ]


def run_exp3(data: list[dict], llm: LLM, sampling: SamplingParams,
             system: str) -> dict:
    # Only records with gold pairs AND a valid other-language context
    data = [
        d for d in data
        if d["gold_annotations"]["relations"]
        and d.get("other_language_context_with_gold")
    ]

    messages = [
        build_chat_messages(
            system,
            build_prompt_exp3(
                d["text"],
                d["other_language_context_with_gold"]["text"],
                d["other_language_context_with_gold"]["language"],
                _collect_other_gold_pairs(d),
            )
        )
        for d in data
    ]
    raw_outputs = run_vllm(llm, sampling, messages)

    results = []
    for d, raw in zip(data, raw_outputs):
        gold_pairs, _, _ = _collect_gold_pairs(d)
        results.append({"id": d["id"], "raw_output": raw,
                        "pred_pairs": parse_pairs(raw), "gold_pairs": gold_pairs})

    return {"metrics": _score_pairs(results), "predictions": results}


def run_exp4(data: list[dict], llm: LLM, sampling: SamplingParams,
             system: str) -> dict:
    # Only records with a valid other-language context
    data = [d for d in data if d.get("other_language_context_with_gold")]

    def _other(d):
        o = d["other_language_context_with_gold"]
        return {"other_text": o["text"], "other_lang": o["language"]}

    instances = _flatten_instances(data, extra_fields_fn=_other)
    messages  = [
        build_chat_messages(
            system,
            build_prompt_exp4(i["text"], i["other_text"], i["other_lang"], i["term"])
        )
        for i in instances
    ]
    raw_outputs = run_vllm(llm, sampling, messages)

    results = []
    for inst, raw in zip(instances, raw_outputs):
        results.append({
            "id":       inst["id"],
            "term":     inst["term"],
            "pred_def": parse_definition(raw),
            "gold_def": inst["gold_def"],
        })

    return {"metrics": _score_definitions(results), "predictions": results}


# ── Exp5/6 helpers ───────────────────────────────────────────────────────────

def build_prompt_exp5(term: str, abstracts: list[str]) -> str:
    abs_block = "\n\n".join(
        f"Abstract {i+1}:\n{text}" for i, text in enumerate(abstracts)
    )
    return (
        f"Target term: {term}\n\n"
        f"{abs_block}\n\n"
        f"Extract the definition of the target term."
    )


def build_prompt_exp6(term: str, abstracts: list[str]) -> str:
    abs_block = "\n\n".join(
        f"Abstract {i+1}:\n{text}" for i, text in enumerate(abstracts)
    )
    return (
        f"Target term: {term}\n\n"
        f"{abs_block}\n\n"
        f"Extract the definition of the target term if it is explicitly defined. "
        f"Return an empty definition if it is not defined."
    )


def _parse_ls_annotation(result: list) -> dict:
    """
    Parse a Label Studio result list into {terms, definitions, relations}.
    Only keeps spans that participate in at least one relation.
    """
    spans, relations = {}, []
    for r in result:
        if r["type"] == "labels":
            label = r["value"]["labels"][0]
            spans[r["id"]] = {
                "id":    r["id"],
                "label": label,
                "text":  r["value"]["text"].replace("\n", " ").strip(),
                "start": r["value"]["start"],
                "end":   r["value"]["end"],
            }
        elif r["type"] == "relation":
            relations.append({
                "from_id":   r["from_id"],
                "to_id":     r["to_id"],
                "direction": r.get("direction", "right"),
            })
    related_ids = {r["from_id"] for r in relations} | {r["to_id"] for r in relations}
    terms       = [{k: v for k, v in s.items() if k != "label"}
                   for s in spans.values() if s["id"] in related_ids and s["label"] == "TERM"]
    definitions = [{k: v for k, v in s.items() if k != "label"}
                   for s in spans.values() if s["id"] in related_ids and s["label"] == "DEFINITION"]
    return {"terms": terms, "definitions": definitions, "relations": relations}


def _normalise_arxiv_record(item: dict) -> dict | None:
    """
    Accept either raw arxiv777.json (Label Studio export) or already-converted
    arxiv_experiments.json and return a unified record dict, or None to skip.
    """
    # ── Already-converted format ──────────────────────────────────────────────
    if "gold_annotations" in item:
        term = (item.get("metadata", {}).get("term") or item.get("term", "")).strip()
        return {
            "id":              item["id"],
            "term":            term,
            "text":            item["text"],
            "gold_annotations": item["gold_annotations"],
        }

    # ── Raw Label Studio format ───────────────────────────────────────────────
    # Find the first non-cancelled annotation (result may be empty → unannotated)
    ann = None
    for a in item.get("annotations", []):
        if not a.get("was_cancelled", False):
            ann = a
            break

    meta = item.get("meta", {})
    term = meta.get("term", "").strip()
    text = item.get("data", {}).get("text", "")

    gold = _parse_ls_annotation(ann["result"]) if ann and ann.get("result") else \
           {"terms": [], "definitions": [], "relations": []}

    return {
        "id":               item["id"],
        "term":             term,
        "text":             text,
        "gold_annotations": gold,
    }


def _build_arxiv_index(raw_data: list[dict]) -> dict:
    """
    Accepts raw arxiv777.json OR converted arxiv_experiments.json.
    Returns:
      positives : term -> list of {id, text, gold_defs}
                  abstracts that contain at least one valid gold definition for the term
      negatives : term -> list of {id, text}
                  abstracts from all OTHER term groups (no gold def for this term)
    """
    # Normalise all records
    records = [_normalise_arxiv_record(item) for item in raw_data]
    records = [r for r in records if r is not None]

    # Group by term
    by_term: dict[str, list[dict]] = {}
    for r in records:
        by_term.setdefault(r["term"], []).append(r)

    all_terms = list(by_term.keys())
    term_positives: dict[str, list[dict]] = {}
    term_negatives: dict[str, list[dict]] = {}

    for term, recs in by_term.items():
        positives = []
        for r in recs:
            ann      = r["gold_annotations"]
            term_map = {t["id"]: t["text"] for t in ann.get("terms", [])}
            def_map  = {t["id"]: t["text"] for t in ann.get("definitions", [])}
            gold_defs = [
                def_map[rel["to_id"]]
                for rel in ann.get("relations", [])
                if rel["from_id"] in term_map and rel["to_id"] in def_map
            ]
            if gold_defs:
                positives.append({
                    "id":        r["id"],
                    "text":      r["text"],
                    "gold_defs": gold_defs,
                })
        term_positives[term] = positives

        # Negatives: all abstracts from every OTHER term group
        term_negatives[term] = [
            {"id": r["id"], "text": r["text"]}
            for other in all_terms if other != term
            for r in by_term[other]
        ]

    return {"positives": term_positives, "negatives": term_negatives}


def run_exp5(arxiv_data: list[dict], llm: LLM, sampling: SamplingParams,
             system: str, pos_per_term: int = 5) -> dict:
    """
    Oracle retrieval: for each term, sample up to pos_per_term positive abstracts
    (known to contain a gold definition). All sampled abstracts are given together
    in one prompt per term. Evaluated with token F1 against all gold definitions.
    """
    index     = _build_arxiv_index(arxiv_data)
    positives = index["positives"]

    instances = []
    for term, records in positives.items():
        if not records:
            continue
        sampled   = random.sample(records, min(pos_per_term, len(records)))
        abstracts = [r["text"] for r in sampled]
        gold_defs = [g for r in sampled for g in r["gold_defs"]]
        ids       = [r["id"] for r in sampled]
        instances.append({
            "term":      term,
            "ids":       ids,
            "abstracts": abstracts,
            "gold_defs": gold_defs,
        })

    if not instances:
        return {"metrics": {}, "predictions": []}

    messages = [
        build_chat_messages(system, build_prompt_exp5(i["term"], i["abstracts"]))
        for i in instances
    ]
    raw_outputs = run_vllm(llm, sampling, messages)

    results, f1s = [], []
    for inst, raw in zip(instances, raw_outputs):
        pred_def = parse_definition(raw)
        best_f1  = max(_instance_token_f1(pred_def, g) for g in inst["gold_defs"])
        f1s.append(best_f1)
        results.append({
            "ids":       inst["ids"],
            "term":      inst["term"],
            "pred_def":  pred_def,
            "gold_defs": inst["gold_defs"],
            "token_f1":  best_f1,
        })

    n = len(f1s)
    metrics = {
        "token_f1":        sum(f1s) / n if n else 0.0,
        "num_instances":   n,
        "num_terms":       len(positives),
        "pos_per_term":    pos_per_term,
    }
    return {"metrics": metrics, "predictions": results}


def run_exp6(arxiv_data: list[dict], llm: LLM, sampling: SamplingParams,
             system: str, neg_per_term: int = 5) -> dict:
    """
    Robustness under irrelevant retrieval: for each term, sample neg_per_term
    negative abstracts (no gold def for this term). All neg abstracts are given
    together in one prompt per term. Model should return empty.
    Reports Hallucination Probability (HP) and Hallucination FPR (HFPR).
    """
    index     = _build_arxiv_index(arxiv_data)
    positives = index["positives"]
    negatives = index["negatives"]

    instances = []
    for term in positives:      # only terms that have at least one positive
        negs    = negatives.get(term, [])
        sampled = random.sample(negs, min(neg_per_term, len(negs)))
        ids     = [r["id"] for r in sampled]
        abstracts = [r["text"] for r in sampled]
        instances.append({
            "term":      term,
            "ids":       ids,
            "abstracts": abstracts,
        })

    if not instances:
        return {"metrics": {}, "predictions": []}

    messages = [
        build_chat_messages(system, build_prompt_exp6(i["term"], i["abstracts"]))
        for i in instances
    ]
    raw_outputs = run_vllm(llm, sampling, messages)

    results = []
    # Exp6 confusion matrix (expected output = empty / no definition)
    # FP (hallucination) : model generates non-empty definition → wrong
    # TN (correct)       : model returns empty → correct
    fp = tn = 0

    for inst, raw in zip(instances, raw_outputs):
        pred_def         = parse_definition(raw)
        is_hallucination = pred_def.strip() != ""
        if is_hallucination:
            fp += 1    # hallucination = false positive
        else:
            tn += 1    # correctly abstained = true negative
        results.append({
            "ids":              inst["ids"],
            "term":             inst["term"],
            "pred_def":         pred_def,
            "is_hallucination": is_hallucination,
        })

    n               = fp + tn          # total term-level instances
    total_negatives = n * neg_per_term # total individual negative abstracts

    # HP  = FP / (FP + TN)  — proportion of terms where model hallucinated
    # HFPR = FP / total_negative_abstracts — false positives per negative abstract
    hp   = fp / n               if n               else 0.0
    hfpr = fp / total_negatives if total_negatives else 0.0

    metrics = {
        "hallucination_probability": hp,
        "hallucination_fpr":         hfpr,
        # Raw counts
        "fp":                        fp,    # hallucinations
        "tn":                        tn,    # correct abstentions
        "total_term_instances":      n,
        "total_negative_abstracts":  total_negatives,
        "neg_per_term":              neg_per_term,
    }
    return {"metrics": metrics, "predictions": results}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",    required=True,
                        help="Path to the local model directory")
    parser.add_argument("--data",     default=DEFAULT_DATA_PATH,
                        help=f"Path to the dataset JSON (default: {DEFAULT_DATA_PATH})")
    parser.add_argument("--outdir",   default=DEFAULT_OUTPUT_DIR,
                        help=f"Directory to save results (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--exps",     nargs="+", type=int,
                        default=[1, 2, 3, 4], choices=[1, 2, 3, 4, 5, 6])
    parser.add_argument("--lang",     choices=["en", "tr", "both"], default="both",
                        help="Which language subset to run for Exp1-4 (default: both)")
    parser.add_argument("--arxiv_data", default=DEFAULT_ARXIV_PATH,
                        help=f"Path to arxiv_experiments.json for Exp5 and Exp6 (default: {DEFAULT_ARXIV_PATH})")
    parser.add_argument("--pos_per_term", type=int, default=5,
                        help="Positive abstracts per term for Exp5 (default: 5)")
    parser.add_argument("--neg_per_term", type=int, default=5,
                        help="Negative abstracts per term for Exp6 (default: 5)")
    parser.add_argument("--max_new_tokens", type=int, default=4096)
    parser.add_argument("--temperature",    type=float, default=0.0)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9,
                        help="Fraction of GPU memory vLLM may use (default: 0.9)")
    parser.add_argument("--max_model_len", type=int, default=None,
                        help="Override model max sequence length (e.g. 32768)")
    parser.add_argument("--seed",     type=int, default=42,
                        help="Random seed for reproducible runs (default: 42)")
    parser.add_argument("--guideline", default=None,
                        help="Path to a Markdown file whose content is appended to "
                             "every system prompt as task guidelines")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    print(f"Seed: {args.seed}")

    guideline_text = None
    if args.guideline:
        with open(args.guideline, encoding="utf-8") as f:
            guideline_text = f.read().strip()
        print(f"Guideline loaded from: {args.guideline}")

    system_prompts = build_system_prompts(guideline_text)

    # ── Load datasets ─────────────────────────────────────────────────────────
    main_exps   = [e for e in args.exps if e in (1, 2, 3, 4)]
    arxiv_exps  = [e for e in args.exps if e in (5, 6)]

    all_data = []
    if main_exps:
        with open(args.data, encoding="utf-8") as f:
            all_data = json.load(f)
        if args.lang != "both":
            all_data = [d for d in all_data if d["language"] == args.lang]
        print(f"Loaded {len(all_data)} records from {args.data}")

    arxiv_data = []
    if arxiv_exps:
        with open(args.arxiv_data, encoding="utf-8") as f:
            arxiv_data = json.load(f)
        print(f"Loaded {len(arxiv_data)} ArXiv records from {args.arxiv_data}")

    print(f"Model: {args.model}")

    llm_kwargs = dict(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )
    if args.max_model_len is not None:
        llm_kwargs["max_model_len"] = args.max_model_len

    llm = LLM(**llm_kwargs)
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    all_metrics = {}
    for exp_id in args.exps:
        print(f"\n{'='*60}\nRunning Exp{exp_id} ...\n{'='*60}")

        if exp_id in (1, 2, 3, 4):
            runners = {1: run_exp1, 2: run_exp2, 3: run_exp3, 4: run_exp4}
            result = runners[exp_id](all_data, llm, sampling, system_prompts[exp_id])
        elif exp_id == 5:
            result = run_exp5(arxiv_data, llm, sampling, system_prompts[5],
                              pos_per_term=args.pos_per_term)
        elif exp_id == 6:
            result = run_exp6(arxiv_data, llm, sampling, system_prompts[6],
                              neg_per_term=args.neg_per_term)

        all_metrics[f"exp{exp_id}"] = result["metrics"]

        out_file = outdir / f"exp{exp_id}_predictions.json"
        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"Exp{exp_id} metrics:")
        for k, v in result["metrics"].items():
            if isinstance(v, float):
                print(f"  {k:25s}: {v:.4f}")
            else:
                print(f"  {k:25s}: {v}")
        print(f"Saved: {out_file}")

    summary_file = outdir / "metrics_summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    print(f"\nAll metrics saved: {summary_file}")


if __name__ == "__main__":
    main()
