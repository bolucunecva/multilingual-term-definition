import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev
from typing import List, Tuple, Dict

# Allow running from term_extraction/ or from parent dir
sys.path.insert(0, str(Path(__file__).parent))
from experiments.data_utils import load_data, extract_gold_pairs, get_language, get_article_id

def _word_count(text: str) -> int:
    return len(text.split())

def _char_count(text: str) -> int:
    return len(text.strip())

def _fmt(values: List[float], decimals: int = 1) -> str:
    if not values:
        return "n/a"
    return (
        f"mean={mean(values):.{decimals}f}  "
        f"median={median(values):.{decimals}f}  "
        f"min={min(values):.{decimals}f}  "
        f"max={max(values):.{decimals}f}  "
        f"stdev={stdev(values):.{decimals}f}" if len(values) > 1
        else f"mean={mean(values):.{decimals}f}  (single value)"
    )

def _section(title: str):
    width = 60
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)

def _row(label: str, value, indent: int = 2):
    print(f"{'  ' * indent}{label:<40} {value}")


def collect_stats(data: List[Dict]) -> Dict:
    stats = {
        "total_docs":        len(data),
        "docs_with_pairs":   0,
        "docs_no_pairs":     0,
        "total_pairs":       0,

        # per-language
        "lang_doc_counts":   Counter(),
        "lang_pair_counts":  Counter(),

        # per-doc pair counts
        "pairs_per_doc":     [],

        # span lengths (chars & words)
        "term_chars":        [],
        "term_words":        [],
        "def_chars":         [],
        "def_words":         [],

        # per-language span lengths
        "term_chars_lang":   defaultdict(list),
        "term_words_lang":   defaultdict(list),
        "def_chars_lang":    defaultdict(list),
        "def_words_lang":    defaultdict(list),

        # article-level (one OAI id = one bilingual article)
        "article_langs":     defaultdict(set),   # oai_id -> {lang, ...}

        # docs per pair count (histogram)
        "pair_count_hist":   Counter(),
    }

    for item in data:
        lang   = get_language(item)
        oai_id = get_article_id(item)
        pairs  = extract_gold_pairs(item)

        stats["lang_doc_counts"][lang] += 1
        stats["article_langs"][oai_id].add(lang)

        n = len(pairs)
        stats["pair_count_hist"][n] += 1
        stats["pairs_per_doc"].append(n)

        if n == 0:
            stats["docs_no_pairs"] += 1
            continue

        stats["docs_with_pairs"] += 1
        stats["total_pairs"]     += n
        stats["lang_pair_counts"][lang] += n

        for term, definition in pairs:
            tc = _char_count(term)
            tw = _word_count(term)
            dc = _char_count(definition)
            dw = _word_count(definition)

            stats["term_chars"].append(tc)
            stats["term_words"].append(tw)
            stats["def_chars"].append(dc)
            stats["def_words"].append(dw)

            stats["term_chars_lang"][lang].append(tc)
            stats["term_words_lang"][lang].append(tw)
            stats["def_chars_lang"][lang].append(dc)
            stats["def_words_lang"][lang].append(dw)

    return stats


def print_stats(stats: Dict):

    _section("CORPUS OVERVIEW")
    _row("Total documents",          stats["total_docs"])
    _row("Documents with pairs",     stats["docs_with_pairs"])
    _row("Documents without pairs",  stats["docs_no_pairs"])
    _row("Total term-def pairs",     stats["total_pairs"])
    _row("Unique articles (OAI)",    len(stats["article_langs"]))

    bilingual = sum(1 for langs in stats["article_langs"].values() if len(langs) >= 2)
    _row("Bilingual articles (en+tr)", bilingual)

    _section("LANGUAGE DISTRIBUTION")
    for lang, cnt in sorted(stats["lang_doc_counts"].items()):
        pairs = stats["lang_pair_counts"].get(lang, 0)
        pct_docs  = cnt  / stats["total_docs"]  * 100
        pct_pairs = pairs / stats["total_pairs"] * 100 if stats["total_pairs"] else 0
        _row(f"[{lang}]  docs",  f"{cnt:>4}  ({pct_docs:.1f}%)")
        _row(f"[{lang}]  pairs", f"{pairs:>4}  ({pct_pairs:.1f}%)")

    _section("PAIRS PER DOCUMENT")
    _row("Distribution", _fmt([float(x) for x in stats["pairs_per_doc"]]))
    print()
    print("    count  docs")
    for k in sorted(stats["pair_count_hist"]):
        bar = "#" * stats["pair_count_hist"][k]
        print(f"    {k:>5}  {stats['pair_count_hist'][k]:>4}  {bar}")

    _section("TERM SPAN LENGTHS  (all languages)")
    _row("Chars", _fmt(stats["term_chars"]))
    _row("Words", _fmt(stats["term_words"]))

    _section("DEFINITION SPAN LENGTHS  (all languages)")
    _row("Chars", _fmt(stats["def_chars"]))
    _row("Words", _fmt(stats["def_words"]))

    for lang in sorted(stats["term_chars_lang"]):
        _section(f"SPAN LENGTHS  [{lang}]")
        _row("Term chars",       _fmt(stats["term_chars_lang"][lang]))
        _row("Term words",       _fmt(stats["term_words_lang"][lang]))
        _row("Definition chars", _fmt(stats["def_chars_lang"][lang]))
        _row("Definition words", _fmt(stats["def_words_lang"][lang]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",   default="dataset/mlonlysum_majority.json")
    parser.add_argument("--output", default=None, help="Optional JSON file to save stats")
    args = parser.parse_args()

    data  = load_data(args.data)
    stats = collect_stats(data)
    print_stats(stats)

    if args.output:
        # Serialisable subset (no defaultdicts / Counters with list values)
        out = {
            "total_docs":       stats["total_docs"],
            "docs_with_pairs":  stats["docs_with_pairs"],
            "docs_no_pairs":    stats["docs_no_pairs"],
            "total_pairs":      stats["total_pairs"],
            "unique_articles":  len(stats["article_langs"]),
            "bilingual_articles": sum(
                1 for l in stats["article_langs"].values() if len(l) >= 2
            ),
            "lang_doc_counts":  dict(stats["lang_doc_counts"]),
            "lang_pair_counts": dict(stats["lang_pair_counts"]),
            "pair_count_hist":  {str(k): v for k, v in stats["pair_count_hist"].items()},
            "term_chars":  {"mean": mean(stats["term_chars"]) if stats["term_chars"] else 0,
                            "median": median(stats["term_chars"]) if stats["term_chars"] else 0,
                            "min": min(stats["term_chars"]) if stats["term_chars"] else 0,
                            "max": max(stats["term_chars"]) if stats["term_chars"] else 0},
            "def_chars":   {"mean": mean(stats["def_chars"]) if stats["def_chars"] else 0,
                            "median": median(stats["def_chars"]) if stats["def_chars"] else 0,
                            "min": min(stats["def_chars"]) if stats["def_chars"] else 0,
                            "max": max(stats["def_chars"]) if stats["def_chars"] else 0},
            "term_words":  {"mean": mean(stats["term_words"]) if stats["term_words"] else 0,
                            "median": median(stats["term_words"]) if stats["term_words"] else 0,
                            "min": min(stats["term_words"]) if stats["term_words"] else 0,
                            "max": max(stats["term_words"]) if stats["term_words"] else 0},
            "def_words":   {"mean": mean(stats["def_words"]) if stats["def_words"] else 0,
                            "median": median(stats["def_words"]) if stats["def_words"] else 0,
                            "min": min(stats["def_words"]) if stats["def_words"] else 0,
                            "max": max(stats["def_words"]) if stats["def_words"] else 0},
        }
        Path(args.output).write_text(
            json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"\nStats saved to {args.output}")


if __name__ == "__main__":
    main()
