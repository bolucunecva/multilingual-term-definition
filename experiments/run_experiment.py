import argparse
import json
from pathlib import Path

from data_utils import (
    load_data,
    get_document_text,
    extract_gold_pairs,
    pairs_to_label_studio,
)





# Arguments
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Label Studio JSON export")
    p.add_argument("--model", required=True, help="Local model path or HF name")
    p.add_argument("--output", required=True, help="Path to save LS-format predictions")
    p.add_argument("--guideline", default=None, help="Plain-text guideline file")
    p.add_argument("--eval_output", default=None, help="Path to save evaluation JSON")
    return p.parse_args()


def main():
    args = parse_args()
    
    guideline = None
    if args.guideline:
        guideline = Path(args.guideline).read_text(encoding="utf-8").strip()

    # Load dataset
    data = load_data(args.data)
    print(f"[+] Loaded {len(data)} documents")