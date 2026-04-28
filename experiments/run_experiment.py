import argparse
import json
from pathlib import Path

from data_utils import (
    load_data,
    get_document_text,
    extract_gold_pairs,
    pairs_to_label_studio,
)

from prompts import build_prompt
from inference import Extractor
from evaluator import evaluate_document, aggregate_scores



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

    # Load model
    extractor = Extractor(
        model_name=args.model,
        tensor_parallel_size=2
    )

    # Build prompts
    prompts = [
        build_prompt(
            text=get_document_text(item),
            guideline=guideline,
            tokenizer=extractor.get_tokenizer(),
        )
        for item in data
    ]

    # Run inference (batched)
    raw_outputs = extractor.generate(prompts)
    print(f"Inference complete")

    # Parse outputs, build LS predictions, evaluate
    ls_predictions = []
    doc_scores = []

    for item, raw in zip(data, raw_outputs):
        source_text = get_document_text(item)
        pred_pairs = Extractor.parse_pairs(raw)
        gold_pairs = extract_gold_pairs(item)

        score = evaluate_document(pred_pairs, gold_pairs)
        doc_scores.append(score)

        ls_pred = pairs_to_label_studio(pred_pairs, source_text, item.get("id"))
        ls_predictions.append(ls_pred)

    # Save LS-format predictions
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(ls_predictions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[+] Predictions saved to {out_path}")

    # Aggregate and print scores
    agg = aggregate_scores(doc_scores)
    print("\n=== Evaluation Results ===")
    print(json.dumps(agg, indent=2))

    if args.eval_output:
        eval_path = Path(args.eval_output)
        eval_path.parent.mkdir(parents=True, exist_ok=True)
        eval_path.write_text(
            json.dumps({"aggregate": agg, "per_doc": doc_scores}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"Evaluation saved to {eval_path}")


if __name__ == "__main__":
    main()