"""Run the full, resumable NER experiment grid."""

import argparse
import json
from pathlib import Path

from cnn_classification import EMBEDDING_NAMES, run as run_classical
from transformers_classification import run as run_bert


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", choices=["classical", "bert", "all"], default="all")
    parser.add_argument("--output-dir", default=str(Path(__file__).parent / "results"))
    parser.add_argument("--bert-model", default="bert-base-multilingual-cased")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for domain in ("EMEA", "MEDLINE"):
        if args.only in ("classical", "all"):
            for architecture in ("cnn", "lstm"):
                for source in ["random"] + EMBEDDING_NAMES:
                    result_path = output / f"{domain}_{architecture}_{source}.json"
                    if result_path.exists() and not args.force:
                        saved = json.loads(result_path.read_text(encoding="utf-8"))
                        if saved.get("epochs", 0) >= 30:
                            print(f"Skipping completed {result_path.name}", flush=True)
                            continue
                    run_classical(domain, architecture, source, epochs=30, output_dir=output)
        if args.only in ("bert", "all"):
            lower_name = args.bert_model.lower()
            model_label = ("drbert" if "drbert" in lower_name else
                           "camembert" if "camembert" in lower_name else "bert")
            result_path = output / f"{domain}_{model_label}.json"
            if result_path.exists() and not args.force:
                saved = json.loads(result_path.read_text(encoding="utf-8"))
                if saved.get("epochs", 0) >= 5 and saved.get("checkpoint") == args.bert_model:
                    print(f"Skipping completed {result_path.name}", flush=True)
                    continue
            run_bert(domain, model_name=args.bert_model, epochs=5, output_dir=output)


if __name__ == "__main__":
    main()
