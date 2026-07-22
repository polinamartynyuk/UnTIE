from __future__ import annotations

import argparse
from pathlib import Path


MODELS = {
    "en-qa": ("qa", "deepset/roberta-base-squad2", "bert_eng_qa_baseroberta_model"),
    "ru-qa": (
        "qa",
        "AlexKay/xlm-roberta-large-qa-multilingual-finedtuned-ru",
        "rubert_ru_qa_model",
    ),
    "en-sentence": (
        "sentence",
        "bert-large-nli-mean-tokens",
        "eng_sentence_transformer_model",
    ),
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a supported UnTIE model")
    parser.add_argument("model", choices=MODELS)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("scripts/models_processing/models"),
    )
    args = parser.parse_args()
    kind, source, directory = MODELS[args.model]
    destination = args.output_root / directory

    if kind == "sentence":
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(source).save(str(destination))
    else:
        from transformers import AutoModelForQuestionAnswering, AutoTokenizer

        AutoTokenizer.from_pretrained(source).save_pretrained(destination)
        AutoModelForQuestionAnswering.from_pretrained(source).save_pretrained(destination)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
