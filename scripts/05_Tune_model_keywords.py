"""Собрать статический словарь ключевых слов для тематического аспекта.

Дорогие QA/attention результаты сохраняются в ``--cache-dir``. Повторный запуск
с теми же fingerprint продолжает SFFS из checkpoint и не повторяет inference.
"""

from __future__ import annotations

import argparse
import ast
import json
import logging
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from untie.attention import AttentionKeywordExtractor
from untie.config import PipelineConfig
from untie.data import load_json_dataframe, replace_underscores
from untie.keyword_evidence import (
    EvidenceStore,
    ExtractionMetricCache,
    build_evidence_fingerprint,
    collect_document_evidence,
)
from untie.keyword_training import (
    StrategyConfig,
    TrainingConfig,
    save_strategy_summary_csv,
    save_tuning_trace,
    tune_global_keywords,
)
from untie.keyword_tuning import ObjectiveConfig, deterministic_document_split
from untie.model_params import (
    create_tuned_copy,
    load_model_params,
    save_model_params_atomic,
)
from untie.models import ModelFactory, profile_for_language
from untie.pipelines import DocumentProcessor
from untie.text import RussianSentenceSplitter, SentenceSplitter


LOGGER = logging.getLogger("untie.keyword_tuning")
ProgressCallback = Callable[[str, int, int, Mapping[str, Any]], None]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune global static keywords for one thematic aspect"
    )
    parser.add_argument("--language", choices=("en", "ru"), default="en")
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--model-params", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--field-id", type=int)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "keyword_tuning_cache",
    )
    parser.add_argument("--trace", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--attention-top-k", type=int, default=100)
    parser.add_argument("--min-document-support", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=150)
    parser.add_argument("--max-keywords", type=int, default=20)
    parser.add_argument("--evaluation-budget", type=int, default=250)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=5)
    parser.add_argument("--stability-runs", type=int, default=5)
    parser.add_argument("--stability-threshold", type=float, default=0.7)
    parser.add_argument("--include-bertscore", action="store_true")
    parser.add_argument(
        "--strategy",
        nargs=3,
        action="append",
        metavar=("SCORE", "CLUSTER", "ANSWER"),
        help=(
            "Tune a specific strategy instead of the complete 27-strategy grid; "
            "repeat the option to tune several strategies"
        ),
    )
    parser.add_argument("--chunk-max-tokens", type=int)
    parser.add_argument("--overlap-tokens", type=int)
    parser.add_argument("--log-level", default="INFO")
    return parser


def _defaults(language: str) -> tuple[Path, Path, Path]:
    if language == "en":
        return (
            PROJECT_ROOT / "datasets" / "scirex_structured.json",
            PROJECT_ROOT / "model_params" / "scart_init_model.json",
            PROJECT_ROOT / "model_params" / "scart_tuned_model.json",
        )
    return (
        PROJECT_ROOT / "datasets" / "ruserrc_structured.csv",
        PROJECT_ROOT / "model_params" / "ruserrc_init_model.json",
        PROJECT_ROOT / "model_params" / "ruserrc_tuned_model.json",
    )


def _parse_string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return [text]
    if isinstance(parsed, (list, tuple)):
        return [str(item).strip() for item in parsed if str(item).strip()]
    return [str(parsed).strip()]


def load_training_dataset(path: Path, language: str) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path, sep=";", encoding="utf-8")
        reference_column = (
            "Task_aspects" if "Task_aspects" in raw.columns else "tasks_cleaned"
        )
        if reference_column not in raw.columns:
            raise ValueError("CSV must contain Task_aspects or tasks_cleaned")
        text_column = (
            "text_clean"
            if "text_clean" in raw.columns
            else "original_text"
            if "original_text" in raw.columns
            else "text"
        )
        id_column = "id" if "id" in raw.columns else "doc_id"
        frame = pd.DataFrame(
            {
                "doc_id": raw[id_column].astype(str),
                "original_text": raw[text_column].fillna(raw.get("text", "")),
                "tasks_cleaned": raw[reference_column].apply(_parse_string_list),
            }
        )
    else:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            frame = (
                pd.DataFrame(payload)
                if isinstance(payload, list)
                else pd.DataFrame.from_dict(payload)
            )
        except json.JSONDecodeError:
            frame = load_json_dataframe(path)
        if "tasks_cleaned" not in frame.columns:
            if "tasks" not in frame.columns:
                raise ValueError("JSON dataset must contain tasks or tasks_cleaned")
            frame = replace_underscores(
                frame, "tasks", destination_column="tasks_cleaned"
            )
    required = {"doc_id", "original_text", "tasks_cleaned"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["doc_id"] = frame["doc_id"].astype(str)
    frame["tasks_cleaned"] = frame["tasks_cleaned"].apply(_parse_string_list)
    frame = frame[frame["tasks_cleaned"].map(bool)]
    return frame.reset_index(drop=True)


def _normalizers(language: str):
    if language == "en":
        try:
            from nltk.stem import SnowballStemmer

            stemmer = SnowballStemmer("english")
            return str.lower, stemmer.stem
        except ImportError:
            return str.lower, str.lower
    try:
        from pymorphy3 import MorphAnalyzer
    except ImportError as error:
        raise RuntimeError("Russian tuning requires pymorphy3") from error
    morph = MorphAnalyzer()
    try:
        from nltk.stem import SnowballStemmer

        stem = SnowballStemmer("russian").stem
    except ImportError:
        stem = str.lower
    return lambda word: morph.parse(word.lower())[0].normal_form, stem


def _field_by_id(model, field_id: int | None):
    if field_id is None:
        if not model.fields:
            raise ValueError("Model params contain no fields")
        return model.fields[0]
    for field in model.fields:
        if field.field_id == field_id:
            return field
    raise ValueError(f"Unknown field_id: {field_id}")


def run(
    args: argparse.Namespace,
    *,
    progress_callback: ProgressCallback | None = None,
):
    dataset_default, params_default, output_default = _defaults(args.language)
    dataset_path = args.dataset or dataset_default
    params_path = args.model_params or params_default
    output_path = args.output or output_default
    trace_path = args.trace or output_path.with_suffix(".tuning.json")

    dataframe = load_training_dataset(dataset_path, args.language)
    if args.limit is not None:
        dataframe = dataframe.iloc[: args.limit].copy()
    model_params = load_model_params(params_path)
    field = _field_by_id(model_params, args.field_id)
    question = field.questions[0]
    split = deterministic_document_split(
        dataframe["doc_id"], seed=args.seed
    )
    train_ids = set(split["train"])

    profile = profile_for_language(args.language)
    if args.language == "ru" and not profile.attention_supported:
        profile = replace(profile, attention_supported=True)
    chunk_max_tokens = args.chunk_max_tokens or (
        128 if args.language == "ru" else 384
    )
    overlap_tokens = args.overlap_tokens
    if overlap_tokens is None:
        overlap_tokens = 24 if args.language == "ru" else 50
    pipeline_config = PipelineConfig(
        profile=profile,
        chunk_max_tokens=chunk_max_tokens,
        overlap_tokens=overlap_tokens,
        device=args.device,
    )
    models = ModelFactory(pipeline_config)
    splitter = (
        RussianSentenceSplitter()
        if args.language == "ru"
        else SentenceSplitter()
    )
    processor = DocumentProcessor(
        models.tokenizer,
        pipeline_config,
        splitter=splitter,
        sentence_encoder=None,
    )
    attention = AttentionKeywordExtractor(
        models.qa_model,
        models.tokenizer,
        models.device,
        limit=args.attention_top_k,
    )
    lemmatize, stem = _normalizers(args.language)
    evidence_store = EvidenceStore(
        args.cache_dir / args.language / f"field-{field.field_id}" / "evidence"
    )
    fingerprint_payload = {
        "language": args.language,
        "qa_model": profile.qa_model,
        "sentence_model": profile.sentence_model,
        "chunk_max_tokens": chunk_max_tokens,
        "overlap_tokens": overlap_tokens,
        "attention_top_k": args.attention_top_k,
        "pipeline_version": "static-keywords-v1",
    }

    evidence = []
    for index, row in dataframe.iterrows():
        doc_id = str(row["doc_id"])
        text = str(row["original_text"])
        references = _parse_string_list(row["tasks_cleaned"])
        collect_candidates = doc_id in train_ids
        expected = build_evidence_fingerprint(
            doc_id=doc_id,
            text=text,
            aspect_id=str(field.field_id),
            question=question,
            references=references,
            collect_candidates=collect_candidates,
            fingerprint_payload=fingerprint_payload,
        )
        cached = evidence_store.load(
            doc_id, expected_fingerprint=expected
        )
        cache_hit = cached is not None
        if cached is None:
            cached = collect_document_evidence(
                doc_id=doc_id,
                text=text,
                aspect_id=str(field.field_id),
                aspect_reference=f"{field.field_name}. {field.text}",
                question=question,
                references=references,
                processor=processor,
                answerer=models.answerer,
                encoder=models.sentence_encoder,
                attention_extractor=attention,
                language=args.language,
                lemmatize=lemmatize,
                stem=stem,
                strict_answer_threshold=pipeline_config.strict_answer_threshold,
                min_answer_threshold=pipeline_config.min_answer_threshold,
                keyword_idf_threshold=pipeline_config.keyword_idf_threshold,
                collect_candidates=collect_candidates,
                fingerprint_payload=fingerprint_payload,
            )
            evidence_store.save(cached)
        evidence.append(cached)
        LOGGER.info(
            "Evidence %s/%s: %s candidates=%s source=%s",
            index + 1,
            len(dataframe),
            doc_id,
            len(cached.candidates),
            "cache" if cache_hit else "computed",
        )
        if progress_callback is not None:
            progress_callback(
                "evidence",
                index + 1,
                len(dataframe),
                {
                    "doc_id": doc_id,
                    "candidate_count": len(cached.candidates),
                    "source": "cache" if cache_hit else "computed",
                    "collect_candidates": collect_candidates,
                },
            )

    strategies = ()
    if args.strategy:
        strategies = tuple(StrategyConfig(*values) for values in args.strategy)
    training_config = TrainingConfig(
        language=args.language,
        min_document_support=args.min_document_support,
        max_candidates=args.max_candidates,
        max_keywords=args.max_keywords,
        evaluation_budget=args.evaluation_budget,
        patience=args.patience,
        beam_width=args.beam_width,
        stability_runs=args.stability_runs,
        stability_threshold=args.stability_threshold,
        seed=args.seed,
        include_bertscore=args.include_bertscore,
        strategies=strategies,
    )
    metric_cache = ExtractionMetricCache(
        args.cache_dir
        / args.language
        / f"field-{field.field_id}"
        / "extraction_metrics.json"
    )
    outcome = tune_global_keywords(
        evidence,
        config=training_config,
        metric_cache=metric_cache,
        objective_config=ObjectiveConfig(
            downside_penalty=0.75,
            harm_penalty=0.5,
            fallback_penalty=0.1,
            size_penalty=0.002,
            harm_threshold=0.01,
            confidence_weight=0.25,
            bootstrap_seed=args.seed,
        ),
        split=split,
        checkpoint_dir=(
            args.cache_dir
            / args.language
            / f"field-{field.field_id}"
            / "checkpoints"
        ),
        progress_callback=progress_callback,
    )
    tuned_field = replace(
        field,
        keywords=list(outcome.keywords),
        keyword_metadata=list(outcome.keyword_metadata),
        tuning_metadata=outcome.tuning_metadata(),
    )
    tuned_fields = [
        tuned_field if item.field_id == field.field_id else item
        for item in model_params.fields
    ]
    tuned_model = create_tuned_copy(
        model_params,
        fields=tuned_fields,
        tuning_metadata={
            "method": "multi-fidelity-sffs-stability",
            "dataset": str(dataset_path),
            "score": outcome.objective,
            "document_count": len(dataframe),
            "random_seed": args.seed,
            "output": str(output_path),
        },
    )
    save_model_params_atomic(tuned_model, output_path)
    save_tuning_trace(outcome, trace_path)
    save_strategy_summary_csv(
        outcome, trace_path.with_suffix(".strategies.csv")
    )
    LOGGER.info(
        "Saved %s keywords, strategy=%s, objective=%.6f to %s",
        len(outcome.keywords),
        outcome.strategy.name,
        outcome.objective,
        output_path,
    )
    return outcome


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
