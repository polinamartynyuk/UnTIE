"""Final train+dev keyword refit with frozen topic routing.

This script does not rebuild clusters and does not re-search the keyword or
routing grids. It reads the already selected keyword trial and the routing
parameters stored in a tuned topic artifact, then refits leaf dictionaries on
the union of train and dev documents. Cluster evidence keeps attention and IDF
but, unlike global static tuning, does not contrast candidates against gold
task labels. Chunk scoring uses attention weight only. The selected trial's
min_keywords floor may be lowered only here so union-fit can keep stable 4-5
word dictionaries. Outputs are written to a separate directory and never
replace the source tuned model.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import shutil
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from untie.attention import AttentionKeywordExtractor
from untie.cli import load_static_keywords
from untie.config import PipelineConfig
from untie.keyword_evidence import (
    CachedDocumentEvidence,
    EvidenceStore,
    ExtractionMetricCache,
    build_evidence_fingerprint,
    collect_document_evidence,
    evidence_fingerprint_payload,
    stable_fingerprint,
)
from untie.keyword_training import MetricWeights, StrategyConfig, TrainingConfig
from untie.keyword_tuning import ObjectiveConfig, deterministic_document_split
from untie.models import ModelFactory, profile_for_language
from untie.pipelines import DocumentProcessor
from untie.topic_keyword_training import (
    DEFAULT_REFIT_MIN_KEYWORDS,
    load_selected_keyword_trial,
    refit_topic_keyword_profiles,
    resolve_refit_keyword_floor,
)
from untie.topics import (
    TopicDocument,
    load_topic_artifact,
    save_topic_artifact_atomic,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Refit hierarchical topic keywords on train+dev using the frozen "
            "best routing gate and selected keyword trial"
        )
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Directory with topic_model.tuned.json, keyword_search.csv and manifest.json",
    )
    parser.add_argument("--tuned-model", type=Path)
    parser.add_argument("--keyword-search", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--static-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--language", choices=("en", "ru"), default="en")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--question", default="Which task was solved?")
    parser.add_argument(
        "--aspect-reference", default="task. main task being solved in paper"
    )
    parser.add_argument("--field-id", type=int, default=1)
    parser.add_argument("--evaluation-budget", type=int, default=120)
    parser.add_argument("--stability-runs", type=int, default=3)
    parser.add_argument("--min-fit-documents", type=int, default=4)
    parser.add_argument(
        "--refit-min-keywords",
        type=int,
        default=DEFAULT_REFIT_MIN_KEYWORDS,
        help=(
            "SFFS floor used only on this union-fit. Never raises the frozen "
            f"trial min_keywords. Default {DEFAULT_REFIT_MIN_KEYWORDS} keeps "
            "stable 4-5 word dictionaries that missed the trial floor of 6."
        ),
    )
    parser.add_argument("--chunk-max-tokens", type=int, default=384)
    parser.add_argument("--overlap-tokens", type=int, default=50)
    parser.add_argument("--attention-top-k", type=int, default=100)
    parser.add_argument(
        "--contrast-against-gold",
        action="store_true",
        help=(
            "Restore aspect-vs-gold contrast when collecting cluster evidence. "
            "Off by default: global static tuning still uses gold contrast."
        ),
    )
    parser.add_argument(
        "--score-chunk-strategy",
        choices=("only_weight", "only_score_diff", "equal_weight_score_diff"),
        default="only_weight",
        help=(
            "Chunk scoring for the cluster dictionary. Default only_weight "
            "matches the no-gold-contrast experiment; static fallback still "
            "uses the frozen static model strategy."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite previous refit outputs in --output-dir only",
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse evidence/SFFS checkpoints already present in output-dir",
    )
    return parser


def _load_payload(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = [json.loads(line) for line in text.splitlines() if line.strip()]
    if isinstance(payload, dict):
        payload = payload.get("documents")
    if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
        raise ValueError("dataset must contain a JSON array or JSONL objects")
    return payload


def _references(raw: dict) -> tuple[str, ...]:
    value = raw.get("tasks_cleaned", raw.get("tasks", ()))
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            value = (value,)
    values = value if isinstance(value, (list, tuple)) else (value,)
    return tuple(
        str(item).replace("_", " ").strip()
        for item in values
        if item is not None and str(item).strip()
    )


def load_fit_documents(
    path: Path, seed: int, *, quick: bool = False
) -> tuple[list[TopicDocument], dict[str, tuple[str, ...]]]:
    payload = _load_payload(path)
    ids = [str(item.get("doc_id", item.get("id", ""))) for item in payload]
    if any(not doc_id for doc_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("dataset document ids must be non-empty and unique")
    explicit = [item.get("split") for item in payload]
    if any(value is not None for value in explicit):
        if not all(value is not None for value in explicit):
            raise ValueError("split must be present on every record or none")
        split_by_id = {
            doc_id: str(value).casefold() for doc_id, value in zip(ids, explicit)
        }
        invalid = sorted(set(split_by_id.values()).difference({"train", "dev", "test"}))
        if invalid:
            raise ValueError(f"unsupported dataset splits: {invalid}")
    else:
        split = deterministic_document_split(ids, seed=seed)
        split_by_id = {
            doc_id: split_name
            for split_name, split_ids in split.items()
            for doc_id in split_ids
        }

    def convert(raw: dict, split_name: str) -> TopicDocument:
        headings = raw.get("headings", ())
        if isinstance(headings, str):
            headings = (headings,)
        return TopicDocument(
            doc_id=str(raw.get("doc_id", raw.get("id"))),
            text=str(raw.get("original_text", raw.get("text", "")) or ""),
            title=str(raw.get("title", "") or ""),
            abstract=str(raw.get("abstract", "") or ""),
            headings=tuple(str(item) for item in headings),
            split=split_name,
        )

    documents = [
        convert(raw, split_by_id[str(raw.get("doc_id", raw.get("id")))])
        for raw in payload
        if split_by_id.get(str(raw.get("doc_id", raw.get("id")))) in {"train", "dev"}
    ]
    if quick:
        train = [item for item in documents if item.split == "train"][:24]
        dev = [item for item in documents if item.split == "dev"][:8]
        documents = [*train, *dev]
    references = {
        str(raw.get("doc_id", raw.get("id"))): _references(raw) for raw in payload
    }
    if not any(item.split == "train" for item in documents) or not any(
        item.split == "dev" for item in documents
    ):
        raise ValueError("refit requires non-empty train and dev partitions")
    missing_references = [
        document.doc_id
        for document in documents
        if not references.get(document.doc_id)
    ]
    if missing_references:
        raise ValueError(
            f"train/dev documents require task references: {missing_references[:5]}"
        )
    return documents, references


def _normalizers(language: str):
    try:
        from nltk.stem import SnowballStemmer
    except ImportError:
        return str.lower, str.lower
    return str.lower, SnowballStemmer(
        "english" if language == "en" else "russian"
    ).stem


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _resolve_source_paths(args: argparse.Namespace) -> tuple[Path, Path, Path | None]:
    source_dir = args.source_dir.resolve() if args.source_dir is not None else None
    tuned = args.tuned_model or (
        source_dir / "topic_model.tuned.json" if source_dir is not None else None
    )
    keyword_search = args.keyword_search or (
        source_dir / "keyword_search.csv" if source_dir is not None else None
    )
    manifest = args.manifest
    if manifest is None and source_dir is not None:
        candidate = source_dir / "manifest.json"
        manifest = candidate if candidate.is_file() else None
    if tuned is None or keyword_search is None:
        raise ValueError("provide --source-dir or both --tuned-model and --keyword-search")
    return Path(tuned), Path(keyword_search), Path(manifest) if manifest else None


def _protected_source_paths(tuned_model: Path, keyword_search: Path, manifest: Path | None) -> set[Path]:
    protected = {tuned_model.resolve(), keyword_search.resolve()}
    if manifest is not None:
        protected.add(manifest.resolve())
    parent = tuned_model.resolve().parent
    for name in (
        "topic_model.auto.json",
        "topic_model.profiled.json",
        "topic_model.tuned.json",
        "keyword_search.csv",
        "routing_search.csv",
        "manifest.json",
        "run_config.json",
        "topic_keyword_coverage.csv",
    ):
        protected.add(parent / name)
    return protected


def default_refit_output_dir(tuned_model: Path, *, contrast_against_gold: bool) -> Path:
    parent = tuned_model.resolve().parent
    if contrast_against_gold:
        return parent / "refit"
    return parent / "refit_no_gold_contrast"


def _sibling_evidence_stores(output_dir: Path, primary_root: Path) -> list[EvidenceStore]:
    cache_root = output_dir / "cache"
    if not cache_root.is_dir():
        return []
    stores = []
    primary = primary_root.resolve()
    for evidence_dir in sorted(cache_root.glob("*/evidence")):
        if evidence_dir.resolve() == primary:
            continue
        stores.append(EvidenceStore(evidence_dir))
    return stores


def _load_evidence_reusing_siblings(
    store: EvidenceStore,
    extra_stores: Sequence[EvidenceStore],
    doc_id: str,
    expected_fingerprint: str,
) -> CachedDocumentEvidence | None:
    cached = store.load(doc_id, expected_fingerprint=expected_fingerprint)
    if cached is not None:
        return cached
    for extra in extra_stores:
        cached = extra.load(doc_id, expected_fingerprint=expected_fingerprint)
        if cached is None:
            continue
        store.save(cached)
        return cached
    return None


def _reuse_metric_cache(output_dir: Path, destination: Path) -> None:
    cache_root = output_dir / "cache"
    if not cache_root.is_dir() or destination.exists():
        return
    candidates = [
        path
        for path in cache_root.glob("*/metrics.json")
        if path.resolve() != destination.resolve() and path.is_file()
    ]
    if not candidates:
        return
    newest = max(candidates, key=lambda path: path.stat().st_mtime)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(newest, destination)


def run(args: argparse.Namespace):
    tuned_model, keyword_search, manifest_path = _resolve_source_paths(args)
    if not tuned_model.is_file():
        raise ValueError(f"tuned topic model not found: {tuned_model}")
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else default_refit_output_dir(
            tuned_model, contrast_against_gold=bool(args.contrast_against_gold)
        )
    )
    output_model = output_dir / "topic_model.refit.json"
    if output_dir.resolve() == tuned_model.resolve():
        raise ValueError(
            "refit refuses to overwrite the source tuned model; choose another --output-dir"
        )
    protected = _protected_source_paths(tuned_model, keyword_search, manifest_path)
    if output_model.resolve() in protected:
        raise ValueError(
            "refit refuses to overwrite the source tuned model; choose another --output-dir"
        )
    colliding = [
        path
        for path in (
            output_model,
            output_dir / "refit_manifest.json",
            output_dir / "refit_keyword_coverage.csv",
            output_dir / "refit_run_config.json",
        )
        if path.exists()
    ]
    if colliding and not args.force and not args.resume:
        raise ValueError(
            "refit outputs already exist; pass --force to overwrite refit files only, "
            f"or choose another --output-dir. Existing: {colliding[0]}"
        )

    trial_index = None
    if manifest_path is not None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "keyword_best_trial" in manifest:
            trial_index = int(manifest["keyword_best_trial"])
    selected_trial = load_selected_keyword_trial(
        keyword_search, trial_index=trial_index
    )
    applied_min_keywords = resolve_refit_keyword_floor(
        int(selected_trial["min_keywords"]),
        refit_min_keywords=int(args.refit_min_keywords),
        max_keywords=int(selected_trial["max_keywords"]),
    )
    source_artifact = load_topic_artifact(tuned_model)
    frozen_routing = source_artifact.routing

    fit_documents, references = load_fit_documents(
        args.dataset, args.seed, quick=args.quick
    )
    profile = profile_for_language(args.language)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_payload = {
        "version": "hierarchical-keyword-refit-v2",
        "tuned_model": str(tuned_model.resolve()),
        "keyword_search": str(keyword_search.resolve()),
        "selected_trial": selected_trial,
        "applied_min_keywords": applied_min_keywords,
        "contrast_against_gold": bool(args.contrast_against_gold),
        "score_chunk_strategy": str(args.score_chunk_strategy),
        "routing": asdict(frozen_routing),
        "dataset": [
            {
                "doc_id": item.doc_id,
                "split": item.split,
                "text": item.text,
            }
            for item in fit_documents
        ],
        "parameters": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"resume", "force", "output_dir", "device"}
        },
    }
    run_fingerprint = stable_fingerprint(run_payload)
    run_config_path = output_dir / "refit_run_config.json"
    previous_run = (
        json.loads(run_config_path.read_text(encoding="utf-8"))
        if run_config_path.exists()
        else None
    )
    if args.resume:
        if previous_run is None:
            raise ValueError("--resume requires an existing refit_run_config.json in output-dir")
        if previous_run.get("fingerprint") != run_fingerprint:
            raise ValueError(
                "--resume configuration/dataset fingerprint does not match "
                "the existing refit run"
            )
        cache_namespace = str(previous_run["cache_namespace"])
    else:
        cache_namespace = f"{run_fingerprint}-{time.time_ns()}"
        run_config_path.write_text(
            json.dumps(
                {
                    "fingerprint": run_fingerprint,
                    "cache_namespace": cache_namespace,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

    cache_root = output_dir / "cache" / cache_namespace
    evidence_store = EvidenceStore(cache_root / "evidence")
    extra_evidence_stores = _sibling_evidence_stores(
        output_dir, cache_root / "evidence"
    )
    metric_cache_path = cache_root / "metrics.json"
    _reuse_metric_cache(output_dir, metric_cache_path)
    pipeline_config = PipelineConfig(
        profile=profile,
        chunk_max_tokens=args.chunk_max_tokens,
        overlap_tokens=args.overlap_tokens,
        device=args.device,
    )
    models = ModelFactory(pipeline_config)
    processor = DocumentProcessor(models.tokenizer, pipeline_config)
    attention = AttentionKeywordExtractor(
        models.qa_model,
        models.tokenizer,
        models.device,
        limit=args.attention_top_k,
    )
    lemmatize, stem = _normalizers(args.language)
    fingerprint_payload = evidence_fingerprint_payload(
        {
            "language": args.language,
            "qa_model": profile.qa_model,
            "sentence_model": profile.sentence_model,
            "chunk_max_tokens": args.chunk_max_tokens,
            "overlap_tokens": args.overlap_tokens,
            "attention_top_k": args.attention_top_k,
            "pipeline_version": "hierarchical-keywords-refit-v2",
        },
        contrast_against_gold=bool(args.contrast_against_gold),
    )
    evidence = []
    for index, document in enumerate(fit_documents, start=1):
        expected = build_evidence_fingerprint(
            doc_id=document.doc_id,
            text=document.text,
            aspect_id=str(args.field_id),
            question=args.question,
            references=references.get(document.doc_id, ()),
            collect_candidates=True,
            fingerprint_payload=fingerprint_payload,
        )
        cached = _load_evidence_reusing_siblings(
            evidence_store,
            extra_evidence_stores,
            document.doc_id,
            expected,
        )
        if cached is None:
            cached = collect_document_evidence(
                doc_id=document.doc_id,
                text=document.text,
                aspect_id=str(args.field_id),
                aspect_reference=args.aspect_reference,
                question=args.question,
                references=references.get(document.doc_id, ()),
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
                collect_candidates=True,
                relaxed_contrast=bool(args.contrast_against_gold),
                contrast_against_gold=bool(args.contrast_against_gold),
                fingerprint_payload=fingerprint_payload,
            )
            evidence_store.save(cached)
        evidence.append(cached)
        if index % 20 == 0 or index == len(fit_documents):
            print(f"Evidence {index}/{len(fit_documents)}", flush=True)

    _, static_strategy = load_static_keywords(args.static_model, args.field_id)
    result = refit_topic_keyword_profiles(
        source_artifact,
        fit_documents,
        evidence,
        models.sentence_encoder,
        config=TrainingConfig(
            language=args.language,
            min_document_support=1,
            max_candidates=80,
            min_keywords=applied_min_keywords,
            max_keywords=int(selected_trial["max_keywords"]),
            max_rescue_keywords=max(12, int(selected_trial["max_keywords"]) + 2),
            evaluation_budget=30 if args.quick else args.evaluation_budget,
            stability_runs=1 if args.quick else args.stability_runs,
            stability_threshold=float(selected_trial["stability_threshold"]),
            selection_policy=str(selected_trial["selection_policy"]),
            harm_cap=0.18,
            seed=args.seed,
            screen_top_k=40,
            strategies=(
                StrategyConfig(
                    str(args.score_chunk_strategy),
                    str(static_strategy.get("choose_cluster_strategy", "weighted_score")),
                    str(static_strategy.get("choose_answer_strategy", "highest_similarity")),
                ),
            ),
        ),
        metric_cache=ExtractionMetricCache(metric_cache_path),
        objective_config=ObjectiveConfig(
            downside_penalty=0.75,
            harm_penalty=0.5,
            fallback_penalty=0.1,
            harm_threshold=0.01,
            confidence_weight=0.25,
            min_activation_rate=0.10,
            bootstrap_seed=args.seed,
        ),
        metric_weights=MetricWeights.exact_match(),
        checkpoint_dir=cache_root / "checkpoints",
        min_fit_documents=2 if args.quick else args.min_fit_documents,
        expected_encoder_name=profile.sentence_model,
    )
    if result.artifact.routing != frozen_routing:
        raise RuntimeError("refit changed frozen routing configuration")
    result = replace(
        result,
        artifact=replace(result.artifact, routing=frozen_routing),
    )
    save_topic_artifact_atomic(result.artifact, output_model)
    coverage_rows = []
    for node in result.artifact.nodes:
        if node.children and node.node_id not in result.coverage:
            continue
        row = result.coverage.get(node.node_id)
        coverage_rows.append(
            {
                "node_id": node.node_id,
                "keywords": " | ".join(
                    item.word for item in node.keyword_profile.keywords
                )
                if node.keyword_profile
                else "",
                "keyword_count": (
                    len(node.keyword_profile.keywords)
                    if node.keyword_profile
                    else 0
                ),
                "objective": (
                    node.keyword_profile.objective if node.keyword_profile else ""
                ),
                "train_documents": row.train_documents if row else "",
                "dev_documents": row.dev_documents if row else "",
                "fit_documents": row.fit_documents if row else "",
                "fit_train_documents": row.fit_train_documents if row else "",
                "fit_dev_documents": row.fit_dev_documents if row else "",
                "candidate_count": row.candidate_count if row else "",
                "selected_keyword_count": row.selected_keyword_count if row else "",
                "covered_train_documents": row.covered_train_documents if row else "",
                "train_document_coverage": row.train_document_coverage if row else "",
                "keyword_floor_reached": row.keyword_floor_reached if row else "",
                "skipped": result.skipped_nodes.get(node.node_id, ""),
            }
        )
    _write_csv(output_dir / "refit_keyword_coverage.csv", coverage_rows)
    refit_manifest = {
        "source_tuned_model": str(tuned_model),
        "refit_model": str(output_model),
        "keyword_trial": selected_trial,
        "refit_min_keywords": int(args.refit_min_keywords),
        "applied_min_keywords": applied_min_keywords,
        "contrast_against_gold": bool(args.contrast_against_gold),
        "score_chunk_strategy": str(args.score_chunk_strategy),
        "routing": asdict(frozen_routing),
        "routing_preserved": True,
        "fit_documents": len(fit_documents),
        "trained_nodes": len(result.outcomes),
        "skipped_nodes": dict(sorted(result.skipped_nodes.items())),
        "run_fingerprint": run_fingerprint,
        "cache_namespace": cache_namespace,
    }
    (output_dir / "refit_manifest.json").write_text(
        json.dumps(refit_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return refit_manifest


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        manifest = run(args)
    except (OSError, RuntimeError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
