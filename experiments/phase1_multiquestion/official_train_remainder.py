"""Label the unlabeled official SciREX TRAIN documents.

The 306-document preprocessing table is official train. Multi-question labels
already exist for the old hash-train slice (178 eligible docs). This runner
labels the remaining eligible documents with the same pipeline as former TRAIN
and official DEV:

1. frozen single-question Phase 1 (canonical q0 spans)
2. multi-question Q0 / Q5 / Q3 using official_frozen_thresholds.json
3. optional merge with the existing 178-doc TRAIN tables

It does not rewrite preprocessing, hash-train artifacts, official DEV/TEST,
or frozen_thresholds.json. TEST is not run. Thresholds are not re-selected.

Usage from the project root:

    ./untie_venv/bin/python -m experiments.phase1_multiquestion.official_train_remainder --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Sequence

import pandas as pd

from experiments.phase1_scirex.tables import eligible_instances, load_preprocessing

from .run import main as multi_main

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPROCESS = PROJECT_ROOT / "datasets" / "scirex_aspect_qa_preprocessing"
EXISTING_TRAIN_Q5 = PROJECT_ROOT / "artifacts" / "phase1_multiquestion" / "train_q5"
EXISTING_TRAIN_Q3 = PROJECT_ROOT / "artifacts" / "phase1_multiquestion" / "train_q3"
EXISTING_TRAIN_Q0 = PROJECT_ROOT / "artifacts" / "phase1_multiquestion" / "train_q0"
THRESHOLDS = PROJECT_ROOT / "artifacts" / "phase1_multiquestion" / "official_frozen_thresholds.json"
PHASE1_OUT = PROJECT_ROOT / "artifacts" / "phase1_scirex" / "official_train_remainder"
MULTI_ROOT = PROJECT_ROOT / "artifacts" / "phase1_multiquestion"
CACHE = MULTI_ROOT / "official_train_remainder_qa_prediction_cache.jsonl"
IDS_FILE = MULTI_ROOT / "official_train_remainder_doc_ids.txt"
MANIFEST = MULTI_ROOT / "official_train_remainder_manifest.json"
Q0 = MULTI_ROOT / "official_train_remainder_q0"
Q3 = MULTI_ROOT / "official_train_remainder_q3"
Q5 = MULTI_ROOT / "official_train_remainder_q5"
MERGED_Q0 = MULTI_ROOT / "official_train_q0"
MERGED_Q3 = MULTI_ROOT / "official_train_q3"
MERGED_Q5 = MULTI_ROOT / "official_train_q5"
REMAINDER_COMPARE_DIR = MULTI_ROOT / "official_train_remainder_compare"
MERGED_COMPARE_DIR = MULTI_ROOT / "official_train_merged_compare"

PROTECTED = (
    PROJECT_ROOT / "datasets" / "scirex_aspect_qa_preprocessing",
    PROJECT_ROOT / "datasets" / "scirex_aspect_qa_preprocessing_dev",
    PROJECT_ROOT / "datasets" / "scirex_aspect_qa_preprocessing_test",
    PROJECT_ROOT / "artifacts" / "phase1_scirex" / "train",
    PROJECT_ROOT / "artifacts" / "phase1_scirex" / "train_exact_match",
    PROJECT_ROOT / "artifacts" / "phase1_scirex" / "official_dev",
    EXISTING_TRAIN_Q0,
    EXISTING_TRAIN_Q3,
    EXISTING_TRAIN_Q5,
    MULTI_ROOT / "debug_q5",
    MULTI_ROOT / "dev_q3",
    MULTI_ROOT / "dev_q5",
    MULTI_ROOT / "official_dev_q3",
    MULTI_ROOT / "official_dev_q5",
    MULTI_ROOT / "frozen_thresholds.json",
    MULTI_ROOT / "q0_q3_q5_comparison.json",
)

STEPS = (
    "phase1",
    "q0",
    "q5",
    "q3",
    "compare_remainder",
    "merge",
    "compare_merged",
)

CORE_TABLES = ("qa_chunk_dataset", "qa_question_predictions", "qa_instances")
CLASS_TABLES = (
    "robust_positive_chunks",
    "question_sensitive_chunks",
    "gold_supporting_robust_negative",
    "false_negative_candidates",
    "hard_negative_chunks",
    "easy_negative_chunks",
)


def _ok(output_dir: Path) -> bool:
    stats = output_dir / "statistics.json"
    sanity = output_dir / "sanity_errors.json"
    if not stats.exists() or not sanity.exists():
        return False
    errors = json.loads(sanity.read_text(encoding="utf-8"))
    return errors == []


def remaining_official_train_document_ids(
    *,
    preprocess_dir: Path = PREPROCESS,
    labeled_dir: Path = EXISTING_TRAIN_Q5,
) -> tuple[str, ...]:
    tables = load_preprocessing(preprocess_dir)
    eligible = eligible_instances(tables["aspect_instances"], split="train")
    eligible_ids = {str(item) for item in eligible["document_id"].astype(str)}
    labeled_path = labeled_dir / "qa_instances.parquet"
    if not labeled_path.exists():
        raise FileNotFoundError(f"missing already-labeled TRAIN Q5 table: {labeled_path}")
    labeled = {
        str(item)
        for item in pd.read_parquet(labeled_path)["document_id"].astype(str).tolist()
    }
    overlap = sorted(eligible_ids & labeled)
    if not overlap:
        raise RuntimeError("existing TRAIN Q5 has no documents in official train eligible set")
    remaining = tuple(sorted(eligible_ids - labeled))
    leaked = labeled - eligible_ids
    if leaked:
        raise RuntimeError(f"labeled TRAIN Q5 documents missing from official train tables: {len(leaked)}")
    return remaining


def write_document_id_file(path: Path, document_ids: Sequence[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{item}\n" for item in document_ids), encoding="utf-8")
    return path


def _assert_isolated() -> None:
    targets = (PHASE1_OUT, Q0, Q3, Q5, MERGED_Q0, MERGED_Q3, MERGED_Q5, CACHE, THRESHOLDS)
    protected = {path.resolve() for path in PROTECTED}
    for path in targets:
        resolved = path.resolve()
        if resolved in protected:
            raise SystemExit(f"refusing to write remainder output onto protected path: {resolved}")
        for banned in protected:
            if banned.is_dir() and (banned == resolved or banned in resolved.parents):
                if path in (CACHE, THRESHOLDS):
                    continue
                raise SystemExit(f"refusing to write {resolved} inside protected {banned}")
    if THRESHOLDS.resolve() == (MULTI_ROOT / "frozen_thresholds.json").resolve():
        raise SystemExit("refusing to use hash-split frozen_thresholds.json")


def _should(name: str, args: argparse.Namespace, done: bool) -> bool:
    if args.only and name not in args.only:
        return False
    if done and not args.force:
        return False
    return True


def _run_multi(argv: list[str]) -> None:
    code = multi_main(argv)
    if code:
        raise SystemExit(code)


def _run_phase1(argv: list[str]) -> None:
    from experiments.phase1_scirex.run import main as phase1_main

    code = phase1_main(argv)
    if code:
        raise SystemExit(code)


def _load_thresholds() -> dict:
    if not THRESHOLDS.exists():
        raise SystemExit(f"missing official DEV thresholds: {THRESHOLDS}")
    payload = json.loads(THRESHOLDS.read_text(encoding="utf-8"))
    for key in ("Q0", "Q3", "Q5"):
        if key not in payload or "chosen_threshold" not in payload[key]:
            raise SystemExit(f"{THRESHOLDS} is missing {key}.chosen_threshold")
    return payload


def _concat_parquet(left: Path, right: Path, dest: Path) -> int:
    frames = []
    if left.exists():
        frames.append(pd.read_parquet(left))
    if right.exists():
        frames.append(pd.read_parquet(right))
    if not frames:
        return 0
    merged = pd.concat(frames, ignore_index=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(dest, index=False)
    return int(len(merged))


def merge_labeled_runs(existing_dir: Path, remainder_dir: Path, dest_dir: Path) -> dict:
    existing_inst = pd.read_parquet(existing_dir / "qa_instances.parquet")
    remainder_inst = pd.read_parquet(remainder_dir / "qa_instances.parquet")
    left_ids = set(existing_inst["document_id"].astype(str))
    right_ids = set(remainder_inst["document_id"].astype(str))
    overlap = left_ids & right_ids
    if overlap:
        raise RuntimeError(f"document overlap while merging {existing_dir} and {remainder_dir}: {len(overlap)}")
    dest_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name in (*CORE_TABLES, *CLASS_TABLES):
        counts[name] = _concat_parquet(
            existing_dir / f"{name}.parquet",
            remainder_dir / f"{name}.parquet",
            dest_dir / f"{name}.parquet",
        )
    from .compare import summarize_run

    summary = summarize_run(
        pd.read_parquet(dest_dir / "qa_instances.parquet"),
        pd.read_parquet(dest_dir / "qa_chunk_dataset.parquet"),
    )
    left_thr = json.loads((existing_dir / "experiment_config.json").read_text(encoding="utf-8"))
    right_thr = json.loads((remainder_dir / "experiment_config.json").read_text(encoding="utf-8"))
    payload = {
        "merged_from": [str(existing_dir), str(remainder_dir)],
        "existing_documents": len(left_ids),
        "remainder_documents": len(right_ids),
        "documents_total": len(left_ids | right_ids),
        "table_rows": counts,
        "summary": summary,
        "existing_threshold": left_thr.get("robust_positive_threshold"),
        "remainder_threshold": right_thr.get("robust_positive_threshold"),
        "official_threshold_file": str(THRESHOLDS),
    }
    if payload["existing_threshold"] != payload["remainder_threshold"]:
        raise RuntimeError(
            "cannot merge runs with different robust_positive_threshold: "
            f"{payload['existing_threshold']} vs {payload['remainder_threshold']}"
        )
    (dest_dir / "merge_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    (dest_dir / "sanity_errors.json").write_text("[]\n", encoding="utf-8")
    (dest_dir / "statistics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    shutil.copy2(remainder_dir / "experiment_config.json", dest_dir / "experiment_config.json")
    config = json.loads((dest_dir / "experiment_config.json").read_text(encoding="utf-8"))
    config["merged_from"] = payload["merged_from"]
    config["document_ids"] = []
    config["n_documents_processed"] = payload["documents_total"]
    (dest_dir / "experiment_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    return payload


def _assert_phase1_covers_remainder(document_ids: Sequence[str]) -> None:
    path = PHASE1_OUT / "qa_chunk_dataset.parquet"
    if not path.exists():
        raise SystemExit(f"missing remainder Phase 1 spans: {path}")
    got = {str(item) for item in pd.read_parquet(path)["document_id"].astype(str).tolist()}
    wanted = {str(item) for item in document_ids}
    extra = sorted(got - wanted)
    missing = sorted(wanted - got)
    if extra:
        raise SystemExit(f"Phase 1 remainder contains {len(extra)} unexpected documents")
    if missing:
        raise SystemExit(f"Phase 1 remainder missed {len(missing)} requested documents")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Label remaining official SciREX TRAIN documents with frozen official DEV thresholds."
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", choices=STEPS, nargs="*", default=())
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-compare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _assert_isolated()
    if not PREPROCESS.exists():
        raise SystemExit(f"missing preprocessing: {PREPROCESS}")
    if not EXISTING_TRAIN_Q5.exists():
        raise SystemExit(f"missing already-labeled TRAIN Q5: {EXISTING_TRAIN_Q5}")
    thresholds = _load_thresholds()
    remaining = remaining_official_train_document_ids()
    write_document_id_file(IDS_FILE, remaining)
    MULTI_ROOT.mkdir(parents=True, exist_ok=True)
    tables = load_preprocessing(PREPROCESS)
    eligible = eligible_instances(tables["aspect_instances"], split="train")
    remainder_instances = eligible[eligible["document_id"].astype(str).isin(set(remaining))]
    manifest = {
        "remaining_documents": len(remaining),
        "remaining_eligible_instances": int(len(remainder_instances)),
        "threshold_file": str(THRESHOLDS),
        "thresholds": {key: thresholds[key]["chosen_threshold"] for key in ("Q0", "Q3", "Q5")},
        "document_ids_file": str(IDS_FILE),
        "phase1_output": str(PHASE1_OUT),
        "cache": str(CACHE),
        "note": (
            "Remainder is official SciREX train documents not present in train_q5. "
            "Hash-split DEV labels are not reused. Thresholds are not re-selected."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("=== official TRAIN remainder ===")
    print(f"remaining documents: {len(remaining)}")
    print(f"remaining eligible instances: {len(remainder_instances)}")
    print(f"thresholds: {manifest['thresholds']}")
    print(f"ids: {IDS_FILE}")
    print("does not rewrite hash-train, official DEV/TEST, or preprocessing.")
    print("TEST is not run. --select-threshold is not used.")

    status = [
        ("phase1", _ok(PHASE1_OUT) and (PHASE1_OUT / "qa_chunk_dataset.parquet").exists(), str(PHASE1_OUT)),
        ("q0", _ok(Q0), str(Q0)),
        ("q5", _ok(Q5), str(Q5)),
        ("q3", _ok(Q3), str(Q3)),
        ("compare_remainder", (REMAINDER_COMPARE_DIR / "q0_q3_q5_comparison.json").exists(), str(REMAINDER_COMPARE_DIR)),
        ("merge", _ok(MERGED_Q5) and (MERGED_Q5 / "merge_manifest.json").exists(), str(MERGED_Q5)),
        ("compare_merged", (MERGED_COMPARE_DIR / "q0_q3_q5_comparison.json").exists(), str(MERGED_COMPARE_DIR)),
    ]
    print("status:")
    for name, done, path in status:
        print(f"  [{'done' if done else 'pending':7}] {name:18}  {path}")
    planned = [name for name, done, _ in status if _should(name, args, done)]
    if args.skip_merge:
        planned = [name for name in planned if name not in {"merge", "compare_merged"}]
    if args.skip_compare:
        planned = [name for name in planned if not name.startswith("compare")]
    if args.dry_run:
        print("dry-run planned:", planned or "(nothing)")
        return 0

    common_multi = [
        "--device",
        str(args.device),
        "--split",
        "train",
        "--input-dir",
        str(PREPROCESS),
        "--cache-path",
        str(CACHE),
        "--canonical-predictions",
        str(PHASE1_OUT / "qa_chunk_dataset.parquet"),
        "--threshold-file",
        str(THRESHOLDS),
        "--document-ids-file",
        str(IDS_FILE),
        "--top-n",
        "5",
        "--gap-bootstrap",
        "10",
        "--seed",
        "42",
        "--bm25-aggregate",
        "mean_normalized",
    ]

    if _should("phase1", args, _ok(PHASE1_OUT) and (PHASE1_OUT / "qa_chunk_dataset.parquet").exists()):
        print("\n[1/7] single-question Phase 1 on remainder (canonical q0, same as former TRAIN / official DEV)")
        _run_phase1(
            [
                "--input-dir",
                str(PREPROCESS),
                "--output-dir",
                str(PHASE1_OUT),
                "--split",
                "train",
                "--device",
                str(args.device),
                "--document-ids-file",
                str(IDS_FILE),
                "--match-method",
                "gold_span_containment",
                "--containment-max-extra-tokens",
                "5",
                "--top-n",
                "5",
                "--gap-bootstrap",
                "10",
                "--seed",
                "42",
            ]
        )
        _assert_phase1_covers_remainder(remaining)
    else:
        print("\n[1/7] skip Phase 1")

    if _should("q0", args, _ok(Q0)):
        if not (PHASE1_OUT / "qa_chunk_dataset.parquet").exists():
            raise SystemExit("Q0 needs remainder Phase 1 spans; run step phase1 first")
        _assert_phase1_covers_remainder(remaining)
        print("\n[2/7] remainder Q0 (canonical question, reuse Phase 1 spans, t=1.0)")
        _run_multi(["--mode", "Q0", "--output-dir", str(Q0), *common_multi])
    else:
        print("\n[2/7] skip Q0")

    if _should("q5", args, _ok(Q5)):
        if not (PHASE1_OUT / "qa_chunk_dataset.parquet").exists():
            raise SystemExit("Q5 needs remainder Phase 1 spans; run step phase1 first")
        print("\n[3/7] remainder Q5 (q0 from Phase 1, q1–q4 frozen QA, official t=0.6)")
        _run_multi(["--mode", "Q5", "--output-dir", str(Q5), *common_multi])
    else:
        print("\n[3/7] skip Q5")

    if _should("q3", args, _ok(Q3)):
        print("\n[4/7] remainder Q3 (cache after Q5, official t=2/3)")
        _run_multi(["--mode", "Q3", "--output-dir", str(Q3), *common_multi])
    else:
        print("\n[4/7] skip Q3")

    if args.skip_compare or (args.only and "compare_remainder" not in args.only):
        print("\n[5/7] skip remainder compare")
    elif (REMAINDER_COMPARE_DIR / "q0_q3_q5_comparison.json").exists() and not args.force:
        print("\n[5/7] skip remainder compare")
    else:
        if not (_ok(Q0) and _ok(Q3) and _ok(Q5)):
            raise SystemExit("remainder compare needs Q0, Q3 and Q5")
        print("\n[5/7] compare remainder Q0 / Q3 / Q5")
        _run_multi(
            [
                "--output-dir",
                str(REMAINDER_COMPARE_DIR),
                "--compare",
                f"Q0={Q0}",
                f"Q3={Q3}",
                f"Q5={Q5}",
            ]
        )

    if args.skip_merge or (args.only and "merge" not in args.only):
        print("\n[6/7] skip merge")
    elif _ok(MERGED_Q5) and (MERGED_Q5 / "merge_manifest.json").exists() and not args.force:
        print("\n[6/7] skip merge")
    else:
        if not (_ok(Q0) and _ok(Q3) and _ok(Q5)):
            raise SystemExit("merge needs remainder Q0, Q3 and Q5")
        if not (_ok(EXISTING_TRAIN_Q0) and _ok(EXISTING_TRAIN_Q3) and _ok(EXISTING_TRAIN_Q5)):
            raise SystemExit("merge needs existing train_q0 / train_q3 / train_q5")
        print("\n[6/7] merge remainder with existing hash-train labels")
        merge_labeled_runs(EXISTING_TRAIN_Q0, Q0, MERGED_Q0)
        merge_labeled_runs(EXISTING_TRAIN_Q3, Q3, MERGED_Q3)
        merge_labeled_runs(EXISTING_TRAIN_Q5, Q5, MERGED_Q5)

    if args.skip_compare or args.skip_merge or (args.only and "compare_merged" not in args.only):
        print("\n[7/7] skip merged compare")
    elif (MERGED_COMPARE_DIR / "q0_q3_q5_comparison.json").exists() and not args.force:
        print("\n[7/7] skip merged compare")
    else:
        if not (_ok(MERGED_Q0) and _ok(MERGED_Q3) and _ok(MERGED_Q5)):
            raise SystemExit("merged compare needs official_train_q0 / q3 / q5")
        print("\n[7/7] compare merged official TRAIN Q0 / Q3 / Q5")
        _run_multi(
            [
                "--output-dir",
                str(MERGED_COMPARE_DIR),
                "--compare",
                f"Q0={MERGED_Q0}",
                f"Q3={MERGED_Q3}",
                f"Q5={MERGED_Q5}",
            ]
        )

    print("\nTEST was not run.")
    print("hash-train artifacts were not rewritten.")
    print("official DEV / TEST were not rewritten.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
