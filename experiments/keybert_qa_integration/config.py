"""All experimental flags and numeric parameters live here, not in scoring functions."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ModeFlags:
    """Independent on/off switches for one evaluated method."""

    use_attention: bool = False
    use_diff: bool = False
    use_keybert: bool = False
    use_keybert_filter: bool = False
    use_keybert_chunk_score: bool = False
    use_answer_consensus: bool = False
    use_keybert_clusters: bool = False
    use_global_local: bool = False
    filter_mode: str = "none"
    keybert_scope: str = "document"
    attention_source: str = "per_doc"
    mix_diff_independent: bool = False

    def is_baseline(self) -> bool:
        return not (
            self.use_attention
            or self.use_diff
            or self.use_keybert_chunk_score
            or self.use_answer_consensus
            or self.use_keybert_clusters
            or self.use_global_local
            or self.use_keybert_filter
        )


@dataclass(frozen=True)
class ExperimentParams:
    """Numeric knobs. Functions must read these instead of literals."""

    keybert_top_k: int = 10
    keybert_extract_k: int = 30
    keybert_ngram_low: int = 1
    keybert_ngram_high: int = 2
    keybert_use_mmr: bool = True
    keybert_diversity: float = 0.5
    keybert_representation: str = "R2"
    keybert_topic_variant: str = "weighted"
    keybert_mean_top_n: int = 3
    attention_weight: float = 1.0
    diff_weight: float = 1.0
    keybert_weight: float = 0.2
    qa_weight: float = 0.0
    keybert_sim_threshold: float = 0.5
    filter_floor: float = 0.25
    global_alpha: float = 0.5
    local_beta: float = 0.5
    consensus_qa_weight: float = 1.0
    consensus_attention_weight: float = 1.0
    consensus_keybert_weight: float = 0.2
    cluster_k: int = 4
    cluster_sim_threshold: float = 0.3
    cluster_top_n: int = 2
    attention_top_chunks: int = 5
    attention_top_keywords: int = 20
    keep_unmatched: bool = True
    similarity_threshold: float = 0.75
    cluster_strategy: str = "weighted_score"
    answer_strategy: str = "combined_score"
    static_weight_ratio: float = 0.5
    hard_negative_ratios: tuple[float, ...] = (0.10, 0.25, 0.50, 0.75)
    alignment_lambdas: tuple[float, ...] = (0.0, 0.01, 0.05, 0.1, 0.2)
    top_k_grid: tuple[int, ...] = (5, 10, 15, 20, 30)
    keybert_weight_grid: tuple[float, ...] = (0.1, 0.2, 0.4)
    topic_variant_grid: tuple[str, ...] = ("max", "mean_top_n", "weighted")
    filter_mode_grid: tuple[str, ...] = ("none", "soft", "hard_threshold", "semantic")
    cluster_k_grid: tuple[int, ...] = (3, 4)
    bootstrap_samples: int = 1000
    gold_substring_min_chars: int = 4


@dataclass(frozen=True)
class ModeSpec:
    name: str
    display_name: str
    flags: ModeFlags
    core_table: bool = False
    experiment: str = ""
    description: str = ""


def default_params(*, quick: bool = False) -> ExperimentParams:
    if not quick:
        return ExperimentParams()
    return replace(
        ExperimentParams(),
        keybert_top_k=10,
        keybert_extract_k=10,
        top_k_grid=(10,),
        keybert_weight_grid=(0.2,),
        topic_variant_grid=("weighted",),
        filter_mode_grid=("none", "soft"),
        cluster_k_grid=(4,),
        attention_top_chunks=3,
        attention_top_keywords=10,
        bootstrap_samples=200,
    )


def core_mode_specs() -> tuple[ModeSpec, ...]:
    return (
        ModeSpec(
            "qa_baseline",
            "QA baseline",
            ModeFlags(),
            core_table=True,
            experiment="core",
            description="Extractive QA with AnswerAggregator; no keywords, no KeyBERT.",
        ),
        ModeSpec(
            "qa_attention",
            "QA + attention",
            ModeFlags(use_attention=True, attention_source="per_doc"),
            core_table=True,
            experiment="core",
            description="Per-document attention keywords, weight_ratio=1.0.",
        ),
        ModeSpec(
            "qa_attention_diff",
            "QA + attention + diff",
            ModeFlags(
                use_attention=True,
                use_diff=True,
                attention_source="static",
            ),
            core_table=True,
            experiment="core",
            description="Frozen global static dictionary, equal_weight_score_diff (0.5).",
        ),
        ModeSpec(
            "qa_keybert",
            "QA + KeyBERT",
            ModeFlags(
                use_keybert=True,
                use_keybert_chunk_score=True,
            ),
            core_table=True,
            experiment="core",
            description="Control: KeyBERT topic score only, no attention.",
        ),
        ModeSpec(
            "qa_attention_keybert",
            "QA + attention + KeyBERT",
            ModeFlags(
                use_attention=True,
                use_keybert=True,
                use_keybert_chunk_score=True,
                attention_source="per_doc",
            ),
            core_table=True,
            experiment="exp2",
            description="Mix per-doc attention chunk scores with KeyBERT topic prior.",
        ),
        ModeSpec(
            "qa_attention_diff_keybert",
            "QA + attention + diff + KeyBERT",
            ModeFlags(
                use_attention=True,
                use_diff=True,
                use_keybert=True,
                use_keybert_chunk_score=True,
                attention_source="static",
            ),
            core_table=True,
            experiment="exp3",
            description="Mix current fullest static att+diff channel with KeyBERT.",
        ),
    )


def extra_mode_specs() -> tuple[ModeSpec, ...]:
    """Priority-1/2 variants beyond the six-row core table."""
    return (
        ModeSpec(
            "qa_attention_soft_filter",
            "QA + attention + KeyBERT soft filter",
            ModeFlags(
                use_attention=True,
                use_keybert=True,
                use_keybert_filter=True,
                filter_mode="soft",
                attention_source="per_doc",
            ),
            experiment="exp4",
            description="new_att = att * (0.5 + 0.5 * max_cosine_to_KeyBERT).",
        ),
        ModeSpec(
            "qa_attention_hard_filter",
            "QA + attention + KeyBERT threshold filter",
            ModeFlags(
                use_attention=True,
                use_keybert=True,
                use_keybert_filter=True,
                filter_mode="hard_threshold",
                attention_source="per_doc",
            ),
            experiment="exp4",
            description="Below-threshold keywords are downweighted, never zeroed.",
        ),
        ModeSpec(
            "qa_attention_semantic_reweight",
            "QA + attention + KeyBERT semantic reweight",
            ModeFlags(
                use_attention=True,
                use_keybert=True,
                use_keybert_filter=True,
                filter_mode="semantic",
                attention_source="per_doc",
            ),
            experiment="exp4",
            description="Same formula as soft filter; listed separately per prompt.",
        ),
        ModeSpec(
            "qa_attention_global_local",
            "QA + attention global×local",
            ModeFlags(
                use_attention=True,
                use_keybert=True,
                use_global_local=True,
                attention_source="per_doc",
            ),
            experiment="exp5",
            description="att * (1+α*global_freq) * (1+β*local_KeyBERT_sim); freq from train only.",
        ),
        ModeSpec(
            "qa_answer_consensus",
            "QA answer consensus (qa+att+KeyBERT)",
            ModeFlags(
                use_attention=True,
                use_keybert=True,
                use_keybert_chunk_score=True,
                use_answer_consensus=True,
                attention_source="per_doc",
            ),
            experiment="exp8",
            description="answer_score = α*qa_conf + β*att + γ*keybert after minmax.",
        ),
        ModeSpec(
            "qa_attention_keybert_clusters",
            "QA + attention + KeyBERT clusters",
            ModeFlags(
                use_attention=True,
                use_keybert=True,
                use_keybert_chunk_score=True,
                use_keybert_clusters=True,
                attention_source="per_doc",
            ),
            experiment="exp1",
            description="Assign attention keywords to KeyBERT phrase clusters; score chunks by top clusters.",
        ),
        ModeSpec(
            "qa_three_way_mix",
            "QA + att + diff + KeyBERT (independent channels)",
            ModeFlags(
                use_attention=True,
                use_diff=True,
                use_keybert=True,
                use_keybert_chunk_score=True,
                attention_source="per_doc",
                mix_diff_independent=True,
            ),
            experiment="exp3",
            description="Independent minmax mix of per-doc att, static diff, and KeyBERT.",
        ),
    )


def ablation_mode_specs() -> tuple[ModeSpec, ...]:
    """Leave-one-out around independent att / diff / KeyBERT channels."""
    return (
        ModeSpec(
            "ablate_att",
            "ablation: attention",
            ModeFlags(use_attention=True, attention_source="per_doc", mix_diff_independent=True),
            experiment="ablation",
        ),
        ModeSpec(
            "ablate_diff",
            "ablation: diff",
            ModeFlags(use_diff=True, mix_diff_independent=True),
            experiment="ablation",
        ),
        ModeSpec(
            "ablate_keybert",
            "ablation: KeyBERT",
            ModeFlags(use_keybert=True, use_keybert_chunk_score=True),
            experiment="ablation",
        ),
        ModeSpec(
            "ablate_att_diff",
            "ablation: attention + diff",
            ModeFlags(
                use_attention=True,
                use_diff=True,
                attention_source="per_doc",
                mix_diff_independent=True,
            ),
            experiment="ablation",
        ),
        ModeSpec(
            "ablate_att_keybert",
            "ablation: attention + KeyBERT",
            ModeFlags(
                use_attention=True,
                use_keybert=True,
                use_keybert_chunk_score=True,
                attention_source="per_doc",
            ),
            experiment="ablation",
        ),
        ModeSpec(
            "ablate_diff_keybert",
            "ablation: diff + KeyBERT",
            ModeFlags(
                use_diff=True,
                use_keybert=True,
                use_keybert_chunk_score=True,
                mix_diff_independent=True,
            ),
            experiment="ablation",
        ),
        ModeSpec(
            "ablate_att_diff_keybert",
            "ablation: attention + diff + KeyBERT",
            ModeFlags(
                use_attention=True,
                use_diff=True,
                use_keybert=True,
                use_keybert_chunk_score=True,
                attention_source="per_doc",
                mix_diff_independent=True,
            ),
            experiment="ablation",
        ),
    )


def params_to_dict(params: ExperimentParams) -> dict[str, Any]:
    payload = asdict(params)
    for key, value in list(payload.items()):
        if isinstance(value, tuple):
            payload[key] = list(value)
    return payload


def flags_to_dict(flags: ModeFlags) -> dict[str, Any]:
    return asdict(flags)


def all_mode_specs() -> tuple[ModeSpec, ...]:
    return (*core_mode_specs(), *extra_mode_specs(), *ablation_mode_specs())


def spec_by_name(name: str, specs: Sequence[ModeSpec] | None = None) -> ModeSpec:
    mapping: Mapping[str, ModeSpec] = {item.name: item for item in (specs or all_mode_specs())}
    if name not in mapping:
        raise KeyError(f"unknown mode {name}")
    return mapping[name]
