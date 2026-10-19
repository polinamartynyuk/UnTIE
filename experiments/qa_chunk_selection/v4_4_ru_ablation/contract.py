"""Frozen search space for the Russian V4.4 ablation cycle."""

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT

EXPERIMENT_ID = "v4_4_ru_ablation_v1"
DEFAULT_INPUT = PROJECT_ROOT / "artifacts/qa_chunk_selection_v4_4_ru_q5_b0_20260912_172431"
SEEDS = (42, 43, 44)
K_VALUES = (3, 4, 5)
WEIGHT_GRID = (0.0, 0.25, 0.5, 1.0, 2.0)
SOURCE_OVERLAP = 0.6
EPSILON_HN = 0.02
HIT1_TOLERANCE = 0.005
ROUTING_QUANTILE = 0.70
MIN_ROUTING_ROWS = 100
MODEL_SPECS = (
    ("bge_m3", "BAAI/bge-reranker-v2-m3"),
)

