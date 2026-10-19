"""Frozen contract for the isolated Russian V4.4 development replay."""

from experiments.qa_chunk_selection.io_util import PROJECT_ROOT
from experiments.qa_chunk_selection.v4.config import FUSION_LAMBDAS

ARTIFACT_ROOT = "artifacts/qa_chunk_selection_v4_4_ru"
DEFAULT_INPUT_ARTIFACT = (
    PROJECT_ROOT / "artifacts/qa_chunk_selection_v4_4_ru_q5_b0_20260912_172431"
)
PROTOCOL = "hypothesis/v4_4_adaptive_reader_consensus_protocol.md"
SEEDS = (42, 43, 44)
K = 5
N_BOOTSTRAP = 5000
MIN_ROUTING_CAL_ROWS = 100
MIN_SUPPORT_CAL_ROWS = 100
SOURCE_OVERLAP = 0.6
EPSILON_HN = 0.02
SEM_FIXED = {"lambda_majority": 0.5, "lambda_semantic": 0.25}
LEX_FIXED = {"lambda_majority": 0.25, "lambda_token_f1": 2.0}
V6_GRID = tuple(float(x) for x in FUSION_LAMBDAS)
SEMANTIC_MODEL = "DeepPavlov/rubert-base-cased-sentence"
NORMALIZATION_VERSION = "unicode-nfc-whitespace-v1"
STATUS = "Русский development replay; результаты не являются оценкой на независимом test."


def contract():
    return {
        "experiment": "V4.4-RU",
        "protocol": PROTOCOL,
        "language": "ru",
        "input": "completed Russian answerable Q5 + three-seed B0 OOF artifact",
        "input_read_only": True,
        "output": ARTIFACT_ROOT,
        "seeds": list(SEEDS),
        "k": K,
        "cv": "document grouped; V6 weights selected on CAL only; routing Q70 uses TRAIN+CAL only",
        "v6_grid": list(V6_GRID),
        "v6_hn_tolerance": EPSILON_HN,
        "sem_fixed": SEM_FIXED,
        "lex_fixed": LEX_FIXED,
        "semantic_model": SEMANTIC_MODEL,
        "rouge_l": "unicode word-token LCS; max over gold answers",
        "qa": "reuse canonical q0 rows from Russian Q5 cache; zero new QA forward passes",
        "bootstrap": "three-seed mean within question, then document-clustered; 5000 replicates",
        "internal_test": "not present and not synthesized",
    }
