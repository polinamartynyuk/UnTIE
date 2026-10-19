"""Language-specific semantic encoder with the established V4.3 disk cache."""
from experiments.qa_chunk_selection.v4_3.encoder import SemanticEncoder
from .contract import NORMALIZATION_VERSION, SEMANTIC_MODEL


def build_encoder(cache_root, *, device, smoke, progress):
    return SemanticEncoder(cache_root=cache_root, device=device, smoke=smoke,
                           model_name=SEMANTIC_MODEL,
                           normalization_version=NORMALIZATION_VERSION,
                           progress=progress)
