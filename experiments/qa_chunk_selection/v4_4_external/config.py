from __future__ import annotations
from dataclasses import asdict, dataclass
from pathlib import Path
from experiments.qa_chunk_selection.io_util import PROJECT_ROOT

SEEDS=(42,43,44)
WEIGHT_GRID=(0.0,0.25,0.5,1.0,2.0)

@dataclass(frozen=True)
class DatasetSpec:
    key:str; language:str; base_artifact:Path; b0_model:str; semantic_model:str; k:int=5

DEFAULT_SPECS={
 "scimdix": DatasetSpec("scimdix","ru",PROJECT_ROOT/"artifacts/qa_chunk_selection_external_data_20260913/scimdix","DeepPavlov/rubert-base-cased-sentence","DeepPavlov/rubert-base-cased-sentence"),
 "scier_384": DatasetSpec("scier_384","en",PROJECT_ROOT/"artifacts/qa_chunk_selection_scier_384_50_20260913/scier","cross-encoder/ms-marco-MiniLM-L6-v2",str(PROJECT_ROOT/"scripts/models_processing/models/eng_sentence_transformer_model")),
 "scier_128": DatasetSpec("scier_128","en",PROJECT_ROOT/"artifacts/qa_chunk_selection_external_data_20260913/scier","cross-encoder/ms-marco-MiniLM-L6-v2",str(PROJECT_ROOT/"scripts/models_processing/models/eng_sentence_transformer_model")),
}

@dataclass(frozen=True)
class Config:
    output_dir:Path
    datasets:tuple[str,...]=( "scimdix","scier_384")
    device:str="auto"; epochs:int=5; patience:int=2; batch_size:int=16
    learning_rate:float=2e-5; max_length:int=512; n_bootstrap:int=5000
    resume:bool=False; smoke:bool=False
    def specs(self):
        unknown=set(self.datasets)-set(DEFAULT_SPECS)
        if unknown or not self.datasets: raise ValueError(f"unknown/empty datasets: {sorted(unknown)}")
        return tuple(DEFAULT_SPECS[x] for x in self.datasets)
    def validate(self):
        self.specs()
        if self.epochs<1 or self.patience<1 or self.batch_size<1: raise ValueError("training parameters must be positive")
        for spec in self.specs():
            required=(spec.base_artifact/"COMPLETED.json",spec.base_artifact/"data/dataset.jsonl",spec.base_artifact/"data/v3_cv_splits.json",spec.base_artifact/"questions_q5.json")
            missing=[str(x) for x in required if not x.is_file()]
            if missing and not self.smoke: raise FileNotFoundError(f"incomplete {spec.key} base artifact: {missing}")
    def to_dict(self):
        row=asdict(self); row["output_dir"]=str(self.output_dir.resolve()); row["datasets"]=list(self.datasets)
        row["specs"]=[{**asdict(x),"base_artifact":str(x.base_artifact.resolve())} for x in self.specs()]
        row["seeds"]=list(SEEDS); row["weight_grid"]=list(WEIGHT_GRID); return row
