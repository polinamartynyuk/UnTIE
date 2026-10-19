from pathlib import Path
import pytest
from experiments.qa_chunk_selection.v4_4_external.config import Config,DEFAULT_SPECS,SEEDS,WEIGHT_GRID
from experiments.qa_chunk_selection.v4_4_external.io_guard import claim_output
from experiments.qa_chunk_selection.v4_4_external.run import parser


def test_specs_keep_language_models_and_inputs_separate(tmp_path:Path):
    cfg=Config(tmp_path/'out')
    specs={x.key:x for x in cfg.specs()}
    assert tuple(specs)==('scimdix','scier_384')
    assert specs['scimdix'].language=='ru'
    assert specs['scimdix'].b0_model=='DeepPavlov/rubert-base-cased-sentence'
    assert specs['scier_384'].language=='en'
    assert specs['scier_384'].b0_model=='cross-encoder/ms-marco-MiniLM-L6-v2'
    assert specs['scier_384'].base_artifact!=DEFAULT_SPECS['scier_128'].base_artifact
    assert SEEDS==(42,43,44) and 0.0 in WEIGHT_GRID


def test_real_base_artifacts_have_q5_answerable_cv_contract(tmp_path:Path):
    Config(tmp_path/'out').validate()
    for spec in Config(tmp_path/'out').specs():
        assert (spec.base_artifact/'data/dataset.jsonl').is_file()
        assert (spec.base_artifact/'data/v3_cv_splits.json').is_file()
        assert (spec.base_artifact/'questions_q5.json').is_file()


def test_cli_exposes_scier_chunk_ablation():
    args=parser().parse_args(['--datasets','scier_128','--output-dir','artifacts/x'])
    assert args.datasets==['scier_128']


def test_output_is_fresh_and_below_artifacts(tmp_path:Path):
    with pytest.raises(ValueError): claim_output(tmp_path/'outside')
    root=Path(__file__).resolve().parents[1]/'artifacts'/'v44_external_guard_test'
    if root.exists(): pytest.skip('guard test output already exists')
    made=claim_output(root)
    assert made.is_dir()
    with pytest.raises(FileExistsError): claim_output(root)
    made.rmdir()
