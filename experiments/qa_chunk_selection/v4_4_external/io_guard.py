from pathlib import Path
from experiments.qa_chunk_selection.io_util import PROJECT_ROOT

def claim_output(path:Path,resume=False):
    out=Path(path).absolute()
    if any(x.is_symlink() for x in (out,*out.parents)): raise ValueError("output path cannot contain symlinks")
    artifacts=(PROJECT_ROOT/"artifacts").resolve(); resolved=out.resolve()
    if artifacts not in resolved.parents: raise ValueError("output must be below project artifacts/")
    protected=[x.resolve() for x in (PROJECT_ROOT/"datasets",PROJECT_ROOT/"untie",PROJECT_ROOT/"experiments")]
    if any(resolved==x or x in resolved.parents or resolved in x.parents for x in protected): raise ValueError("protected output path")
    if resolved.exists():
        if not resume or (resolved/"COMPLETED.json").exists(): raise FileExistsError(f"fresh output required: {resolved}")
    else: resolved.mkdir(parents=True)
    return resolved
