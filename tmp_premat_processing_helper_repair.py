# vvv THOG
from pathlib import Path

path = Path("tmp_premat_processing_tranche.py")
text = path.read_text()
old = '''replace_once(\n    "sheet/trainer_step.py",\n    'from .training_model import TrainingModel\\n',\n    'from .training_model import TrainingModel\\n# vvv THOG opt-in bounded Nsight processing capture\\nfrom .premat_processing import processing_capture_scope\\n# ^^^ THOG\\n',\n)'''
new = '''replace_once(\n    "sheet/trainer_step.py",\n    'from .semantic_materializer import ATTENTION_QUERY_WEIGHT                                                                                                  # <<< THOG fixed generated scalar family for transition-only gauge visibility\\n',\n    'from .semantic_materializer import ATTENTION_QUERY_WEIGHT                                                                                                  # <<< THOG fixed generated scalar family for transition-only gauge visibility\\n# vvv THOG opt-in bounded Nsight processing capture\\nfrom .premat_processing import processing_capture_scope\\n# ^^^ THOG\\n',\n)'''
if text.count(old) != 1:
    raise RuntimeError(f"expected one helper import anchor, found {text.count(old)}")
path.write_text(text.replace(old, new, 1))
print("processing helper import anchor repaired")
# ^^^ THOG
