from __future__ import annotations

import base64
import gzip
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PAYLOAD = ROOT / ".thog_apply_script.py.gz.b64"
source = gzip.decompress(base64.b64decode(PAYLOAD.read_text(encoding="utf-8"))).decode("utf-8")
namespace = {
    "__file__": str(ROOT / ".thog_apply.py"),
    "__name__": "__main__",
}
exec(compile(source, namespace["__file__"], "exec"), namespace)

for temporary in (
    PAYLOAD,
    ROOT / ".thog_apply_jpeg_like_v1_trigger",
    ROOT / ".github" / "workflows" / "apply_jpeg_like_v1_now.yml",
):
    temporary.unlink(missing_ok=True)

for script in ROOT.glob("*.sh"):
    script.chmod(script.stat().st_mode | 0o111)
