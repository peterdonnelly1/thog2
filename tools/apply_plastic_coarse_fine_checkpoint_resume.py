from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    path = ROOT / "sheet/trainer_checkpoint_resume.py"
    text = path.read_text(encoding="utf-8")
    old = (
        '        restore_rng_state(payload["rng_state"])\n'
        '        trainer.distributed.barrier()\n'
    )
    new = (
        '        restore_rng_state(payload["rng_state"])\n'
        '        # vvv THOG restore optional COARSE/FINE phase state without changing legacy checkpoint semantics\n'
        '        trainer.plastic_coarse_fine_state = payload.get("plastic_coarse_fine_state")\n'
        '        trainer.plastic_coarse_provenance = payload.get("plastic_coarse_fine_state")\n'
        '        # ^^^ THOG\n'
        '        trainer.distributed.barrier()\n'
    )
    count = text.count(old)
    if count < 1:
        raise RuntimeError("trainer checkpoint restore anchor was not found")
    path.write_text(text.replace(old, new), encoding="utf-8")


if __name__ == "__main__":
    main()
