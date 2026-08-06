from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one disabled-compatibility anchor, found {count}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def main() -> None:
    replace_once(
        "sheet/plastic_depth_audit_patch.py",
        '    self.state.plastic_depth_count_audit.append(copy.deepcopy(audit))\n'
        '    self._record("plastic_depth_count_audit", **audit)\n',
        '    audit_rows = getattr(self, "plastic_depth_count_audit", None)\n'
        '    if audit_rows is None:\n'
        '        audit_rows = []\n'
        '        self.plastic_depth_count_audit = audit_rows\n'
        '    audit_rows.append(copy.deepcopy(audit))\n'
        '    self._record("plastic_depth_count_audit", **audit)\n',
    )
    replace_once(
        "sheet/trainer_checkpoint_save.py",
        '            # vvv THOG COARSE/FINE phase, trial table, selected count and review-pause remainder are durable resume state\n'
        '            "plastic_coarse_fine_state": getattr(\n'
        '                self,\n'
        '                "plastic_coarse_fine_state",\n'
        '                None,\n'
        '            ),\n'
        '            # ^^^ THOG\n',
        '            # vvv THOG enabled PLASTIC checkpoints alone carry COARSE/FINE phase and replayable count audits; disabled payloads remain unchanged\n'
        '            **(\n'
        '                {\n'
        '                    "plastic_coarse_fine_state": getattr(\n'
        '                        self,\n'
        '                        "plastic_coarse_fine_state",\n'
        '                        None,\n'
        '                    ),\n'
        '                    "plastic_depth_count_audit": list(\n'
        '                        getattr(self, "plastic_depth_count_audit", ())\n'
        '                    ),\n'
        '                }\n'
        '                if self.config.plastic__enabled\n'
        '                else {}\n'
        '            ),\n'
        '            # ^^^ THOG\n',
    )
    path = ROOT / "sheet/trainer_checkpoint_resume.py"
    text = path.read_text(encoding="utf-8")
    old = (
        '        trainer.plastic_coarse_fine_state = payload.get("plastic_coarse_fine_state")\n'
        '        trainer.plastic_coarse_provenance = payload.get("plastic_coarse_fine_state")\n'
    )
    new = old + (
        '        trainer.plastic_depth_count_audit = list(\n'
        '            payload.get("plastic_depth_count_audit", ())\n'
        '        )\n'
    )
    count = text.count(old)
    if count != 2:
        raise RuntimeError(
            "trainer_checkpoint_resume.py: expected two lifecycle restore anchors, "
            f"found {count}"
        )
    path.write_text(text.replace(old, new), encoding="utf-8")


if __name__ == "__main__":
    main()
