from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one M2 anchor, found {count}: {old[:120]!r}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def add_runtime_phase_to_training_config() -> None:
    path = "sheet/training_config.py"
    replace_once(
        path,
        'PLASTIC_TRAINING_CONFIG_FIELDS = (\n'
        '    "plastic__enabled",\n'
        '    "plastic__coarse_phase",\n',
        'PLASTIC_TRAINING_CONFIG_FIELDS = (\n'
        '    "plastic__enabled",\n'
        '    "plastic__runtime_phase",\n'
        '    "plastic__coarse_phase",\n',
    )
    replace_once(
        path,
        '    plastic__enabled: bool = False\n'
        '    plastic__coarse_phase: str = "disabled"\n',
        '    plastic__enabled: bool = False\n'
        '    plastic__runtime_phase: str = "fine"\n'
        '    plastic__coarse_phase: str = "disabled"\n',
    )
    replace_once(
        path,
        '        if not isinstance(self.plastic__enabled, bool):\n'
        '            raise ValueError(f"plastic__enabled must be bool; got {self.plastic__enabled!r}")\n'
        '        if not isinstance(self.plastic__do_learn_layer_count, bool):\n',
        '        if not isinstance(self.plastic__enabled, bool):\n'
        '            raise ValueError(f"plastic__enabled must be bool; got {self.plastic__enabled!r}")\n'
        '        if self.plastic__runtime_phase not in {"coarse", "fine"}:\n'
        '            raise ValueError(\n'
        '                "plastic__runtime_phase must be coarse or fine; "\n'
        '                f"got {self.plastic__runtime_phase!r}"\n'
        '            )\n'
        '        if not isinstance(self.plastic__do_learn_layer_count, bool):\n',
    )


def integrate_fresh_factory() -> None:
    path = "run_thog2_lifecycle.py"
    replace_once(
        path,
        'from sheet.run_manifest import write_run_manifest\n'
        'from sheet.training_config import TrainingConfig\n',
        'from sheet.run_manifest import write_run_manifest\n'
        'from sheet.plastic_depth_fresh_state import build_fresh_training_state\n'
        'from sheet.training_config import TrainingConfig\n',
    )
    replace_once(
        path,
        '    train_tokens = core.load_tokens(dataset_dir / "train.bin")\n'
        '    validation_tokens = core.load_tokens(dataset_dir / "val.bin")\n'
        '    if context["mode"] == "fresh":\n'
        '        trainer = core.OwtTrainer(training_config, train_tokens, validation_tokens)\n'
        '    else:\n',
        '    train_tokens = core.load_tokens(dataset_dir / "train.bin")\n'
        '    validation_tokens = core.load_tokens(dataset_dir / "val.bin")\n'
        '    fresh_state = None\n'
        '    if context["mode"] == "fresh":\n'
        '        # vvv THOG PLASTIC fresh runs and every future COARSE trial share one deterministic step-zero constructor; the disabled path remains byte-for-byte on the established constructor\n'
        '        if training_config.plastic__enabled:\n'
        '            fresh_state = build_fresh_training_state(\n'
        '                trainer_factory=core.OwtTrainer,\n'
        '                resolved_config=training_config,\n'
        '                train_tokens=train_tokens,\n'
        '                validation_tokens=validation_tokens,\n'
        '                phase="fine",\n'
        '                active_layer_count=int(training_config.plastic__initial_active_layers),\n'
        '                instrumentation_namespace="fine",\n'
        '            )\n'
        '            trainer = fresh_state.trainer\n'
        '        else:\n'
        '            trainer = core.OwtTrainer(training_config, train_tokens, validation_tokens)\n'
        '        # ^^^ THOG\n'
        '    else:\n',
    )
    replace_once(
        path,
        '        trainer.close()\n'
        '\n'
        '\n'
        'if __name__ == "__main__":\n',
        '        trainer.close()\n'
        '        if fresh_state is not None:\n'
        '            fresh_state.trainer = None\n'
        '\n'
        '\n'
        'if __name__ == "__main__":\n',
    )


def main() -> None:
    add_runtime_phase_to_training_config()
    integrate_fresh_factory()


if __name__ == "__main__":
    main()
