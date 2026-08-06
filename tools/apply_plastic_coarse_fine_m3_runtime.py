from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one M3 runtime anchor, found {count}: {old[:120]!r}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def enforce_fresh_phase_semantics() -> None:
    path = "sheet/plastic_depth_fresh_state.py"
    replace_once(
        path,
        '    changes: Dict[str, Any] = {\n'
        '        "plastic__initial_layer_count": int(active_layer_count),\n'
        '    }\n'
        '    if hasattr(config, "plastic__runtime_phase"):\n'
        '        changes["plastic__runtime_phase"] = phase\n',
        '    changes: Dict[str, Any] = {\n'
        '        "plastic__initial_layer_count": int(active_layer_count),\n'
        '    }\n'
        '    if hasattr(config, "plastic__coarse_phase"):\n'
        '        changes["plastic__coarse_phase"] = "disabled"\n'
        '    if hasattr(config, "plastic__runtime_phase"):\n'
        '        changes["plastic__runtime_phase"] = phase\n',
    )
    replace_once(
        path,
        '    trainer = trainer_factory(fresh_config, train_tokens, validation_tokens)\n'
        '    completed_updates = int(getattr(trainer.state, "completed_updates", 0))\n',
        '    trainer = trainer_factory(fresh_config, train_tokens, validation_tokens)\n'
        '    if phase == "coarse":\n'
        '        trajectory = getattr(trainer.raw_model, "trajectory", None)\n'
        '        lattice = getattr(trajectory, "plastic_sampling", None)\n'
        '        if lattice is None:\n'
        '            close = getattr(trainer, "close", None)\n'
        '            if callable(close):\n'
        '                close()\n'
        '            raise RuntimeError("PLASTIC COARSE trainer has no sampling lattice")\n'
        '        for parameter in lattice.parameters():\n'
        '            parameter.requires_grad_(False)\n'
        '        if int(lattice.current_active_layers) != int(active_layer_count):\n'
        '            close = getattr(trainer, "close", None)\n'
        '            if callable(close):\n'
        '                close()\n'
        '            raise RuntimeError(\n'
        '                "PLASTIC COARSE active count differs from its candidate: "\n'
        '                f"candidate={active_layer_count}, active={lattice.current_active_layers}"\n'
        '            )\n'
        '    completed_updates = int(getattr(trainer.state, "completed_updates", 0))\n',
    )


def disable_coarse_controller_paths() -> None:
    path = "sheet/trainer_step.py"
    replace_once(
        path,
        '    def _begin_plastic_depth_inline_update(self) -> Optional[Dict[str, Any]]:\n'
        '        if not self.config.plastic__enabled or not self.config.plastic__do_learn_layer_count:\n'
        '            return None\n',
        '    def _begin_plastic_depth_inline_update(self) -> Optional[Dict[str, Any]]:\n'
        '        if not self.config.plastic__enabled or not self.config.plastic__do_learn_layer_count:\n'
        '            return None\n'
        '        if getattr(self.config, "plastic__runtime_phase", "fine") == "coarse":\n'
        '            return None\n',
    )
    replace_once(
        path,
        '            if self.config.plastic__do_learn_layer_count:\n'
        '                lattice.record_training_time(\n',
        '            if (\n'
        '                self.config.plastic__do_learn_layer_count\n'
        '                and getattr(self.config, "plastic__runtime_phase", "fine") == "fine"\n'
        '            ):\n'
        '                lattice.record_training_time(\n',
    )
    path = "sheet/plastic_depth_lookahead_patch.py"
    replace_once(
        path,
        'def _begin_plastic_depth_inline_update_with_lookahead(self: Any) -> Optional[Dict[str, Any]]:\n'
        '    if not self.config.plastic__enabled or not self.config.plastic__do_learn_layer_count:\n'
        '        return None\n',
        'def _begin_plastic_depth_inline_update_with_lookahead(self: Any) -> Optional[Dict[str, Any]]:\n'
        '    if not self.config.plastic__enabled or not self.config.plastic__do_learn_layer_count:\n'
        '        return None\n'
        '    if getattr(self.config, "plastic__runtime_phase", "fine") == "coarse":\n'
        '        return None\n',
    )


def main() -> None:
    enforce_fresh_phase_semantics()
    disable_coarse_controller_paths()


if __name__ == "__main__":
    main()
