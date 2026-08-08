from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    text = target.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(
            f"{path}: expected one telemetry anchor, found {count}: {old[:120]!r}"
        )
    target.write_text(text.replace(old, new, 1), encoding="utf-8")


def add_coarse_training_loss_history() -> None:
    replace_once(
        "sheet/plastic_depth_coarse.py",
        '    validation_losses: Tuple[float, ...] = ()\n'
        '    training_elapsed_seconds: Optional[float] = None\n',
        '    validation_losses: Tuple[float, ...] = ()\n'
        '    training_losses: Tuple[float, ...] = ()\n'
        '    training_elapsed_seconds: Optional[float] = None\n',
    )
    replace_once(
        "sheet/plastic_depth_coarse.py",
        '            if not all(math.isfinite(float(value)) for value in self.validation_losses):\n'
        '                raise ValueError("COARSE validation losses must be finite")\n'
        '            if self.training_elapsed_seconds is None or not math.isfinite(self.training_elapsed_seconds) or self.training_elapsed_seconds <= 0.0:\n',
        '            if not all(math.isfinite(float(value)) for value in self.validation_losses):\n'
        '                raise ValueError("COARSE validation losses must be finite")\n'
        '            if len(self.training_losses) != self.training_steps:\n'
        '                raise ValueError(\n'
        '                    "successful COARSE trials require one training loss per step"\n'
        '                )\n'
        '            if not all(math.isfinite(float(value)) for value in self.training_losses):\n'
        '                raise ValueError("COARSE training losses must be finite")\n'
        '            if self.training_elapsed_seconds is None or not math.isfinite(self.training_elapsed_seconds) or self.training_elapsed_seconds <= 0.0:\n',
    )
    replace_once(
        "sheet/plastic_depth_coarse_runner.py",
        '    _synchronize(trainer)\n'
        '    started = clock()\n'
        '    try:\n'
        '        for local_step in range(1, n_steps + 1):\n',
        '    _synchronize(trainer)\n'
        '    started = clock()\n'
        '    training_losses = []\n'
        '    try:\n'
        '        for local_step in range(1, n_steps + 1):\n',
    )
    replace_once(
        "sheet/plastic_depth_coarse_runner.py",
        '            completed = int(trainer.state.completed_updates)\n'
        '            if completed != local_step:\n',
        '            training_loss = float(metrics["training_loss"])\n'
        '            training_losses.append(training_loss)\n'
        '            completed = int(trainer.state.completed_updates)\n'
        '            if completed != local_step:\n',
    )
    replace_once(
        "sheet/plastic_depth_coarse_runner.py",
        '                    f"layers={state.active_layer_count:<4d} "\n'
        '                    f"loss={float(metrics[\'training_loss\']):.6f}"\n',
        '                    f"layers={state.active_layer_count:<4d} "\n'
        '                    f"loss={training_loss:.6f}"\n',
    )
    replace_once(
        "sheet/plastic_depth_coarse_runner.py",
        '            status="success",\n'
        '            validation_losses=validation_losses,\n'
        '            training_elapsed_seconds=training_elapsed,\n',
        '            status="success",\n'
        '            validation_losses=validation_losses,\n'
        '            training_losses=tuple(training_losses),\n'
        '            training_elapsed_seconds=training_elapsed,\n',
    )
    replace_once(
        "sheet/plastic_depth_coarse_runner.py",
        '            status="failed",\n'
        '            training_elapsed_seconds=(\n',
        '            status="failed",\n'
        '            training_losses=tuple(training_losses),\n'
        '            training_elapsed_seconds=(\n',
    )


def add_phase_qualified_telemetry() -> None:
    path = "sheet/wandb_telemetry.py"
    replace_once(
        path,
        '        for metric in (\n'
        '            "tokens/*",\n'
        '            "time/*",\n'
        '            "train/*",\n'
        '            "val/*",\n'
        '            "optim/*",\n'
        '            "perf/*",\n'
        '            "model/*",\n'
        '            "resource/*",\n'
        '            "gpu/*",\n'
        '            "sheet/*",\n'
        '        ):\n'
        '            define_metric(metric, step_metric="optimizer/update")\n'
        '        self.module = module\n',
        '        for metric in (\n'
        '            "tokens/*",\n'
        '            "time/*",\n'
        '            "train/*",\n'
        '            "val/*",\n'
        '            "optim/*",\n'
        '            "perf/*",\n'
        '            "model/*",\n'
        '            "resource/*",\n'
        '            "gpu/*",\n'
        '            "sheet/*",\n'
        '        ):\n'
        '            define_metric(metric, step_metric="optimizer/update")\n'
        '        define_metric("fine/update")\n'
        '        define_metric("fine/*", step_metric="fine/update")\n'
        '        self.module = module\n',
    )
    replace_once(
        path,
        '    def log_event(self, event: str, payload: Mapping[str, Any]) -> None:\n'
        '        if not self.enabled or self.backend == "none":\n'
        '            return\n'
        '        metrics = _event_metrics(event, payload)\n'
        '        if not metrics:\n'
        '            return\n'
        '        step = int(metrics["optimizer/update"])\n'
        '        metrics.update(self.sampler.sample(step))\n'
        '        self._log_scalars(metrics, step)\n'
        '\n'
        '    def _log_scalars(self, metrics: Mapping[str, Any], step: int) -> None:\n',
        '    def log_event(self, event: str, payload: Mapping[str, Any]) -> None:\n'
        '        if not self.enabled or self.backend == "none":\n'
        '            return\n'
        '        metrics = _event_metrics(event, payload)\n'
        '        if not metrics:\n'
        '            return\n'
        '        step = int(metrics["optimizer/update"])\n'
        '        metrics.update(self.sampler.sample(step))\n'
        '        if bool(self.config.get("plastic__enabled", False)):\n'
        '            metrics["fine/update"] = step\n'
        '            for name, value in tuple(metrics.items()):\n'
        '                if name in {"optimizer/update", "fine/update"}:\n'
        '                    continue\n'
        '                metrics[f"fine/{name.replace(\'/\', \'_\')}"] = value\n'
        '        self._log_scalars(metrics, step)\n'
        '\n'
        '    def log_plastic_coarse_fine(self, provenance: Mapping[str, Any]) -> None:\n'
        '        if not self.enabled or self.backend == "none":\n'
        '            return\n'
        '        for trial in provenance.get("trials", ()):\n'
        '            trial_index = int(trial["trial_index"])\n'
        '            axis = f"coarse/trial_{trial_index}/step"\n'
        '            loss_name = f"coarse/trial_{trial_index}/training_loss"\n'
        '            if self.run is not None:\n'
        '                define_metric = (\n'
        '                    self.run.define_metric\n'
        '                    if hasattr(self.run, "define_metric")\n'
        '                    else self.module.define_metric\n'
        '                )\n'
        '                define_metric(axis)\n'
        '                define_metric(f"coarse/trial_{trial_index}/*", step_metric=axis)\n'
        '            for local_step, loss in enumerate(trial.get("training_losses", ()), start=1):\n'
        '                metrics = {\n'
        '                    axis: local_step,\n'
        '                    "coarse/trial_index": trial_index,\n'
        '                    f"coarse/trial_{trial_index}/layers": int(trial["layers"]),\n'
        '                    loss_name: float(loss),\n'
        '                }\n'
        '                scalars = _scalar_metrics(metrics)\n'
        '                if self.run is not None:\n'
        '                    self.run.log(scalars)\n'
        '                if self.writer is not None:\n'
        '                    self.writer.add_scalar(loss_name, float(loss), local_step)\n'
        '            summary_step = max(1, int(trial.get("training_steps", 1)))\n'
        '            summary = {\n'
        '                axis: summary_step,\n'
        '                f"coarse/trial_{trial_index}/mean_validation_loss": trial.get("mean_validation_loss"),\n'
        '                f"coarse/trial_{trial_index}/validation_loss_std": trial.get("validation_loss_std"),\n'
        '                f"coarse/trial_{trial_index}/seconds_per_step": trial.get("seconds_per_step"),\n'
        '                f"coarse/trial_{trial_index}/tokens_per_second": trial.get("tokens_per_second"),\n'
        '                f"coarse/trial_{trial_index}/score": trial.get("score"),\n'
        '            }\n'
        '            scalars = _scalar_metrics(summary)\n'
        '            if self.run is not None:\n'
        '                self.run.log(scalars)\n'
        '            if self.writer is not None:\n'
        '                for name, value in scalars.items():\n'
        '                    if name != axis:\n'
        '                        self.writer.add_scalar(name, value, summary_step)\n'
        '        selected_layers = provenance.get("selected_layers")\n'
        '        if selected_layers is not None:\n'
        '            self._log_scalars(\n'
        '                {"coarse/selected_layers": int(selected_layers)},\n'
        '                0,\n'
        '            )\n'
        '\n'
        '    def _log_scalars(self, metrics: Mapping[str, Any], step: int) -> None:\n',
    )
    replace_once(
        "run_thog2_lifecycle.py",
        '            telemetry.add_initial_summary(trainer.parameter_report)\n'
        '\n'
        '        gathered_lifecycle = trainer.distributed.all_gather_object(\n',
        '            telemetry.add_initial_summary(trainer.parameter_report)\n'
        '            coarse_telemetry = lifecycle.get("plastic_coarse_fine")\n'
        '            if isinstance(coarse_telemetry, Mapping):\n'
        '                telemetry.log_plastic_coarse_fine(coarse_telemetry)\n'
        '\n'
        '        gathered_lifecycle = trainer.distributed.all_gather_object(\n',
    )


def main() -> None:
    add_coarse_training_loss_history()
    add_phase_qualified_telemetry()


if __name__ == "__main__":
    main()
