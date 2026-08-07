# vvv THOG
from __future__ import annotations

import torch

from sheet.checkpointing import execute_logical_layer_checkpoints


def _block(hidden: torch.Tensor, layer_index: int) -> torch.Tensor:
    return torch.tanh(hidden @ (torch.eye(hidden.shape[-1]) * (1.0 + 0.05 * layer_index)))


def test_prefix_checkpoints_match_one_unshared_reference_chain() -> None:
    torch.manual_seed(811)
    hidden = torch.randn(2, 3, 4, requires_grad=True)
    reference = hidden
    expected = {}
    for layer_index in range(6):
        reference = _block(reference, layer_index)
        if layer_index + 1 in {4, 5, 6}:
            expected[layer_index + 1] = reference

    outputs, report = execute_logical_layer_checkpoints(
        hidden,
        n_layer=8,
        segment_size=0,
        logical_block=_block,
        training=True,
        layer_indices=tuple(range(6)),
        checkpoint_counts=(4, 5, 6),
    )

    assert not report.checkpointing_used
    assert tuple(count for count, _ in outputs) == (4, 5, 6)
    for count, value in outputs:
        torch.testing.assert_close(value, expected[count])


def test_prefix_checkpoints_preserve_checkpointed_selected_gradient() -> None:
    torch.manual_seed(812)
    reference_input = torch.randn(2, 3, 4, requires_grad=True)
    checkpointed_input = reference_input.detach().clone().requires_grad_(True)

    reference = reference_input
    for layer_index in range(5):
        reference = _block(reference, layer_index)
    reference.square().mean().backward()

    outputs, report = execute_logical_layer_checkpoints(
        checkpointed_input,
        n_layer=8,
        segment_size=3,
        logical_block=_block,
        training=True,
        layer_indices=tuple(range(6)),
        checkpoint_counts=(4, 5, 6),
    )
    selected = dict(outputs)[5]
    selected.square().mean().backward()

    assert report.checkpointing_used
    assert report.checkpoint_segments == 4
    torch.testing.assert_close(checkpointed_input.grad, reference_input.grad)


def test_prefix_checkpoints_reject_nonfinal_maximum() -> None:
    hidden = torch.randn(1, 2, 3)
    try:
        execute_logical_layer_checkpoints(
            hidden,
            n_layer=5,
            segment_size=2,
            logical_block=_block,
            training=True,
            layer_indices=(0, 1, 2),
            checkpoint_counts=(1, 2),
        )
    except ValueError as error:
        assert "final checkpoint" in str(error)
    else:
        raise AssertionError("expected a final-checkpoint validation failure")
# ^^^ THOG

from sheet.plastic_depth_inline import PlasticDepthInlineProbeRequest
from sheet.semantic_materializer import ATTENTION_QUERY_WEIGHT
from sheet.training_model import TrainingSheetGPT
from tests.test_plastic_depth import plastic_sheet_config, plastic_training_config


def _plastic_training_model(*, checkpoint_segment_size: int = 0) -> TrainingSheetGPT:
    model = TrainingSheetGPT(
        plastic_sheet_config(
            n_layer=5,
            depth_order=4,
            plastic__layers_to_sample=None,
            plastic__do_learn_layer_count=True,
            plastic__initial_layer_count=3,
            plastic__max_permitted_layers=5,
            dropout=0.0,
        )
    )
    model.set_checkpoint_segment_size(checkpoint_segment_size)
    model.train(True)
    return model


def _gradient_snapshot(model: TrainingSheetGPT):
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


def test_model_inline_probe_runs_shared_chain_once_and_returns_selected_head() -> None:
    torch.manual_seed(813)
    model = _plastic_training_model()
    indices = torch.arange(8, dtype=torch.long).view(1, 8) % model.config.vocab_size
    targets = (indices + 1) % model.config.vocab_size
    calls = []
    original = model._logical_block

    def counted(hidden: torch.Tensor, layer_index: int) -> torch.Tensor:
        calls.append(layer_index)
        return original(hidden, layer_index)

    model._logical_block = counted
    observed = {}

    def select(candidates):
        observed["candidates"] = tuple((count, float(loss.item())) for count, loss in candidates)
        return 3

    request = PlasticDepthInlineProbeRequest(
        candidate_counts=(2, 3, 4),
        sampled_token_indices=torch.tensor([0, 2, 5, 7], dtype=torch.long),
        selector=select,
    )
    logits, loss = model(indices, targets, plastic_depth_probe_request=request)

    assert loss is not None
    assert logits.shape[:2] == indices.shape
    assert calls == [0, 1, 2, 3]
    assert tuple(count for count, _ in observed["candidates"]) == (2, 3, 4)
    assert model.last_plastic_depth_inline_probe_report is not None
    assert model.last_plastic_depth_inline_probe_report.selected_count == 3
    assert model.last_plastic_depth_inline_probe_report.sampled_token_count == 4


def test_inline_selected_gradient_matches_direct_selected_prefix_with_checkpointing() -> None:
    torch.manual_seed(814)
    inline = _plastic_training_model(checkpoint_segment_size=2)
    direct = _plastic_training_model(checkpoint_segment_size=2)
    direct.load_state_dict(inline.state_dict())
    indices = torch.arange(16, dtype=torch.long).view(2, 8) % inline.config.vocab_size
    targets = (indices + 3) % inline.config.vocab_size
    request = PlasticDepthInlineProbeRequest(
        candidate_counts=(2, 3, 4),
        sampled_token_indices=torch.tensor([0, 3, 7, 8, 12, 15], dtype=torch.long),
        selector=lambda candidates: 3,
    )

    inline_logits, inline_loss = inline(
        indices,
        targets,
        plastic_depth_probe_request=request,
    )
    direct_logits, direct_loss = direct(
        indices,
        targets,
        plastic_depth_active_layers_override=3,
    )
    assert inline_loss is not None and direct_loss is not None
    inline_loss.backward()
    direct_loss.backward()

    torch.testing.assert_close(inline_logits, direct_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(inline_loss, direct_loss, rtol=0.0, atol=0.0)
    inline_gradients = _gradient_snapshot(inline)
    direct_gradients = _gradient_snapshot(direct)
    assert set(inline_gradients) == set(direct_gradients)
    for name in inline_gradients:
        inline_gradient = inline_gradients[name]
        direct_gradient = direct_gradients[name]
        if inline_gradient is None or direct_gradient is None:
            assert inline_gradient is None and direct_gradient is None, name
        else:
            torch.testing.assert_close(
                inline_gradient,
                direct_gradient,
                rtol=2.0e-6,
                atol=2.0e-7,
                msg=name,
            )


def test_inline_probe_rejects_selector_non_candidate() -> None:
    model = _plastic_training_model()
    indices = torch.zeros((1, 4), dtype=torch.long)
    request = PlasticDepthInlineProbeRequest(
        candidate_counts=(2, 3, 4),
        sampled_token_indices=None,
        selector=lambda candidates: 5,
    )

    try:
        model(indices, indices, plastic_depth_probe_request=request)
    except RuntimeError as error:
        assert "non-candidate" in str(error)
    else:
        raise AssertionError("expected non-candidate selector rejection")

# vvv THOG direct trainer coverage for the shared first-microstep controller path
import os
from unittest.mock import patch

from sheet.trainer import SharedTrainer
from tests.stage3_test_support import token_splits


def _learned_trainer(**overrides) -> SharedTrainer:
    train_tokens, validation_tokens = token_splits(length=1024)
    values = dict(
        plastic__layers_to_sample=None,
        plastic__do_learn_layer_count=True,
        plastic__initial_layer_count=3,
        plastic__max_permitted_layers=5,
        plastic__layer_count_objective="lowest_loss",
        plastic__layer_count_update_brake=0,
        plastic__layer_count_probe__window_size_as_number_of_probes=8,
        plastic__layer_count_probe_noise_lambda=0.0,
        n_layer=5,
        depth_order=4,
        gradient_accumulation_steps=3,
        max_updates=2,
        warmup_updates=0,
        checkpoint_segment_size=2,
    )
    values.update(overrides)
    return SharedTrainer(plastic_training_config(**values), train_tokens, validation_tokens)


def _select_count(active_layers: int):
    def choose(measurements, **_kwargs):
        measurements = tuple(measurements)
        selected = next(
            measurement
            for measurement in measurements
            if measurement.active_layers == active_layers
        )
        report = tuple(
            {
                "active_layers": measurement.active_layers,
                "validation_loss": measurement.validation_loss,
                "feasible": True,
                # vvv THOG force the requested robust-count outcome independently of stochastic model loss
                "score": 0.0 if measurement.active_layers == active_layers else 1.0,
                # ^^^ THOG
            }
            for measurement in measurements
        )
        return selected, report

    return choose


def test_trainer_uses_inline_probe_then_selected_prefix_for_remaining_microsteps() -> None:
    trainer = _learned_trainer()
    forward_calls = []
    original_forward = trainer.model.forward

    def observed_forward(*args, **kwargs):
        lattice = trainer.raw_model.trajectory.plastic_sampling
        forward_calls.append(
            {
                "persistent_count": lattice.current_active_layers,
                "probe_request": kwargs.get("plastic_depth_probe_request"),
                "override": kwargs.get("plastic_depth_active_layers_override"),
            }
        )
        return original_forward(*args, **kwargs)

    try:
        trainer.model.forward = observed_forward
        with patch.object(
            trainer,
            "_prepare_plastic_depth_for_update",
            side_effect=AssertionError("obsolete external controller was called"),
        ), patch(
            "sheet.trainer_step.choose_plastic_depth_candidate",
            side_effect=_select_count(3),
        ):
            metrics = trainer.train_one_update()

        assert len(forward_calls) == 3
        assert forward_calls[0]["probe_request"] is not None
        assert forward_calls[0]["override"] is None
        assert [row["override"] for row in forward_calls[1:]] == [3, 3]
        assert [row["persistent_count"] for row in forward_calls] == [3, 3, 3]
        assert metrics["plastic_active_layers"] == 3.0
        assert "plastic_transition_report" not in metrics
    finally:
        trainer.close()


def test_trainer_commits_count_only_after_stock_adamw_step() -> None:
    trainer = _learned_trainer(
        gradient_accumulation_steps=2,
        plastic__layer_count_probe__window_size_as_number_of_probes=1,
    )
    forward_counts = []
    optimizer_step_counts = []
    original_forward = trainer.model.forward
    original_optimizer_step = trainer.optimizer.step

    def observed_forward(*args, **kwargs):
        forward_counts.append(
            trainer.raw_model.trajectory.plastic_sampling.current_active_layers
        )
        return original_forward(*args, **kwargs)

    def observed_optimizer_step(*args, **kwargs):
        optimizer_step_counts.append(
            trainer.raw_model.trajectory.plastic_sampling.current_active_layers
        )
        return original_optimizer_step(*args, **kwargs)

    try:
        trainer.model.forward = observed_forward
        trainer.optimizer.step = observed_optimizer_step
        with patch(
            "sheet.trainer_step.choose_plastic_depth_candidate",
            side_effect=_select_count(4),
        ):
            metrics = trainer.train_one_update()

        assert forward_counts == [3, 3]
        assert optimizer_step_counts == [3]
        assert trainer.raw_model.trajectory.plastic_sampling.current_active_layers == 4
        assert metrics["plastic_active_layers"] == 4.0
        decisions = [
            event
            for event in trainer.events
            if event.name == "plastic_depth_count_decision"
        ]
        assert len(decisions) == 1
        decision = decisions[0].payload
        assert decision["previous_active_layers"] == 3
        assert decision["selected_active_layers"] == 4
        assert decision["transition"]["adamw_state_migration_mode"] in {"transform", "reset"}
        assert decision["transition"]["new_active_layers"] == 4
        # vvv THOG the transition sample is post-gauge, scalar-only, transient, and absent from persistent metrics
        sampled_values = trainer._plastic_depth_pending_console_sampled_values
        assert sampled_values is not None
        assert len(sampled_values) == 4
        expected_values = tuple(
            float(
                trainer.raw_model.semantic_materializer.direct_matrix_value(
                    ATTENTION_QUERY_WEIGHT,
                    layer_index,
                    0,
                    0,
                )
                .detach()
                .to(dtype=torch.float64)
                .item()
            )
            for layer_index in range(4)
        )
        torch.testing.assert_close(
            torch.tensor(sampled_values, dtype=torch.float64),
            torch.tensor(expected_values, dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )
        assert "plastic_sampled_values" not in metrics
        assert "sampled_values" not in metrics
        # ^^^ THOG
    finally:
        trainer.close()


def test_inline_probe_sample_positions_are_deterministic_per_update_and_rank() -> None:
    trainer = _learned_trainer()
    targets = torch.arange(512, dtype=torch.long).reshape(16, 32) % 32
    targets[0, :8] = -1
    try:
        first = trainer._plastic_depth_sampled_token_indices(targets)
        second = trainer._plastic_depth_sampled_token_indices(targets)
        torch.testing.assert_close(first, second, rtol=0.0, atol=0.0)
        assert first.numel() == 256
        assert bool((targets.reshape(-1).index_select(0, first) != -1).all().item())

        trainer.state.completed_updates += 1
        next_update = trainer._plastic_depth_sampled_token_indices(targets)
        assert not torch.equal(first, next_update)
    finally:
        trainer.close()


def test_fast_discard_false_retains_maximum_candidate_prefix() -> None:
    with patch.dict(os.environ, {"THOG2_FAST_DISCARD": "false"}):
        trainer = _learned_trainer(
            gradient_accumulation_steps=2,
            plastic__layer_count_probe__window_size_as_number_of_probes=1,
        )
    observed_layer_indices = []
    original_materialize = trainer.raw_model._materialize_block_parameters_for_update

    def observed_materialize(layer_indices):
        observed_layer_indices.append(tuple(layer_indices))
        return original_materialize(layer_indices)

    try:
        trainer.raw_model._materialize_block_parameters_for_update = observed_materialize
        with patch(
            "sheet.trainer_step.choose_plastic_depth_candidate",
            side_effect=_select_count(3),
        ):
            trainer.train_one_update()

        assert observed_layer_indices == [(0, 1, 2, 3)]
        report = trainer.raw_model.update_retained_materialization_report()
        assert report["enabled"]
        assert not report["active"]
        assert trainer.raw_model._plastic_depth_update_layer_count is None
    finally:
        trainer.close()

# vvv THOG robust paired-evidence integration and update-brake coverage
def test_failed_update_discards_uncommitted_paired_evidence() -> None:
    trainer = _learned_trainer(
        gradient_accumulation_steps=1,
        nonfinite_update_policy="skip",
        max_nonfinite_update_skips=1,
    )
    try:
        with patch(
            "sheet.trainer_step.choose_plastic_depth_candidate",
            side_effect=_select_count(4),
        ), patch.object(trainer, "_local_gradients_are_finite", return_value=False):
            metrics = trainer.train_one_update()

        assert metrics["skipped_update"] == 1.0
        assert trainer.state.completed_updates == 0
        assert trainer.state.plastic_depth_probe_histories == {}
        assert trainer.state.plastic_depth_last_count_change_update == -1
        assert trainer._plastic_depth_inline_update_context is None
        assert trainer.raw_model.trajectory.plastic_sampling.current_active_layers == 3
    finally:
        trainer.close()


def test_five_update_brake_collects_evidence_and_enforces_spacing() -> None:
    trainer = _learned_trainer(
        gradient_accumulation_steps=1,
        max_updates=6,
        plastic__layer_count_update_brake=5,
        plastic__layer_count_probe__window_size_as_number_of_probes=5,
    )
    trainer.state.plastic_depth_probe_histories = {
        "3:-1": [1.0, 1.0, 1.0, 1.0],
        "3:+1": [-1.0, -1.0, -1.0, -1.0],
        "3:@LRA": [1.0, 1.0, 1.0, 1.0],
    }
    try:
        with patch(
            "sheet.trainer_step.choose_plastic_depth_candidate",
            side_effect=_select_count(4),
        ):
            trainer.train_one_update()
        assert trainer.raw_model.trajectory.plastic_sampling.current_active_layers == 4
        assert trainer.state.plastic_depth_last_count_change_update == 1

        for expected_update in range(2, 6):
            with patch(
                "sheet.trainer_step.choose_plastic_depth_candidate",
                side_effect=_select_count(3),
            ):
                trainer.train_one_update()
            assert trainer.state.completed_updates == expected_update
            assert trainer.raw_model.trajectory.plastic_sampling.current_active_layers == 4
            assert len(trainer.state.plastic_depth_probe_histories["4:-1"]) == expected_update - 1
            decision = [
                event.payload
                for event in trainer.events
                if event.name == "plastic_depth_count_decision"
            ][-1]
            assert decision["brake_active"] is True

        with patch(
            "sheet.trainer_step.choose_plastic_depth_candidate",
            side_effect=_select_count(3),
        ):
            trainer.train_one_update()
        assert trainer.state.completed_updates == 6
        assert trainer.raw_model.trajectory.plastic_sampling.current_active_layers == 3
        assert trainer.state.plastic_depth_last_count_change_update == 6
    finally:
        trainer.close()
# ^^^ THOG

# ^^^ THOG

# vvv THOG recoverable adjacent N+1 model execution preserves lower candidates after local or distributed CUDA infeasibility
def test_recoverable_upward_oom_preserves_lower_candidates() -> None:
    torch.manual_seed(815)
    model = _plastic_training_model()
    indices = torch.arange(8, dtype=torch.long).view(1, 8) % model.config.vocab_size
    targets = (indices + 1) % model.config.vocab_size
    original = model._logical_block
    calls = []
    prepared = []
    synchronized = []
    observed = {}

    def fail_upward(hidden: torch.Tensor, layer_index: int) -> torch.Tensor:
        calls.append(layer_index)
        if layer_index == 3:
            raise RuntimeError("CUDA out of memory in recoverable N+1 layer")
        return original(hidden, layer_index)

    def synchronize(local_feasible: bool) -> bool:
        synchronized.append(local_feasible)
        return False

    def select(candidates):
        observed["counts"] = tuple(count for count, _ in candidates)
        return 3

    model._logical_block = fail_upward
    request = PlasticDepthInlineProbeRequest(
        candidate_counts=(2, 3, 4),
        sampled_token_indices=torch.tensor([0, 2, 5, 7], dtype=torch.long),
        selector=select,
        recoverable_upward_count=4,
        prepare_recoverable_upward=lambda: prepared.append(True),
        synchronize_recoverable_upward=synchronize,
    )
    _, loss = model(indices, targets, plastic_depth_probe_request=request)

    assert loss is not None
    assert calls == [0, 1, 2, 3]
    assert prepared == [True]
    assert synchronized == [False]
    assert observed["counts"] == (2, 3)
    assert model.last_execution_report.logical_layers == 3
    assert model.last_plastic_depth_inline_probe_report.candidate_counts == (2, 3)


def test_distributed_upward_rejection_discards_successful_local_candidate() -> None:
    torch.manual_seed(816)
    model = _plastic_training_model(checkpoint_segment_size=2)
    indices = torch.arange(8, dtype=torch.long).view(1, 8) % model.config.vocab_size
    targets = (indices + 1) % model.config.vocab_size
    observed = {}

    def select(candidates):
        observed["counts"] = tuple(count for count, _ in candidates)
        return 3

    request = PlasticDepthInlineProbeRequest(
        candidate_counts=(2, 3, 4),
        sampled_token_indices=None,
        selector=select,
        recoverable_upward_count=4,
        prepare_recoverable_upward=lambda: None,
        synchronize_recoverable_upward=lambda local_feasible: False,
    )
    _, loss = model(indices, targets, plastic_depth_probe_request=request)

    assert loss is not None
    assert observed["counts"] == (2, 3)
    assert model.last_execution_report.logical_layers == 3
    assert model.last_plastic_depth_inline_probe_report.candidate_counts == (2, 3)
# ^^^ THOG

# vvv THOG successful recoverable N+1 remains selectable and preserves the direct-prefix gradient
def test_recoverable_upward_success_matches_direct_prefix_gradient() -> None:
    torch.manual_seed(817)
    inline = _plastic_training_model(checkpoint_segment_size=2)
    direct = _plastic_training_model(checkpoint_segment_size=2)
    direct.load_state_dict(inline.state_dict())
    indices = torch.arange(16, dtype=torch.long).view(2, 8) % inline.config.vocab_size
    targets = (indices + 3) % inline.config.vocab_size
    prepared = []
    synchronized = []
    request = PlasticDepthInlineProbeRequest(
        candidate_counts=(2, 3, 4),
        sampled_token_indices=torch.tensor([0, 3, 7, 8, 12, 15], dtype=torch.long),
        selector=lambda candidates: 4,
        recoverable_upward_count=4,
        prepare_recoverable_upward=lambda: prepared.append(True),
        synchronize_recoverable_upward=lambda local_feasible: synchronized.append(local_feasible) or local_feasible,
    )

    inline_logits, inline_loss = inline(indices, targets, plastic_depth_probe_request=request)
    direct_logits, direct_loss = direct(indices, targets, plastic_depth_active_layers_override=4)
    assert inline_loss is not None and direct_loss is not None
    inline_loss.backward()
    direct_loss.backward()

    assert prepared == [True]
    assert synchronized == [True]
    assert inline.last_execution_report.logical_layers == 4
    assert inline.last_plastic_depth_inline_probe_report.candidate_counts == (2, 3, 4)
    torch.testing.assert_close(inline_logits, direct_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(inline_loss, direct_loss, rtol=0.0, atol=0.0)
    inline_gradients = _gradient_snapshot(inline)
    direct_gradients = _gradient_snapshot(direct)
    for name in inline_gradients:
        inline_gradient = inline_gradients[name]
        direct_gradient = direct_gradients[name]
        if inline_gradient is None or direct_gradient is None:
            assert inline_gradient is None and direct_gradient is None, name
        else:
            torch.testing.assert_close(
                inline_gradient,
                direct_gradient,
                rtol=2.0e-6,
                atol=2.0e-7,
                msg=name,
            )
# ^^^ THOG
