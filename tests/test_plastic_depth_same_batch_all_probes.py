# vvv THOG
from __future__ import annotations

import copy
import os
import subprocess
from pathlib import Path

import pytest
import torch

import run_thog2_owt_core as runner
from sheet import plastic_depth_same_batch_all_probes_patch as same_batch
from sheet.trainer import SharedTrainer
from tests.stage3_test_support import token_splits
from tests.test_plastic_depth import plastic_training_config


@pytest.fixture(autouse=True)
def _isolated_same_batch_environment(monkeypatch):
    monkeypatch.delenv(same_batch._RUNTIME_ENV, raising=False)
    monkeypatch.delenv(same_batch._EXPLICIT_ENV, raising=False)
    try:
        yield
    finally:
        # vvv THOG the runtime flag deliberately survives for one real process; tests must not leak that process-global selection into unrelated suites
        os.environ.pop(same_batch._RUNTIME_ENV, None)
        os.environ.pop(same_batch._EXPLICIT_ENV, None)
        # ^^^ THOG


def _same_batch_trainer(*, window_size: int = 2, max_updates: int = 8):
    same_batch._set_runtime_enabled(True)
    train_tokens, validation_tokens = token_splits(length=2048)
    config = plastic_training_config(
        plastic__layers_to_sample=None,
        plastic__do_learn_layer_count=True,
        plastic__initial_layer_count=3,
        plastic__max_permitted_layers=5,
        plastic__layer_count_objective="lowest_loss",
        plastic__layer_count_update_brake=0,
        plastic__layer_count_probe__probe_every_n_steps=1,
        plastic__layer_count_probe__window_size_as_number_of_probes=window_size,
        plastic__layer_count_probe_noise_lambda=1.0e9,
        plastic__layer_count_probe__number_of_sampled_valid_tokens=16,
        gradient_accumulation_steps=1,
        batch_size=4,
        block_size=8,
        max_updates=max_updates,
        warmup_updates=0,
        checkpoint_segment_size=0,
        device="cpu",
        dtype="float32",
    )
    return SharedTrainer(config, train_tokens, validation_tokens), train_tokens, validation_tokens


def _same_batch_events(trainer: SharedTrainer, name: str):
    return [event for event in trainer.events if event.name == name]


def test_public_flag_is_exact_and_persists_only_when_enabled():
    parser = runner.build_parser()
    arguments = parser.parse_args(
        [
            "--model-type",
            "sheet",
            "--o-depth",
            "3",
            "--plastic__enabled",
            "--plastic__do_learn_layer_count",
            "--plastic__initial_layer_count",
            "3",
            "--plastic__max_permitted_layers",
            "5",
            "--plastic__layer_count__same_batch_all_probes",
        ]
    )
    config = runner.config_from_arguments(arguments)

    assert arguments.plastic__layer_count__same_batch_all_probes is True
    assert config.plastic__layer_count__same_batch_all_probes is True
    assert config.persistent_dict()[same_batch._CONFIG_KEY] is True
    assert config.compact_identity()["plastic_depth"][same_batch._CONFIG_KEY] is True
    assert same_batch._PUBLIC_OPTION in parser.format_help()

    with pytest.raises(SystemExit):
        runner.build_parser().parse_args(
            ["--model-type", "sheet", "--plastic-layer-count-same-batch-all-probes"]
        )


# vvv THOG reproduce the public wrapper failure where the Python-native same-batch flag reached getopts before the wrapper's -- delimiter
def test_train_owt_wrapper_routes_same_batch_with_wall_time_controls_after_separator() -> None:
    repository_root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            "bash",
            "./train_OWT.sh",
            "-h",
            "--plastic__layer_count__same_batch_all_probes",
            "--plastic__wall_time_equivalent_time_gain_discount",
            "0.9",
            "--plastic__wall_time_equivalent_time_gain_loss_rate_window",
            "64",
            "--plastic__wall_time_equivalent_time_gain_loss_rate_min_observations",
            "16",
        ],
        cwd=repository_root,
        text=True,
        capture_output=True,
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "Unknown option: --" not in output
    assert same_batch._PUBLIC_OPTION in output
# ^^^ THOG


def test_false_mode_preserves_existing_metadata_shape():
    same_batch._set_runtime_enabled(False)
    train_tokens, validation_tokens = token_splits(length=1024)
    config = plastic_training_config()
    trainer = SharedTrainer(config, train_tokens, validation_tokens)
    try:
        assert config.plastic__layer_count__same_batch_all_probes is False
        assert same_batch._CONFIG_KEY not in config.persistent_dict()
        payload = trainer.checkpoint_payload()
        assert same_batch._CHECKPOINT_STATE_KEY not in payload
        assert same_batch._CONFIG_KEY not in payload["trainer_config"]
    finally:
        trainer.close()


def test_dedicated_evidence_probe_is_no_grad_and_does_not_advance_training_batch_stream():
    trainer, _train_tokens, _validation_tokens = _same_batch_trainer(window_size=2)
    parameter_before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.raw_model.named_parameters()
    }
    train_generator_before = trainer.batch_source.train_generator.get_state().clone()
    optimizer_state_before = copy.deepcopy(trainer.optimizer.state_dict())

    try:
        context = trainer._begin_plastic_depth_inline_update()
        assert context is not None
        assert context["plastic_same_batch_precomputed"] is True
        assert context["plastic_same_batch_window_ordinal"] == 1
        assert context["plastic_same_batch_window_complete"] is False
        assert trainer.batch_source.training_trace() == ()
        torch.testing.assert_close(
            trainer.batch_source.train_generator.get_state(),
            train_generator_before,
            rtol=0.0,
            atol=0.0,
        )
        assert trainer.optimizer.state_dict() == optimizer_state_before
        for name, parameter in trainer.raw_model.named_parameters():
            torch.testing.assert_close(
                parameter.detach(),
                parameter_before[name],
                rtol=0.0,
                atol=0.0,
                msg=name,
            )
            assert parameter.grad is None, name
    finally:
        trainer._clear_plastic_depth_inline_update()
        trainer.close()


def test_window_reuses_one_batch_then_retires_on_stay_and_starts_fresh():
    trainer, _train_tokens, _validation_tokens = _same_batch_trainer(window_size=2)
    try:
        first = trainer.train_one_update()
        assert first["plastic_active_layers"] == 3.0
        state_after_first = copy.deepcopy(same_batch._window_state(trainer))
        assert state_after_first["active"]["probe_count"] == 1
        assert state_after_first["active"]["probe_sequences"] == [1]

        second = trainer.train_one_update()
        assert second["plastic_active_layers"] == 3.0
        state_after_second = copy.deepcopy(same_batch._window_state(trainer))
        assert state_after_second["active"] is None
        assert state_after_second["last_window_id"] == 1
        assert trainer.state.plastic_depth_probe_histories == {}

        third = trainer.train_one_update()
        assert third["plastic_active_layers"] == 3.0
        state_after_third = copy.deepcopy(same_batch._window_state(trainer))
        assert state_after_third["active"]["window_id"] == 2
        assert state_after_third["active"]["probe_count"] == 1
        assert state_after_third["active"]["probe_sequences"] == [3]

        committed = _same_batch_events(trainer, "plastic_depth_same_batch_probe_committed")
        assert len(committed) == 3
        assert committed[0].payload["batch_digest"] == committed[1].payload["batch_digest"]
        assert committed[2].payload["batch_digest"] != committed[1].payload["batch_digest"]
        assert committed[0].payload["probe_provenance"] == (1,)
        assert committed[1].payload["probe_provenance"] == (1, 2)
        assert committed[2].payload["probe_provenance"] == (3,)

        decisions = _same_batch_events(trainer, "plastic_depth_same_batch_window_decision")
        assert len(decisions) == 1
        assert decisions[0].payload["disposition"] == "stay"
        assert decisions[0].payload["probe_provenance"] == (1, 2)

        assert len(trainer.batch_source.training_trace()) == 3
        audits = trainer.plastic_depth_count_audit
        assert audits[0]["probe_batch_digest"] == audits[1]["probe_batch_digest"]
        assert audits[0]["sampled_tokens_by_rank"] == audits[1]["sampled_tokens_by_rank"]
        assert audits[1]["probe_window_complete"] is True
        assert audits[1]["histories_after_window_retirement"] == {}
    finally:
        trainer.close()


def test_partial_window_checkpoint_resumes_same_batch_and_provenance(tmp_path: Path):
    trainer, train_tokens, validation_tokens = _same_batch_trainer(window_size=3)
    checkpoint_path = tmp_path / "same_batch_partial.pt"
    try:
        trainer.train_one_update()
        before = copy.deepcopy(same_batch._window_state(trainer))
        assert before["active"]["probe_count"] == 1
        trainer.save_checkpoint(checkpoint_path)
    finally:
        trainer.close()

    same_batch._set_runtime_enabled(False)
    resumed = SharedTrainer.from_checkpoint(
        checkpoint_path,
        train_tokens,
        validation_tokens,
    )
    try:
        assert resumed.config.plastic__layer_count__same_batch_all_probes is True
        restored = copy.deepcopy(same_batch._window_state(resumed))
        assert restored == before
        resumed.train_one_update()
        after = copy.deepcopy(same_batch._window_state(resumed))
        assert after["active"]["window_id"] == before["active"]["window_id"]
        assert after["active"]["batch_digest"] == before["active"]["batch_digest"]
        assert after["active"]["probe_count"] == 2
        assert after["active"]["probe_sequences"] == [1, 2]
        committed = _same_batch_events(resumed, "plastic_depth_same_batch_probe_committed")
        assert committed[-1].payload["probe_provenance"] == (1, 2)
        assert committed[-1].payload["batch_digest"] == before["active"]["batch_digest"]
    finally:
        resumed.close()


def test_window_state_rejects_inconsistent_checkpoint_provenance():
    with pytest.raises(ValueError, match="inconsistent probe provenance"):
        same_batch._validate_window_state(
            {
                "version": 1,
                "last_window_id": 0,
                "active": {
                    "window_id": 1,
                    "current_count": 3,
                    "probe_count": 2,
                    "probe_sequences": [1],
                    "global_starts": [10, 20, 30, 40],
                    "batch_digest": "x",
                    "sample_seed_base": 123,
                },
            }
        )
# ^^^ THOG
