from pathlib import Path


def test_stage8_top_level_wrappers_exist() -> None:
    for wrapper in ["current_scruffy_train_OWT.sh", "current_scruffy_inference_OWT.sh", "current_dreedle_train_OWT.sh", "current_dreedle_inference_OWT.sh"]:
        assert Path(wrapper).exists()
