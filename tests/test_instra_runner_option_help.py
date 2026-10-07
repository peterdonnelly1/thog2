# vvv THOG audit visible option help and preserve open-ended registered compressor versions
import importlib.util
import json
from pathlib import Path

import pytest

from sheet.bases import BASIS_REGISTRY
from thog_grid_runner import CATALOGUE


def test_every_visible_field_has_help_and_a_typed_or_explicit_option_description():
    for key, spec in CATALOGUE.items():
        if spec.get("ui_hidden") or spec.get("kind") in {"manual", "automatic"}:
            continue
        assert spec.get("help"), key
        assert spec.get("choices") or spec.get("options_help") or spec["type"] in {"int","float","flag","bool_value"}, key


@pytest.mark.parametrize("field",["--basis-version","DEPTH.compressor_version","WIDTH.compressor_version","--hyperblock-compressor-version"])
def test_registered_versions_and_equivalence_are_present_without_restricting_parameterised_versions(field):
    spec = CATALOGUE[field]
    assert "auto" in spec["options_help"]
    for definition in BASIS_REGISTRY.definitions():
        assert definition.version in spec["options_help"]
    assert "_wN_o500" in spec["options_help"]
    assert "equivalent to DCT-II up to ordering and signs" in spec["help"]
    assert not spec.get("choices")


def test_help_enrichment_is_idempotent_and_keeps_boolean_controls_typed():
    path = Path(__file__).parents[1]/"tools/build_runner_catalogue_help.py"
    module_spec = importlib.util.spec_from_file_location("runner_help_builder",path)
    module = importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
    catalogue = json.loads(json.dumps(CATALOGUE));module.enrich_options(catalogue)
    initial = json.loads(json.dumps(catalogue));module.enrich_options(catalogue)
    assert catalogue == initial
    assert catalogue["--reset-optimizer"]["type"] == "flag"
    assert catalogue["--direct-factorised-hyperblock-mlp"]["type"] == "flag"
    assert catalogue["--direct-factorised-hyperblock-mlp"]["boolean_optional"]
    assert "options_help" not in catalogue["--reset-optimizer"]
# ^^^ THOG
