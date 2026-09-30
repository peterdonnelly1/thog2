# vvv THOG honour disabled plastic telemetry while retaining unconfigured legacy data
from sheet.local_dashboard_wandb_charts_patch import _plastic_charts_enabled
from thog_grid_runner import CATALOGUE, COMMON
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock


def test_disabled_plastic_configuration_suppresses_chart_groups():
    assert not _plastic_charts_enabled({"plastic__enabled": False, "plastic__coarse_phase": "disabled",
                                       "plastic__do_learn_layer_count": False})
    assert _plastic_charts_enabled({"plastic__enabled": True})
    assert _plastic_charts_enabled({"plastic__coarse_phase": "enabled"})
    assert _plastic_charts_enabled({"plastic__do_learn_layer_count": True})
    assert _plastic_charts_enabled({})


def test_common_fields_retain_original_category_membership():
    for key in ("--log-interval", "--eval-iters", "--eval-interval"):
        assert key in COMMON
        assert CATALOGUE[key]["category"] == "Run Control Parameters"


def test_chart_routes_hide_disabled_plastic_and_preserve_enabled_or_legacy_data(monkeypatch, tmp_path):
    import sheet.local_dashboard_wandb_charts_patch as charts

    class Handler:
        def do_GET(self):
            raise AssertionError("unexpected fallback route")

        def _send_json(self, payload, **kwargs):
            self.response = payload

    configuration = {}
    state = SimpleNamespace(status=lambda: {"configuration": configuration})
    catalogue = SimpleNamespace(state_for_run=lambda name: state)
    plastic = {"name": "plastic", "revision": 1, "charts": [{"id": "plastic/ncols",
               "series": [{"x": [1], "y": [8]}]}]}
    scanner = SimpleNamespace(path=tmp_path/"run.wandb", record_count=1, catching_up=False, error="",
        group_summaries=lambda: [{"name": "train", "chart_count": 1}, {"name": "plastic", "chart_count": 1}],
        group_payload=Mock(return_value=plastic))
    live = SimpleNamespace(path=None, values={"train": {}}, refresh=lambda *args: None,
        summaries=lambda groups, scanner: groups, merge=lambda payload: payload)
    monkeypatch.setattr(charts, "_ScannerCatalog", lambda catalogue: SimpleNamespace(scanner_for=lambda state: scanner))
    monkeypatch.setattr(charts, "LiveLossReader", lambda: live)
    dashboard = SimpleNamespace(_handler_for=lambda catalogue: Handler, HTTPStatus=HTTPStatus)
    charts.install(dashboard)
    handler = dashboard._handler_for(catalogue)()
    for enabled in (False, True, None):
        configuration.clear()
        if enabled is not None:
            configuration["plastic__enabled"] = enabled
        handler.path = "/api/chart-groups?run=fixture"
        handler.do_GET()
        assert {group["name"] for group in handler.response["groups"]} == (
            {"train"} if enabled is False else {"train", "plastic"})
        scanner.group_payload.reset_mock()
        handler.path = "/api/chart-group?run=fixture&group=plastic"
        handler.do_GET()
        if enabled is False:
            assert handler.response["group"]["charts"] == []
            scanner.group_payload.assert_not_called()
        else:
            assert handler.response["group"] == plastic
# ^^^ THOG
