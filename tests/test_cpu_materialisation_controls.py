# vvv THOG canonical CLI, dormant GPU identities, Runner conditional dimensions and shared capture controls
import copy
from dataclasses import replace
import json
from pathlib import Path
import pytest
from run_thog2_owt_core import build_parser, config_from_arguments
from sheet.premat_cpu_config import CPU_DEFAULTS, cpu_config_dict, cpu_identity, validate_cpu_arguments
from sheet.training_config import TrainingConfig
from tests.test_dashboard_processing_companion_pairing import _State
import run_thog2_local_dashboard as dashboard


def args(extra=()):
    return build_parser().parse_args(["--model-type","sheet","--geometry-preset","depth","--n-layer","4","--n-head","2","--n-embd","8","--o-depth","3",*extra])


def test_gpu_default_identity_and_model_arguments_have_no_cpu_fields():
    config=TrainingConfig(model_type="thog2_sheet",geometry_preset="depth",n_layer=4,n_embd=8,n_head=2,depth_order=3)
    assert not any(name in config.persistent_dict() for name in CPU_DEFAULTS)
    assert "premat_cpu" not in config.compact_identity_metadata()
    assert not any(name in config.model_arguments() for name in CPU_DEFAULTS)


@pytest.mark.parametrize("name",tuple(key for key in CPU_DEFAULTS if key.startswith("premat_cpu_")))
def test_gpu_cli_rejects_even_explicit_default_cpu_only_controls(name):
    argv=["--"+name,str(CPU_DEFAULTS[name])]
    parsed=args(argv)
    with pytest.raises(ValueError,match="CPU-only"):validate_cpu_arguments(parsed,argv)


def test_cpu_all_controls_reach_model_identity_and_resolved_limits():
    argv=["--premat","enabled","--premat_materialisation_device","cpu_and_gpu","--premat_cpu_preparation","scheduled","--premat_cpu_layer_batch_size","all_layers","--premat_cpu_workers","1","--premat_cpu_threads_per_worker","1","--premat_cpu_transfer_timing","predicted_gemm_start","--premat_cpu_transfer_lead_ms","2","--premat_cpu_staging_limit_mb","0.5","--premat_cpu_checkpoint_replay","enabled"]
    parsed=args(argv);validate_cpu_arguments(parsed,argv)
    config=config_from_arguments(parsed)
    canonical=config.canonical_dict(world_size=1)
    assert canonical["premat_cpu_layer_batch_size"]=="all_layers" and canonical["premat_cpu_all_layer_count"]==4
    assert canonical["premat_cpu_staging_limit_bytes_resolved"]==524288
    assert canonical["premat_cpu_threads_per_worker_resolved"]==1 and canonical["premat_schema_version"]==5
    assert canonical["premat_gpu_timing_active"] is False and canonical["premat_target_offset_active"] is False
    assert cpu_identity(canonical).upper() in config.artifact_name.upper()


def test_inactive_lead_and_conflicting_gpu_timing_are_rejected_explicitly():
    for extra,reason in ((["--premat_cpu_transfer_lead_ms","0"],"explicit"),(["--premat_timing","previous_gemm_leading_edge"],"conflicts")):
        argv=["--premat","enabled","--premat_materialisation_device","cpu_and_gpu",*extra]
        with pytest.raises(ValueError,match=reason):validate_cpu_arguments(args(argv),argv)


def test_cpu_provider_and_experiments_cannot_pair_with_gpu_even_with_manual_artifact_name():
    gpu=_State("260909_scruffy_NSYS___MANUAL")
    cpu=_State("260909_scruffy_NSYS___MANUAL")
    cpu.reader._metadata["config_json"]=json.dumps({"host_label":"scruffy","premat_materialisation_device":"cpu_and_gpu"})
    assert dashboard._processing_pair_key(cpu)!=dashboard._processing_pair_key(gpu)
    different=_State("260909_scruffy_NCU___MANUAL")
    different.reader._metadata["config_json"]=json.dumps({"host_label":"scruffy","premat_materialisation_device":"cpu_and_gpu","premat_cpu_workers":2})
    assert dashboard._processing_pair_key(cpu)!=dashboard._processing_pair_key(different)
# ^^^ THOG

# vvv THOG executable field harness configuration and shared profiler-selector regressions
from types import SimpleNamespace
from tools.benchmark_cpu_materialisation import parser as field_parser, training_config as field_config, optimizer_values

@pytest.mark.parametrize("provider",("off","gpu","cpu_and_gpu"))
def test_field_harness_builds_valid_matched_trainer_configs_without_cuda_allocation(provider):
    options=field_parser().parse_args(["--smoke","--threads","1"])
    config=field_config(options,provider)
    assert config.nonfinite_update_policy=="raise"
    assert config.gradient_accumulation_steps==2 and config.checkpoint_segment_size==2
    assert config.model_seed==171 and config.data_seed==272 and config.dropout==0.1
    assert config.premat==("disabled" if provider=="off" else "enabled")
    assert config.premat_materialisation_device==("cpu_and_gpu" if provider=="cpu_and_gpu" else "gpu")


def test_field_harness_captures_optimizer_steps_and_both_moments():
    import torch
    model=torch.nn.Linear(2,2);optimizer=torch.optim.AdamW(model.parameters())
    model(torch.ones(1,2)).sum().backward();optimizer.step()
    values=optimizer_values(SimpleNamespace(raw_model=model,optimizer=optimizer))
    assert set(values)=={name+"."+kind for name in ("weight","bias") for kind in ("step","exp_avg","exp_avg_sq")}
    assert values["weight.step"].item()==1


@pytest.mark.parametrize("device,full,selectors,update,scope",(
    ("cpu_and_gpu","enable",[],7,"complete optimizer update"),
    ("cpu_and_gpu","enable",["--premat_processing_logging_capture_update","3"],3,"complete optimizer update"),
    ("cpu_and_gpu","enable",["--premat_instra__full_step_timing_capture_and_chart_capture","step","4"],4,"complete optimizer update"),
    ("cpu_and_gpu","enable",["--premat_processing_logging_capture_update","3","--premat_instra__full_step_timing_capture_and_chart_capture","step","3"],3,"complete optimizer update"),
    ("cpu_and_gpu","disable",[],1,"first forward microstep"),
    ("gpu","enable",[],1,"first forward microstep"),
))
def test_public_full_step_selectors_share_one_update_only_for_enabled_cpu_capture(monkeypatch,tmp_path,capsys,device,full,selectors,update,scope):
    import sheet.premat_processing as processing
    monkeypatch.setenv(processing._PROCESSING_CHILD_ENV,"0")
    monkeypatch.setattr(processing.tempfile,"mkdtemp",lambda **kwargs:str(tmp_path))
    monkeypatch.setattr(processing,"_find_nsys",lambda:"mock-nsys")
    commands=[]
    monkeypatch.setattr(processing.subprocess,"run",lambda command,**kwargs:commands.append(command) or SimpleNamespace(returncode=7))
    argv=["--premat_processing_logging","enabled","--premat_materialisation_device",device,"--max-iters","7","--premat_instra__full_step_timing_capture_and_chart",full,*selectors]
    assert processing.maybe_reexec_under_nsys(argv,entrypoint=Path("run_thog2_owt.py"))==7
    output=capsys.readouterr().out
    assert ("capturing update "+str(update)) in output and scope in output
    if scope=="complete optimizer update":
        command=commands[0]
        internal="--processing_logging_capture_update_internal"
        assert command[command.index(internal)+1]==str(update)
        assert command[-3:]==["--premat_instra__full_step_timing_capture_and_chart_capture","step",str(update)]


def test_conflicting_cpu_capture_selectors_fail_before_launch(monkeypatch,tmp_path):
    import sheet.premat_processing as processing
    monkeypatch.setenv(processing._PROCESSING_CHILD_ENV,"0")
    monkeypatch.setattr(processing.tempfile,"mkdtemp",lambda **kwargs:str(tmp_path))
    launches=[]
    monkeypatch.setattr(processing.subprocess,"run",lambda *args,**kwargs:launches.append(args))
    argv=["--premat_processing_logging","enabled","--premat_materialisation_device","cpu_and_gpu","--premat_instra__full_step_timing_capture_and_chart","enable","--premat_instra__full_step_timing_capture_and_chart_capture","step","2","--premat_processing_logging_capture_update","3"]
    with pytest.raises(ValueError,match="selectors must agree"):processing.maybe_reexec_under_nsys(argv,entrypoint=Path("run_thog2_owt.py"))
    assert not launches
# ^^^ THOG

# vvv THOG active profiler launcher must preserve CPU evidence through post-run normalization
@pytest.mark.parametrize("cpu_report_present",(False,True))
def test_active_instra_launcher_passes_cpu_capture_into_normalizer(monkeypatch,tmp_path,cpu_report_present):
    import sheet.premat_processing as processing
    monkeypatch.setenv(processing._PROCESSING_CHILD_ENV,"0")
    monkeypatch.setattr(processing.tempfile,"mkdtemp",lambda **kwargs:str(tmp_path))
    monkeypatch.setattr(processing,"_find_nsys",lambda:"mock-nsys")
    directory=tmp_path/"run"
    (tmp_path/"handoff.json").write_text(json.dumps({"run_directory":str(directory)}))
    (tmp_path/"processing_trace.nsys-rep").write_text("fixture")
    report={"schema_version":5,"cpu_lifecycle":[{"event":"matrix_use","matrix_use_id":"fixture"}]}
    if cpu_report_present:(tmp_path/"cpu_capture.json").write_text(json.dumps(report))
    def launch(command,**kwargs):
        if "export" in command:(tmp_path/"processing_trace.sqlite").write_text("normalized by fixture")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(processing.subprocess,"run",launch)
    normalizations=[]
    def normalize(*args,**kwargs):
        normalizations.append(kwargs)
        return {"metadata":{"files":{"raw_trace":"artifact_trace.nsys-rep","bundle":"artifact_bundle.zip"}}}
    monkeypatch.setattr(processing,"normalize_nsys_sqlite",normalize)
    assert processing.maybe_reexec_under_nsys(["--premat_processing_logging","enabled","--premat_materialisation_device","cpu_and_gpu"],entrypoint=Path("run_thog2_owt.py"))==0
    if cpu_report_present:assert normalizations[0]["cpu_report"]==report
    else:assert "cpu_report" not in normalizations[0]
    assert (directory/"processing"/"artifact_trace.nsys-rep").read_text()=="fixture"
# ^^^ THOG
