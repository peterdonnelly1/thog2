# vvv THOG evidence tests use actual SQLite copy rows, explicit clocks and immutable semantic IDs
import copy
import json
import sqlite3
import zipfile
from types import SimpleNamespace
import pytest
from tests.test_premat_processing import _synthetic_nsys_database
from sheet.premat_processing import normalize_nsys_sqlite, processing_operation_range, processing_capture_scope
from sheet.processing_cpu_evidence import copy_activities, cpu_task_rows, extend_processing, update_timing_overlay


def report():
    task={"snapshot_id":"snapshot","cpu_task_id":"task","cpu_matrix_id":"matrix","family":"O","layer_index":0,"optimizer_step":1,"micro_step":0,"phase":"cpu_preparation","start_ns":8000,"end_ns":9000}
    uses=[{"event":"matrix_use","snapshot_id":"snapshot","cpu_matrix_id":"matrix","matrix_use_id":str(i),"optimizer_step":2,"micro_step":i,"phase":"original_forward" if i<2 else "checkpoint_recompute","final_outcome":"FULL HIT","targeted":True} for i in range(3)]
    return {"cpu_configuration":{"premat_materialisation_device":"cpu_and_gpu"},"cpu_runtime":{"staging_peak_bytes":256,"staging_limit_bytes":512},"cpu_lifecycle":[{**task,"event":"cpu_matrix_ready"},*uses,{"event":"matrix_use","optimizer_step":2,"snapshot_id":"snapshot","matrix_use_id":"excluded","phase":"original_forward","targeted":False,"final_outcome":"NOT TARGETED"}]}


def test_actual_copy_rows_keep_raw_clocks_censoring_ids_and_unknown_bandwidth(tmp_path):
    with sqlite3.connect(":memory:") as db:
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER,end INTEGER,copyKind INTEGER,bytes INTEGER,streamId INTEGER,contextId INTEGER,deviceId INTEGER,correlationId INTEGER)")
        db.executemany("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,?,?,?,?,?,?)",[(500,2000,2,4096,9,1,0,2),(2000,4000,1,4096,9,1,0,3),(9000,11000,1,4096,9,1,0,4)])
        rows,covered=copy_activities(db,{"CUPTI_ACTIVITY_KIND_MEMCPY"},{3:{"upload_id":"upload","snapshot_id":"snapshot","cpu_matrix_id":"matrix"}},1000,10000)
    assert covered and len(rows)==3
    assert rows[0]["raw_start_ns"]==500 and rows[0]["start_us"]==0 and rows[0]["left_censored"]
    assert rows[0]["full_transfer_bandwidth_gb_s"] is None and not rows[0]["logical_identity_known"]
    assert rows[1]["upload_id"]=="upload" and rows[1]["full_transfer_bandwidth_gb_s"]>0
    assert rows[2]["right_censored"]


def test_unknown_copy_table_is_unknown_not_zero():
    with sqlite3.connect(":memory:") as db:assert copy_activities(db,set(),{},0,100)==([],False)


def test_cpu_clock_unknown_preserves_precursors_without_fake_positions():
    rows=cpu_task_rows(report()["cpu_lifecycle"],{},1000,10000)
    assert len(rows)==1 and rows[0]["start_us"] is None and not rows[0]["clock_alignment_known"]
    assert rows[0]["start_ns"]==8000 and rows[0]["cpu_service_us"]==1


def test_cpu_work_counted_once_while_reused_uses_and_exclusions_remain_separate(tmp_path):
    data={"metadata":{"capture":{"optimizer_update":2,"host_nvtx_lower_ns":10000,"host_nvtx_upper_ns":10000},"run":{},"warnings":[],"files":{}},"intervals":[],"stream_resources":[{"start_us":0,"end_us":10,"attribution_state":"IDLE"}]}
    copyrow={"start_us":1,"end_us":3,"direction":"H2D","bytes":256}
    extend_processing(data,tmp_path,"artifact_",report(),[copyrow],True,1000,11000)
    s=data["cpu_summary"]
    assert s["unique_cpu_matrices"]==1 and s["eligible_matrix_uses"]==3 and s["full_hits"]==3 and s["not_targeted_uses"]==1
    assert s["cpu_service_ms"]==0.001 and s["cpu_matrix_use_counts"]["matrix"]==3
    assert data["cpu_tasks"][0]["precursor"] and data["stream_resources"][0]["attribution_state"]=="IDLE" and data["stream_resources"][0]["copy_activity_present"]
    assert data["premat_compatibility"]["reason"]=="N/A: CPU materialisation"
    assert all((tmp_path/name).exists() and name.startswith("artifact_") for name in data["metadata"]["files"].values())


def test_normalizer_preserves_kernel_counts_metrics_and_bundles_four_cpu_downloads(tmp_path):
    database=tmp_path/"trace.sqlite";_synthetic_nsys_database(database)
    with sqlite3.connect(database) as db:
        db.execute("CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY (start INTEGER,end INTEGER,copyKind INTEGER,bytes INTEGER,streamId INTEGER,correlationId INTEGER)")
        db.execute("INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (5000,6000,1,256,9,99)")
    gpu=normalize_nsys_sqlite(database,tmp_path/"gpu",capture_frequency_hz=10000)
    cpu=normalize_nsys_sqlite(database,tmp_path/"cpu",capture_frequency_hz=10000,handoff={"run_name":"cpu_artifact","config":{"premat_materialisation_device":"cpu_and_gpu"}},capture_metadata={"optimizer_update":2},cpu_report=report())
    assert gpu["intervals"]==cpu["intervals"] and gpu["samples"]==cpu["samples"]
    assert gpu["metadata"]["kernel_owner_counts"]==cpu["metadata"]["kernel_owner_counts"]
    assert gpu["metadata"]["schema_version"]==4 and cpu["metadata"]["schema_version"]==5
    files=cpu["metadata"]["files"]
    with zipfile.ZipFile(tmp_path/"cpu"/files["bundle"]) as bundle:
        assert all(files[key] in bundle.namelist() for key in ("transfers","cpu_tasks","cpu_memory","cpu_lifecycle"))


def test_missing_cpu_report_is_unknown_even_when_copy_export_available(tmp_path):
    database=tmp_path/"trace.sqlite";_synthetic_nsys_database(database)
    data=normalize_nsys_sqlite(database,tmp_path/"cpu",capture_frequency_hz=10000,handoff={"config":{"premat_materialisation_device":"cpu_and_gpu"}})
    assert data["cpu_summary"]["full_hits"] is None and data["cpu_summary"]["eligible_matrix_uses"] is None
    assert data["cpu_summary"]["cuda_copy_coverage"]=="unknown"


def test_update_cpu_overlay_does_not_add_to_exclusive_host_phases():
    runtime=SimpleNamespace(report=report)
    payload=update_timing_overlay(runtime,2,10000)
    assert payload["cpu_overlay"]["additive"] is False and len(payload["cpu_overlay"]["lifecycle"])==5
    assert payload["cpu_overlay"]["tasks"][0]["precursor"]


def test_copy_auxiliary_thread_does_not_change_capture_thread_nvtx_depth(monkeypatch):
    import threading
    import sheet.premat_processing as module
    monkeypatch.setattr(module,"_capture_active",True)
    monkeypatch.setattr(module,"_capture_thread",threading.get_ident())
    monkeypatch.setattr(module,"_capture_stack_depth",0)
    labels=[]
    monkeypatch.setattr(module,"_nvtx_push",labels.append)
    monkeypatch.setattr(module,"_nvtx_pop",lambda:None)
    def copy():
        with processing_operation_range("COPY","matrix_h2d",snapshot_id="s",upload_id="u"):assert module._capture_stack_depth==0
    worker=threading.Thread(target=copy);worker.start();worker.join()
    assert module._capture_stack_depth==0 and "upload_id=u" in labels[0]
# ^^^ THOG
