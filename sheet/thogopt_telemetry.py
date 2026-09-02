# vvv THOG
"""Passive, versioned optimizer histories using Instra's existing coupling selection."""
from __future__ import annotations

import os
import json
import time
from pathlib import Path
import torch

from .thogopt import Thogopt
from .thogopt_math import comparison_errors
from . import local_chart_store as storage


def _families():
    from .depth_weight_curves_v2_patch import _CHART_FAMILIES
    return _CHART_FAMILIES


def history_quantities(m, v, *, step, betas, eps, lr):
    b1, b2 = betas
    m_hat = m / (1-b1**step) if step else m
    v_hat = v.clamp_min(0) / (1-b2**step) if step else v.clamp_min(0)
    rms = v_hat.sqrt()
    scaling = 1 / (rms + eps)
    return {"momentum": m_hat, "raw_momentum": m, "second_moment": v_hat,
            "raw_second_moment": v, "rms": rms, "scaling": scaling,
            "adaptive_update": -lr*m_hat*scaling}


def configure_reference(trainer, telemetry):
    from .depth_weight_curves_v2_patch import _selected_scalar_coordinates_v2
    selection = _selected_scalar_coordinates_v2(trainer, telemetry)
    optimizer = trainer.optimizer
    optimizer.configure_reference({name: [r*optimizer.families[name].shape[1]+c for r,c in selection[chart]]
        for chart,name in _families().items() if name in optimizer.families})
    for reference in optimizer.reference_histories.values():
        reference.setdefault("run_origin",int(trainer.state.completed_updates))
    return selection


@torch.no_grad()
def history_snapshot(trainer, telemetry):
    from .dense_weight_curves_patch import _dense_selection, _dense_family_weight, _dense_model
    from .depth_weight_curves_v2_patch import _selected_scalar_coordinates_v2
    optimizer = trainer.optimizer
    compact = isinstance(optimizer, Thogopt)
    if not compact and not (isinstance(optimizer, torch.optim.AdamW) and _dense_model(trainer.raw_model)):
        return {}
    selection = _selected_scalar_coordinates_v2(trainer, telemetry) if compact else _dense_selection(trainer, telemetry)
    layers = trainer.raw_model.config.n_layer
    groups = {p:g for g in optimizer.param_groups for p in g["params"]}
    snapshot = {"schema_version":1, "optimizer_update":int(trainer.state.completed_updates),
        "kind":"thogopt" if compact else "dense_adamw", "layers":layers,
        "state_timing":"post_commit_histories_used_for_post_step_weights", "encoding":"lossless_float64_json_values_from_source_moments", "evaluation_dtype":"torch.float64",
        "full_matrix_cadence":getattr(trainer.config,"instrumentation__optimizer_histories__full_matrix_every_n_steps",0),
        "selection_seed":selection["seed"], "attention_head":selection["attention_head"], "families":{}}
    for chart,name in _families().items():
        coordinates = selection[chart]
        if compact:
            parameter = optimizer.families[name]
            state = optimizer.state.get(parameter)
            if not state: continue
            indices = [r*parameter.shape[1]+c for r,c in coordinates]
            m_coeff = state["exp_avg"][:,indices].detach().cpu().double()
            v_coeff = state["exp_avg_sq"][:,indices].detach().cpu().double()
            q_m, q_v = optimizer.q_m.cpu().double(), optimizer.q_v.cpu().double()
            m, v = q_m@m_coeff, q_v@v_coeff
            group = groups[parameter]
            step = int(state["step"])
            shape = list(parameter.shape[:2])
            extra = {"momentum_coefficients":m_coeff.T.tolist(), "scaling_coefficients":v_coeff.T.tolist(),
                     "q_m":q_m.tolist(), "q_v":q_v.tolist()}
        else:
            width = trainer.raw_model.config.n_embd
            m_rows, v_rows, steps = [], [], []
            for block in trainer.raw_model.transformer.h:
                parameter, offset = _dense_family_weight(block,chart,width)
                state = optimizer.state.get(parameter)
                if not state: break
                m_rows.append(torch.stack([state["exp_avg"][r+offset,c] for r,c in coordinates]).cpu().double())
                v_rows.append(torch.stack([state["exp_avg_sq"][r+offset,c] for r,c in coordinates]).cpu().double())
                steps.append(int(state["step"]))
            if len(m_rows)!=layers or len(set(steps))!=1: continue
            m, v, step = torch.stack(m_rows), torch.stack(v_rows), steps[0]
            group = groups[parameter]
            shape = [width if chart.startswith("attn_") else parameter.shape[0], parameter.shape[1]]
            extra = {}
        settings = {"step":step, "betas":list(group["betas"]), "eps":group["eps"], "lr":group["lr"]}
        values = history_quantities(m,v,**settings)
        family = {"semantic_family":name,"shape":shape,"coordinates":[list(x) for x in coordinates],
                  **settings, **extra, "values":{k:t.T.tolist() for k,t in values.items()}}
        reference = optimizer.reference_histories.get(name) if compact else None
        if reference is not None and all(index in reference["indices"] for index in indices):
            lookup = {index:i for i,index in enumerate(reference["indices"])}
            columns = [lookup[index] for index in indices]
            # A late-start reference is explicitly scoped to its own history origin.
            reference_settings = dict(settings,step=int(reference["step"])-int(reference["origin"]))
            reference_values = history_quantities(reference["exp_avg"][:,columns],reference["exp_avg_sq"][:,columns],**reference_settings)
            family["reference"] = {"kind":"same_gradient_adamw", "origin":int(reference.get("run_origin",reference["origin"])), "optimizer_origin":int(reference["origin"]),
                "step":int(reference["step"]), "full_history":reference["origin"]==0,
                "values":{k:t.T.tolist() for k,t in reference_values.items()}}
            family["errors"] = {key:comparison_errors(value,reference_values[key]) for key,value in values.items()}
            update = values["adaptive_update"]
            oracle = reference_values["adaptive_update"]
            projection = optimizer.weight_basis.cpu().double()@optimizer.weight_analysis.cpu().double()
            family["errors"]["projected_adaptive_update"] = comparison_errors(projection@update,projection@oracle)
        snapshot["families"][chart] = family
    if compact:
        snapshot["diagnostics"] = optimizer.last_step_metrics
        snapshot["history_counts"] = {"momentum":optimizer.momentum_count,"scaling":optimizer.scaling_count}
    return snapshot


@torch.no_grad()
def full_history_snapshot(trainer):
    from .dense_weight_curves_patch import _dense_family_weight
    optimizer = trainer.optimizer
    compact = isinstance(optimizer,Thogopt)
    groups = {p:g for g in optimizer.param_groups for p in g["params"]}
    payload = {"schema_version":1,"optimizer_update":int(trainer.state.completed_updates),
        "kind":"thogopt" if compact else "dense_adamw","layers":trainer.raw_model.config.n_layer,"families":{}}
    for chart,name in _families().items():
        if compact:
            parameter = optimizer.families[name]
            state = optimizer.state.get(parameter)
            if not state: continue
            m,v = state["exp_avg"].cpu().clone(),state["exp_avg_sq"].cpu().clone()
            shape = tuple(parameter.shape[:2])
            extra = {"q_m":optimizer.q_m.cpu(),"q_v":optimizer.q_v.cpu()}
        else:
            m_rows,v_rows,steps = [],[],[]
            width = trainer.raw_model.config.n_embd
            for block in trainer.raw_model.transformer.h:
                parameter,offset = _dense_family_weight(block,chart,width)
                state = optimizer.state.get(parameter)
                if not state: break
                rows = width if chart.startswith("attn_") else parameter.shape[0]
                m_rows.append(state["exp_avg"][offset:offset+rows].cpu().clone())
                v_rows.append(state["exp_avg_sq"][offset:offset+rows].cpu().clone())
                steps.append(int(state["step"]))
            if len(m_rows)!=payload["layers"] or len(set(steps))!=1: continue
            shape = tuple(m_rows[0].shape)
            m,v = torch.stack(m_rows).flatten(1),torch.stack(v_rows).flatten(1)
            extra = {}
        group = groups[parameter]
        payload["families"][chart] = {"shape":shape,"step":int(state["step"]),"betas":group["betas"],
            "eps":group["eps"],"lr":group["lr"],"m":m,"v":v,**extra}
    return payload


def append_history(store, snapshot, *, full=None, history_length=20):
    if not snapshot: return
    step = int(snapshot["optimizer_update"])
    connection = store.connection
    connection.execute("CREATE TABLE IF NOT EXISTS optimizer_histories (optimizer_update INTEGER PRIMARY KEY,payload BLOB NOT NULL,full_path TEXT)")
    relative = None
    if full is not None:
        relative = f"optimizer_histories/step_{step:012d}.pt"
        target = store.path.parent / relative
        target.parent.mkdir(parents=True,exist_ok=True)
        temporary = target.with_suffix(".tmp")
        torch.save(full,temporary)
        temporary.replace(target)
    connection.execute("INSERT OR REPLACE INTO optimizer_histories VALUES (?,?,COALESCE(?,(SELECT full_path FROM optimizer_histories WHERE optimizer_update=?)))",
        (step,storage._encode_payload(snapshot),relative,step))
    obsolete = connection.execute("SELECT optimizer_update,full_path FROM optimizer_histories ORDER BY optimizer_update DESC LIMIT -1 OFFSET ?",(max(1,history_length),)).fetchall()
    for row in obsolete:
        connection.execute("DELETE FROM optimizer_histories WHERE optimizer_update=?",(row["optimizer_update"],))
    store._touch()
    connection.commit()
    for row in obsolete:
        if row["full_path"]: (store.path.parent/row["full_path"]).unlink(missing_ok=True)


def history_rows(path, *, step_min=0, step_max=2**63-1):
    connection = storage._open_database(Path(path),readonly=True)
    try:
        if not connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='optimizer_histories'").fetchone(): return []
        rows = connection.execute("SELECT * FROM optimizer_histories WHERE optimizer_update BETWEEN ? AND ? ORDER BY optimizer_update",(step_min,step_max)).fetchall()
        result = [{**storage._decode_payload(row["payload"]),"full_path":row["full_path"]} for row in rows]
        if any(row.get("schema_version")!=1 for row in result):
            raise ValueError("Unsupported optimizer history schema; update Instra to the capture version")
        return result
    finally:
        connection.close()


def attach_history_telemetry(trainer,telemetry):
    from . import depth_weight_curves_and_observational_probes_patch as weights
    from .dense_weight_curves_patch import _dense_model
    import constants
    if weights._destination()!="local" or int(constants.DEBUG)<=2 or not trainer.distributed.is_primary: return
    supported = isinstance(trainer.optimizer,Thogopt) or (isinstance(trainer.optimizer,torch.optim.AdamW) and _dense_model(trainer.raw_model))
    if not supported: return
    if isinstance(trainer.optimizer,Thogopt):
        configure_reference(trainer,telemetry)
        store = storage.ensure_local_chart_store(telemetry)
        configuration = trainer.optimizer.configuration()
        configuration["weight_basis"] = configuration["weight_basis"].tolist()
        configuration["resources"] = trainer.optimizer.resource_report()
        store.connection.execute("INSERT OR REPLACE INTO metadata(key,value) VALUES (?,?)",("thogopt_configuration",json.dumps(configuration)))
        store.connection.commit()
    cadence = int(getattr(trainer.config,"instrumentation__optimizer_histories__full_matrix_every_n_steps",0))
    original = trainer._timed
    train_step = trainer.train_one_update
    def timed(function):
        if function==train_step and isinstance(trainer.optimizer,Thogopt):
            configure_reference(trainer,telemetry)
        metrics,elapsed = original(function)
        update = int(trainer.state.completed_updates)
        if function==train_step and not metrics.get("skipped_update") and update>0:
            full_due = cadence>0 and update%cadence==0
            if weights._weight_snapshot_due(update) or full_due:
                started = time.perf_counter()
                try:
                    snapshot = history_snapshot(trainer,telemetry)
                    full = full_history_snapshot(trainer) if full_due else None
                    snapshot["capture_seconds"] = time.perf_counter()-started
                    append_history(storage.ensure_local_chart_store(telemetry),snapshot,full=full,history_length=weights._history_length())
                except Exception as error:
                    print(f"THOG2 optimizer history capture failed at step {update}: {error}",flush=True)
        return metrics,elapsed
    trainer._timed = timed
# ^^^ THOG
