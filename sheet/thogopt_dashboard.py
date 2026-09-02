# vvv THOG
"""Instra read-only optimizer-history curves and bounded matrix windows."""
from __future__ import annotations

from pathlib import Path
from urllib.parse import parse_qs,urlparse
import numpy as np
import json
import torch

from .thogopt_telemetry import history_rows,history_quantities
from .thogopt_math import comparison_errors

QUANTITIES = ("momentum","raw_momentum","second_moment","raw_second_moment","rms","scaling","adaptive_update")


def _curve(family,quantity,index):
    values = family["values"][quantity][index]
    layers = len(values)
    x = np.arange(1,layers+1,dtype=float)
    if quantity not in ("momentum","raw_momentum","second_moment","raw_second_moment"):
        return x.tolist(),values
    first = quantity in ("momentum","raw_momentum")
    q = family.get("q_m" if first else "q_v")
    coeff = family.get("momentum_coefficients" if first else "scaling_coefficients")
    if q is None or len(q[0])==layers: return x.tolist(),values
    order = len(q[0])
    nodes = np.linspace(-1,1,layers)
    vandermonde = np.polynomial.chebyshev.chebvander(nodes,order-1)
    polynomial = np.linalg.lstsq(vandermonde,np.asarray(q),rcond=None)[0] @ np.asarray(coeff[index])
    dense_x = np.linspace(-1,1,max(layers,256))
    y = np.polynomial.chebyshev.chebval(dense_x,polynomial)
    if quantity in ("momentum","second_moment"):
        y /= 1-family["betas"][0 if first else 1]**family["step"]
    return (1+(dense_x+1)*(layers-1)/2).tolist(),y.tolist()


def history_payload(path,*,quantity="momentum",step_min=0,step_max=2**63-1,reference_path=None,latest=False):
    if quantity not in QUANTITIES: raise ValueError("unknown optimizer history quantity")
    rows = history_rows(path,step_min=step_min,step_max=step_max)
    if latest and rows: rows=rows[-1:]
    reference_rows = {row["optimizer_update"]:row for row in history_rows(reference_path,step_min=step_min,step_max=step_max)} if reference_path else {}
    figures, errors = {}, {}
    for snapshot in rows:
        step = snapshot["optimizer_update"]
        for chart,family in snapshot["families"].items():
            traces = figures.setdefault(chart,{"data":[],"layout":{"xaxis":{"title":"Executed layer"},"yaxis":{"title":quantity.replace('_',' ')},"hovermode":"closest","margin":{"t":25,"r":20,"b":45,"l":70}}})["data"]
            reference = family.get("reference")
            reference_label = f"same gradients, history origin {reference['origin']}" if reference else ""
            if reference_path:
                other = reference_rows.get(step,{}).get("families",{}).get(chart)
                if other is None: reference = None
                elif snapshot["layers"]!=reference_rows[step]["layers"] or family["coordinates"]!=other["coordinates"]:
                    raise ValueError("comparison requires identical layer and coupling coordinates at the same captured step")
                else:
                    reference = {"values":other["values"]}
                    reference_label = "separate training run"
            for index,coordinate in enumerate(family["coordinates"]):
                label = f"r{coordinate[0]} c{coordinate[1]} · step {step}"
                x,y = _curve(family,quantity,index)
                fraction = (step-rows[0]["optimizer_update"])/max(1,rows[-1]["optimizer_update"]-rows[0]["optimizer_update"])
                colour = f"hsl({(index*137+215)%360},65%,{70-35*fraction:.1f}%)"
                traces.append({"x":x,"y":y,"mode":"lines" if len(x)>snapshot["layers"] else "lines+markers","name":label,"line":{"color":colour},"marker":{"color":colour},
                    "meta":{"step":step,"coordinate":coordinate,"source":snapshot["kind"]},"hovertemplate":"layer %{x:.4g}<br>%{y:.8g}<extra>%{fullData.name}</extra>"})
                if reference:
                    actual = family["values"][quantity][index]
                    expected = reference["values"][quantity][index]
                    traces.append({"x":list(range(1,len(expected)+1)),"y":expected,"mode":"markers","name":f"{label} · {reference_label}","marker":{"symbol":"x","size":6,"color":colour}})
                    traces.append({"x":list(range(1,len(actual)+1)),"y":[a-b for a,b in zip(actual,expected)],"mode":"lines","line":{"dash":"dot","color":colour},"visible":"legendonly","name":f"{label} · thogopt − reference"})
            if reference:
                errors.setdefault(chart,[]).append({"step":step,"source":reference_label,
                    **comparison_errors(torch.tensor(family["values"][quantity],dtype=torch.float64),torch.tensor(reference["values"][quantity],dtype=torch.float64))})
    return {"schema_version":1,"quantity":quantity,"steps":[r["optimizer_update"] for r in rows],
            "full_steps":[r["optimizer_update"] for r in rows if r["full_path"]],"figures":figures,"errors":errors,
            "snapshots":rows,"comparison":"separate training run" if reference_path else "same-gradient reference when captured"}


def _load_full(path,step):
    rows = history_rows(path,step_min=step,step_max=step)
    if not rows or not rows[0]["full_path"]: raise FileNotFoundError("Full history matrices were not captured at this step. Sampled reference histories are not full matrices.")
    root = Path(path).parent.resolve()
    target = (root/rows[0]["full_path"]).resolve()
    if not target.is_relative_to(root/"optimizer_histories") or target.suffix!=".pt": raise ValueError("invalid snapshot path")
    result = torch.load(target,map_location="cpu",weights_only=True)
    if result.get("schema_version")!=1: raise ValueError("Unsupported full optimizer snapshot schema; update Instra")
    return result


def matrix_window(path,*,step,chart,quantity="momentum",layer=1,coefficient=None,row_start=0,column_start=0,row_count=32,column_count=16,reference_path=None):
    if quantity not in QUANTITIES: raise ValueError("unknown optimizer history quantity")
    if min(row_start,column_start)<0 or not (1<=row_count<=64 and 1<=column_count<=64): raise ValueError("matrix window must contain 1..64 rows and columns")
    def extract(source):
        full = _load_full(source,step)
        family = full["families"][chart]
        rows,columns = family["shape"]
        if not 1<=layer<=full["layers"]: raise ValueError("layer outside snapshot")
        if row_start>=rows or column_start>=columns: raise ValueError("matrix window outside snapshot")
        r = torch.arange(row_start,min(rows,row_start+row_count))
        c = torch.arange(column_start,min(columns,column_start+column_count))
        indices = (r[:,None]*columns+c[None,:]).flatten()
        m,v = family["m"][:,indices].double(),family["v"][:,indices].double()
        basis_signature = None
        if coefficient is not None:
            if full["kind"]!="thogopt": raise ValueError("dense AdamW has no history coefficients")
            first = quantity in ("momentum","raw_momentum")
            tensor = m if first else v
            if quantity not in ("momentum","raw_momentum","second_moment","raw_second_moment"): raise ValueError("coefficient view requires a moment quantity")
            if not 0<=coefficient<tensor.shape[0]: raise ValueError("coefficient index outside history")
            values = tensor[coefficient]
            basis_signature = family["q_m" if first else "q_v"]
        else:
            if full["kind"]=="thogopt":
                m = family["q_m"][layer-1].double()@m
                v = family["q_v"][layer-1].double()@v
            else: m,v = m[layer-1],v[layer-1]
            values = history_quantities(m,v,**{key:family[key] for key in ("step","betas","eps","lr")})[quantity]
        return values.reshape(len(r),len(c)),[rows,columns],basis_signature,full["kind"],full["layers"]
    values,shape,basis,kind,layers = extract(path)
    comparison = None
    if reference_path:
        other,other_shape,other_basis,other_kind,other_layers = extract(reference_path)
        if shape!=other_shape or layers!=other_layers or (basis is not None and (other_basis is None or not torch.equal(basis,other_basis))): raise ValueError("matrix comparison requires matching shapes, layer coordinates and coefficient basis")
        values -= other
        comparison = f"{kind} minus {other_kind}; separate training runs"
    return {"values":values.tolist(),"shape":shape,"layers":layers,"row_start":row_start,"column_start":column_start,
        "step":step,"layer":layer,"quantity":quantity,"coefficient":coefficient,"comparison":comparison,"kind":kind}


def install(dashboard):
    if getattr(dashboard,"_thogopt_installed",False): return
    dashboard._thogopt_installed=True
    original_status = dashboard.RunDashboardState.status
    def status(self):
        result = original_status(self)
        configuration = self.reader.metadata().get("thogopt_configuration")
        if configuration:
            result = {**result,"configuration":{**result.get("configuration",{}),"thogopt":json.loads(configuration)}}
        return result
    dashboard.RunDashboardState.status=status
    original = dashboard._handler_for
    def handler_for(catalog):
        class HistoryHandler(original(catalog)):
            def do_GET(self):
                parsed = urlparse(self.path)
                if parsed.path not in ("/api/optimizer-history","/api/optimizer-history-matrix"): return super().do_GET()
                query = parse_qs(parsed.query)
                def value(key,default=""): return query.get(key,[default])[0]
                try:
                    state = catalog.state_for_run(value("run"))
                    reference = catalog.state_for_run(value("reference_run")).reader.path if value("reference_run") else None
                    if parsed.path.endswith("-matrix"):
                        payload = matrix_window(state.reader.path,step=int(value("step")),chart=value("chart"),quantity=value("quantity","momentum"),layer=int(value("layer","1")),
                            coefficient=int(value("coefficient")) if value("coefficient") else None,
                            row_start=int(value("row_start","0")),column_start=int(value("column_start","0")),row_count=int(value("row_count","32")),column_count=int(value("column_count","16")),reference_path=reference)
                    else:
                        payload = history_payload(state.reader.path,quantity=value("quantity","momentum"),step_min=int(value("step_min","0")),step_max=int(value("step_max",str(2**63-1))),reference_path=reference,latest=value("latest","false")=="true")
                    self._send_json(payload)
                except (KeyError,FileNotFoundError) as error:
                    self._send_json({"error":str(error)},status=dashboard.HTTPStatus.NOT_FOUND)
                except (ValueError,IndexError) as error:
                    self._send_json({"error":str(error)},status=dashboard.HTTPStatus.BAD_REQUEST)
                except Exception as error:
                    self._send_json({"error":str(error)},status=dashboard.HTTPStatus.INTERNAL_SERVER_ERROR)
        return HistoryHandler
    dashboard._handler_for=handler_for
# ^^^ THOG
