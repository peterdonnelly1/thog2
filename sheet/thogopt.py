# vvv THOG
"""Layer-space AdamW with independently compressed, raw-gradient histories.

Raw step gradients and transaction candidates are staged on the host. Device
history evaluation is tiled. No dense layer-moment replica is retained.
"""
from __future__ import annotations

from copy import deepcopy
import math
import os
import time
from pathlib import Path
from types import MethodType

import torch
import torch.distributed as dist
from torch import Tensor

from .depth_trajectory import DepthTrajectory
from .thogopt_math import comparison_errors, fit_nonnegative, history_basis, resolve_history_count


SCHEMA_VERSION = 1
MOMENTUM_SETTING = "thogopt__momentum_history_coefficients"
SCALING_SETTING = "thogopt__scaling_history_coefficients"


class Thogopt(torch.optim.Optimizer):
    def __init__(self, params, *, trajectory: DepthTrajectory, lr=1e-3,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                 momentum_history_coefficients="auto", scaling_history_coefficients="auto",
                 tile_elements=1048576):
        if not isinstance(trajectory, DepthTrajectory) or trajectory.legacy_sheet_col_vectors:
            raise ValueError("thogopt requires the public DEPTH materialiser")
        if trajectory.plastic_enabled or trajectory.basis_family != "chebyshev":
            raise ValueError("thogopt initially requires fixed-depth Chebyshev geometry")
        if lr < 0 or eps <= 0 or weight_decay < 0 or not all(0 <= b < 1 for b in betas):
            raise ValueError("invalid thogopt AdamW hyperparameters")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))
        self.trajectory = trajectory
        self.layers = trajectory.config.n_layer
        self.order = trajectory.config.depth_order
        self.momentum_count = resolve_history_count(momentum_history_coefficients, nominal=self.order, layers=self.layers)
        self.scaling_count = resolve_history_count(scaling_history_coefficients, nominal=2*self.order-1, layers=self.layers)
        self.tile_columns = max(1, int(tile_elements) // self.layers)
        self.families = {item.name: parameter for name, parameter, item in trajectory.named_semantic_parameters()
                         if trajectory._representation(item) == "depth_coefficients" and parameter.requires_grad}
        self.parameter_families = {parameter: name for name, parameter in self.families.items()}
        owned = [p for group in self.param_groups for p in group["params"]]
        if len({id(p) for p in owned}) != len(owned) or not set(self.families.values()).issubset(set(owned)):
            raise ValueError("thogopt optimizer ownership is incomplete or duplicated")
        for group in self.param_groups:
            if group.get("amsgrad", False):
                raise ValueError("thogopt does not support AMSGrad")
        if not self.families:
            raise ValueError("thogopt requires at least one trainable depth-coefficient family")
        reference = next(iter(self.families.values()))
        self.history_dtype = torch.float64 if reference.dtype == torch.float64 else torch.float32
        self.weight_basis = trajectory.depth_basis.detach().to(reference.device, dtype=self.history_dtype).clone()
        if torch.linalg.matrix_rank(self.weight_basis.double()) != self.order:
            raise ValueError("thogopt requires a full-column-rank weight materialiser")
        self.weight_analysis = torch.linalg.pinv(self.weight_basis.double()).to(self.history_dtype)
        self.q_m = history_basis(self.layers, self.momentum_count["effective"], dtype=self.history_dtype, device=reference.device)
        self.q_v = history_basis(self.layers, self.scaling_count["effective"], dtype=self.history_dtype, device=reference.device)
        required = self.resource_report()["transaction_host_bytes_estimate"]
        budget_mib = os.environ.get("THOG2_THOGOPT_HOST_BUDGET_MIB")
        if budget_mib is not None:
            available = float(budget_mib)*2**20
            if not math.isfinite(available) or available<=0:
                raise ValueError("THOG2_THOGOPT_HOST_BUDGET_MIB must be positive and finite")
        else:
            available = os.sysconf("SC_AVPHYS_PAGES")*os.sysconf("SC_PAGE_SIZE") if hasattr(os,"sysconf") else math.inf
            try:
                limit = Path("/sys/fs/cgroup/memory.max").read_text().strip()
                if limit!="max": available = min(available,int(limit)-int(Path("/sys/fs/cgroup/memory.current").read_text()))
            except (OSError,ValueError): pass
        if required>available:
            raise MemoryError(f"thogopt host staging needs approximately {required/2**20:.1f} MiB; budget/available {available/2**20:.1f} MiB")
        self.raw_gradients = {}
        self.seen_layers = {}
        self.reference_histories = {}
        self.last_step_metrics = {}
        self.prepared = False
        self.gradient_staging_seconds = 0.
        self.gradient_preparation_seconds = 0.
        self.capture_active = True
        self._original_materialize = trajectory._materialize_depth_parameter
        optimizer = self

        def materialize_with_raw_gradient(this, name, layer_index):
            value = optimizer._original_materialize(name, layer_index)
            if value.requires_grad and name in optimizer.families and optimizer.capture_active:
                def capture(gradient):
                    optimizer.accumulate_layer_gradient(name, layer_index, gradient)
                    return gradient
                value.register_hook(capture)
            return value

        trajectory._materialize_depth_parameter = MethodType(materialize_with_raw_gradient, trajectory)

    def configuration(self):
        return {"schema_version": SCHEMA_VERSION, "optimizer": "thogopt", "layers": self.layers,
                "weight_coefficients": self.order, "momentum_history": dict(self.momentum_count),
                "scaling_history": dict(self.scaling_count), "history_dtype": str(self.history_dtype),
                "fit": "unweighted_nonnegative_layer_ls_v1", "staging": "host_accumulation_and_transaction",
                "device_tile_columns": self.tile_columns, "weight_basis": self.weight_basis.detach().cpu(),
                "history_basis_version":"chebyshev_equispaced_qr_positive_diagonal_or_full_samples_v1",
                "history_coordinates":torch.linspace(-1,1,self.layers,dtype=torch.float64).tolist(),
                "weight_basis_condition":float(torch.linalg.cond(self.weight_basis.double())),
                "momentum_basis_condition":float(torch.linalg.cond(self.q_m.double())),
                "scaling_basis_condition":float(torch.linalg.cond(self.q_v.double())),
                "family_shapes": {name: tuple(p.shape) for name, p in self.families.items()}}

    def resource_report(self):
        n = sum(p.numel() // self.order for p in self.families.values())
        ordinary_bytes = sum(p.numel()*p.element_size() for group in self.param_groups for p in group["params"] if p not in self.parameter_families)
        ordinary_transaction_bytes = sum(p.numel()*(p.element_size()+2*(8 if p.dtype==torch.float64 else 4)) for group in self.param_groups for p in group["params"] if p not in self.parameter_families)
        size = torch.empty((), dtype=self.history_dtype).element_size()
        return {"moment_entries": n * (self.q_m.shape[1] + self.q_v.shape[1]),
                "moment_bytes": size * n * (self.q_m.shape[1] + self.q_v.shape[1]),
                "raw_gradient_host_bytes": size * n * self.layers,
                "weight_coefficient_bytes": sum(p.numel()*p.element_size() for p in self.families.values()),
                "dense_adamw_moment_bytes": 2 * size * n * self.layers,
                "tile_layer_elements": self.layers * self.tile_columns,
                "transaction_host_bytes_estimate": size*n*(self.layers+self.q_m.shape[1]+self.q_v.shape[1])+sum(p.numel()*p.element_size() for p in self.families.values())+ordinary_transaction_bytes,
                "ordinary_parameter_bytes": ordinary_bytes,
                "shared_basis_bytes":sum(t.numel()*t.element_size() for t in (self.weight_basis,self.weight_analysis,self.q_m,self.q_v)),
                "diagnostic_reference_bytes":sum(r[k].numel()*r[k].element_size() for r in getattr(self,"reference_histories",{}).values() for k in ("exp_avg","exp_avg_sq"))}

    @torch.no_grad()
    def accumulate_layer_gradient(self, name, layer_index, gradient):
        started = time.perf_counter()
        if self.prepared:
            raise RuntimeError("thogopt received a gradient after step preparation")
        if name not in self.raw_gradients:
            n = self.families[name].numel() // self.order
            self.raw_gradients[name] = torch.zeros((self.layers, n), dtype=self.history_dtype, device="cpu")
            self.seen_layers[name] = torch.zeros(self.layers, dtype=torch.bool)
        self.raw_gradients[name][layer_index].add_(gradient.detach().reshape(-1).to(device="cpu", dtype=self.history_dtype))
        self.seen_layers[name][layer_index] = True
        self.gradient_staging_seconds += time.perf_counter()-started

    def zero_grad(self, set_to_none=True):
        super().zero_grad(set_to_none=set_to_none)
        self.raw_gradients.clear()
        self.seen_layers.clear()
        self.prepared = False
        self.gradient_staging_seconds = 0.
        self.gradient_preparation_seconds = 0.

    def _distributed_all_true(self, value):
        if not dist.is_available() or not dist.is_initialized():
            return bool(value)
        device = self.weight_basis.device if dist.get_backend() == "nccl" else torch.device("cpu")
        flag = torch.tensor(int(value), dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        return bool(flag.item())

    @torch.no_grad()
    def prepare_gradients(self, *, loss_scale=1., grad_clip=0.):
        started = time.perf_counter()
        if self.prepared:
            raise RuntimeError("thogopt gradients were already prepared")
        if not math.isfinite(loss_scale) or loss_scale <= 0:
            raise ValueError("thogopt loss scale must be positive and finite")
        distributed = dist.is_available() and dist.is_initialized()
        norm_squared = torch.zeros((), dtype=torch.float64)
        local_valid = True
        for name, parameter in self.families.items():
            seen = self.seen_layers.get(name, torch.zeros(self.layers, dtype=torch.bool))
            # Presence is checked collectively before any data-dependent collective.
            all_present = self._distributed_all_true(bool(seen.all()))
            all_absent = self._distributed_all_true(not bool(seen.any()) and parameter.grad is None)
            if all_absent:
                continue
            if not all_present:
                raise ValueError(f"thogopt missing raw layer gradients for {name}; mixed missing/present layers or a bypassed materialiser is unsupported")
            raw = self.raw_gradients[name]
            raw.div_(loss_scale)
            if distributed:
                collective_device = parameter.device if dist.get_backend() == "nccl" else torch.device("cpu")
                for start in range(0, raw.shape[1], self.tile_columns):
                    tile = raw[:, start:start+self.tile_columns].to(collective_device).contiguous()
                    dist.all_reduce(tile)
                    tile.div_(dist.get_world_size())
                    raw[:, start:start+self.tile_columns].copy_(tile.cpu())
            local_valid = local_valid and bool(torch.isfinite(raw).all())
            norm_squared += torch.linalg.vector_norm(raw, dtype=torch.float64).square()
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter in self.parameter_families or parameter.grad is None:
                    continue
                local_valid = local_valid and bool(torch.isfinite(parameter.grad).all())
                norm_squared += torch.linalg.vector_norm(parameter.grad.detach(), dtype=torch.float64).square().cpu()
        if not self._distributed_all_true(local_valid):
            raise FloatingPointError("thogopt non-finite raw optimizer-step gradient")
        norm = float(norm_squared.sqrt())
        if not math.isfinite(norm):
            raise FloatingPointError("thogopt non-finite physical gradient norm")
        if grad_clip > 0:
            factor = min(1., float(grad_clip) / (norm + 1e-6))
            for raw in self.raw_gradients.values():
                raw.mul_(factor)
            for group in self.param_groups:
                for parameter in group["params"]:
                    if parameter.grad is not None:
                        parameter.grad.mul_(factor)
        self.prepared = True
        self.gradient_preparation_seconds = time.perf_counter()-started
        return norm

    def configure_reference(self, coordinates):
        """coordinates maps family names to flattened physical coupling indices."""
        for name, indices in coordinates.items():
            if name not in self.families:
                continue
            chosen = tuple(sorted(set(int(i) for i in indices)))
            if not chosen:
                continue
            n = self.families[name].numel() // self.order
            if min(chosen) < 0 or max(chosen) >= n:
                raise ValueError("reference coupling index outside matrix")
            previous = self.reference_histories.get(name)
            if previous is not None and tuple(previous["indices"]) == chosen:
                continue
            step = int(self.state.get(self.families[name], {}).get("step", 0))
            self.reference_histories[name] = {"indices": chosen, "origin": step, "step": step,
                "exp_avg": torch.zeros(self.layers, len(chosen), dtype=torch.float64),
                "exp_avg_sq": torch.zeros(self.layers, len(chosen), dtype=torch.float64)}

    @torch.no_grad()
    def _compressed_candidate(self, parameter, name, group):
        old = self.state.get(parameter, {})
        step = int(old.get("step", 0)) + 1
        beta1, beta2 = group["betas"]
        raw = self.raw_gradients[name]
        columns = raw.shape[1]
        # Read weights directly from their device tile; do not round-trip the
        # entire family through CPU before calculating its candidate.
        candidate = torch.empty((columns, self.order), dtype=parameter.dtype, device="cpu")
        parameter_values = parameter.detach().reshape(columns, self.order)
        full_momentum = self.q_m.shape[1] == self.layers
        full_scaling = self.q_v.shape[1] == self.layers
        first = torch.empty((self.q_m.shape[1], columns), dtype=self.history_dtype, device="cpu")
        second = torch.empty((self.q_v.shape[1], columns), dtype=self.history_dtype, device="cpu")
        metrics = {"constrained_columns": 0, "roundoff_corrections": 0, "maximum_roundoff_correction": 0., "fit_sweeps": 0, "fit_seconds": 0., "tiles": 0, "full_momentum": full_momentum, "full_scaling": full_scaling}
        for start in range(0, columns, self.tile_columns):
            end = min(start + self.tile_columns, columns)
            metrics["tiles"] += 1
            gradient = raw[:, start:end].to(parameter.device)
            m = torch.zeros((self.q_m.shape[1], end-start), device=parameter.device, dtype=self.history_dtype) if not old else old["exp_avg"][:, start:end].clone()
            v = torch.zeros((self.q_v.shape[1], end-start), device=parameter.device, dtype=self.history_dtype) if not old else old["exp_avg_sq"][:, start:end]
            m.mul_(beta1).add_(gradient if full_momentum else self.q_m.T @ gradient, alpha=1-beta1)
            previous_values = v if full_scaling else self.q_v @ v
            tolerance = 64 * torch.finfo(self.history_dtype).eps * previous_values.abs().amax(dim=0).clamp_min(torch.finfo(self.history_dtype).tiny)
            if bool((previous_values < -tolerance).any()):
                raise FloatingPointError(f"thogopt negative previous second moment: {name}")
            target = previous_values.clamp_min(0).mul_(beta2).add_(gradient.square(), alpha=1-beta2)
            fit_started = time.perf_counter()
            if full_scaling:
                # With H_v=L, Q_v is the identity: the nonnegative target is
                # already the exact least-squares solution. No fit is needed.
                v = target
            else:
                try:
                    v, fit_metrics = fit_nonnegative(self.q_v, target)
                except (ValueError,FloatingPointError) as error:
                    raise FloatingPointError(f"thogopt {name} step {step}: {error}") from error
                metrics["fit_seconds"] += time.perf_counter()-fit_started
                for key in ("constrained_columns", "roundoff_corrections"):
                    metrics[key] += fit_metrics[key]
                for key in ("maximum_roundoff_correction", "fit_sweeps"):
                    metrics[key] = max(metrics[key], fit_metrics[key])
            m_hat = (m if full_momentum else self.q_m @ m) / (1-beta1**step)
            v_hat = (v if full_scaling else self.q_v @ v).clamp_min(0) / (1-beta2**step)
            update = -group["lr"] * m_hat / (v_hat.sqrt() + group["eps"])
            coefficient_delta = (self.weight_analysis @ update).T
            values = parameter_values[start:end].clone()
            values.mul_(1-group["lr"]*group["weight_decay"]).add_(coefficient_delta.to(values.dtype))
            if not bool(torch.stack([torch.isfinite(value).all() for value in (m, v, values)]).all()):
                raise FloatingPointError(f"thogopt non-finite candidate: {name}")
            first[:, start:end].copy_(m.cpu())
            second[:, start:end].copy_(v.cpu())
            candidate[start:end].copy_(values.cpu())
        return candidate.reshape(parameter.shape), {"step": step, "exp_avg": first, "exp_avg_sq": second}, metrics

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        if not self.prepared:
            self.prepare_gradients()
        started = time.perf_counter()
        if self.weight_basis.device.type=="cuda": torch.cuda.reset_peak_memory_stats(self.weight_basis.device)
        pending = []
        references = deepcopy(self.reference_histories)
        diagnostics = {}
        failure = None
        try:
            for group in self.param_groups:
                beta1, beta2 = group["betas"]
                for parameter in group["params"]:
                    name = self.parameter_families.get(parameter)
                    if name is not None:
                        if name not in self.raw_gradients:
                            continue
                        values, state, metrics = self._compressed_candidate(parameter, name, group)
                        diagnostics[name] = metrics
                        if name in references:
                            reference = references[name]
                            gradient = self.raw_gradients[name][:, reference["indices"]].double()
                            reference["exp_avg"].mul_(beta1).add_(gradient, alpha=1-beta1)
                            reference["exp_avg_sq"].mul_(beta2).add_(gradient.square(), alpha=1-beta2)
                            reference["step"] = state["step"]
                    else:
                        if parameter.grad is None:
                            continue
                        state_dtype = torch.float64 if parameter.dtype==torch.float64 else torch.float32
                        gradient = parameter.grad.detach().to(dtype=state_dtype)
                        if gradient.is_sparse:
                            raise ValueError("thogopt does not support sparse gradients")
                        old = self.state.get(parameter, {})
                        step = int(old.get("step", 0)) + 1
                        m = old["exp_avg"].clone() if old else torch.zeros_like(parameter,dtype=state_dtype)
                        v = old["exp_avg_sq"].clone() if old else torch.zeros_like(parameter,dtype=state_dtype)
                        m.lerp_(gradient, 1-beta1)
                        v.mul_(beta2).addcmul_(gradient, gradient, value=1-beta2)
                        denominator = v.sqrt() / math.sqrt(1-beta2**step) + group["eps"]
                        values = parameter.detach().mul(1-group["lr"]*group["weight_decay"])
                        values.addcdiv_(m, denominator, value=-group["lr"]/(1-beta1**step))
                        if not bool(torch.stack([torch.isfinite(value).all() for value in (m, v, values)]).all()):
                            raise FloatingPointError("thogopt non-finite ordinary parameter candidate")
                        state = {"step": step, "exp_avg": m.cpu(), "exp_avg_sq": v.cpu()}
                        values = values.cpu()
                    pending.append((parameter, values, state))
        except Exception as error:
            failure = error
        if not self._distributed_all_true(failure is None):
            if failure is not None:
                raise failure
            raise FloatingPointError("thogopt candidate rejected on another rank")
        # Allocate missing persistent buffers before changing any parameter.
        commit_states = []
        allocation_failure = None
        try:
            for parameter, values, state in pending:
                dtype = self.history_dtype if parameter in self.parameter_families else (torch.float64 if parameter.dtype==torch.float64 else torch.float32)
                existing = self.state.get(parameter, {})
                destination = {key: existing[key] if key in existing else torch.empty_like(state[key], device=parameter.device, dtype=dtype)
                               for key in ("exp_avg", "exp_avg_sq")}
                commit_states.append(destination)
        except Exception as error:
            allocation_failure = error
        if not self._distributed_all_true(allocation_failure is None):
            if allocation_failure is not None:
                raise allocation_failure
            raise FloatingPointError("thogopt state allocation failed on another rank")
        for (parameter, values, state), destination in zip(pending, commit_states):
            parameter.copy_(values)
            destination["exp_avg"].copy_(state["exp_avg"])
            destination["exp_avg_sq"].copy_(state["exp_avg_sq"])
            destination["step"] = state["step"]
            self.state[parameter] = destination
        self.reference_histories = references
        self.last_step_metrics = {"families": diagnostics, "optimizer_seconds": time.perf_counter()-started,
                                  "gradient_staging_seconds":self.gradient_staging_seconds, "gradient_preparation_seconds":self.gradient_preparation_seconds,
                                  "device_peak_allocated_bytes":torch.cuda.max_memory_allocated(self.weight_basis.device) if self.weight_basis.device.type=="cuda" else 0,
                                  **self.resource_report()}
        return loss

    def state_dict(self):
        payload = super().state_dict()
        payload["thogopt"] = self.configuration()
        payload["thogopt_reference"] = deepcopy(self.reference_histories)
        return payload

    def load_state_dict(self, payload):
        saved = payload.get("thogopt")
        current = self.configuration()
        if saved is None:
            raise ValueError("checkpoint has no thogopt raw-gradient histories; start an explicit weights-only fork")
        for key in ("schema_version", "layers", "weight_coefficients", "momentum_history", "scaling_history", "fit", "family_shapes", "history_dtype", "history_basis_version", "history_coordinates"):
            if saved.get(key) != current[key]:
                raise ValueError(f"incompatible thogopt checkpoint {key}; history changes require an explicit fork/reset")
        if not torch.equal(saved["weight_basis"].double().cpu(), current["weight_basis"].double().cpu()):
            raise ValueError("incompatible thogopt weight basis")
        restored = []
        for group, saved_group in zip(self.param_groups, payload["param_groups"], strict=True):
            for parameter, index in zip(group["params"], saved_group["params"], strict=True):
                state = payload["state"].get(index)
                if not state:
                    continue
                if parameter in self.parameter_families:
                    n = parameter.numel() // self.order
                    if tuple(state["exp_avg"].shape) != (self.q_m.shape[1], n) or tuple(state["exp_avg_sq"].shape) != (self.q_v.shape[1], n):
                        raise ValueError("incompatible thogopt history tensor shape")
                if int(state["step"])<0 or not all(bool(torch.isfinite(state[key]).all()) for key in ("exp_avg","exp_avg_sq")):
                    raise ValueError("invalid thogopt checkpoint moment state")
                restored.append((parameter, state))
        super().load_state_dict(payload)
        for parameter, state in restored:
            dtype = self.history_dtype if parameter in self.parameter_families else (torch.float64 if parameter.dtype==torch.float64 else torch.float32)
            self.state[parameter] = {"step": int(state["step"]), "exp_avg": state["exp_avg"].to(parameter.device, dtype=dtype).clone(),
                                     "exp_avg_sq": state["exp_avg_sq"].to(parameter.device, dtype=dtype).clone()}
        self.reference_histories = deepcopy(payload.get("thogopt_reference", {}))
        self.zero_grad()


def build_thogopt(model, parameter_groups, *, learning_rate, betas, weight_decay, config=None):
    trajectory = model.trajectory if hasattr(model, "trajectory") else None
    if not isinstance(trajectory, DepthTrajectory):
        raise ValueError("thogopt requires a public DEPTH model; use AdamW for DENSE")
    if config is not None and (config.layer_dropout_enabled or config.chaos_bump__sampling__enabled):
        raise ValueError("thogopt initially requires all fixed layer coordinates: disable layer dropout and chaos bump sampling")
    momentum = config.thogopt__momentum_history_coefficients if config is not None else os.environ.get("THOG2_THOGOPT_MOMENTUM_HISTORY_COEFFICIENTS", "auto")
    scaling = config.thogopt__scaling_history_coefficients if config is not None else os.environ.get("THOG2_THOGOPT_SCALING_HISTORY_COEFFICIENTS", "auto")
    return Thogopt(parameter_groups, trajectory=trajectory, lr=learning_rate, betas=betas, weight_decay=weight_decay,
        momentum_history_coefficients=momentum, scaling_history_coefficients=scaling)
# ^^^ THOG
