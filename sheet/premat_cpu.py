# vvv THOG spawned native-thread workers, immutable shared snapshots and bounded prediction histories
from __future__ import annotations

import atexit
from collections import OrderedDict, deque
from dataclasses import dataclass, field
import multiprocessing as mp
from multiprocessing import shared_memory
import queue
import resource
import statistics
import threading
import time
import uuid

import torch

from .depth_numerical_policy import materialize_cpu
from .premat_cpu_config import resolve_cpu_threads

FAMILY_NAMES = {
    "QKV": ("attention_query_weight", "attention_key_weight", "attention_value_weight"),
    "QK": ("attention_query_weight", "attention_key_weight"),
    "V": ("attention_value_weight",),
    "O": ("attention_output_weight",),
    "UP": ("mlp_expansion_weight",),
    "DOWN": ("mlp_contraction_weight",),
}


class SharedTensorOwner:
    def __init__(self, memory, unlink):
        self.memory, self.unlink = memory, unlink

    def __del__(self):
        try:
            self.memory.close()
        except (BufferError, OSError):
            pass
        if self.unlink:
            try:
                self.memory.unlink()
            except FileNotFoundError:
                pass


def shared_tensor(value, name=None):
    if hasattr(value, "_cpu_shared_owner"):
        return value
    value = value.detach().contiguous()
    memory = shared_memory.SharedMemory(create=True, name=name, size=max(1, value.numel() * value.element_size()))
    result = torch.frombuffer(memory.buf, dtype=value.dtype, count=value.numel()).reshape(value.shape)
    result.copy_(value)
    result._cpu_shared_owner = SharedTensorOwner(memory, True)
    return result


def tensor_descriptor(value):
    return {"name": value._cpu_shared_owner.memory.name, "shape": tuple(value.shape), "dtype": str(value.dtype).split(".")[-1]}


def open_shared_tensor(descriptor, unlink=False):
    memory = shared_memory.SharedMemory(name=descriptor["name"])
    count = 1
    for extent in descriptor["shape"]:
        count *= extent
    result = torch.frombuffer(memory.buf, dtype=getattr(torch, descriptor["dtype"]), count=count).reshape(descriptor["shape"])
    result._cpu_shared_owner = SharedTensorOwner(memory, unlink)
    return result


def _worker(tasks, results, generation, threads):
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    results.put({"event": "cpu_worker_ready", "worker_pid": mp.current_process().pid, "cuda_initialized": torch.cuda.is_initialized(), "native_threads": torch.get_num_threads(), "host_time_ns": time.perf_counter_ns()})
    while True:
        task = tasks.get()
        if task is None:
            return
        if task["generation"] != generation.value:
            results.put({"event": "cpu_batch_finished", "cpu_task_id": task["cpu_task_id"], "generation": task["generation"], "discarded": True})
            continue
        try:
            sources = tuple(open_shared_tensor(item) for item in task["coefficients"])
            rows = open_shared_tensor(task["rows"])
            for layer in task["layers"]:
                if task["generation"] != generation.value:
                    break
                identity = f'{task["snapshot_id"]}:{task["family"]}:{layer}'
                fields = {"generation": task["generation"], "snapshot_id": task["snapshot_id"], "family": task["family"], "layer_index": layer, "cpu_matrix_id": identity, "cpu_task_id": task["cpu_task_id"], "worker_pid": mp.current_process().pid, "native_threads": threads, "batch_layers": task["layers"], "request_ns": task["request_ns"]}
                started = time.perf_counter_ns()
                cpu_started = time.process_time_ns()
                results.put({**fields, "event": "cpu_task_start", "host_time_ns": started, "start_ns": started})
                parts = [materialize_cpu(source, rows[layer], task["policy"]) for source in sources]
                output = shared_tensor(torch.cat(parts, dim=0) if len(parts) > 1 else parts[0], name=f"thog_cpu_{task['cpu_task_id']}_{layer}")
                ended = time.perf_counter_ns()
                results.put({**fields, "event": "cpu_matrix_ready", "host_time_ns": ended, "start_ns": started, "end_ns": ended, "worker_cpu_service_ns": time.process_time_ns()-cpu_started, "worker_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, "tensor": tensor_descriptor(output), "bytes": output.numel() * output.element_size()})
                output._cpu_shared_owner.unlink = False
                del output, parts
            del sources, rows
        except Exception as error:
            results.put({"event": "cpu_task_failed", "host_time_ns": time.perf_counter_ns(), "cpu_task_id": task["cpu_task_id"], "generation": task["generation"], "snapshot_id": task["snapshot_id"], "family": task["family"], "batch_layers": task["layers"], "error": f"{type(error).__name__}: {error}"})

        finally:
            results.put({"event": "cpu_batch_finished", "cpu_task_id": task["cpu_task_id"], "generation": task["generation"]})

def source_fingerprint(trajectory, policy, families):
    names = tuple(dict.fromkeys(name for family in families for name in FAMILY_NAMES[family]))
    sources = [(name, trajectory.coefficients[name]) for name in names] + [("depth_basis", trajectory.depth_basis)]
    return (policy.policy_id, tuple((name, id(tensor), tensor.data_ptr(), tensor._version, tuple(tensor.shape), str(tensor.dtype), str(tensor.device)) for name, tensor in sources))


@dataclass
class CpuSnapshot:
    snapshot_id: str
    fingerprint: tuple
    policy: object
    generation: int
    coefficients: dict = field(default_factory=dict)
    rows: object = None
    ready_families: set = field(default_factory=set)


class CpuPreparationProvider:
    def __init__(self, trajectory, *, families, preparation, batch_size, workers, threads, record=None, wake=None):
        self.trajectory, self.families = trajectory, tuple(families)
        self.preparation, self.batch_size = preparation, batch_size
        self.threads, self.thread_budget = resolve_cpu_threads(workers, threads)
        self.worker_count, self.record, self.wake = workers, record or (lambda event: None), wake or (lambda: None)
        self.n_layer = trajectory.config.n_layer
        self.lock = threading.RLock()
        self.context = mp.get_context("spawn")
        self.generation = self.context.Value("q", 0)
        self.queue_limit = max(16, self.n_layer * len(self.families) * 3 + workers)
        self.tasks = self.context.Queue(maxsize=self.queue_limit)
        self.results = self.context.Queue(maxsize=self.queue_limit)
        self.workers = []
        self.snapshot, self.cache = None, {}
        self.requested, self.submitted, self.failures = set(), set(), {}
        self.closed, self.started = False, False
        self.cache_peak, self.snapshot_peak = 0, 0
        self.worker_health = {}
        self.pending_outputs = {}
        self.collector = None
        atexit.register(self.close)

    def start(self):
        if self.started:
            return
        self.started = True
        for _ in range(self.worker_count):
            process = self.context.Process(target=_worker, args=(self.tasks, self.results, self.generation, self.threads), daemon=True)
            process.start()
            self.workers.append(process)
        self.collector = threading.Thread(target=self._collect, daemon=True, name="premat-cpu-completions")
        self.collector.start()

    def begin_snapshot(self, policy):
        fingerprint = source_fingerprint(self.trajectory, policy, self.families)
        with self.lock:
            if self.snapshot is not None and self.snapshot.fingerprint == fingerprint:
                return self.snapshot, False
            self.generation.value += 1
            self.snapshot = CpuSnapshot(uuid.uuid4().hex, fingerprint, policy, self.generation.value)
            self.cache.clear()
            self.requested.clear()
            self.submitted.clear()
            self.failures.clear()
            snapshot = self.snapshot
        self.record({"event": "snapshot_created", "snapshot_id": snapshot.snapshot_id, "host_time_ns": time.perf_counter_ns(), "numerical_policy": vars(policy)})
        return snapshot, True

    def publish_source(self, snapshot_id, name, tensor):
        # The full host copy and shared-bank allocation happen on a coordinator, outside dispatch locks.
        bank = shared_tensor(tensor)
        with self.lock:
            snapshot = self.snapshot
            if snapshot is None or snapshot.snapshot_id != snapshot_id:
                return
            if name == "depth_basis":
                snapshot.rows = bank
            else:
                snapshot.coefficients[name] = bank
            if snapshot.rows is not None:
                for family in self.families:
                    if all(source in snapshot.coefficients for source in FAMILY_NAMES[family]):
                        snapshot.ready_families.add(family)
                        if self.preparation == "eager":
                            self.requested.update((family, layer) for layer in range(self.n_layer))
            self.snapshot_peak = max(self.snapshot_peak, self.snapshot_bytes())
            self._submit_ready()
        self.wake()

    def batch_layers(self, layer):
        if self.batch_size == "all_layers":
            return tuple(range(self.n_layer))
        size = 1 if self.batch_size == "single_layer" else int(self.batch_size)
        start = (int(layer) // size) * size
        return tuple(range(start, min(self.n_layer, start + size)))

    def request(self, family, layer):
        with self.lock:
            if family not in self.families or self.closed:
                return
            self.requested.update((selected, item) for selected in self.families for item in self.batch_layers(layer))
            self._submit_ready()

    def _submit_ready(self):
        if self.snapshot is None:
            return
        for family in self.families:
            if family not in self.snapshot.ready_families:
                continue
            for layer in range(self.n_layer):
                key = (family, layer)
                if key not in self.requested or key in self.submitted or key in self.failures:
                    continue
                layers = tuple(item for item in self.batch_layers(layer) if (family, item) not in self.submitted and (family, item) not in self.failures)
                task = {"generation": self.snapshot.generation, "snapshot_id": self.snapshot.snapshot_id, "family": family, "layers": layers, "coefficients": tuple(tensor_descriptor(self.snapshot.coefficients[name]) for name in FAMILY_NAMES[family]), "rows": tensor_descriptor(self.snapshot.rows), "policy": self.snapshot.policy, "cpu_task_id": uuid.uuid4().hex, "request_ns": time.perf_counter_ns()}
                try:
                    self.tasks.put_nowait(task)
                except queue.Full:
                    return
                self.pending_outputs[task["cpu_task_id"]] = {f"thog_cpu_{task['cpu_task_id']}_{item}" for item in layers}
                self.submitted.update((family, item) for item in layers)
                self.record({key: value for key, value in {**task, "event": "cpu_task_queued", "host_time_ns": task["request_ns"], "batch_layers": layers}.items() if key not in ("coefficients", "rows", "policy")})

    def _collect(self):
        while not self.closed:
            try:
                event = self.results.get(timeout=0.1)
            except queue.Empty:
                continue
            except (ValueError, OSError, EOFError):
                return
            descriptor = event.pop("tensor", None)
            try:
                matrix = open_shared_tensor(descriptor, unlink=True) if descriptor is not None else None
            except FileNotFoundError:
                matrix = None
            with self.lock:
                current = self.snapshot is not None and event.get("generation") == self.snapshot.generation
                if descriptor is not None:
                    self.pending_outputs.get(event.get("cpu_task_id"), set()).discard(descriptor["name"])
                if event["event"] == "cpu_batch_finished":
                    self.pending_outputs.pop(event["cpu_task_id"], None)
                if event.get("worker_pid") in self.worker_health:
                    self.worker_health[event["worker_pid"]].update({key:event[key] for key in ("host_time_ns", "worker_peak_rss_bytes", "worker_cpu_service_ns") if key in event})
                if event["event"] == "cpu_worker_ready":
                    self.worker_health[event["worker_pid"]] = dict(event)
                elif current and matrix is not None:
                    self.cache[(event["family"], event["layer_index"])] = (matrix, dict(event))
                    self.cache_peak = max(self.cache_peak, self.cache_bytes())
                elif current and event["event"] == "cpu_task_failed":
                    self.failures.update({(event["family"], layer): event["error"] for layer in event["batch_layers"]})
                self._submit_ready()
            self.record({**event, "discarded": descriptor is not None and not current, "published_ns": time.perf_counter_ns() if matrix is not None else None})
            del matrix
            self.wake()

    def lookup(self, family, layer, policy):
        with self.lock:
            if self.snapshot is None or self.snapshot.fingerprint != source_fingerprint(self.trajectory, policy, self.families):
                return None, {"fallback_reason": "stale_snapshot"}
            if (family, layer) in self.failures or any(process.exitcode not in (None, 0) for process in self.workers):
                return None, {"fallback_reason": "cpu_worker_failed"}
            return self.cache.get((family, layer), (None, {"fallback_reason": "cpu_preparation_not_ready"}))

    def cache_bytes(self):
        return sum(matrix.numel() * matrix.element_size() for matrix, _ in self.cache.values())

    def snapshot_bytes(self):
        if self.snapshot is None:
            return 0
        tensors = list(self.snapshot.coefficients.values()) + ([self.snapshot.rows] if self.snapshot.rows is not None else [])
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def report(self):
        with self.lock:
            dtype_bytes = 4 if self.snapshot is None or self.snapshot.policy.output_dtype == "float32" else 2
            sizes = [sum(self.trajectory.coefficients[name].shape[0] * self.trajectory.coefficients[name].shape[1] for name in FAMILY_NAMES[family]) * dtype_bytes for family in self.families]
            largest = max(sizes, default=0)
            return {"snapshot_id": None if self.snapshot is None else self.snapshot.snapshot_id, "cpu_cache_bytes": self.cache_bytes(), "cpu_cache_peak_bytes": self.cache_peak, "snapshot_bytes": self.snapshot_bytes(), "snapshot_peak_bytes": self.snapshot_peak, "cpu_cache_bound_bytes": self.n_layer * sum(sizes), "cpu_pending_output_bound_bytes": self.queue_limit * largest, "snapshot_storage_bound_bytes": (self.worker_count + 1) * self.snapshot_peak, "cpu_worker_workspace_bound_bytes": self.worker_count * (2 * self.snapshot_peak + 4 * largest), "workers_requested": self.worker_count, "workers_resolved": len(self.workers), "threads_per_worker_resolved": self.threads, "affinity_thread_budget": self.thread_budget, "worker_health": list(self.worker_health.values()), "dead_worker_count": sum(process.exitcode not in (None, 0) for process in self.workers), "task_failure_count": len(self.failures)}

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.generation.value += 1
        for _ in self.workers:
            try:
                self.tasks.put_nowait(None)
            except queue.Full:
                break
        for process in self.workers:
            process.join(timeout=0.25)
            if process.is_alive():
                process.terminate()
                process.join(timeout=0.5)
        if self.collector is not None and self.collector is not threading.current_thread():
            self.collector.join(timeout=0.25)
        # Outputs not yet received are named before submission; cleanup remains bounded after worker termination.
        for names in self.pending_outputs.values():
            for name in names:
                try:
                    memory = shared_memory.SharedMemory(name=name)
                    memory.unlink()
                    memory.close()
                except FileNotFoundError:
                    pass
        self.pending_outputs.clear()
        self.cache.clear()
        self.snapshot = None
        for pipe in (self.tasks, self.results):
            pipe.cancel_join_thread()
            pipe.close()
        atexit.unregister(self.close)


class GemmPredictor:
    def __init__(self, limit=2048):
        self.limit, self.samples, self.uploads = limit, OrderedDict(), deque(maxlen=8)
        self.reset_reason = "cold_start"

    def reset(self, reason):
        self.samples.clear()
        self.uploads.clear()
        self.reset_reason = reason

    def observe(self, context, offset_ms, uncertainty_ms):
        if uncertainty_ms is None or uncertainty_ms > 5 or offset_ms < 0:
            return
        self.samples.setdefault(context, deque(maxlen=8)).append(float(offset_ms))
        self.samples.move_to_end(context)
        if len(self.samples) > self.limit:
            self.samples.popitem(last=False)

    def predict(self, context, origin_ns, lead_ms, uncertainty_ms):
        samples = self.samples.get(context, ())
        if len(samples) < 3 or uncertainty_ms is None or uncertainty_ms > 5:
            return {"prediction_available": False, "prediction_fallback_reason": self.reset_reason if not samples else "insufficient_qualified_observations", "effective_fallback_trigger": "as_the_code_flies"}
        predicted = origin_ns + int(statistics.median(samples) * 1e6)
        availability = predicted - int(lead_ms * 1e6)
        cost = statistics.median(self.uploads) if self.uploads else 0.0
        return {"prediction_available": True, "predicted_gemm_start_ns": predicted, "intended_gpu_available_ns": availability, "intended_upload_submission_ns": availability - int(cost * 1e6), "transfer_lead_ms": lead_ms, "prediction_uncertainty_ms": uncertainty_ms, "estimated_upload_ms": cost, "qualified_observation_count": len(samples)}
# ^^^ THOG
