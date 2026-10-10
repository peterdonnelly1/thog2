# vvv THOG matched unprofiled DEPTH / GPU PREMAT / CPU+GPU smoke and preliminary field measurements
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import psutil
from sheet.trainer import SharedTrainer
from sheet.training_config import TrainingConfig


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--smoke",action="store_true",help="three tiny updates/provider, A=2, checkpoint segment=2")
    p.add_argument("--device",default="cuda")
    p.add_argument("--backend",choices=("matmul","einsum"),default="matmul",help="effective DEPTH materialiser, installed identically for all providers")
    p.add_argument("--dtype",choices=("float32","float16","bfloat16"),default="float32")
    p.add_argument("--layers",type=int,default=4)
    p.add_argument("--width",type=int,default=32)
    p.add_argument("--heads",type=int,default=4)
    p.add_argument("--depth-order",type=int,default=3)
    p.add_argument("--batch",type=int,default=2)
    p.add_argument("--context",type=int,default=16)
    p.add_argument("--accumulation",type=int,default=2)
    p.add_argument("--checkpoint",type=int,default=2)
    p.add_argument("--updates",type=int,default=20)
    p.add_argument("--warmup",type=int,default=3)
    p.add_argument("--repeats",type=int,default=3)
    p.add_argument("--workers",type=int,default=1)
    p.add_argument("--threads",type=int,default=0)
    p.add_argument("--preparation",choices=("eager","scheduled","demand_driven"),default="eager")
    p.add_argument("--layer-batch",default="single_layer")
    p.add_argument("--transfer",choices=("as_the_code_flies","previous_gemm_leading_edge","as_soon_as_ready","demand_driven","predicted_gemm_start"),default="as_the_code_flies")
    p.add_argument("--lead-ms",type=float,default=0)
    p.add_argument("--staging-mb",type=float,default=0)
    p.add_argument("--replay",choices=("disabled","enabled"),default="disabled")
    p.add_argument("--logging",choices=("disabled","enabled"),default="disabled")
    p.add_argument("--training-seed",type=int,default=373,help="dropout RNG seed; each repeat adds its index, identically for every provider")
    p.add_argument("--label",default=platform.node())
    p.add_argument("--output",type=Path,default=Path("evidence/cpu_materialisation_field.json"))
    return p


def memory(runtime):
    parent=psutil.Process()
    processes=[parent]
    if runtime is not None and getattr(runtime,"cpu_provider",None) is not None:
        for worker in runtime.cpu_provider.workers:
            if worker.is_alive():processes.append(psutil.Process(worker.pid))
    rows=[]
    for process in processes:
        try:
            info=process.memory_full_info()
            rows.append({"pid":process.pid,"rss_bytes":info.rss,"pss_bytes":getattr(info,"pss",None)})
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    return {"processes":rows,"total_pss_bytes":sum(row["pss_bytes"] for row in rows) if rows and all(row["pss_bytes"] is not None for row in rows) else None,"accounting":"RSS is reported per process; shared pages are not summed as unique memory. PSS is summed only when available."}


def training_config(options,mode):
    cpu=mode=="cpu_and_gpu"
    kwargs={}
    if cpu:
        kwargs={"premat_materialisation_device":mode,"premat_cpu_workers":options.workers,"premat_cpu_threads_per_worker":options.threads,"premat_cpu_preparation":options.preparation,"premat_cpu_layer_batch_size":options.layer_batch,"premat_cpu_transfer_timing":options.transfer,"premat_cpu_transfer_lead_ms":options.lead_ms,"premat_cpu_staging_limit_mb":options.staging_mb,"premat_cpu_checkpoint_replay":options.replay}
    return TrainingConfig(model_type="thog2_sheet",geometry_preset="depth",basis_family="chebyshev",block_size=options.context,vocab_size=128,n_layer=options.layers,n_embd=options.width,n_head=options.heads,depth_order=options.depth_order,base_row_order=1,batch_size=options.batch,gradient_accumulation_steps=options.accumulation,checkpoint_segment_size=options.checkpoint,max_updates=options.warmup+options.updates,learning_rate=1e-3,min_learning_rate=1e-3,decay_learning_rate=False,decay_updates=options.warmup+options.updates,weight_decay=0,grad_clip=0,dropout=0.1,eval_interval=0,checkpoint_interval=0,device=options.device,dtype=options.dtype,model_seed=171,data_seed=272,premat="disabled" if mode=="off" else "enabled",premat_gpu_memory_buffer_gb=0,premat_headroom_stay_below_current_peak=False,premat_headroom_stay_within_global_buffer=True,premat_logging=options.logging,premat_instra="disabled",premat_processing_logging="disabled",nonfinite_update_policy="raise",**kwargs)


def optimizer_values(trainer):
    names={id(parameter):name for name,parameter in trainer.raw_model.named_parameters()}
    result={}
    def collect(prefix,value):
        if torch.is_tensor(value):result[prefix]=value.detach().cpu().clone()
        elif isinstance(value,dict):
            for key,item in value.items():collect(prefix+"."+str(key),item)
        elif isinstance(value,(int,float,bool)):result[prefix]=torch.tensor(value,dtype=torch.float64)
        elif isinstance(value,(list,tuple)):
            for index,item in enumerate(value):collect(prefix+"."+str(index),item)
        else:raise TypeError("Unqualified optimizer-state value at "+prefix+": "+type(value).__name__)
    for parameter,state in trainer.optimizer.state.items():collect(names[id(parameter)],state)
    return result


def tensor_digest(value):
    value=value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(value.numpy().tobytes()).hexdigest()


def model_digest(model):
    digest=hashlib.sha256()
    for name,value in sorted(model.state_dict().items()):
        digest.update(json.dumps([name,list(value.shape),str(value.dtype)],separators=(",",":")).encode())
        digest.update(tensor_digest(value).encode())
    return digest.hexdigest()


def rng_fingerprints(trainer):
    return {"cpu_rng_sha256":tensor_digest(torch.get_rng_state()),"cuda_rng_sha256":tensor_digest(torch.cuda.get_rng_state(trainer.device)) if trainer.device.type=="cuda" else None,"batch_rng_sha256":tensor_digest(trainer.batch_source.train_generator.get_state())}


def seed_training_rng(seed,device):
    # Model construction uses fork_rng and restores the ambient training RNG.
    # Reset after construction so dropout does not depend on provider execution order.
    torch.default_generator.manual_seed(seed)
    if device.type=="cuda":
        with torch.cuda.device(device):torch.cuda.manual_seed(seed)


def compare_run_results(reference,candidate,*,repeat,mode,atol,rtol):
    rows=[]
    for kind in ("initial_conditions","final_conditions"):
        expected=reference[0][kind];observed=candidate[0][kind]
        rows.append({"repeat":repeat,"mode":mode,"kind":kind,"passed":expected==observed,"reference":expected,"candidate":observed})
    collections=(("losses",{"updates":torch.tensor(reference[0]["losses"],dtype=torch.float64)},{"updates":torch.tensor(candidate[0]["losses"],dtype=torch.float64)}),*((label,reference[index],candidate[index]) for label,index in (("parameters",1),("gradients",2),("optimizer_state",3))))
    for kind,expected,observed in collections:
        row={"repeat":repeat,"mode":mode,"kind":kind,"atol":atol,"rtol":rtol,"passed":True,"max_absolute_difference":0.0,"checked_tensors":0,"failed_tensors":0,"errors":[]}
        if expected.keys()!=observed.keys():
            row.update(passed=False,missing_tensors=sorted(expected.keys()-observed.keys()),extra_tensors=sorted(observed.keys()-expected.keys()))
        for name in sorted(expected.keys()&observed.keys()):
            value=expected[name];actual=observed[name]
            row["checked_tensors"]+=1
            if value.shape==actual.shape and value.numel():
                difference=float((actual-value).abs().max())
                if math.isfinite(difference):row["max_absolute_difference"]=max(row["max_absolute_difference"],difference)
            try:torch.testing.assert_close(actual,value,atol=atol,rtol=rtol)
            except AssertionError as error:
                row["passed"]=False;row["failed_tensors"]+=1
                if len(row["errors"])<3:row["errors"].append({"tensor":name,"message":str(error)})
        rows.append(row)
    return rows


def write_evidence(path,payload):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\n")
    temporary.replace(path)


def run(options,mode,repeat):
    cpu=mode=="cpu_and_gpu"
    config=training_config(options,mode)
    tokens=torch.arange(max(65536,options.context*16),dtype=torch.long).remainder(128)
    trainer=SharedTrainer(config,tokens,tokens)
    runtime=getattr(trainer.raw_model,"_premat_runtime",None)
    try:
        training_seed=options.training_seed+repeat
        seed_training_rng(training_seed,trainer.device)
        initial_conditions={"training_seed":training_seed,"model_state_sha256":model_digest(trainer.raw_model),**rng_fingerprints(trainer)}
        for _ in range(options.warmup):
            result=trainer.train_one_update()
            if result.get("skipped_update"):raise AssertionError("warmup update was skipped")
        torch.cuda.synchronize(trainer.device)
        torch.cuda.reset_peak_memory_stats(trainer.device)
        before=memory(runtime)
        started=time.perf_counter_ns()
        losses=[]
        for _ in range(options.updates):
            result=trainer.train_one_update()
            if result.get("skipped_update"):raise AssertionError("measured update was skipped")
            losses.append(float(result["training_loss"]))
        # Same final device drain for all providers; no CPU-job completion wait is added.
        torch.cuda.synchronize(trainer.device)
        elapsed=(time.perf_counter_ns()-started)/1e9
        peak_allocated=torch.cuda.max_memory_allocated(trainer.device)
        peak_reserved=torch.cuda.max_memory_reserved(trainer.device)
        after=memory(runtime)
        report={} if runtime is None else runtime.report()
        final_conditions={**rng_fingerprints(trainer),"batch_trace_sha256":trainer.batch_source.trace_digest("train"),"recorded_microbatches":len(trainer.batch_source.training_trace())}
        if cpu:
            health=report.get("cpu_runtime",{})
            assert health.get("staging_peak_bytes",0)<=health.get("staging_limit_bytes",0)
            assert not any(worker.get("cuda_initialized") for worker in health.get("worker_health",()))
        values={name:parameter.detach().cpu().clone() for name,parameter in trainer.raw_model.named_parameters()}
        gradients={name:parameter.grad.detach().cpu().clone() for name,parameter in trainer.raw_model.named_parameters() if parameter.grad is not None}
        payload={"mode":mode,"repeat":repeat,"completed_updates":trainer.state.completed_updates,"initial_conditions":initial_conditions,"final_conditions":final_conditions,"losses":losses,"elapsed_seconds":elapsed,"milliseconds_per_update":elapsed*1000/options.updates,"tokens_per_second":options.updates*options.batch*options.context*options.accumulation/elapsed,"gpu_peak_allocated_bytes":peak_allocated,"gpu_peak_reserved_bytes":peak_reserved,"cpu_memory_before":before,"cpu_memory_after":after,"runtime":report}
        return payload,values,gradients,optimizer_values(trainer)
    finally:
        trainer.close()
        del trainer
        gc.collect()
        torch.cuda.empty_cache()


def main():
    options=parser().parse_args()
    if not torch.cuda.is_available() or not options.device.startswith("cuda"):
        raise SystemExit("An actual CUDA GPU is required; this script does not substitute CPU measurements.")
    if options.dtype=="bfloat16" and not torch.cuda.is_bf16_supported():raise SystemExit("Selected GPU does not support BF16")
    if options.smoke:
        options.updates=3;options.warmup=0;options.repeats=1
    for name in ("updates","repeats","layers","width","heads","batch","context","accumulation"):
        if vars(options)[name]<1:raise SystemExit(name+" must be positive")
    if options.training_seed<0 or options.training_seed+options.repeats-1>2**64-1:raise SystemExit("training seeds must lie in [0, 2**64-1]")
    torch.backends.cuda.matmul.allow_tf32=False
    os.environ["THOG2_DEPTH_MATERIALISATION_MATMUL"]="true" if options.backend=="matmul" else "false"
    from sheet.depth_materialisation_runtime import install_depth_materialisation_runtime
    install_depth_materialisation_runtime()
    payload={"schema_version":2,"status":"running","purpose":"matched unprofiled smoke" if options.smoke else "matched unprofiled preliminary field experiment","host":options.label,"platform":platform.platform(),"python":sys.version,"torch":str(torch.__version__),"cuda":torch.version.cuda,"gpu":torch.cuda.get_device_name(),"configuration":{key:str(value) if isinstance(value,Path) else value for key,value in vars(options).items()},"runs":[],"parity":[],"qualification":"Timing results are unqualified until all parity checks pass. No speedup or memory improvement is assumed. CPU deadlines use ordinary immediate GPU fallback."}
    modes=("off","gpu","cpu_and_gpu")
    atol,rtol=(3e-5,3e-4) if options.dtype=="float32" else (4e-4,0.02) if options.dtype=="float16" else (3e-3,0.06)
    for repeat in range(options.repeats):
        results={}
        for mode in modes[repeat%3:]+modes[:repeat%3]:
            try:record,parameters,gradients,optimizer_state=run(options,mode,repeat)
            except Exception as error:
                payload.update(status="runtime_failed",error={"repeat":repeat,"mode":mode,"type":type(error).__name__,"message":str(error)})
                write_evidence(options.output,payload)
                raise
            payload["runs"].append(record);results[mode]=(record,parameters,gradients,optimizer_state)
            write_evidence(options.output,payload)
            print(json.dumps({**{key:record[key] for key in ("mode","repeat","milliseconds_per_update","tokens_per_second","gpu_peak_allocated_bytes")},"parity_status":"pending"}),flush=True)
        for mode in ("gpu","cpu_and_gpu"):
            payload["parity"].extend(compare_run_results(results["off"],results[mode],repeat=repeat,mode=mode,atol=atol,rtol=rtol))
        failed=[row for row in payload["parity"] if not row["passed"]]
        if failed:
            payload["status"]="parity_failed";write_evidence(options.output,payload)
            raise SystemExit("FAIL numerical parity: "+", ".join(row["mode"]+"/"+row["kind"] for row in failed)+"; timing results unqualified; evidence: "+str(options.output))
        write_evidence(options.output,payload)
    payload["summary"]={mode:{"median_tokens_per_second":statistics.median(item["tokens_per_second"] for item in payload["runs"] if item["mode"]==mode),"median_ms_per_update":statistics.median(item["milliseconds_per_update"] for item in payload["runs"] if item["mode"]==mode),"max_gpu_peak_allocated_bytes":max(item["gpu_peak_allocated_bytes"] for item in payload["runs"] if item["mode"]==mode)} for mode in modes}
    payload["status"]="passed";write_evidence(options.output,payload)
    print("PASS matched losses/gradients/parameters/optimizer state and CPU staging cap; evidence: "+str(options.output),flush=True)


if __name__=="__main__":main()
# ^^^ THOG
