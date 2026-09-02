# vvv THOG
"""Small reproducible optimizer overhead measurement; synthetic tokens, no quality claim."""
import argparse
import contextlib
import io
import json
import os
import resource
import time
from pathlib import Path
import torch
from sheet.stage4_trainer import Stage4Trainer
from sheet.training_config import TrainingConfig


def benchmark(*,layers=8,order=3,width=16,steps=20,device='cpu'):
    torch.set_num_threads(1)
    reports=[]
    def synchronize():
        if device.startswith('cuda'):torch.cuda.synchronize()
    for model_type,optimizer_name in [('dense','adamw'),('thog2_sheet','adamw'),('thog2_sheet','thogopt')]:
        os.environ['THOG2_OPTIMIZER']=optimizer_name
        os.environ['THOG2_FAST_DISCARD']='true'
        config=TrainingConfig(model_type=model_type,geometry_preset='depth' if model_type=='thog2_sheet' else None,
            n_layer=layers,n_head=2,n_embd=width,depth_order=order,base_row_order=1,
            block_size=8,vocab_size=32,batch_size=2,gradient_accumulation_steps=1,
            max_updates=steps+3,decay_updates=steps+3,device=device,dtype='float32',
            eval_interval=0,eval_batches=1,checkpoint_interval=0,dropout=0.,weight_decay=.01)
        tokens=torch.arange(2048)%32
        with contextlib.redirect_stdout(io.StringIO()): trainer=Stage4Trainer(config,tokens,tokens)
        optimizer_time=[0.]
        original=trainer.optimizer.step
        def timed(*args,**kwargs):
            synchronize();started=time.perf_counter()
            result=original(*args,**kwargs)
            synchronize();optimizer_time[0]+=time.perf_counter()-started
            return result
        trainer.optimizer.step=timed
        for _ in range(3):trainer.train_one_update()
        optimizer_time[0]=0.
        totals={'fit_seconds':0.,'gradient_staging_seconds':0.,'gradient_preparation_seconds':0.}
        synchronize();started=time.perf_counter()
        for _ in range(steps):
            trainer.train_one_update()
            metrics=getattr(trainer.optimizer,'last_step_metrics',{})
            totals['fit_seconds']+=sum(item['fit_seconds'] for item in metrics.get('families',{}).values())
            for key in ('gradient_staging_seconds','gradient_preparation_seconds'):totals[key]+=metrics.get(key,0.)
        synchronize();elapsed=time.perf_counter()-started
        moment_bytes=sum(t.numel()*t.element_size() for state in trainer.optimizer.state.values() for key,t in state.items() if key in ('exp_avg','exp_avg_sq'))
        report={'model':model_type,'optimizer':optimizer_name,'layers':layers,'weight_order':order,'width':width,'steps':steps,
            'dtype':'float32','device':device,'elapsed_seconds':elapsed,'optimizer_seconds':optimizer_time[0],
            'forward_backward_and_other_seconds':elapsed-optimizer_time[0],'tokens_per_second':steps*2*8/elapsed,
            'all_parameter_moment_bytes':moment_bytes,'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            'instrumentation':'disabled; no telemetry attached','synthetic_tokens':True,**totals}
        if hasattr(trainer.optimizer,'resource_report'):report['compressed_families']=trainer.optimizer.resource_report()
        reports.append(report)
        trainer.close()
    return {'torch_version':torch.__version__,'reports':reports,'limitations':['Synthetic tiny CPU workload is not OpenWebText training quality evidence.','Process peak RSS is cumulative across cases; not an isolated optimizer allocation peak.','Host gradient staging remains O(L * couplings); device moments use O((H_m + H_v) * couplings).']}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--layers',type=int,default=8);parser.add_argument('--order',type=int,default=3)
    parser.add_argument('--width',type=int,default=16);parser.add_argument('--steps',type=int,default=20);parser.add_argument('--device',default='cpu');parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();output=args.output;values={key:vars(args)[key] for key in ('layers','order','width','steps','device')}
    result=benchmark(**values);output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n');print(output)
if __name__=='__main__':main()
# ^^^ THOG
