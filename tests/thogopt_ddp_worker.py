# vvv THOG
"""Two-rank Gloo validation: reduce before squaring, then real accumulated model updates."""
import os
import torch
import torch.distributed as dist
from sheet.stage4_trainer import Stage4Trainer
from tests.stage4_test_support import stage4_training_config,stage4_tokens
from tests.test_thogopt import make_optimizer,supply


def main():
    torch.set_num_threads(1)
    dist.init_process_group('gloo')
    rank=dist.get_rank()
    _,parameter,optimizer=make_optimizer()
    supply(optimizer,torch.full((4,4),1. if rank==0 else 3.,dtype=torch.float64))
    optimizer.step()
    torch.testing.assert_close(optimizer.state[parameter]['exp_avg_sq'],torch.full((4,4),.2,dtype=torch.float64))
    # One rank's invalid input must reject the update on both ranks.
    before=parameter.detach().clone()
    supply(optimizer,torch.full((4,4),float('nan') if rank==0 else 1.,dtype=torch.float64))
    try: optimizer.step()
    except FloatingPointError: pass
    else: raise AssertionError('non-finite rank was not rejected')
    torch.testing.assert_close(parameter,before,atol=0,rtol=0)
    os.environ['THOG2_OPTIMIZER']='thogopt'
    os.environ['THOG2_FAST_DISCARD']='true'
    trainer=Stage4Trainer(stage4_training_config(geometry_preset='depth',gradient_accumulation_steps=2),*stage4_tokens())
    for _ in range(2): trainer.train_one_update()
    for parameter in trainer.raw_model.parameters():
        reference=parameter.detach().clone()
        dist.broadcast(reference,src=0)
        torch.testing.assert_close(parameter,reference,atol=0,rtol=0)
    for state in trainer.optimizer.state.values():
        for name in ('exp_avg','exp_avg_sq'):
            reference=state[name].clone();dist.broadcast(reference,src=0)
            torch.testing.assert_close(state[name],reference,atol=0,rtol=0)
    if rank==0: print('THOGOPT_DDP_PASS: reduction before square, collective rejection, accumulated model and moment parity',flush=True)
    dist.destroy_process_group()

if __name__=='__main__':main()
# ^^^ THOG
