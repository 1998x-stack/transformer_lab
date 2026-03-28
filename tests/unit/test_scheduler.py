import pytest
import torch
import math
from optim.scheduler import NoamScheduler


class TestNoamScheduler:
    def test_noam_scheduler_initial_lr(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(10))], lr=1.0)
        scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=4000)
        assert scheduler._step_num == 1
        lr = scheduler.get_lr()[0]
        assert scheduler._step_num == 2
        assert lr > 0

    def test_noam_scheduler_warmup_phase(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(10))], lr=1.0)
        scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=4000)
        lrs = []
        for _ in range(10):
            lr = scheduler.get_lr()[0]
            lrs.append(lr)
        assert len(lrs) == 10
        assert lrs[0] < lrs[1] < lrs[2]
        assert lrs[-1] > lrs[0]

    def test_noam_scheduler_post_warmup(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(10))], lr=1.0)
        scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=10)
        lrs = []
        for _ in range(20):
            lr = scheduler.get_lr()[0]
            lrs.append(lr)
        assert lrs[0] < lrs[5]
        assert lrs[15] < lrs[9]

    def test_noam_scheduler_formula(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(10))], lr=1.0)
        scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=4000)
        for _ in range(99):
            scheduler.get_lr()
        lr = scheduler.get_lr()[0]
        assert scheduler._step_num == 101
        d_model = scheduler.d_model
        warmup = scheduler.warmup
        step_num = scheduler._step_num
        scale = (d_model**-0.5) * min(step_num**-0.5, step_num * (warmup**-1.5))
        assert math.isclose(lr, scale, rel_tol=1e-6)

    def test_noam_scheduler_state_dict(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(10))], lr=1.0)
        scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=4000)
        for _ in range(4):
            scheduler.get_lr()
        assert scheduler._step_num == 5
        state = scheduler.state_dict()
        assert "_step_num" in state
        assert state["_step_num"] == 5
        new_optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(10))], lr=1.0)
        new_scheduler = NoamScheduler(new_optimizer, d_model=256, warmup_steps=2000)
        new_scheduler.load_state_dict(state)
        assert new_scheduler._step_num == 5

    def test_noam_scheduler_different_d_model(self):
        param1 = torch.nn.Parameter(torch.randn(10))
        param2 = torch.nn.Parameter(torch.randn(10))
        optimizer1 = torch.optim.SGD([param1], lr=1.0)
        optimizer2 = torch.optim.SGD([param2], lr=1.0)
        scheduler_small = NoamScheduler(optimizer1, d_model=256, warmup_steps=4000)
        scheduler_large = NoamScheduler(optimizer2, d_model=1024, warmup_steps=4000)
        lr_small = scheduler_small.get_lr()[0]
        lr_large = scheduler_large.get_lr()[0]
        assert lr_large < lr_small

    def test_noam_scheduler_one_warmup(self):
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(10))], lr=1.0)
        scheduler = NoamScheduler(optimizer, d_model=512, warmup_steps=1)
        lr1 = scheduler.get_lr()[0]
        lr2 = scheduler.get_lr()[0]
        assert lr2 < lr1

    def test_noam_scheduler_ignores_base_lr(self):
        param1 = torch.nn.Parameter(torch.randn(10))
        param2 = torch.nn.Parameter(torch.randn(10))
        optimizer_high = torch.optim.SGD([param1], lr=100.0)
        optimizer_low = torch.optim.SGD([param2], lr=0.01)
        scheduler_high = NoamScheduler(optimizer_high, d_model=512, warmup_steps=4000)
        scheduler_low = NoamScheduler(optimizer_low, d_model=512, warmup_steps=4000)
        lr_high = scheduler_high.get_lr()[0]
        lr_low = scheduler_low.get_lr()[0]
        assert math.isclose(lr_high, lr_low, rel_tol=1e-10)
