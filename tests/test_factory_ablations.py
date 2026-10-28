"""CPU-only contracts for factory mode and paper ablation machinery."""
from dataclasses import replace
import json
import math
import torch
import pytest
from models.factory import build_model, Registry
from models.transformer import MultiHeadAttention, PositionalEncoding
from models.label_smoothing import LabelSmoothingLoss
from research.ablations import paper_matrix, mechanism_matrix, materialize
from research.tracking import RunRecorder
from utils.config import ModelConfig, load_config
from optim.scheduler import NoamScheduler


def test_factories_validate_component_names():
    config = replace(ModelConfig(), architecture="unknown")
    with pytest.raises(ValueError, match="Unknown architecture"):
        build_model(config, 24)
    reg = Registry("test")
    reg.register("one")(lambda: 1)
    with pytest.raises(ValueError, match="Duplicate"):
        reg.register("one")(lambda: 2)


def tiny(**kwargs):
    cfg = ModelConfig(N=1, d_model=32, d_ff=64, num_heads=4,
                      dropout=0.0, attn_dropout=0.0,
                      label_smoothing=0.0, **kwargs)
    return build_model(cfg, 31).eval()


def test_independent_key_value_projection_dims():
    model = tiny(d_k=2, d_v=3, final_norm=False)
    attn = model.encoder[0].self_attn
    assert attn.q_proj.out_features == 8
    assert attn.k_proj.out_features == 8
    assert attn.v_proj.out_features == 12
    assert attn.o_proj.in_features == 12
    x = torch.randint(1, 31, (2, 5))
    y = torch.randint(1, 31, (2, 4))
    out = model(x, y, x.ne(0), y.ne(0))
    assert out.shape == (2, 4, 31)
    out.mean().backward()
    assert torch.isfinite(attn.q_proj.weight.grad).all()


def test_future_target_tokens_do_not_leak_to_prefix():
    torch.manual_seed(0)
    model = tiny(d_k=2, d_v=2)
    x = torch.tensor([[3, 4, 5, 6]])
    target_a = torch.tensor([[1, 2, 3, 4]])
    target_b = torch.tensor([[1, 2, 17, 29]])
    with torch.no_grad():
        result_a = model(x, target_a, x.ne(0), target_a.ne(0))
        result_b = model(x, target_b, x.ne(0), target_b.ne(0))
    assert torch.allclose(result_a[:, :2], result_b[:, :2], atol=1e-5)


def test_empty_mask_and_capture_are_finite():
    m = MultiHeadAttention(32, 4, dropout=0.0)
    m.capture_attention = True
    output = m(torch.ones(1, 2, 32), torch.ones(1, 2, 32),
               torch.ones(1, 2, 32), torch.zeros(1, 1, 2, 2, dtype=torch.bool))
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(m.last_attention) == 0


def test_attention_scaling_control_and_position_none():
    model = tiny(attention_kind="unscaled_dot", pos_encoding="none", norm_style="pre")
    assert not model.encoder[0].self_attn.attention_scale
    encoding = PositionalEncoding(32, dropout=0, mode="none")
    x = torch.randn(1, 7, 32)
    assert torch.equal(encoding(x), x)
    src = torch.randint(1, 31, (1, 3))
    pred = model(src, src, src.ne(0), src.ne(0))
    assert torch.isfinite(pred).all()


def test_paper_matrix_preserves_dv_and_is_not_mechanism():
    names = [x.name for x in paper_matrix()]
    assert len(names) == 20
    assert len(set(names)) == len(names)
    assert len(mechanism_matrix()) == 8
    base = {"model": {"num_heads": 8, "d_model": 512, "d_ff": 2048,
                      "attn_dropout": 0.0}, "optim": {"max_steps": 100000},
            "runtime": {}}
    item = next(x for x in paper_matrix() if x.name == "B_dk_16")
    cfg = materialize(base, item, 42, __import__("pathlib").Path("work/a"))
    assert cfg["model"]["d_k"] == 16
    assert "d_v" not in cfg["model"]
    assert cfg["model"]["final_norm"] is False
    assert cfg["model"]["ffn_inner_dropout"] == 0.0
    assert base["model"]["num_heads"] == 8


def test_all_padding_label_smoothing_has_zero_grad():
    loss = LabelSmoothingLoss(classes=10, smoothing=.1, ignore_index=0)
    pred = torch.randn(2, 3, 10, requires_grad=True)
    value = loss(pred, torch.zeros((2, 3), dtype=torch.long))
    value.backward()
    assert value.item() == 0
    assert torch.equal(pred.grad, torch.zeros_like(pred))


def test_noam_get_lr_has_no_side_effect():
    param = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.Adam([param], lr=1e-3)
    sched = NoamScheduler(opt, d_model=512, warmup_steps=4000)
    assert sched.get_lr() == sched.get_lr()
    before = sched.get_last_lr()[0]
    opt.step()
    sched.step()
    assert sched.get_last_lr()[0] > before


def test_run_recorder_emits_real_records(tmp_path):
    cfg = load_config("configs/copy_en_en.yaml")
    cfg = replace(cfg, runtime=replace(cfg.runtime, log_dir=str(tmp_path)))
    record = RunRecorder(cfg, parameters=123)
    record.record("train", 5, nll=2.0, ppl=math.exp(2))
    manifest = json.loads((record.directory / "manifest.json").read_text())
    row = json.loads(record.metrics_path.read_text().strip())
    assert manifest["parameters"] == 123
    assert manifest["config_sha256"]
    assert row["step"] == 5
    assert row["metrics"]["nll"] == 2.0
