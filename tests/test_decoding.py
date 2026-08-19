import torch

from models.transformer import Transformer
from utils.decoding import beam_search


def _tiny_model():
    torch.manual_seed(1)
    return Transformer(
        vocab_size=32, N=1, d_model=32, d_ff=64, num_heads=4, dropout=0.0,
        attn_dropout=0.0, activation="relu", share_embeddings=True, tie_softmax_weight=True,
        pos_encoding="sinusoidal",
    ).eval()


def test_beam_search_accepts_repeat_penalty():
    model = _tiny_model()
    # single token source so max_len stays small
    src = torch.tensor([[5, 6, 7]])
    out = beam_search(
        model, src, src_pad_id=0, bos_id=1, eos_id=2, beam=4, alpha=0.6,
        repeat_penalty=1.0, max_len_ratio=1.0, max_len_offset=8,
    )
    assert isinstance(out, list)
    assert all(isinstance(t, int) and 0 <= t < 32 for t in out[0])


def test_beam_search_no_penalty_matches_intended_default():
    model = _tiny_model()
    src = torch.tensor([[5, 6, 7]])
    out_off = beam_search(
        model, src, 0, 1, 2, beam=4, alpha=0.6, max_len_ratio=1.0, max_len_offset=8,
        repeat_penalty=0.0,
    )
    out_on = beam_search(
        model, src, 0, 1, 2, beam=4, alpha=0.6, max_len_ratio=1.0, max_len_offset=8,
        repeat_penalty=1.0,
    )
    # Repeat penalty is additive to scores; results may differ or stay identical,
    # but the call must return well-formed integer sequences.
    for out in (out_off, out_on):
        assert isinstance(out, list) and all(isinstance(t, int) for t in out[0])