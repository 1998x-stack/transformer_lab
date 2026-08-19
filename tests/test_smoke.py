import torch

from data.collate import Sample, collate
from data.tokenization import SPTokenizer
from models.label_smoothing import LabelSmoothingLoss
from models.transformer import Transformer


def _tiny_tokenizer(tmp_path):
    import sentencepiece as spm

    lines = [f"This is sentence number {i} in the fairy corpus." for i in range(60)]
    f = tmp_path / "c.txt"
    f.write_text("\n".join(lines), encoding="utf-8")
    spm.SentencePieceTrainer.train(
        input=str(f), model_prefix=str(tmp_path / "spm"), vocab_size=150,
        character_coverage=1.0, model_type="bpe", bos_id=1, eos_id=2, pad_id=0, unk_id=3,
    )
    return SPTokenizer(str(tmp_path / "spm.model"))


def test_smoke_training_decreases_loss(tmp_path):
    tok = _tiny_tokenizer(tmp_path)
    torch.manual_seed(0)
    model = Transformer(
        vocab_size=tok.vocab_size, N=1, d_model=64, d_ff=128, num_heads=4, dropout=0.0,
        attn_dropout=0.0, activation="relu", share_embeddings=True, tie_softmax_weight=True,
        pos_encoding="sinusoidal",
    )
    crit = LabelSmoothingLoss(classes=tok.vocab_size, smoothing=0.0, ignore_index=-100)
    opt = torch.optim.Adam(model.parameters(), lr=3e-2)

    phrases = [
        "this is the first sample sentence",
        "the second line carries more meaning",
        "a third fairy tale begins here",
        "and finally the fourth one ends",
    ]
    samples = [Sample(tok.encode(p), tok.encode(p, add_bos=True, add_eos=True)) for p in phrases]
    batch = collate(samples, pad_id=tok.pad_id)

    losses = []
    for _ in range(20):
        opt.zero_grad()
        pad = tok.pad_id
        src = batch["src_ids"]
        tgt_in = batch["tgt_in_ids"]
        tgt_out = batch["tgt_out_ids"]
        target = tgt_out.masked_fill(tgt_out == pad, -100)
        logits = model(src, tgt_in, src != pad, tgt_in != pad)
        loss = crit(logits, target)
        loss.backward()
        opt.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], f"loss must decrease, got {losses[0]} -> {losses[-1]}"