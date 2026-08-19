def test_parse_copy_corpus_src_equals_tgt(tmp_path):
    from data.corpus import parse_copy_corpus

    f = tmp_path / "c.txt"
    f.write_text("The quick brown fox jumps over the lazy dog.\n\nFOO BAR\n\nOnce upon a time there was a king.\n")
    recs = parse_copy_corpus(str(f), min_len=4, max_len=100)
    assert len(recs) == 2  # two real sentences; the ALL-CAPS 'FOO BAR' is dropped
    for r in recs:
        assert r["src"] == r["tgt"]
        assert r["src"].strip()


def test_load_copy_corpus_splits(tmp_path):
    from data.datasets import load_copy_corpus

    f = tmp_path / "c.txt"
    lines = [f"line {i} eleven words here to make length" for i in range(100)]
    f.write_text("\n".join(lines) + "\n")
    ddict = load_copy_corpus(str(f), min_len=4, max_len=100, seed=0)
    assert set(ddict.keys()) == {"train", "validation", "test"}
    assert len(ddict["train"]) > 0
    assert sum(len(ddict[k]) for k in ("train", "validation", "test")) == 100