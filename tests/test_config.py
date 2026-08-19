def test_dataconfig_accepts_copy_and_corpus_path():
    from utils.config import DataConfig

    cfg = DataConfig(dataset="copy", corpus_path="sample_corpus.txt")
    assert cfg.dataset == "copy"
    assert cfg.corpus_path == "sample_corpus.txt"