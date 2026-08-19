def test_dataconfig_accepts_copy_and_corpus_path():
    from utils.config import DataConfig

    cfg = DataConfig(dataset="copy", corpus_path="sample_corpus.txt")
    assert cfg.dataset == "copy"
    assert cfg.corpus_path == "sample_corpus.txt"


def test_load_copy_config():
    from utils.config import load_config

    cfg = load_config("configs/copy_en_en.yaml")
    assert cfg.data.dataset == "copy"
    assert cfg.data.corpus_path == "sample_corpus.txt"
    assert cfg.model.N == 6
    assert cfg.optim.max_steps == 6000