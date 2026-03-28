import tempfile
from pathlib import Path

import pytest
import torch
from datasets import Dataset, DatasetDict

from data.collate import DynamicBatcher, Sample, collate
from data.datasets import filter_and_rename
from data.tokenization import SPTokenizer, train_or_load_spm
from utils.config import DataConfig


def create_mock_dataset():
    data = {
        "train": [
            {
                "translation": {
                    "de": "die katze sitzt auf der matte",
                    "en": "the cat sits on the mat",
                }
            },
            {
                "translation": {
                    "de": "der hund läuft im park",
                    "en": "the dog runs in the park",
                }
            },
            {
                "translation": {
                    "de": "ein vogel fliegt am himmel",
                    "en": "a bird flies in the sky",
                }
            },
            {
                "translation": {
                    "de": "der fisch schwimmt im wasser",
                    "en": "the fish swims in the water",
                }
            },
            {
                "translation": {
                    "de": "das kind spielt mit spielzeug",
                    "en": "the child plays with toys",
                }
            },
        ],
        "validation": [
            {
                "translation": {
                    "de": "die sonne scheint hell",
                    "en": "the sun shines bright",
                }
            },
            {
                "translation": {
                    "de": "der mond leuchtet nachts",
                    "en": "the moon glows at night",
                }
            },
        ],
    }
    return DatasetDict(
        {split: Dataset.from_list(examples) for split, examples in data.items()}
    )


def create_minimal_data_config():
    return DataConfig(
        dataset="wmt14",
        lang_pair="de-en",
        max_src_len=32,
        max_tgt_len=32,
        min_len=1,
        vocab_size=50,
        use_shared_vocab=True,
        max_tokens_per_batch=1000,
        num_buckets=2,
        cache_dir=None,
        tokenizer_dir="work/tokenizer",
    )


def create_minimal_data_config_with_tokenizer(tokenizer_dir):
    """Create minimal data config with specific tokenizer directory."""
    return DataConfig(
        dataset="wmt14",
        lang_pair="de-en",
        max_src_len=32,
        max_tgt_len=32,
        min_len=1,
        vocab_size=50,
        use_shared_vocab=True,
        max_tokens_per_batch=1000,
        num_buckets=2,
        cache_dir=None,
        tokenizer_dir=tokenizer_dir,
    )


def preprocess_for_tokenizer(dataset):
    """Convert dataset from translation format to src/tgt format."""

    def _convert(example):
        return {
            "src": example["translation"]["de"],
            "tgt": example["translation"]["en"],
        }

    return dataset.map(_convert, remove_columns=dataset.column_names)


@pytest.fixture
def temp_data_dirs():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        tokenizer_dir = tmpdir / "tokenizer"
        data_dir = tmpdir / "data"
        cache_dir = tmpdir / "cache"

        tokenizer_dir.mkdir()
        data_dir.mkdir()
        cache_dir.mkdir()

        yield {
            "base": tmpdir,
            "tokenizer_dir": tokenizer_dir,
            "data_dir": data_dir,
            "cache_dir": cache_dir,
        }


@pytest.mark.integration
@pytest.mark.slow
class TestDataPipeline:
    def test_data_pipeline_tokenizer_training(self, temp_data_dirs):
        """Test that SentencePiece tokenizer can be trained on a small corpus."""
        config = create_minimal_data_config()
        config = create_minimal_data_config_with_tokenizer(
            str(temp_data_dirs["tokenizer_dir"])
        )

        mock_dataset = create_mock_dataset()

        tokenizer = train_or_load_spm(
            train_ds=preprocess_for_tokenizer(mock_dataset["train"]),
            save_dir=config.tokenizer_dir,
            vocab_size=config.vocab_size,
        )

        assert isinstance(tokenizer, SPTokenizer), "Should return SPTokenizer instance"
        assert tokenizer.vocab_size > 0, "Vocabulary size should be positive"
        assert (
            tokenizer.vocab_size <= config.vocab_size + 10
        ), "Vocab size should be close to target"

        assert tokenizer.pad_id >= 0, "PAD ID should be defined"
        assert tokenizer.bos_id >= 0, "BOS ID should be defined"
        assert tokenizer.eos_id >= 0, "EOS ID should be defined"

        test_sentence = "the cat sits on the mat"
        encoded = tokenizer.encode(test_sentence)
        assert isinstance(encoded, list), "Encoded output should be a list"
        assert len(encoded) > 0, "Encoded output should not be empty"
        assert all(
            isinstance(tok, int) for tok in encoded
        ), "All tokens should be integers"

        decoded = tokenizer.decode(encoded)
        assert isinstance(decoded, str), "Decoded output should be a string"

    def test_data_pipeline_dataset_loading(self):
        """Test that dataset can be loaded and processed."""
        mock_dataset = create_mock_dataset()

        assert "train" in mock_dataset, "Dataset should have train split"
        assert "validation" in mock_dataset, "Dataset should have validation split"

        train_examples = list(mock_dataset["train"])
        assert len(train_examples) == 5, "Train split should have 5 examples"

        first_example = train_examples[0]
        assert "translation" in first_example, "Example should have translation field"
        assert "de" in first_example["translation"], "Should have German text"
        assert "en" in first_example["translation"], "Should have English text"

    def test_data_pipeline_filter_and_rename(self):
        """Test dataset filtering and renaming."""
        mock_dataset = create_mock_dataset()

        filtered = filter_and_rename(
            ddict=mock_dataset,
            src_lang="de",
            tgt_lang="en",
            min_len=1,
            max_src=32,
            max_tgt=32,
        )

        assert "train" in filtered, "Filtered dataset should have train split"
        assert "validation" in filtered, "Filtered dataset should have validation split"

        train_example = filtered["train"][0]
        assert "src" in train_example, "Should have 'src' field"
        assert "tgt" in train_example, "Should have 'tgt' field"
        assert "src_len" in train_example, "Should have 'src_len' field"
        assert "tgt_len" in train_example, "Should have 'tgt_len' field"

        assert isinstance(train_example["src"], str), "Source should be string"
        assert isinstance(train_example["tgt"], str), "Target should be string"
        assert isinstance(train_example["src_len"], int), "Source length should be int"
        assert isinstance(train_example["tgt_len"], int), "Target length should be int"

    def test_data_pipeline_tokenization(self, temp_data_dirs):
        """Test tokenization of source and target sentences."""
        config = create_minimal_data_config()
        config = create_minimal_data_config_with_tokenizer(
            str(temp_data_dirs["tokenizer_dir"])
        )

        mock_dataset = create_mock_dataset()
        tokenizer = train_or_load_spm(
            train_ds=preprocess_for_tokenizer(mock_dataset["train"]),
            save_dir=config.tokenizer_dir,
            vocab_size=config.vocab_size,
        )

        test_sentence = "the cat sits on the mat"
        encoded = tokenizer.encode(test_sentence)
        assert len(encoded) > 0, "Should produce token IDs"

        decoded = tokenizer.decode(encoded)
        assert isinstance(decoded, str), "Should decode to string"

        encoded_with_bos = tokenizer.encode(test_sentence, add_bos=True)
        assert len(encoded_with_bos) == len(encoded) + 1, "Should add BOS token"
        assert encoded_with_bos[0] == tokenizer.bos_id, "First token should be BOS"

        encoded_with_eos = tokenizer.encode(test_sentence, add_eos=True)
        assert len(encoded_with_eos) == len(encoded) + 1, "Should add EOS token"
        assert encoded_with_eos[-1] == tokenizer.eos_id, "Last token should be EOS"

    def test_data_pipeline_collate(self):
        """Test batch collation with padding."""
        pad_id = 0

        samples = [
            Sample(src_ids=[1, 2, 3], tgt_ids=[4, 5, 6]),
            Sample(src_ids=[1, 2], tgt_ids=[4, 5]),
            Sample(src_ids=[1, 2, 3, 4], tgt_ids=[4, 5, 6, 7]),
        ]

        batch = collate(samples, pad_id)

        assert "src_ids" in batch, "Batch should have src_ids"
        assert "tgt_in_ids" in batch, "Batch should have tgt_in_ids"
        assert "tgt_out_ids" in batch, "Batch should have tgt_out_ids"

        src_ids = batch["src_ids"]
        assert isinstance(src_ids, torch.Tensor), "src_ids should be torch.Tensor"
        assert src_ids.shape[0] == 3, "Batch size should be 3"
        assert src_ids.shape[1] == 4, "Sequence length should be 4 (max)"
        assert src_ids[1, 2].item() == pad_id, "Padded positions should be pad_id"

        tgt_in_ids = batch["tgt_in_ids"]
        assert tgt_in_ids.shape == (3, 3), "tgt_in_ids should be (3, 3)"

        tgt_out_ids = batch["tgt_out_ids"]
        assert tgt_out_ids.shape == (3, 3), "tgt_out_ids should be (3, 3)"

    def test_data_pipeline_dynamic_batcher_initialization(self, temp_data_dirs):
        """Test DynamicBatcher initialization."""
        config = create_minimal_data_config()
        config = create_minimal_data_config_with_tokenizer(
            str(temp_data_dirs["tokenizer_dir"])
        )

        mock_dataset = create_mock_dataset()
        tokenizer = train_or_load_spm(
            train_ds=preprocess_for_tokenizer(mock_dataset["train"]),
            save_dir=config.tokenizer_dir,
            vocab_size=config.vocab_size,
        )

        batcher = DynamicBatcher(
            dataset=preprocess_for_tokenizer(mock_dataset["train"]),
            tokenizer=tokenizer,
            max_tokens=100,
            num_buckets=2,
            shuffle=False,
        )

        assert hasattr(batcher, "ds"), "Should have dataset"
        assert hasattr(batcher, "tok"), "Should have tokenizer"
        assert hasattr(batcher, "max_tokens"), "Should have max_tokens"
        assert hasattr(batcher, "lengths"), "Should have precomputed lengths"
        assert len(batcher.lengths) == len(
            mock_dataset["train"]
        ), "Should have length for each example"
