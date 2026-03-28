import pytest
import math
import torch
import time
from utils.metrics import perplexity, compute_bleu, SpeedMeter, estimate_train_flops


class TestSpeedMeter:
    def test_speed_meter_initialization(self):
        meter = SpeedMeter()
        assert meter.tokens == 0
        assert meter.t0 > 0

    def test_speed_meter_update(self):
        meter = SpeedMeter()
        meter.update(100)
        assert meter.tokens == 100
        meter.update(50)
        assert meter.tokens == 150

    def test_speed_meter_rate(self):
        meter = SpeedMeter()
        time.sleep(0.01)
        meter.update(100)
        rate = meter.rate()
        assert rate > 0
        assert isinstance(rate, float)

    def test_speed_meter_rate_with_multiple_updates(self):
        meter = SpeedMeter()
        time.sleep(0.01)
        meter.update(100)
        time.sleep(0.01)
        meter.update(100)
        rate = meter.rate()
        assert rate > 0
        expected_rate = 200 / (time.time() - meter.t0)
        assert math.isclose(rate, expected_rate, rel_tol=0.1)


class TestPerplexity:
    def test_perplexity_basic(self):
        loss = 2.0
        expected_ppl = math.exp(loss)
        result = perplexity(loss)
        assert math.isclose(result, expected_ppl, rel_tol=1e-6)
        assert result > 7.38 and result < 7.40

    def test_perplexity_zero_loss(self):
        loss = 0.0
        result = perplexity(loss)
        assert result == 1.0

    def test_perplexity_negative_loss(self):
        loss = -1.0
        result = perplexity(loss)
        expected = math.exp(-1.0)
        assert math.isclose(result, expected, rel_tol=1e-6)
        assert result < 1.0

    def test_perplexity_tensor(self):
        loss_tensor = torch.tensor(2.5, dtype=torch.float32)
        result = perplexity(loss_tensor.item())
        expected = math.exp(2.5)
        assert math.isclose(result, expected, rel_tol=1e-6)

    def test_perplexity_small_loss(self):
        loss = 0.1
        result = perplexity(loss)
        expected = math.exp(0.1)
        assert math.isclose(result, expected, rel_tol=1e-6)
        assert result < 2.0

    def test_perplexity_large_loss(self):
        loss = 10.0
        result = perplexity(loss)
        expected = math.exp(10.0)
        assert math.isclose(result, expected, rel_tol=1e-6)
        assert result > 20000


class TestComputeBLEU:
    def test_compute_bleu_identical(self):
        preds = ["the cat sat on the mat"]
        refs = ["the cat sat on the mat"]
        result = compute_bleu(preds, refs)
        assert math.isclose(result, 100.0, rel_tol=1e-10)

    def test_compute_bleu_completely_different(self):
        preds = ["apple orange banana"]
        refs = ["the cat sat on the mat"]
        result = compute_bleu(preds, refs)
        assert result < 10.0

    def test_compute_bleu_partial_match(self):
        preds = ["the cat sat on the mat"]
        refs = ["the dog sat on the rug"]
        result = compute_bleu(preds, refs)
        assert result > 20.0 and result < 80.0

    def test_compute_bleu_empty_prediction(self):
        preds = [""]
        refs = ["the cat sat on the mat"]
        result = compute_bleu(preds, refs)
        assert result == 0.0

    def test_compute_bleu_partial_multiple_sentences(self):
        preds = ["the cat sat", "the dog ran fast"]
        refs = ["the cat sat", "the dog quickly ran"]
        result = compute_bleu(preds, refs)
        assert result >= 0.0 and result <= 100.0


class TestEstimateTrainFLOPS:
    def test_estimate_train_flops_basic(self):
        hours = 1.0
        num_gpus = 8
        sustained_tflops_per_gpu = 100.0
        result = estimate_train_flops(hours, num_gpus, sustained_tflops_per_gpu)
        expected = 1.0 * 3600 * 8 * 100.0 * 1e12
        assert math.isclose(result, expected, rel_tol=1e-10)

    def test_estimate_train_flops_zero_hours(self):
        hours = 0.0
        num_gpus = 8
        sustained_tflops_per_gpu = 100.0
        result = estimate_train_flops(hours, num_gpus, sustained_tflops_per_gpu)
        assert result == 0.0

    def test_estimate_train_flops_single_gpu(self):
        hours = 24.0
        num_gpus = 1
        sustained_tflops_per_gpu = 50.0
        result = estimate_train_flops(hours, num_gpus, sustained_tflops_per_gpu)
        expected = 24.0 * 3600 * 1 * 50.0 * 1e12
        assert math.isclose(result, expected, rel_tol=1e-10)
