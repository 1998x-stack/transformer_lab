import torch


def test_resolve_cpu():
    from utils.torch_utils import resolve_device

    assert resolve_device("cpu").type == "cpu"


def test_resolve_fallback_on_unavailable_backend():
    from utils.torch_utils import resolve_device

    dev = resolve_device("cuda") if not torch.cuda.is_available() else resolve_device("cpu")
    assert isinstance(dev, torch.device)
    assert dev.type in ("cuda", "mps", "cpu")


def test_resolve_none_returns_something():
    from utils.torch_utils import resolve_device

    assert resolve_device(None).type in ("cuda", "mps", "cpu")