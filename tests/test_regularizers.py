import pytest

torch = pytest.importorskip("torch")

from src.regularizers import SpectralNormPenalty, nuclear_norm_penalty


def test_nuclear_norm_known_value():
    x = torch.diag(torch.tensor([1.0, 2.0, 3.0]))
    val = nuclear_norm_penalty(x, center=False)
    assert abs(val.item() - 6.0 / 3 ** 0.5) < 1e-5


def test_nuclear_norm_has_gradient():
    x = torch.randn(16, 8, requires_grad=True)
    nuclear_norm_penalty(x).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_spectral_penalty_matches_svd():
    torch.manual_seed(0)
    lin = torch.nn.Sequential(torch.nn.Linear(8, 8, bias=False))
    pen = SpectralNormPenalty(lin, scope="linear", n_iter=50)
    true = torch.linalg.svdvals(lin[0].weight).max().item()
    assert abs(pen().item() - true) < 1e-3
