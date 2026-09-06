import pytest
import torch
import tropical_gemm
from tropical_gemm import pytorch as tg


@pytest.mark.parametrize("op", ["maxplus", "minplus", "maxmul"])
def test_batched_double_precision_preserves_winner(op):
    values = [1.0, 1.0 + 1e-8] if op != "minplus" else [1.0, 1.0 - 1e-8]
    a = torch.tensor([[values]], dtype=torch.float64, requires_grad=True)
    b = torch.full(
        (1, 2, 1),
        1.0 if op == "maxmul" else 0.0,
        dtype=torch.float64,
        requires_grad=True,
    )
    c = getattr(tg, f"tropical_{op}_matmul_batched")(a, b)
    ref = getattr(tg, f"tropical_{op}_matmul")(a[0], b[0])
    assert c.dtype == torch.float64
    torch.testing.assert_close(c[0], ref, rtol=0, atol=0)
    c.sum().backward()
    torch.testing.assert_close(
        a.grad, torch.tensor([[[0.0, 1.0]]], dtype=torch.float64)
    )


@pytest.mark.parametrize("batched", [False, True])
def test_maxmul_detects_input_mutation(batched):
    shape = (1, 1, 1) if batched else (1, 1)
    a = torch.full(shape, 2.0, requires_grad=True)
    b = torch.full(shape, 3.0, requires_grad=True)
    fn = tg.tropical_maxmul_matmul_batched if batched else tg.tropical_maxmul_matmul
    c = fn(a, b)
    with torch.no_grad():
        b.fill_(100.0)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        c.sum().backward()


@pytest.mark.skipif(
    not torch.cuda.is_available() or not tropical_gemm.cuda_available(),
    reason="CUDA required",
)
@pytest.mark.parametrize("batched", [False, True])
def test_dlpack_nondefault_producer_and_consumer_streams(batched):
    shape = (1, 64, 64) if batched else (64, 64)
    fn = (
        tropical_gemm.maxplus_matmul_batched_dlpack
        if batched
        else tropical_gemm.maxplus_matmul_dlpack
    )
    a = torch.zeros(shape, device="cuda")
    b = torch.zeros_like(a)
    torch.cuda.synchronize()
    producer = torch.cuda.Stream()
    consumer = torch.cuda.Stream()
    with torch.cuda.stream(producer):
        torch.cuda._sleep(20_000_000)
        a.fill_(1.0)
        b.fill_(2.0)
        # The binding must tell the producer which stream will consume the inputs.
        caps = fn(a, b)
    with torch.cuda.stream(consumer):
        c = torch.from_dlpack(caps[0]).clone()
        idx = torch.from_dlpack(caps[1])
        assert idx.dtype == torch.int32
    torch.cuda.synchronize()
    torch.testing.assert_close(c, torch.full(shape, 3.0, device="cuda"))
    # Also test a delayed default-stream producer with an immediate side-stream consumer.
    torch.cuda._sleep(20_000_000)
    caps = fn(a, b)
    with torch.cuda.stream(consumer):
        c = torch.from_dlpack(caps[0]).clone()
    torch.cuda.synchronize()
    torch.testing.assert_close(c, torch.full(shape, 3.0, device="cuda"))


def test_package_version_matches_rust_extension():
    from tropical_gemm import _core

    assert tropical_gemm.__version__ == _core.__version__
