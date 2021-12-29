import pytest
import torch

from zero_layer_transformer import Embedder, Unembedder, ZeroLayerTransformer


def test_embedder():
    n_vocab = 3
    d_model = 2
    embedder = Embedder(n_vocab=n_vocab, d_model=d_model)
    embedder.param = torch.nn.Parameter(
        torch.tensor([[1.0, 1.0], [10.0, 10.0], [100.0, 100.0]])
    )
    tokens = torch.tensor([1, 2, 1])

    acts = embedder.forward(tokens)
    assert acts.tolist() == [[10.0, 10.0], [100.0, 100.0], [10.0, 10.0]]

    act_grads = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 4.0]])
    embedder.backward(act_grads)
    assert torch.allclose(
        embedder.grad, torch.tensor([[0.0, 0.0], [4.0, 5.0], [2.0, 2.0]])
    )


def test_unembedder():
    n_vocab = 3
    d_model = 2
    n_tokens = 4
    # acts = 1,4,2 param = 2,3

    unembedder = Unembedder(n_vocab=n_vocab, d_model=d_model)
    unembedder.param = torch.nn.Parameter(
        torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    )
    acts = torch.tensor([[[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]]])

    logits = unembedder(acts)
    assert logits.shape == torch.Size([1, n_tokens, n_vocab])

    logit_grads = torch.ones_like(logits)
    act_grads = unembedder.backward(logit_grads)

    assert torch.allclose(
        unembedder.param.grad, torch.tensor([[10.0, 10.0, 10.0], [10.0, 10.0, 10.0]])
    )
    assert torch.allclose(
        act_grads, torch.tensor([[[3.0, 6.0], [3.0, 6.0], [3.0, 6.0], [3.0, 6.0]]])
    )
