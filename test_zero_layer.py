import pytest
import torch

from zero_layer_transformer import Embedder


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
        embedder.param_grads, torch.tensor([[0.0, 0.0], [4.0, 5.0], [2.0, 2.0]])
    )
