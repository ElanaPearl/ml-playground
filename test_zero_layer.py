import pytest
import torch

from zero_layer_transformer import Embedder


def test_embedder():
    n_vocab = 3
    d_model = 2
    embedder = Embedder(n_vocab=n_vocab, d_model=d_model)
    embedder.param = torch.nn.Parameter(torch.tensor([[1, 1], [10, 10], [100, 100]]))
    tokens = [1, 2, 1]

    acts = embedder.forward(tokens)
    assert acts.tolist() == [[10, 10], [100, 100], [10, 10]]
