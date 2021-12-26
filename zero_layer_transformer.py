import torch
from torch import Tensor
from torchtyping import TensorType

Tokens = TensorType["n_batch", "n_tokens"]
Acts = TensorType["n_batch", "n_tokens", "d_model"]
Logits = TensorType["n_batch", "n_tokens", "n_vocab"]


class ZeroLayerTransformer(torch.nn.Module):
    def __init__(self, embedder: torch.nn.Module, unembedder: torch.nn.Module):
        super().__init__()
        self.embedder = embedder
        self.unembedder = unembedder

    def forward(self, tokens: Tokens) -> Logits:
        embeddings = self.embedder(tokens)
        logits = self.unembedder(embeddings)
        return logits

    def backward(self, logit_grads: Logits):
        embedding_grads = self.unembedder.backward(logit_grads)
        self.embedder.backward(embedding_grads)


class Embedder(torch.nn.Module):
    def __init__(self, n_vocab: int, d_model: int):
        super().__init__()
        self.n_vocab = n_vocab
        self.d_model = d_model
        self.param = torch.nn.Parameter(torch.empty((n_vocab, d_model)))
        torch.nn.init.normal_(self.param)

    def forward(self, tokens: Tokens) -> Acts:
        self.tokens = tokens
        return self.param[tokens]

    def backward(self, embedding_grads: Acts):
        self.param_grads = torch.zeros(self.n_vocab, self.d_model)
        self.param_grads.index_add_(dim=0, index=self.tokens, source=embedding_grads)
        del self.tokens


class Unembedder(torch.nn.Module):
    def __init__(self, n_vocab: int, d_model: int):
        super().__init__()
        self.n_vocab = n_vocab
        self.param = torch.nn.Parameter(torch.empty((d_model, n_vocab)))

    def forward(self, embedding: Acts) -> Logits:
        self.embedding = embedding
        return torch.einsum("bnd,dv->bnv", embedding, self.param)

    def backward(self, logit_grads: Logits) -> Acts:
        self.param_grads = torch.einsum("bnd,bnv->dv", self.embedding, logit_grads)
        return torch.einsum("bnv,dv->bnd", logit_grads, self.param)
