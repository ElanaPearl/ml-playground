from torch._C import TensorType
from torch.functional import Tensor

from optimizers import Adam
from zero_layer_transformer import (
    Embedder,
    Logits,
    Tokens,
    Unembedder,
    ZeroLayerTransformer,
)


def softmax(logits: Logits) -> Tensor:
    logits -= logits.max()
    exp_logits = logits.exp()
    return exp_logits / exp_logits.sum()


class SoftmaxCrossEntropyLoss:
    def forward(self, logits: Logits, labels: Tokens) -> Tensor:
        probs = softmax(logits)
        vocab_dim = 2
        # probs are BxNxV but labels are BxN so labels need an extra dim for gather
        reshaped_labels = labels.unsqueeze(vocab_dim)
        predicted_probs_for_labels = probs.gather(dim=vocab_dim, index=reshaped_labels)
        return -predicted_probs_for_labels.log().sum()


def train_step(
    model: ZeroLayerTransformer,
    token_batch: Tokens,
    optimizer: Adam,
    loss_fn: SoftmaxCrossEntropyLoss,
) -> float:
    # Slice out last token since there is no label for it
    inputs: Tokens = token_batch[:, :-1]
    # Slice out first token since there is no input token before it
    labels: Tokens = token_batch[:, 1:]

    logits = model.forward(inputs)
    loss = loss_fn.forward(logits, labels)
    logit_grads = loss_fn.backward(loss)
    model.backward(logit_grads=logit_grads)

    # this updates the model parameters accordingly
    optimizer.step()

    return loss.item()
