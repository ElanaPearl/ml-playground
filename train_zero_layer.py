import torch
from torch.functional import Tensor

from optimizers import Adam
from zero_layer_transformer import (
    Embedder,
    Logits,
    Tokens,
    Unembedder,
    ZeroLayerTransformer,
)


class Softmax:
    def forward(self, logits: Logits) -> Tensor:
        # subtract mean for numeric stability, doesn't affect results
        normalized_logits = logits - logits.max()
        exp_logits = normalized_logits.exp()
        self.probs = exp_logits / exp_logits.sum()
        return self.probs

    def backward(self, softmax_grads: Tensor) -> Logits:
        """Equation to calculate derivative of softmax:
        dL/dx_i = dL/d_Si * dS_i/dx_i + ∑_{i!=j} dL/d_Sj * dS_j/dx_i
        where S_i = softmax(x_i) and:
            dS_i/dx_i = S_i(1 - S_i)
            dS_i/dx_j = S_i*S_j
        """

        kronecker_delta = torch.eye(self.probs.shape[2])
        outer = torch.einsum("bnv,bnw->bnvw", self.probs, self.probs)
        jacobian = -outer + torch.einsum("vw,bnv->bnvw", kronecker_delta, self.probs)
        return torch.einsum("bnvw,bnv->bnv", jacobian, softmax_grads)


class SoftmaxCrossEntropyLoss:
    def forward(self, logits: Logits, labels: Tokens) -> Tensor:
        probs = Softmax.forward(logits)
        # probs are BxNxV but labels are BxN so labels need an extra dim for gather
        labels_3d = labels[:, :, None]
        predicted_probs_for_labels = probs.gather(dim=2, index=labels_3d)
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
