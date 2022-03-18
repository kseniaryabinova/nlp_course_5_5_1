import torch
from torch import nn
from torch.nn import CrossEntropyLoss


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            torch.nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            torch.nn.init.constant_(param.data, 0)


class FocalLoss(nn.Module):
    def __init__(
        self,
        pad_index=0,
        alpha=0.25,
        gamma=2.,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.criterion = CrossEntropyLoss(
            reduction='none',
            ignore_index=pad_index,
        )

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        logprobs = self.criterion(inputs, targets)
        pt = torch.exp(-logprobs)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * logprobs
        return focal_loss.mean()
