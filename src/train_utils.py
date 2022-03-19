import heapq

import numpy as np
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F


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
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.criterion = CrossEntropyLoss(
            reduction='none',
            ignore_index=pad_index,
            label_smoothing=label_smoothing,
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


class BeamGenerator:
    def __init__(self):
        self.eos_token_id = 2

    def __call__(self, logits, max_steps_n=40, return_hypotheses_n=5, beamsize=5):
        initial_length = 1
        counter = 0

        partial_hypotheses = [(0, [1])]
        final_hypotheses = []

        while counter != logits.shape[0]:
            cur_partial_score, cur_partial_hypothesis = heapq.heappop(partial_hypotheses)

            # in_batch = torch.tensor(cur_partial_hypothesis).unsqueeze(0).to(self.device)
            next_tokens_logits = logits[counter]
            counter += 1
            next_tokens_logproba = F.log_softmax(next_tokens_logits, dim=0)
            topk_continuations = next_tokens_logproba.topk(beamsize)

            for token_score, token_idx in zip(topk_continuations.values, topk_continuations.indices):
                token_score = float(token_score)
                token_idx = int(token_idx)

                old_denorm_score = cur_partial_score * np.sqrt(len(cur_partial_hypothesis))
                new_score = (old_denorm_score - token_score) / np.sqrt(len(cur_partial_hypothesis) + 1)

                new_hypothesis = cur_partial_hypothesis + [token_idx]
                new_item = (new_score, new_hypothesis)

                if token_idx == self.eos_token_id or len(new_hypothesis) - initial_length >= max_steps_n:
                    final_hypotheses.append(new_item)
                else:
                    heapq.heappush(partial_hypotheses, new_item)

            if len(partial_hypotheses) > beamsize:
                partial_hypotheses = heapq.nsmallest(beamsize, partial_hypotheses)
                heapq.heapify(partial_hypotheses)

        final_scores, final_token_lists = zip(*final_hypotheses)
        result = list(zip(final_scores, final_token_lists))
        result.sort()
        result = result[:return_hypotheses_n]

        return result
