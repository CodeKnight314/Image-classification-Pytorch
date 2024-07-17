import torch 
import torch.nn as nn 
import torch.optim as opt 
from torch.nn import functional as F
from typing import Tuple

class CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__() 
    
    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Calculates the cross-entropy loss between logits and true labels.

        Args:
            logits (torch.Tensor): The logits output from a model (unnormalized scores).
            labels (torch.Tensor): The true labels, expected to be class indices.

        Returns:
            torch.Tensor: The calculated loss value, averaged over the batch.
        """
        probs = F.log_softmax(logits, dim=-1)
        loss = -(probs[range(logits.shape[0]), labels] + 1e-9)
        return loss.mean()
    
class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.classes = classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
    
    def forward(self, x, target):
        log_probs = torch.nn.functional.log_softmax(x, dim=-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.classes - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

class BCELoss(nn.Module): 
    def __init__(self): 
        super().__init__()

    def forward(self, logits : torch.Tensor, labels : torch.Tensor) -> torch.Tensor: 
        """
        Calculates the binary cross-entropy loss between logits and true labels.

        Args:
            logits (torch.Tensor): The logits output from a model (unnormalized scores), expected to have shape [N, 1].
            labels (torch.Tensor): The true labels, expected to have the same shape as logits.

        Returns:
            torch.Tensor: The calculated loss value, averaged over all elements in the batch.
        """
        probs = torch.sigmoid(logits)
        probs_log = torch.log(probs + 1e-9)
        neg_probs_log = torch.log(1-probs + 1e-9)
        return - (labels * probs_log + (1-labels) * neg_probs_log).sum(dim=-1).mean()

class HingeLoss(nn.Module): 
    def __init__(self): 
        super().__init__() 
    
    def forward(self, logits : torch.Tensor, labels : torch.Tensor): 
        """
        Calculates the hinge loss for binary classification.

        Args:
            logits (torch.Tensor): The logits or scores output from a model.
            labels (torch.Tensor): The true labels, expected to have the same shape as logits and values of 1 or -1.

        Returns:
            float: The calculated loss value, averaged over all elements in the batch. The loss is constrained to be non-negative.
        """
        assert logits.shape == labels.shape, "[ERROR] Logits and labels have incompatiable shapes."
        probs = torch.tanh(logits)
        loss = (1 - labels * probs).sum(dim=-1).mean().item()
        return max(0, loss)