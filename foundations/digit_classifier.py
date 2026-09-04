import torch
import torch.nn as nn
from torchtyping import TensorType


class Solution(nn.Module):

    def __init__(self):
        super().__init__()

        torch.manual_seed(0)

        self.model = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 10),
            nn.Sigmoid()
        )

    def forward(self, images: TensorType[float]) -> TensorType[float]:
        torch.manual_seed(0)

        output = self.model(images)

        return torch.round(output * 10000) / 10000