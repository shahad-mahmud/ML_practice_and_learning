import torch


class Model(torch.nn.Module):
    def __init__(
        self,
        weights: torch.Tensor,
        bias: torch.Tensor,
    ):
        super().__init__()
        self.weights = torch.nn.Parameter(weights)
        self.bias = torch.nn.Parameter(bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.weights) + self.bias