import torch


class Model(torch.nn.Module):
    def __init__(
        self,
        input_size: int = None,
        output_size: int = None,
    ):
        super().__init__()
        self.input_size = input_size

        self.linear = torch.nn.Linear(
            in_features=input_size, out_features=output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)