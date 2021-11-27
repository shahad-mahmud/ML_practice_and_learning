import torch


def create_data(weights: torch.Tensor, bias: torch.Tensor, n_examples):
    inputs = torch.normal(0, 1, (n_examples, weights.shape[0]))
    outputs = torch.matmul(inputs, weights) + bias
    outputs += torch.normal(0, 0.01, outputs.shape)
    outputs = outputs.reshape(-1, 1)

    return inputs, outputs


def get_weights_and_bias(input_size):
    weights = torch.normal(0, 1, size=(input_size, 1), requires_grad=True)
    bias = torch.zeros(1, requires_grad=True)

    return weights, bias


def get_dataloader(data_sets, batch_size):
    dataset = torch.utils.data.TensorDataset(*data_sets)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    return dataloader
