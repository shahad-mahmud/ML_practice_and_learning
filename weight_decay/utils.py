import torch

def create_data(weights: torch.Tensor, bias: torch.Tensor, n_examples):
    inputs = torch.normal(0, 1, (n_examples, weights.shape[0]))
    outputs = torch.matmul(inputs, weights) + bias
    outputs += torch.normal(0, 0.01, outputs.shape)
    outputs = outputs.reshape(-1, 1)
    
    return inputs, outputs

def get_weights_and_bias(input_size):
    weights = torch.ones((input_size, 1)) * 0.01
    bias = torch.randn(1)
    
    return weights, bias

def get_dataloader(*data_sets, batch_size):
    data_set = torch.utils.data.TensorDataset(*data_sets)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size)
    
    return data_loader