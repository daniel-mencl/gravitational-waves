import torch
import torch.nn as nn


def predict_next_values(model: nn.Module, values: torch.Tensor, sequence_length: int, count: int):
    """
    Runs a model's prediction repeatedly to predict the evolution of a wave
    """
    assert values.size(
        0) == sequence_length, f"length of values should be {sequence_length}"

    device = next(model.parameters()).device
    values = values.to(device)

    all_values = torch.empty(sequence_length + count, device=device)
    all_values[:sequence_length] = values

    for i in range(count):
        current_values = all_values[i:i+sequence_length]
        with torch.no_grad():
            predicted_next_value = model(current_values.reshape(1, -1)).item()
        all_values[i + sequence_length] = predicted_next_value

    return all_values.cpu().split((sequence_length, count))[1]
