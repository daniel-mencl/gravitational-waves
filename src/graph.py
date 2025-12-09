import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

cmap = sns.color_palette("Paired")

def lengthen_tensors(first: torch.Tensor, second: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    first_length = first.size(0)
    second_length = second.size(0)

    first_lengthened = np.empty(first_length + second_length)
    first_lengthened[:first_length] = first
    first_lengthened[first_length:] = np.nan

    second_lengthened = np.empty(first_length + second_length)
    second_lengthened[:first_length] = np.nan
    second_lengthened[first_length:] = second

    return first_lengthened, second_lengthened

def plot_prediction(values: torch.Tensor, expected_values: torch.Tensor, predicted_values: torch.Tensor, time_step: float):
    values_long, expected_values_long = lengthen_tensors(values, expected_values)
    _, predicted_values_long = lengthen_tensors(values, predicted_values)
    time = np.arange(values_long.shape[0]) * time_step

    fig, ax = plt.subplots()
    ax.plot(time, values_long, color=cmap[1])
    ax.plot(time, expected_values_long, color=cmap[0], label="Actual")
    ax.plot(time, predicted_values_long, color=cmap[9], label="Predicted")
    ax.legend()
    return fig