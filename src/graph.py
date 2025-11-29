import torch
import numpy as np
import matplotlib.pyplot as plt

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

def plot_prediction(values: torch.Tensor, predicted_values: torch.Tensor, time_step: float):
    first, second = lengthen_tensors(values, predicted_values)
    time = np.arange(first.shape[0]) * time_step

    fig, ax = plt.subplots()
    ax.plot(time, first)
    ax.plot(time, second)
    return fig