import torch
import torch.nn as nn


def train_one_epoch(model: nn.Module, optimizer: torch.optim.Optimizer, loss_function, device, batched_samples: torch.Tensor, batched_params: torch.Tensor):
    model.train()

    for samples, params in zip(batched_samples, batched_params):
        samples = samples.to(device)
        params = params.to(device)

        optimizer.zero_grad()
        predictions = model(samples)
        loss = loss_function(predictions, params)
        loss.backward()
        optimizer.step()


def evaluate(model: nn.Module, loss_function, device, batched_eval_samples: torch.Tensor, batched_eval_params: torch.Tensor):
    model.eval()

    total_loss = 0.0
    for samples, params in zip(batched_eval_samples, batched_eval_params):
        samples = samples.to(device)
        params = params.to(device)

        with torch.no_grad():
            predictions = model(samples)
            loss = loss_function(predictions, params).item()
            total_loss += loss

    return total_loss / (batched_eval_samples.size(0) * batched_eval_samples.size(1))
