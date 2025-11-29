import torch
import torch.nn as nn
import datetime
from .data import create_batches

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

def train(model: nn.Module, optimizer: torch.optim.Optimizer, loss_function, device, batched_samples: torch.Tensor, batched_params: torch.Tensor, epochs: int, eval_step: int | None = None):
    start = datetime.datetime.now()
    for epoch in range(epochs):
        train_one_epoch(model, optimizer, loss_function, device, batched_samples, batched_params)
        if eval_step is not None and (epoch + 1) % eval_step == 0:
            evaluation = evaluate(model, loss_function, device, batched_samples, batched_params)
            print(f"Epoch {epoch + 1}: {evaluation} after {datetime.datetime.now() - start}")

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

def predict_next_values(model: nn.Module, values: torch.Tensor, sequence_length: int, count: int):
    assert values.size(0) == sequence_length, f"length of values should be {sequence_length}"

    device = next(model.parameters()).device
    values = values.to(device)
    
    all_values = torch.empty(sequence_length + count, device=device)
    all_values[:sequence_length] = values

    for i in range(count):
        current_values = all_values[i:i+sequence_length]
        with torch.no_grad():
            predicted_next_value = model(current_values.reshape(1, -1)).item()
        all_values[i + sequence_length] = predicted_next_value

    return all_values.cpu().split((sequence_length, count))

def make_prediction(model: nn.Module, values: torch.Tensor, batch_size: int) -> torch.Tensor:
    predictions = []
    batched = create_batches([values], batch_size)[0]
    device = next(model.parameters()).device
    
    for batch in batched:
        with torch.no_grad():
            pred = model(batch.to(device)).cpu()

        predictions.append(pred)

    return torch.cat(predictions).reshape(-1)