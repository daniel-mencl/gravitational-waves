import torch

def random_values(count: int, minimum, maximum):
    return torch.rand(count) * (maximum - minimum) + minimum

def create_signals(omegas: torch.Tensor, signal_function, length: int, time_step: float, phases: torch.Tensor | None = None) -> torch.Tensor:
    count = omegas.size(0)
    waves = torch.empty(count, length)

    if phases is None:
        phases = torch.zeros(count)

    times = torch.arange(0, length) * time_step
    for i in range(count):
        wave = times * omegas[i] + phases[i]
        waves[i] = signal_function(wave)
    
    return waves

def sine(inputs: torch.Tensor) -> torch.Tensor:
    return inputs.sin()

def cosine_squared(inputs: torch.Tensor) -> torch.Tensor:
    return inputs.cos().square()

def create_batches(unbatched_list: list[torch.Tensor], batch_size: int):
    batch_count = unbatched_list[0].size(0) // batch_size
    result = []

    for unbatched in unbatched_list:
        result.append(unbatched.reshape(batch_count, batch_size, *unbatched.shape[1:]))
    
    return tuple(result)

def add_noise(values: torch.Tensor, noise_strength: float):
    noise = noise_strength * torch.randn_like(values)
    return values + noise