import torch


def random_values(count: int, minimum, maximum):
    """
    Creates a tensor with values randomly distributed between minimum and maximum
    """
    return torch.rand(count) * (maximum - minimum) + minimum


def create_signals(omegas: torch.Tensor, signal_function, length: int, time_step: float, phases: torch.Tensor | None = None) -> torch.Tensor:
    """
    Creates waves with a certain length using the provided frequencies, phases and function to create them
    """
    count = omegas.size(0)
    waves = torch.empty(count, length)

    if phases is None:
        phases = torch.zeros(count)

    times = torch.arange(0, length) * time_step
    for i in range(count):
        wave = times * omegas[i] + phases[i]
        waves[i] = signal_function(wave)

    return waves


def sine(inputs: torch.Tensor):
    return inputs.sin()


def cosine_squared(inputs: torch.Tensor):
    return inputs.cos().square()


def create_batches(unbatched_list: list[torch.Tensor], batch_size: int):
    batch_count = unbatched_list[0].size(0) // batch_size
    result = []

    for unbatched in unbatched_list:
        result.append(unbatched.reshape(
            batch_count, batch_size, *unbatched.shape[1:]))

    return tuple(result)


def add_noise(values: torch.Tensor, noise_strength: float):
    noise = noise_strength * torch.randn_like(values)
    return values + noise


def remove_samples(values: torch.Tensor, missing_segment_length: int, missing_segment_count: int):
    """
    Masks out random parts of the input waves, with each wave having a certain amount of missing segments with acertain length (these can overlap).
    """
    sample_count = values.size(0)
    sample_length = values.size(1)
    new_values = values.detach().clone()

    random_indices = torch.randint(
        sample_length - missing_segment_length, (sample_count, missing_segment_count))

    for i in range(sample_count):
        for j in range(missing_segment_count):
            random_index = random_indices[i, j]
            new_values[i, random_index:random_index +
                       missing_segment_length] = -1e9

    return new_values
