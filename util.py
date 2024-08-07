import numpy as np
import torch

def check_eos_occurrence(outputs: torch.Tensor, eos_id: int) -> np.ndarray:
    flattened_list = np.array([item for sublist in outputs.to("cpu") for item in sublist])
    return np.where(flattened_list == eos_id)