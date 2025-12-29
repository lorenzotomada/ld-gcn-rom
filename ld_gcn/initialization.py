import os
import torch
import numpy as np
import random
import warnings


def set_device():
    """
    Identifies the primary compute device and configures global tensor properties.

    The function prioritizes CUDA-capable GPUs but defaults to the CPU if no 
    compatible hardware is found. It also enforces `torch.float32` as the 
    default floating-point precision to maintain consistency across different 
    hardware architectures.

    Returns:
        device (str): Execution context identifier ('cuda' or 'cpu').
    """

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Device used: ", device)
    torch.set_default_dtype(torch.float32)
    warnings.filterwarnings("ignore")
    return device


def set_reproducibility(HyperParams):
    """
    Enforces deterministic behavior across all random number generators.

    Technical Details:
    1. **Seed Propagation:** Synchronizes seeds for Python's native `random`, 
       `numpy`, and `torch` (both CPU and all available GPU devices).
    2. **Algorithmic Determinism:** Sets `cudnn.deterministic = True`. This 
       forces the CuDNN backend to use deterministic convolution algorithms, 
       avoiding the variance introduced by non-deterministic atomic operations 
       in parallel kernels (e.g., `atomicAdd`).
    3. **Benchmark Disabling:** Disables the cuDNN auto-tuner (`benchmark = False`) 
       to prevent the selection of different algorithms based on system load 
       during the initial iterations.

    Args:
        HyperParams (object): Configuration object containing the `seed` attribute (int).
    """

    seed = HyperParams.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_path(HyperParams):
    """
    Ensures the existence of the output directory for model artifacts.

    Verifies the existence of `HyperParams.net_dir`. If the directory does not 
    exist, it is created. If the directory already exists, the `exist_ok=False`
    flag ensures that subsequent operations do not silently overwrite existing
    experimental results without manual intervention.

    Args:
        HyperParams (object): Configuration object containing `net_dir` (str).
    """
    path = HyperParams.net_dir
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path, exist_ok=False)


def initialize(HyperParams):
    """
    Orchestrates the environment setup for the neural network execution.

    This high-level wrapper sequence:
    1. Selects and prints the compute hardware.
    2. Fixes the global seed for reproducibility.
    3. Prepares the filesystem for result storage.

    Args:
        HyperParams (object): The hyperparameter and path configuration object.

    Returns:
        device (str): The device string to be passed to model.to(device) and 
                     data.to(device) calls.
    """
    device = set_device()
    set_reproducibility(HyperParams)
    set_path(HyperParams)
    return device