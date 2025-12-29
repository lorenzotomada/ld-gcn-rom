import torch
from typing import Optional, Tuple
import matplotlib.pyplot as plt

def POD(matrix: torch.Tensor, r: Optional[int] = None, center: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simple POD in PyTorch.

    Args:
        matrix : (n_features, n_snapshots) tensor
        r      : optional, number of modes to keep
        center : whether to subtract mean of snapshots

    Returns:
        U : POD modes (n_features, r or full)
        s : singular values (r or full)
    """
    # Ensure float32 for SVD
    X = matrix.float()

    # Centering
    if center:
        mean = X.mean(dim=1, keepdim=True)
        Xc = X - mean
    else:
        Xc = X

    # Compute SVD
    U, s, Vh = torch.linalg.svd(Xc, full_matrices=False)

    # Truncate if requested
    if r is not None:
        U = U[:, :r]
        s = s[:r]

    return U, s


def projection(matrix: torch.Tensor, U: torch.Tensor, N_pod: Optional[int] = None) -> torch.Tensor:
    """
    Project snapshots onto POD modes.

    Args:
        matrix : (n_features, n_snapshots)
        U      : POD modes (n_features, r)
        N_pod  : optional, number of modes to use for projection

    Returns:
        coeffs : (r, n_snapshots) POD coefficients
    """
    # Choose number of modes
    if N_pod is not None:
        U = U[:, :N_pod]

    # Project
    coeffs = U.T @ matrix.float()
    return coeffs


def return_ic_s(VAR, n_sim):
    print("Warning: only works in 1D for now.")
    step = VAR.shape[0] // n_sim
    ic_s = VAR[::step, :,0]
    return ic_s


# ---------------------------
# Minimal example
# ---------------------------
if __name__ == "__main__":
    # Create example data: 50 features, 20 snapshots
    torch.manual_seed(0)
    X = torch.randn(50, 20)

    # Compute POD keeping 5 modes, centering
    U, s = POD(X, r=5, center=False)
    coeffs = projection(X, U)

    print("U shape:", U.shape)
    print("s shape:", s.shape)
    print("coeffs shape:", coeffs.shape)

    plt.plot(s.cpu().numpy(), marker='o')
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('singular_values.png')
    plt.show()
    plt.clf()
    plt.close()