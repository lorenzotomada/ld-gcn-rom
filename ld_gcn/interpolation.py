import torch
from scipy.interpolate import LinearNDInterpolator
#from scipy.interpolate import RBFInterpolator
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, Matern
import numpy as np


def prepare_latents_for_interpolation(latents, n_sim, n_times_fine, indices, extrapolation=None):
    """
    Reformats flat latent tensors into a structured NumPy format for interpolation purposes.

    Technical transformation:
    The input `latents` tensor is expected to be a concatenated sequence of (latent) snapshots. 
    This function reshapes it to [Simulations, Time, Latent_Dim], subsets the 
    relevant indices (typically training sims), and transposes it for axis-consistent 
    interpolation.

    Args:
        latents (torch.Tensor): Raw latent vectors from the encoder or propagator.
        n_sim (int): Total number of simulations in the dataset.
        n_times_fine (int): Number of temporal snapshots per simulation.
        indices (list[int]): Indices of the simulations to include (e.g., training set).
        extrapolation (int, optional): If provided, limits the time window to the first 
            `extrapolation` snapshots.

    Returns:
        np.ndarray: Reshaped latents of shape [Latent_Dim, Time, Selected_Sims].
    """
    latents_interpolation = latents.reshape(n_sim, n_times_fine, -1).detach().cpu().numpy()
    if extrapolation is None:
        latents_interpolation = latents_interpolation[indices,:,:]
    else:
        latents_interpolation = latents_interpolation[indices, :extrapolation,:]
    latents_interpolation = latents_interpolation.transpose(2, 1, 0)
    return latents_interpolation


def build_coords_and_vals(latents_interpolation, params, times, indices):
    """
    Constructs the input coordinate grid (mu1, mu2, t) for multidimensional interpolation.

    Mathematical Logic:
    For each latent point $z_{i,t,s}$ (where $i$ is latent dim, $t$ is time index, 
    and $s$ is simulation index), we associate a coordinate vector:
    $$\mathbf{c}_{t,s} = [\mu_{1,s}, \mu_{2,s}, t]$$
    
    Args:
        latents_interpolation (np.ndarray): Structured latents from `prepare_latents_for_interpolation`.
        params (torch.Tensor): Physical parameter tensor [Simulations, Time, Param_Dim].
        times (np.ndarray): Temporal grid.
        indices (list[int]): Indices identifying the simulations used for training the surrogate.

    Returns:
        coords (np.ndarray): Matrix of shape [N_samples, 3] where each row is (mu1, mu2, t).
        latents_interpolation (np.ndarray): The original latent values (returned for consistency).
        times_fine (np.ndarray): The reconstructed fine temporal grid.
    """
    _, n_times_fine, _ = latents_interpolation.shape
    n_params_interp = latents_interpolation.shape[2]
    # n params train missing

    # Extract (mu1, mu2) at t=0 for training sims
    mu_params = params[indices, :, :2].detach().cpu().numpy()
    mu1_vals = mu_params[:, 0, 0] # shape (n_train_sim,), mu1 parameter at time zero
    mu2_vals = mu_params[:, 0, 1]

    # Repeat and tile to build full grid: (mu1, mu2, t) per latent point
    times_fine = np.linspace(times[0], times[-1], n_times_fine)
    mu1_grid = np.repeat(mu1_vals, n_times_fine)
    mu2_grid = np.repeat(mu2_vals, n_times_fine)
    t_grid = np.tile(times_fine, n_params_interp)
    coords = np.stack([mu1_grid, mu2_grid, t_grid], axis=1)

    return coords, latents_interpolation, times_fine


def create_spline_interpolators(interpolators, coords_train, latents_train_interpolation, latent_dim, n_train_sim, n_times_fine):
    """
    Initializes N-dimensional linear interpolators for each latent dimension.

    Uses Delaunay triangulation (via scipy) to perform piecewise linear 
    interpolation across the parameter-time space.

    Args:
        interpolators (list): List to be populated with SciPy interpolator objects.
        coords_train (np.ndarray): Coordinate grid from `build_coords_and_vals`.
        latents_train_interpolation (np.ndarray): Ground truth latent values.
    """
    for i in range(latent_dim):
        vals = []
        for sim in range(n_train_sim):
            for t in range(n_times_fine):
                vals.append(latents_train_interpolation[i, t, sim])
        vals = np.array(vals)

        interp = LinearNDInterpolator(coords_train, vals)
        interpolators.append(interp)


def create_rbf_interpolators(interpolators, coords_train, latents_train_interpolation, latent_dim, n_train_sim, n_times_fine):
    # TODO
    pass


def create_gpr_interpolators(interpolators, coords_train, latents_train_interpolation, latent_dim, n_train_sim, n_times_fine):
    """
    Trains Gaussian Process Regressors for latent state prediction.

    GPR Analysis:
    - **Kernel:** Matern kernel with $\nu=1.5$ (once-differentiable) + WhiteKernel 
      to model observation noise ($\epsilon \approx 10^{-3}$).
    - **Scaling:** Uses `normalize_y=True` to handle varying latent magnitudes 
      without manual feature scaling.

    Args:
        interpolators (list): List to be populated with trained GPR models.
    """
    for i in range(latent_dim):
        vals = []
        for sim in range(n_train_sim):
            for t in range(n_times_fine):
                vals.append(latents_train_interpolation[i, t, sim])
        vals = np.array(vals)

        kernel = Matern(length_scale=[0.2, 0.5, 0.5], nu=1.5)+WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-5, 1.))
        #kernel = RBF(length_scale=0.05) + WhiteKernel(noise_level=1e-5)
        gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-10, normalize_y=True)
        #gpr = GaussianProcessRegressor(kernel=kernel, alpha=0.0, normalize_y=True)
        gpr.fit(coords_train, vals)

        interpolators.append(gpr)


def interpolate_spline_latents(mu1, mu2, t, interpolators):
    """
    Queries spline interpolators for a specific parameter-time coordinate.
    """
    pt = np.array([mu1, mu2, t])
    return np.array([interp(pt) for interp in interpolators]).reshape(-1)
    # pt = np.array([[mu1, mu2, t]])
    # return np.array([interp(pt)[0] for interp in interpolators]).reshape(-1)


def interpolate_gpr_latents(mu1, mu2, t, interpolators):
    """
    Queries trained GPR models for a specific parameter-time coordinate.
    """
    pt = np.array([[mu1, mu2, t]])  # Shape (1, 3) for sklearn predict
    return np.array([gpr.predict(pt)[0] for gpr in interpolators])


def get_index(mu_space, mu1_index, mu2_index, t_index, n_times):
    """
    Utility for mapping discrete parameter indices to a flattened simulation list.
    """
    return n_times*(mu1_index * len(mu_space[1]) + mu2_index) +  t_index
