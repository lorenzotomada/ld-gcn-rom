import torch
from tqdm import tqdm
import numpy as np
from .utils import prepare_data, time_stepping, decode


def evaluate(VAR, model_decoder, model_dyn, loader, params, HyperParams, ic_s=None):
    """
    Performs full-trajectory inference.

    This function evolves the latent states for all simulations in the provided 
    loader using the dynamics model and reconstructs the physical field using 
    the decoder. It accounts for time-dependent parameters and optional 
    sub-stepping (interpolation).

    Mathematical formulation:
    1.  **Initial Condition:** $\mathbf{s}_0$ is either provided via `ic_s` or initialized to $\mathbf{0}$.
    2.  **Dynamics:** $\mathbf{s}_{t+1} = \int_{t}^{t+dt} f_{dyn}(\mathbf{s}, \mu, \tau) d\tau$.
    3.  **Decoding:** $\hat{\mathbf{x}}_{t+1} = \mathcal{D}(\mathbf{s}_{t+1}, \mu_{t+1}, \mathcal{G})$, 
        where $\mathcal{G}$ represents the graph topology.

    Args:
        VAR (torch.Tensor): Template tensor of the full dataset, used to extract 
            spatial dimensions and store results. Shape: [Total_Snapshots, Nodes, Comp].
        model_decoder (nn.Module): The GCN-based decoder network.
        model_dyn (nn.Module): The latent dynamics propagator.
        loader (DataLoader): PyG DataLoader providing the graph objects.
        params (torch.Tensor): Physical parameters/signals. Shape: [n_sim, n_times, n_params].
        HyperParams (object): Namespace containing bottleneck_dim, dt, and architectural flags.
        ic_s (np.ndarray, optional): Pre-defined latent initial conditions. 
            Expected shape: [Latent_Dim, n_sim].

    Returns:
        results (torch.Tensor): Reconstructed physical field in the original 
            snapshot-based format. Shape: [Total_Snapshots, Nodes, Comp].
        latents (torch.Tensor): The complete history of the latent manifold 
            evolution. Shape: [n_sim * n_times_interp, Latent_Dim].
    """
    with torch.no_grad():
        device = "cpu"
        data_list, times, ratio, params = prepare_data(device, loader, HyperParams, params)
        dt = HyperParams.dt

        n_simulations = len(params)
        n_times = len(times)
        interpolate = HyperParams.interpolate_signals
        parameter_multiplier = ratio if interpolate else 1

        results = torch.zeros(VAR.shape[0], VAR.shape[1], HyperParams.comp, device=device)

        print("Evaluating the model...")

        # Initialize latent states
        if ic_s is None:
            stn = torch.zeros(n_simulations, HyperParams.bottleneck_dim, device=device, dtype=torch.float32)
        else:
            stn = torch.tensor(ic_s.T, device=device, dtype=torch.float32)
            assert stn.shape[0] == n_simulations, \
                f"ic_s mismatch: got {stn.shape[0]} ICs, expected {n_simulations}"

        simulation_latents = [[] for _ in range(n_simulations)]

        # First time step: compute IC prediction
        for alpha_idx in range(n_simulations):
            exact_ic = data_list[alpha_idx * n_times]
            prediction = decode(model_decoder, HyperParams, stn[alpha_idx], params[alpha_idx][0], exact_ic)
            results[alpha_idx * n_times] = prediction
            simulation_latents[alpha_idx].append(stn[alpha_idx].clone())

        # Time stepping and predictions
        for t_idx in range(len(times[1:])):
            alpha_t = params[:, parameter_multiplier * t_idx:parameter_multiplier * (t_idx + 1)]

            stn = time_stepping(
                model_dyn,
                ratio,
                stn,
                alpha_t,
                dt,
                interpolate,
                latents=simulation_latents  # List of lists, one per simulation
            )

            for alpha_idx in range(n_simulations):
                time_idx = alpha_idx * n_times + t_idx + 1
                data = data_list[time_idx]
                param_t = params[alpha_idx][t_idx * parameter_multiplier + 1]
                prediction = decode(model_decoder, HyperParams, stn[alpha_idx], param_t, data)
                results[time_idx] = prediction
                #simulation_latents[alpha_idx].append(stn[alpha_idx].clone())

        # Stack all latent trajectories
        latents = torch.stack([torch.stack(lat_seq) for lat_seq in simulation_latents])
        latents = latents.view(-1, latents.shape[-1])

        print("Evaluation complete!")

        return results, latents