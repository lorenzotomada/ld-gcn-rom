import torch
import torch.nn.functional as F
import yaml
from ld_gcn import network, pde, preprocessing
import os
import numpy as np

def time_stepping(model_dyn, num_integrations, stns, alpha_values, dt, interpolate=True, latents=None):
    """
    Function to perform time stepping using a Runge-Kutta method.
    Designed to work in parallel (i.e. integrating for all the values of the parameter at once).
    """
    for i in range(num_integrations):
        if interpolate:
            current_alpha = alpha_values[:, i]
        else:
            current_alpha = alpha_values[:, 0]

        dyn_input = cat_input(stns, current_alpha)
        stn_derivatives = model_dyn(dyn_input)
        stns = RungeKutta(stns, stn_derivatives, dt)

        if latents is not None:
            for sim_idx in range(stns.shape[0]):
                latents[sim_idx].append(stns[sim_idx].clone())

    return stns    
    

def prepare_data(device, loader, HyperParams, params):
    """
    Function used in training and testing to prepare data and parameters.
    It preloads data to the specified device and interpolates parameters if needed.
    Args:
        device: The device to load data onto.
        loader: DataLoader providing the data batches.
        HyperParams: HyperParams object containing configuration.
        params (torch.Tensor): Tensor of shape [n_simulations, n_times, n_params]
    """
    data_list = preload_data(loader, device)

    times = params[..., -1][0].to(dtype=torch.float32, device=device)
    ratio = round(float(times[1] - times[0]) / HyperParams.dt)

    if HyperParams.interpolate_signals:
        params = interpolate_params(params, ratio)

    params = params.to(dtype=torch.float32, device=device)
    return data_list, times, ratio, params


def interpolate_params(params: torch.Tensor, ratio: int) -> torch.Tensor:
    """
    Linearly interpolates parameters along the time axis to a finer resolution.

    Args:
        params (torch.Tensor): Tensor of shape [n_simulations, n_times, n_params]
        ratio (int): Interpolation ratio (e.g., 3 means insert 2 steps between every original pair)

    Returns:
        torch.Tensor: Interpolated tensor of shape [n_simulations, new_n_times, n_params]
    """
    n_sim, n_times, n_params = params.shape
    new_n_times = (n_times - 1) * ratio + 1

    original_times = torch.arange(n_times, device=params.device, dtype=torch.float32)

    interpolated_times = torch.linspace(0, n_times - 1, steps=new_n_times, device=params.device)

    # Interpolate along time dimension using linear interpolation
    interpolated = torch.nn.functional.interpolate(
        params.permute(0, 2, 1),  # [n_sim, n_params, n_times]
        size=new_n_times,
        mode='linear',
        align_corners=True
    ).permute(0, 2, 1)  # Back to [n_sim, new_n_times, n_params]

    return interpolated


def preload_data(loader, device):
    """
    Function to preload all data from the DataLoader to the specified device.
    """
    data_list = []
    for batch in loader:
        data_list.append(batch.to(device))
    return data_list


def cat_input(tensor, param):
    """
    Concatenate the input tensor with the parameter tensor.
    If tensor is 1D: concat along dim=0 (latent + param).
    If tensor is 2D: concat along dim=1 (batch, latent + param).
    """
    if tensor.dim() not in (1, 2):
        raise ValueError("Invalid tensor dimensions in cat_input")
    return torch.cat((tensor, param), dim=tensor.dim() - 1)


def RungeKutta(x_k, x_prime_k, dt):
    """
    Function to perform time stepping using the explicit Euler method.
    Actually, not needed, as it just consists of a line of code.
    However, this way it is easier to change the time stepping method if neede, e.g. using RK4.
    
    :param x_k: The current state.
    :param x_k1: The next state.
    :param dt: The time step size.
    :return: The updated state after the time stepping.
    """
    x_k = x_k + dt * x_prime_k
    return x_k


def decode(model_decoder, HyperParams, stn, alpha_value, data):
    """
    Function to decode the state tensor using the decoder model.
    It handles the case of continuous dependence on parameters and time.
    """
    if HyperParams.continuous_dependence:
        if not HyperParams.decoder_dynamics:
            stn = exclude_time(stn) # defined below
        decoder_input = cat_input(stn, alpha_value)
    else:
        decoder_input = stn
    prediction = model_decoder(decoder_input, data)

    return prediction


def exclude_time(tensor):
    """
    In case we want the decoder not to depend on time, we exclude the last dimension of the tensor, corresponding to time.
    """
    parameters_only = tensor[..., :-1]
    return parameters_only



def create_param_list(mu_space, device):
    """
    TODO: generalized to other test cases
    """
    params = []

    if len(mu_space)==2:
        u_vals = mu_space[0]
        time_vals = mu_space[1]
        for i in range(u_vals.shape[0]):
            paired = torch.tensor(np.stack([u_vals[i], time_vals], axis=1), device=device)
            params.append(paired)

    elif len(mu_space)==3:
        for i in range(len(mu_space[0])):
            for j in range(len(mu_space[1])):
                batch = torch.stack([
                    torch.tensor([mu_space[0][i], mu_space[1][j], mu_space[2][k]])
                    for k in range(len(mu_space[2]))
                ])
                params.append(batch)
    else:
        raise NotImplementedError("This function has not been implemented yet. We leave it as trivial exercise to the reader.")

    params = torch.stack(params)
    params = params.to(device)

    return params


def prepare_HyperParams(pde_problem):
    """
    Function to prepare the HyperParams object based on the PDE problem.
    :param pde_problem: The PDE problem identifier.
    :return: A tuple containing problem details and the HyperParams object.
    """
    problem_name, variable, mu_space, n_param, dim_pde, n_comp = pde.problem(pde_problem)
    print(f"Problem: {problem_name}\nVariable: {variable}\nParameters: {n_param}")

    file_path = os.path.join('../config', f'config_{problem_name}')
    with open(file_path + ".yaml", "r") as file:
        config = yaml.safe_load(file)

    config["variable"] = variable
    config["scaler_name"] = "custom_scaling"
    config["sparse_method"] = 'L1_mean'
    config["comp"] = n_comp # TODO check if this is = comp
    config["param_dim"] = n_param+1 # in this case, including time! size of the signal u(t)
    config["recnet_act"] = 'elu'
    config["dynnet_act"] = 'tanh'
    config["epsilon"] = 1e-4
    config["delta"] = 1e-1

  
    HyperParams = network.HyperParams(config)

    try:
        n_sim = np.prod(np.array([len(mu_space[i]) for i in range(len(mu_space)-1)])) # here assuming exactly one simulation per parameter -1 bc we are excluding
                                                                              # time (does not impact on the number of simulations)
                                                                              #  TODO: update in case of changing ic
    except:
        n_sim = None # for Coanda test case
    return problem_name, variable, mu_space, n_param, dim_pde, n_comp, n_sim, HyperParams


def maxabs(tensor1, tensor2):
    """ Computes the maximum absolute error between two tensors."""
    return (torch.max(torch.abs(tensor1-tensor2))).item()


def relerr(approx, exact):
    """ Computes the relative error between two tensors."""
    return (torch.norm(approx-exact)/torch.norm(exact)).item()


def renormalize_single_snapshot(scaler_all, snapshot):
    """
    Function to renormalize a single snapshot.
    """
    snapshot = snapshot.reshape(1, -1, 1)
    return preprocessing.inverse_normalize_input(snapshot, scaler_all)[0,:,0]