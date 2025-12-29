import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import random


def graphs_dataset(dataset, HyperParams, params):
    """
    Handles the conversion of mesh-based simulation data into PyTorch Geometric datasets.

    This function performs three major tasks:
    1. **Data Selection:** Splits simulations into training and test sets based on `HyperParams.rate`.
    2. **Temporal Windowing:** Limits training snapshots to a fraction of the total time using `HyperParams.time_rate`.
    3. **Graph Construction:** Builds a list of `Data` objects, where each object represents a single 
       temporal snapshot with node features, spatial positions, and connectivity.

    Args:
        dataset (Namespace): Object containing mesh data:
            - xx, yy, zz: Nodal coordinates.
            - U or VX, VY: Physical field components.
            - E: Connectivity matrix (elements).
        HyperParams (object): Model configuration including `rate`, `time_rate`, and `comp`.
        params (torch.Tensor): Parameter tensor of shape [Simulations, Time, Param_Dim].

    Returns:
        tuple: (loader, train_loader, test_loader, val_loader, ...) 
               Standard PyG DataLoaders and scaling objects.
    """
    xx = dataset.xx.to(dtype=torch.float32)
    yy = dataset.yy.to(dtype=torch.float32)
    xyz = [xx, yy]
    if dataset.dim == 3:
       zz = dataset.zz
       xyz.append(zz)
    if HyperParams.comp == 1:
        var = dataset.U
    else:
        var1 = dataset.VX
        var2 = dataset.VY
        var = torch.stack((dataset.VX, dataset.VY), dim=2)

    # PROCESSING DATASET
    num_nodes = var.shape[0]
    num_graphs = var.shape[1]

    print("Number of nodes processed: ", num_nodes)
    print("Number of graphs processed: ", num_graphs)

    
    rate = HyperParams.rate / 100
    time_rate = HyperParams.time_rate / 100

    total_sims, n_times, _param_dim = params.shape

    n_train_params = int(total_sims * rate)
    train_param_indices = sorted(random.sample(range(total_sims), n_train_params))
    test_param_indices = sorted(set(range(total_sims)) - set(train_param_indices))

    time_cutoff = int(n_times * time_rate)

    # Construct params_train. only the first `time_rate`% of training parameters
    params_train = torch.stack([params[i, :time_cutoff] for i in train_param_indices])

    # Construct params_test, full trajectories of test parameters only
    params_test = torch.stack([params[i] for i in test_param_indices])

    # Test snapshots: all from test parameters
    test_snapshots_indices = set()
    for p in test_param_indices:
        for t in range(n_times):
            test_snapshots_indices.add(p * n_times + t)

    # Train snapshots: only from training parameters within time cutoff
    train_snapshots_indices = set()
    for p in train_param_indices:
        for t in range(time_cutoff):
            train_snapshots_indices.add(p * n_times + t)
    
    validation_snapshots_indices = set()
    for p in train_param_indices:
        for t in range(time_cutoff, n_times):
            validation_snapshots_indices.add(p * n_times + t)
    
    test_snapshots_indices = sorted(test_snapshots_indices)
    train_snapshots_indices = sorted(train_snapshots_indices)
    validation_snapshots_indices = sorted(validation_snapshots_indices)

    # SCALING DATASET
    if HyperParams.comp == 1:
        var_test = var[:, test_snapshots_indices]
        var_val = var[:, validation_snapshots_indices]

    elif HyperParams.comp == 2:
        var_test = var[:, test_snapshots_indices, :]
        var_val = var[:, validation_snapshots_indices, :]

    else:
        raise NotImplementedError("This function has not been implemented yet.")

    alphas_snapshots = compute_normalization_constants(var[:, train_snapshots_indices])
    VAR_all, scaler_all = normalize_input(var, alphas_snapshots)
    VAR_test, scaler_test = normalize_input(var_test, alphas_snapshots)
    VAR_val, scaler_val = normalize_input(var_val, alphas_snapshots)
    VAR_all = VAR_all.view(VAR_all.shape[0], VAR_all.shape[1], HyperParams.comp).permute(1, 0, 2)
    VAR_test = VAR_test.view(VAR_test.shape[0], VAR_test.shape[1], HyperParams.comp).permute(1, 0, 2)
    VAR_val = VAR_val.view(VAR_val.shape[0], VAR_val.shape[1], HyperParams.comp).permute(1, 0, 2)

    graphs = []
    edge_index = torch.t(dataset.E) - 1
    
    for graph in range(num_graphs):
        if dataset.dim == 2:
            pos = torch.cat((xx[:, graph].unsqueeze(1), yy[:, graph].unsqueeze(1)), 1)
        elif dataset.dim == 3:
            pos = torch.cat((xx[:, graph].unsqueeze(1), yy[:, graph].unsqueeze(1), zz[:, graph].unsqueeze(1)), 1)
        ei = torch.index_select(pos, 0, edge_index[0, :])
        ej = torch.index_select(pos, 0, edge_index[1, :])
        edge_attr = torch.abs(ej - ei)
        if dataset.dim == 2:
            edge_weight = torch.sqrt(torch.pow(edge_attr[:, 0], 2) + torch.pow(edge_attr[:, 1], 2)).unsqueeze(1)
        elif dataset.dim == 3:
            edge_weight = torch.sqrt(torch.pow(edge_attr[:, 0], 2) + torch.pow(edge_attr[:, 1], 2) + torch.pow(edge_attr[:, 2], 2)).unsqueeze(1)
        if HyperParams.comp == 1:
            node_features = VAR_all[graph, :]
        else:
            node_features = VAR_all[graph, :, :]
        dataset_graph = Data(x=node_features, edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr, pos=pos)
        graphs.append(dataset_graph)

    HyperParams.num_nodes = dataset_graph.num_nodes
    train_dataset = [graphs[i] for i in train_snapshots_indices]
    test_dataset = [graphs[i] for i in test_snapshots_indices]

    print("Length of train dataset: ", len(train_dataset))
    print("Length of test dataset: ", len(test_dataset))

    loader = DataLoader(graphs, batch_size=1)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    val_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    
    return loader, train_loader, test_loader, \
            val_loader, scaler_all, scaler_test, xyz, VAR_all, VAR_val, VAR_test, \
                train_snapshots_indices, validation_snapshots_indices, test_snapshots_indices, params_train, params_test


def compute_normalization_constants(tensor):
    """
    Defined to be able to scale test and training snapshots separately.
    Calculates the Mid-range (alpha0) and Half-range (alphaw) for node-wise scaling.

    For each node $i$:
    $$\alpha_{0,i} = \frac{\max(x_i) + \min(x_i)}{2}, \quad \alpha_{w,i} = \frac{\max(x_i) - \min(x_i)}{2}$$
    This maps the data range to $[-1, 1]$.

    Returns:
        Tensor: [2, num_nodes] containing normalization offsets and scales.
    """
    alpha0 = (tensor.max(dim=1)[0] + tensor.min(dim=1)[0]) / 2 # this used to be dim = 0! The [0] is because torch returns (max, index)
    alphaw = (tensor.max(dim=1)[0] - tensor.min(dim=1)[0]) / 2

    # To avoid division by 0:
    alpha_w_mean = alphaw[alphaw != 0].mean()
    alphaw = torch.where(alphaw == 0, alpha_w_mean, alphaw)
    
    return torch.stack((alpha0, alphaw)) # this is scaler_all for the next method


def normalize_input(tensor, scaler_all):
    """ 
    Normalization according to the paper. Use this on the whole dataset, and the other one only on the training set
    alpha0 = scaler_all[0]
    alphaw = scaler_all[1]
    normalized_tensor = (tensor - alpha0) / alphaw
    """
    alpha0 = scaler_all[0][:, None]  # Shape [num_nodes, 1]
    alphaw = scaler_all[1][:, None]  # Shape [num_nodes, 1]
    
    normalized_tensor = (tensor - alpha0) / alphaw
    
    return normalized_tensor, scaler_all


def inverse_normalize_input(normalized_tensor, scaler_all):
    """
    Inverse normalization of the input tensor.
    This function is used to reconstruct the original tensor from the normalized tensor.
    
    :param normalized_tensor: The normalized tensor to be inverse normalized.
    :param scaler_all: The scaling parameters used for normalization.
    :return: The inverse normalized tensor.
    """
    alpha0 = scaler_all[0][:, None]  # Shape [num_nodes, 1]
    alphaw = scaler_all[1][:, None]  # Shape [num_nodes, 1]
    
    tensor = (normalized_tensor * alphaw) + alpha0
    
    return tensor


def delete_initial_condition(dataset, mu_space, n_comp, n_snap_time, n_delete=1, shrink_param_space=False):
    """
    Removes the first n_delete snapshots from each trajectory.

    This is often used to exclude the $t=0$ state which might be trivial (zeros) 
    or numerically inconsistent with the learned dynamics.
    """
    # Delete first `n_delete` time samples per trajectory in mu_space[-1]
    mu_space[-1] = np.concatenate([
        mu_space[-1][i * n_snap_time + n_delete : (i + 1) * n_snap_time] 
            for i in range(len(mu_space[-1]) // n_snap_time)
    ])

    if shrink_param_space:
        mu_space[0] = np.array([mu_space[0][i][n_delete:] for i in range(len(mu_space[0]))])

    # Determine number of total snapshots and trajectories
    total_snapshots = dataset.U.shape[1] if n_comp == 1 else dataset.VX.shape[1]
    n_trajectories = total_snapshots // n_snap_time

    # Create mask to delete the first `n_delete` snapshots from each trajectory
    indices = torch.ones(total_snapshots, dtype=torch.bool)
    for i in range(n_trajectories):
        start_idx = i * n_snap_time
        indices[start_idx:start_idx + n_delete] = 0

    # Apply mask to the dataset
    if n_comp == 1:
        dataset.U = dataset.U[:, indices]
    elif n_comp == 2:
        dataset.VX = dataset.VX[:, indices]
        dataset.VY = dataset.VY[:, indices]
    else:
        print("Invalid dimension. Please enter 1 or 2.")
        return dataset, mu_space

    dataset.xx = dataset.xx[:, indices]
    dataset.yy = dataset.yy[:, indices]
    
    return dataset, mu_space


def shrink_dataset(dataset, mu_space, n_sim, n_snap2keep, n_comp, shrink_param_space=False):
    """
    TODO: generalization to mixed kinds of parameters, different number of parameters and so on.
    """
    time = mu_space[-1]
    n_time = len(time)
    idx_time = np.round(np.linspace(0, n_time-1, n_snap2keep)).astype(int)
    mu_space[-1] = time[idx_time]
    if shrink_param_space:
        mu_space[0][i] = mu_space[0][i][idx_time]

    idx = np.copy(idx_time)
    for i in range(1, n_sim):
        idx_time += n_time
        idx = np.hstack((idx, idx_time))

    if n_comp == 1:
        dataset.U = dataset.U[:, idx]
    elif n_comp == 2:
        dataset.VX = dataset.VX[:, idx]
        dataset.VY = dataset.VY[:, idx]
    dataset.xx = dataset.xx[:, idx]
    dataset.yy = dataset.yy[:, idx]

    return dataset, mu_space


def cut_dataset(dataset, mu_space, n_sim, n_snap2keep, n_comp):
    """
    Similar to the previous function, but this one keeps the *first* n_snap2keep snapshots
    """
    time = mu_space[-1]
    n_time = len(time)
    idx_time = np.arange(0, n_snap2keep).astype(int)
    mu_space[-1] = time[idx_time]

    idx = np.copy(idx_time)
    
    for i in range(1, n_sim):
        idx_time += n_time
        idx = np.hstack((idx, idx_time))

    if n_comp == 1:
        dataset.U = dataset.U[:, idx]
    elif n_comp == 2:
        dataset.VX = dataset.VX[:, idx]
        dataset.VY = dataset.VY[:, idx]
    dataset.xx = dataset.xx[:, idx]
    dataset.yy = dataset.yy[:, idx]

    return dataset, mu_space