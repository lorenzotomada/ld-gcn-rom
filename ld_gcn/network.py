import torch
from torch import nn
from ld_gcn import gca
from datetime import datetime
import numpy as np
import torch.nn.functional as F


def get_activation(act_str):
    """Retrieves a functional activation from torch.nn.functional by string name."""
    return getattr(F, act_str)


class HyperParams:
    """
    Class that holds the hyperparameters for the autoencoder model.
    Remark on changing_ic (see below): it not refer to the fact that we are dealing with multiple ICs, but rather defines whether or not the
    initial condition should be used to compute the loss. 
    Indeed, assume that NN_rec does not depend on the signal mu and that the initial condition is fixed for each simulation: then the network
    is told n_sim times that NN_rec(s=0)=ic, and this specific fact is weighted much more than other snapshots, potentially overlooking dynamical
    info.
    Set to True if working with multiple initial conditions.

    Args:
        comp (int): Dimension of the values of the solution.
        param_dim (int): Size of the signal u(t), including time.
        variable (str): Name of the variable (e.g., 'U' for scalar problems).
        seed (int): Seed for random number generators (NumPy and PyTorch).
        skip (int): Whether to use skip connections (1 for True, 0 for False).
        conv (str): Type of convolution used.
        ffn (int): Size of the intermediate layer in the decoder feed-forward network.
        nodes (int): Number of nodes in each layer of NN_dyn (not used, kept for compatibility). TODO: remove in future versions.
        bottleneck_dim (int): Latent dimension of the bottleneck layer.
        in_channels (int): Number of input channels.
        hidden_channels (list): Number of hidden channels for each layer (size: in_channels*comp).
        layer_vec (list): Structure of NN_dyn layers (input: bottleneck+parameter dim, output: bottleneck dim). The network takes s,u, but returns \dot(s), which is of shape bottleneck_dim
        scaler_name (str): Name of the scaler used for preprocessing (only 'custom_scaler' supported).
        num_nodes (int): Number of nodes in the network (currently not used). TODO: remove in future versions.
        recnet_act (function): Activation function for NN_rec.
        dynnet_act (function): Activation function for NN_dyn.
        continuous_dependence (bool): Whether NN_rec depends on parameter mu.
        decoder_dynamics (bool): Whether NN_rec also depends on time. TODO: Maybe rename 'exclude_time'
        changing_ic (bool): Whether to consider varying initial conditions when computing the loss.
        rate (int): Percentage of simulations used for training (e.g., 70 means 70%).
        time_rate (int): Percentage of snapshots in each simulation used for training (e.g., 50 means first 50% of each simulation is used).
        interpolate_signals (bool): Whether to interpolate signals when using time-dependent parameters and small stepsize.
        max_epochs (int): Maximum number of epochs for training.
        dt (float): Time step used for integration.
        minibatch (int or None): Batch size for decoding (None if not used).
        cross_validation (bool): Whether to compute test loss during training.
        sparse_method (str): Type of sparsity constraint ('L^1' or 'L^2').
        weight_decay (float): Weight decay for the optimizer.
        learning_rate (float): Learning rate for the optimizer.
        tolerance (float): Threshold for stopping training if loss falls below this value.
        miles (list): Epoch milestones for learning rate scheduler.
        gamma (float): Factor for learning rate scheduler.
        epsilon (float): Regularization parameter in the NS loss.
        delta (float): Regularization parameter in the NS loss.
        net_name (str): Name of the network for saving.
        net_dir (str): Directory for saving the network.
    """

    def __init__(self, param_dict):
        # ----- Differential problem data -----
        self.comp = int(param_dict["comp"])
        self.param_dim = int(param_dict["param_dim"])
        self.variable = param_dict["variable"]

        # ----- Architectural details
        self.seed = int(param_dict["seed"])
        self.skip = int(param_dict["skip"])
        self.conv = param_dict["conv"]
        self.ffn = int(param_dict["ffn"])
        self.nodes = int(param_dict["nodes"])
        self.bottleneck_dim = int(param_dict["bottleneck_dim"])
        self.in_channels = int(param_dict["in_channels"])
        self.hidden_channels = [self.comp] * self.in_channels
        self.layer_vec = [self.bottleneck_dim + self.param_dim] + param_dict["layer_vec"] + [self.bottleneck_dim]
        self.scaler_name = param_dict["scaler_name"]
        self.num_nodes = int(param_dict["num_nodes"])
        self.recnet_act = param_dict["recnet_act"]
        self.dynnet_act = param_dict["dynnet_act"]
        self.continuous_dependence = bool(param_dict["continuous_dependence"])
        self.decoder_dynamics = bool(param_dict["decoder_dynamics"])
        self.changing_ic = bool(param_dict["changing_ic"])

        # ----- Training hyperparams -----
        self.rate = int(param_dict["rate"])
        self.time_rate = int(param_dict["time_rate"])
        self.interpolate_signals = bool(param_dict["interpolate_signals"])
        self.max_epochs = int(param_dict["max_epochs"])
        self.dt = float(param_dict["dt"])
        self.minibatch = None if param_dict["minibatch"] is None else int(param_dict["minibatch"])

        # ----- Loss, learning rate, and scheduler hyperparameters -----
        self.cross_validation = bool(param_dict["cross_validation"])
        self.sparse_method = param_dict["sparse_method"]
        self.weight_decay = float(param_dict["weight_decay"])
        self.learning_rate = float(param_dict["learning_rate"])
        self.tolerance = float(param_dict["tolerance"])
        self.miles = param_dict["miles"]
        self.gamma = float(param_dict["gamma"])
        self.epsilon = float(param_dict["epsilon"])
        self.delta = float(param_dict["delta"])

        # ----- To save the network -----
        self.net_name = param_dict["net_name"]
        self.net_run = '_' + self.scaler_name
        self.net_dir = './' + self.net_name + '/' + self.net_run + '/' + self.variable + '_' + self.net_name + '_btt' + str(self.bottleneck_dim) \
                            + '_seed' + str(self.seed) + '_lv' + str(len(self.layer_vec)-2) + '_hc' + str(len(self.hidden_channels)) + '_nd' + str(self.nodes) \
                            + '_ffn' + str(self.ffn) + '_skip' + str(self.skip) + '_lr' + str(self.learning_rate) + '_sc' + '_rate' + str(self.rate) + '_time_rate' + str(self.time_rate) +'/' \
                            + f'_changing_ic_{str(self.changing_ic)}_' + f'decoder_dynamics_{str(self.decoder_dynamics)}'+ f'_layer_vec_{str(self.layer_vec)}'



class DynNet(nn.Module):
    """
    This MLP approximates the time derivative (rhs) of the latent trajectory. 
    It maps the concatenated vector of the current latent state $\mathbf{s}(t)$ 
    and parameters $\mu(t)$ to the latent velocity $\dot{\mathbf{s}}(t)$.
    
    The network structure is defined by the tuple HyperParams.layer_vec, where each entry 
    represents the size of a layer. The number of layers is inferred from this list.

    Input: $\mathbb{R}^{d + p}$ (Latent dim + Parameter dim)
    Output: $\mathbb{R}^{d}$ (Latent velocity)
    """

    def __init__(self, HyperParams):
        super().__init__()
        self.dynnet_act = get_activation(HyperParams.dynnet_act)
        self.layer_vec = HyperParams.layer_vec

        self.maptovec = nn.ModuleList([
            nn.Linear(self.layer_vec[k], self.layer_vec[k+1]) 
            for k in range(len(self.layer_vec) - 1)
        ])

    def forward(self, x):
        """
        Computes the latent derivative.
        Activation is applied to all hidden layers; the output layer remains linear 
        to allow for arbitrary derivative magnitudes.
        """
        for i, layer in enumerate(self.maptovec):
            x = layer(x) if i == len(self.maptovec) - 1 else self.dynnet_act(layer(x))
        return x  # Expected to represent the derivative of s(t)


class RecNet(torch.nn.Module):
    """
    Graph Convolutional decoder ($NN_{rec}$).

    Acts as the Decoder in the autoencoder framework, transforming latent 
    representations back into spatial node features on the computational graph.

    Architectural Logic:
    1. **Parameter Conditioning:** If `continuous_dependence` is True, the 
       input dimension is expanded to include $\mu$, effectively allowing 
       the decoder to morph the reconstruction basis based on parameters.
    2. **Temporal Masking:** If `decoder_dynamics` is False, the time component 
       of the parameter vector is stripped, forcing the decoder to be time-invariant.
    """
    
    def __init__(self, HyperParams):
        super().__init__()
        self.latent_dim = HyperParams.bottleneck_dim
        self.param_dim = HyperParams.param_dim
        if not HyperParams.decoder_dynamics:
            self.param_dim -=1
        if HyperParams.continuous_dependence:
            self.latent_dim += self.param_dim
        self.decoder = gca.Decoder(
            HyperParams.hidden_channels,
            self.latent_dim,
            HyperParams.num_nodes,
            ffn=HyperParams.ffn,
            skip=HyperParams.skip,
            act=get_activation(HyperParams.recnet_act),
            conv=HyperParams.conv
        )

    def solo_decoder(self, x, data):
        """Direct call to the underlying GCN decoder."""
        x = self.decoder(x, data)
        return x
    
    def forward(self, z, data):
        """
        Forward pass.
        Args:
            z (Tensor): Latent vector (possibly concatenated with parameters).
            data (Data): PyG batch object containing graph connectivity.
        """
        x = self.solo_decoder(z, data)
        return x