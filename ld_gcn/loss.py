import torch
import torch.nn.functional as F



def mse_loss(prediction, data, HyperParams):
    """
    Computes Mean Squared Error (MSE) with graph-level normalization.

    This function is written in such a way that each graph in the batch contributes equally to the loss,
    regardless of the number of nodes in each graph. This is useful to ensure the loss is computed
    consistently when using minibatches of graphs.

    For a batch of $G$ graphs, the loss is:
    $$\mathcal{L} = \sum_{g=1}^{G} \left( \frac{1}{N_g \cdot F} \sum_{i=1}^{N_g} \sum_{j=1}^{F} (\hat{x}_{i,j} - x_{i,j})^2 \right)$$

    Args:
        prediction (Tensor): Predicted node features $[N_{total}, F]$.
        data (Data): PyG Data object containing ground truth `.x` and `.batch` index.
        HyperParams (object): Configuration object. Uses `.minibatch` to toggle logic.

    Returns:
        Tensor: Scalar loss value.
    """
    if HyperParams.minibatch is None:
        loss_update = F.mse_loss(prediction, data.x, reduction='mean')
        return loss_update
    else:
        se = F.mse_loss(prediction, data.x, reduction='none')  # [N, F]
        se_per_node = se.mean(dim=1)  # mean over features: shape [N]

        # Get graph indices
        batch = data.batch  # [N], values in [0, num_graphs-1]

        # Compute per-graph mean MSE
        num_graphs = batch.max().item() + 1
        per_graph_mse = torch.zeros(num_graphs, device=prediction.device).scatter_add_(
            0, batch, se_per_node
        )
        counts = torch.bincount(batch, minlength=num_graphs).float()
        per_graph_mse /= counts  # average per graph

        # Sum across all graphs.
        # The reason for using this instead of F.mse_loss(prediction, data.x, reduction='mean') is to avoid to change the value of the loss when using minibatches
        # Indeed, the previous command would just average over the whole batch, changing the value of the loss
        return per_graph_mse.sum()


def rmse_loss(prediction, data, HyperParams):
    """
    Computes a Relative Mean Square Error loss.

    Formulation for graph $g$:
    $$\epsilon_g = \frac{\text{MSE}(\hat{x}_g, x_g)}{\text{Mean}(x_g^2) + \epsilon}$$

    Args:
        prediction (Tensor): Predicted node features.
        data (Data): Ground truth features and batch mapping.
        HyperParams (object): Included for interface consistency.

    Returns:
        Tensor: Sum of normalized squared errors across graphs.
    """
    epsilon = 1e-8

    if HyperParams.minibatch is None:
        numerator = F.mse_loss(prediction, data.x, reduction='mean')
        denominator = torch.mean(data.x ** 2) + epsilon
        return numerator / denominator
    else:
        se = F.mse_loss(prediction, data.x, reduction='none')  # [N, F]
        se_per_node = se.mean(dim=1)  # [N]

        # Compute squared norm of true data (denominator), also per node
        target_squared = data.x ** 2
        target_energy_per_node = target_squared.mean(dim=1)  # [N]

        batch = data.batch  # [N]
        num_graphs = batch.max().item() + 1

        # Sum over nodes per graph
        mse_per_graph = torch.zeros(num_graphs, device=prediction.device).scatter_add_(
            0, batch, se_per_node
        )
        energy_per_graph = torch.zeros(num_graphs, device=prediction.device).scatter_add_(
            0, batch, target_energy_per_node
        )

        counts = torch.bincount(batch, minlength=num_graphs).float()
        mse_per_graph /= counts
        energy_per_graph /= counts

        # Normalize MSE by energy and sum across graphs
        normalized_rmse_squared = mse_per_graph / (energy_per_graph + epsilon)
        return normalized_rmse_squared.sum()


def physics_loss(prediction, data, HyperParams):
    """
    Multi-objective loss combining Magnitude MSE and directional similarity. Used in the lid-driven
    test case.

    The loss consists of two terms:
    1. **Term 1 (Magnitude):** Standard graph-averaged MSE.
    2. **Term 2 (Direction):** MSE between $L_2$-normalized vectors, forcing 
       the model to learn the correct orientation of field vectors.

    $$\mathcal{L} = \text{MSE}(\hat{\mathbf{v}}, \mathbf{v}) + \delta \cdot \text{MSE}\left(\frac{\hat{\mathbf{v}}}{\|\hat{\mathbf{v}}\|+\epsilon}, \frac{\mathbf{v}}{\|\mathbf{v}\|+\epsilon}\right)$$

    Args:
        prediction (Tensor): Predicted vectors $[N, d]$.
        data (Data): Target vectors and batch mapping.
        HyperParams (object): Must contain `.epsilon` (stability) and `.delta` (term weight).

    Returns:
        Tensor: Combined physics-informed loss.
    """
    epsilon = HyperParams.epsilon
    delta = HyperParams.delta

    v_pred = prediction     # [N, d]
    v_target = data.x       # [N, d]

    if HyperParams.minibatch is None:
        # global mean (no batching)
        term1 = F.mse_loss(v_pred, v_target, reduction='mean')

        v_t_dir = v_target / (v_target.norm(dim=-1, keepdim=True) + epsilon)
        v_p_dir = v_pred   / (v_pred.norm(dim=-1, keepdim=True)   + epsilon)
        term2 = delta * F.mse_loss(v_p_dir, v_t_dir, reduction='mean')

        return term1 + term2

    # --- minibatch-aware path ---
    batch = data.batch              # [N]
    num_graphs = int(batch.max().item()) + 1

    # Term 1: per-node MSE → per-graph average
    se = F.mse_loss(v_pred, v_target, reduction='none')  # [N, d]
    se_node = se.mean(dim=1)                             # [N]
    term1_pg = torch.zeros(num_graphs, device=v_pred.device).scatter_add_(0, batch, se_node)
    counts   = torch.bincount(batch, minlength=num_graphs).float()
    term1_pg /= counts                                   # [num_graphs]

    # Term 2: directional MSE → per-graph average
    v_t_dir = v_target / (v_target.norm(dim=-1, keepdim=True) + epsilon)
    v_p_dir = v_pred   / (v_pred.norm(dim=-1, keepdim=True)   + epsilon)
    se2 = F.mse_loss(v_p_dir, v_t_dir, reduction='none')  # [N, d]
    se2_node = se2.mean(dim=1)                            # [N]
    term2_pg = torch.zeros(num_graphs, device=v_pred.device).scatter_add_(0, batch, se2_node)
    term2_pg /= counts                                    # [num_graphs]

    # Sum contributions from all graphs
    return term1_pg.sum() + delta * term2_pg.sum()