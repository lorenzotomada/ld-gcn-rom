import numpy as np
from ld_gcn.preprocessing import inverse_normalize_input
import torch


def save_error(error, norm, HyperParams, vars):
    """
    Computes snapshot-wise relative errors and exports statistical aggregates to a text file.
    For each snapshot $i$, the relative error is calculated as:
    $$\epsilon_{i} = \frac{e_i}{n_i}$$
    where $e_i$ is the absolute $L_2$ error and $n_i$ is the norm of the ground truth.
    The function saves the vector $[\max(\epsilon), \text{mean}(\epsilon), \min(\epsilon)]$.

    Parameters:
    -----------
    error : List[float] or np.ndarray
        Vector of absolute errors for each snapshot.
    norm : List[float] or np.ndarray
        Vector of ground truth norms for each snapshot.
    HyperParams : object
        HyperParams object containing network parameters, including `net_dir` and `net_run`.
    vars : str
        The variable/field name (e.g., 'U') to identify the file.

    Returns:
    --------
    None : Saves a .txt file.
    """

    error = np.array(error)
    norm = np.array(norm)
    rel_error = error/norm
    np.savetxt(HyperParams.net_dir+'relative_errors'+HyperParams.net_run+vars+'.txt', [max(rel_error), sum(rel_error)/len(rel_error), min(rel_error)])



def print_error(error, norm, vars):
    """
    Computes and logs descriptive statistics of absolute and relative errors to the console.

    Parameters:
    -----------
    error : List[float] or np.ndarray
        The absolute error residuals $\|\mathbf{z} - \hat{\mathbf{z}}\|_2$.
    norm : List[float] or np.ndarray
        The norms of the reference signals $\|\mathbf{z}\|_2$.
    vars : str
        Descriptor of the field being analyzed (e.g., "U").

    Returns:
    --------
    None : Prints statistics directly to stdout.
    """

    error = np.array(error)
    norm = np.array(norm)
    rel_error = error/norm
    print("\nMaximum absolute error for field "+vars+" = ", max(error))
    print("Mean absolute error for field "+vars+" = ", sum(error)/len(error))
    print("Minimum absolute error for field "+vars+" = ", min(error))
    print("\nMaximum relative error for field "+vars+" = ", max(rel_error))
    print("Mean relative error for field "+vars+" = ", sum(rel_error)/len(rel_error))
    print("Minimum relative error for field "+vars+" = ", min(rel_error))



def compute_error(res, VAR, scaler, reference_value = None):
    """
    Inverse-transforms model output and computes error metrics against ground truth.

    The function performs three steps:
    1.  **Tensor Reshaping:** Adjusts input dimensions based on the feature count (VAR.shape[2]).
    2.  **Inverse Scaling:** Maps the data from the scaled range (e.g., [0, 1] or [-1, 1]) 
        back to physical units using the provided scaler.
    3.  **Error Calculation:** - If `reference_value` is None: Calculates snapshot-wise $L_2$ absolute errors 
          and norms.
        - If `reference_value` is not None: Calculates a range-normalized Root Mean 
          Square Error (NRMSE):
          $$\text{NRMSE} = \frac{\sqrt{\text{MSE}(\mathbf{Z}, \hat{\mathbf{Z}})}}{\max(\mathbf{Z}) - \min(\mathbf{Z})}$$

    Parameters:
    -----------
    res : torch.Tensor or np.ndarray
        The reconstructed output from LD-GCN.
    VAR : torch.Tensor or np.ndarray
        The ground truth (target) tensor.
    scaler : object
        The scaling object (e.g., MinMaxScaler) used during preprocessing.
    reference_value : Any, optional
        Flag to trigger NRMSE calculation. If provided, the return behavior changes.

    Returns:
    --------
    error_abs_list : List[float] or float
        If reference_value is None: A list of $L_2$ errors per snapshot.
        If reference_value is not None: A single NRMSE scalar.
    norm_z_list : List[float]
        A list of $L_2$ norms for each snapshot of the ground truth. 
        (Note: remains empty if reference_value is not None).
    """

    error_abs_list = list()
    norm_z_list = list()

    if VAR.shape[2] == 1:
        VAR = VAR[:, :, 0].T
        res = res[:, :, 0].T
    elif VAR.shape[2] == 2:
        VAR = VAR.permute(1, 0, 2)
        res = res.permute(1, 0, 2)

    Z_net = inverse_normalize_input(res, scaler).detach().numpy()
    Z = inverse_normalize_input(VAR, scaler).detach().numpy()

    if reference_value is None:
        for snap in range(Z.shape[1]):
            error_abs = np.linalg.norm((Z[:, snap] - Z_net[:, snap]), 2)
            norm_z = np.linalg.norm(Z[:, snap], 2)
            error_abs_list.append(error_abs)
            norm_z_list.append(norm_z)
    else:
        error_abs_list = np.sqrt(np.mean(np.square(Z_net-Z)))/(np.max(Z)-np.min(Z))
    return error_abs_list, norm_z_list