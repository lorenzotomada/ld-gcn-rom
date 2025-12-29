import torch
import torch.nn.functional as F
from torch_geometric.data import Batch
import numpy as np
from tqdm import tqdm
from .utils import prepare_data, time_stepping, decode
from .loss import mse_loss, physics_loss


def train(model_decoder, model_dyn, optimizer, device, scheduler, \
          train_loader, test_loader, HyperParams, params_train, \
          params_test, loss=mse_loss, ic_s_train=None, ic_s_test=None):
    """
    Function to train a LD-GCN model.
    The training process optimizes two models simultaneously:
    1. **model_dyn**: A latent propagator that evolves state $\mathbf{s}_t$ to $\mathbf{s}_{t+1}$ 
       given parameters $\mu$.
    2. **model_decoder**: A Graph Neural Network that maps $\mathbf{s}_t$ back to 
       high-dimensional mesh space.

    Technical Features:
    - **Optimization:** Supports both first-order (Adam/SGD) and second-order (LBFGS) 
      optimizers via closure functions.
    - **Checkpointing:** Implements "best model" tracking based on the minimum training 
      loss encountered during the epoch loop.
    - **Early Stopping:** Terminates if `train_loss` falls below `HyperParams.tolerance`.

    Args:
        model_decoder (nn.Module): The graph-based reconstruction network.
        model_dyn (nn.Module): The neural dynamical system (e.g., Neural ODE or RNN).
        optimizer (torch.optim.Optimizer): Optimizer instance.
        device (str): Compute device ('cuda' or 'cpu').
        scheduler (torch.optim.lr_scheduler): Learning rate adjustment strategy.
        train_loader/test_loader: Iterables providing raw data snapshots.
        HyperParams (object): Namespace containing training constraints (max_epochs, 
            net_dir, tolerance, etc.).
        params_train/test (np.ndarray): Physical parameters associated with 
            each trajectory.
        loss (function): Loss function (defaults to mse_loss).
        ic_s_train/test (Tensor, optional): Pre-computed initial conditions for the 
            latent trajectories.
    """
    train_history = dict(train=[])
    test_history = dict(test=[])

    min_train_loss = np.inf
    best_epoch = 0

    model_decoder.train()
    model_dyn.train()

    loop = tqdm(range(HyperParams.max_epochs))

    train_data_list, times_train, ratio_train, params_train = prepare_data(
                                                                          device,
                                                                          train_loader,
                                                                          HyperParams,
                                                                          params_train,
                                                                          )
    test_data_list, times_test, ratio_test, params_test = prepare_data(
                                                                      device,
                                                                      test_loader,
                                                                      HyperParams,
                                                                      params_test,
                                                                      )

    is_lbfgs = isinstance(optimizer, torch.optim.LBFGS)

    for epoch in loop:
        optimizer.zero_grad()

        if is_lbfgs:
            def closure():
                optimizer.zero_grad()
                train_loss = compute_loss_on_epoch(model_decoder, model_dyn, device, HyperParams, train_data_list, params_train, times_train, ratio_train, loss, ic_s_train)
                train_loss.backward()
                return train_loss

            train_loss = optimizer.step(closure)
        else:
            train_loss = compute_loss_on_epoch(
                                          model_decoder,
                                          model_dyn,
                                          device,
                                          HyperParams,
                                          train_data_list,
                                          params_train,
                                          times_train,
                                          ratio_train,
                                          loss,
                                          ic_s_train,
                                          )
            train_loss.backward()
            optimizer.step()
            scheduler.step()
 
        train_history['train'].append(train_loss.item())

        if HyperParams.cross_validation:
            with torch.no_grad():
                test_loss = compute_loss_on_epoch(
                                                  model_decoder,
                                                  model_dyn,
                                                  device,
                                                  HyperParams,
                                                  test_data_list,
                                                  params_test,
                                                  times_test,
                                                  ratio_test,
                                                  loss,
                                                  ic_s_test
                                                  )
                test_history['test'].append(test_loss.item())

            loop.set_postfix({"Loss(training)": train_history['train'][-1], "Loss(validation)": test_history['test'][-1]})
        else:
            loop.set_postfix({"Loss(training)": train_history['train'][-1]})


        if HyperParams.tolerance >= train_loss:
            print(f'Early stopping! Stopped at epoch {epoch}.')
            break


        if train_loss < min_train_loss:
            np.save(HyperParams.net_dir+'history' + HyperParams.net_run+'.npy', train_history)
            np.save(HyperParams.net_dir+'history_test' + HyperParams.net_run+'.npy', test_history)
        
            best_epoch = epoch
            min_train_loss = train_loss

            torch.save(model_decoder.state_dict(), HyperParams.net_dir+HyperParams.net_name+HyperParams.net_run+'_decoder.pt')
            torch.save(model_dyn.state_dict(), HyperParams.net_dir+HyperParams.net_name+HyperParams.net_run+'_dyn.pt')

    print(f'Best epoch: {best_epoch}, min train loss: {min_train_loss}')
    model_decoder.load_state_dict(torch.load(HyperParams.net_dir+HyperParams.net_name+HyperParams.net_run+'_decoder.pt', map_location=torch.device('cpu')))
    model_dyn.load_state_dict(torch.load(HyperParams.net_dir+HyperParams.net_name+HyperParams.net_run+'_dyn.pt', map_location=torch.device('cpu')))


def compute_loss_on_epoch(model_decoder, model_dyn, device, HyperParams,
                           data_list, params, times, ratio, loss, ic_s=None):
    """
    Computes the cumulative loss across all temporal snapshots of a dataset.

    1. **Initialization:** Defines the starting latent state $\mathbf{s}_0$. If `ic_s` 
       is provided, it uses the predefined state; otherwise, starts at the origin.
    2. **IC Loss:** Reconstructs the initial condition and compares it to ground truth.
    3. **Unrolling:** Iterates through time steps. For each step:
        a. Evolves latent state: $\mathbf{s}_{t+1} = \text{model\_dyn}(\mathbf{s}_t, \mu_t)$
        b. Reconstructs field: $\hat{\mathbf{x}}_{t+1} = \text{model\_decoder}(\mathbf{s}_{t+1}, \mu_{t+1})$
        c. Computes loss $\mathcal{L}(\hat{\mathbf{x}}_{t+1}, \mathbf{x}_{t+1})$.

    Handling of Minibatches:
    - If `HyperParams.minibatch` is set, graphs for different trajectories are 
      aggregated into a single `Batch` object for efficient GPU decoding.

    Returns:
        torch.Tensor: Mean loss over all trajectories and time steps.
    """
    current_loss = 0.0
    total_examples = 0
    dt = HyperParams.dt

    n_trajectories = params.shape[0]
    interpolate = HyperParams.interpolate_signals
    parameter_multiplier = ratio if interpolate else 1

    # Initialize latent states for all trajectories
    if ic_s is None:
        stn = torch.zeros(n_trajectories, HyperParams.bottleneck_dim,
                          device=device, dtype=torch.float32)
    else:
        stn = torch.tensor(ic_s.T, device=device, dtype=torch.float32)
        assert stn.shape[0] == n_trajectories, \
            f"ic_s mismatch: got {stn.shape[0]} ICs, expected {n_trajectories}"

    # ---- Loss on initial condition ----
    if HyperParams.minibatch is None or not HyperParams.changing_ic:
        # Full batch or serial IC handling
        if HyperParams.changing_ic:
            for alpha_idx in range(n_trajectories):
                exact_ic = data_list[alpha_idx * len(times)]
                pred_ic = decode(model_decoder, HyperParams,
                                 stn[alpha_idx], params[alpha_idx][0], exact_ic)
                current_loss += loss(pred_ic, exact_ic, HyperParams)
                total_examples += 1
        else:
            exact_ic = data_list[0]
            pred_ic = decode(model_decoder, HyperParams,
                             stn[0], params[0][0], exact_ic)
            current_loss += loss(pred_ic, exact_ic, HyperParams)
            total_examples += 1
    else:
        # Minibatch IC handling
        mini_batch_size = HyperParams.minibatch
        for i in range(0, n_trajectories, mini_batch_size):
            end = min(i + mini_batch_size, n_trajectories)

            # Prepare batch data
            batch_data_list = [data_list[traj_idx * len(times)]
                               for traj_idx in range(i, end)]
            data_batch = Batch.from_data_list(batch_data_list).to(device)

            stn_batch = stn[i:end]
            param_batch = params[i:end, 0]

            predictions = decode(model_decoder, HyperParams,
                                 stn_batch, param_batch, data_batch)

            current_loss += loss(predictions, data_batch, HyperParams)
            total_examples += (end - i)

    # ---- Time-stepping loop ----
    for t_idx in range(len(times) - 1):
        # Evolve latent states for all trajectories
        stn = time_stepping(model_dyn, ratio, stn,
                            params[:, parameter_multiplier * t_idx:
                                      parameter_multiplier * (t_idx + 1)],
                            dt, interpolate)

        if HyperParams.minibatch is None:
            # Serial decoding
            for alpha_idx in range(n_trajectories):
                data = data_list[alpha_idx * len(times) + t_idx + 1]
                prediction = decode(model_decoder, HyperParams,
                                    stn[alpha_idx],
                                    params[alpha_idx][t_idx * parameter_multiplier + 1],
                                    data)
                current_loss += loss(prediction, data, HyperParams)
                total_examples += 1
        else:
            # Minibatch decoding
            mini_batch_size = HyperParams.minibatch
            for i in range(0, n_trajectories, mini_batch_size):
                end = min(i + mini_batch_size, n_trajectories)

                batch_data_list = [data_list[traj_idx * len(times) + t_idx + 1]
                                   for traj_idx in range(i, end)]
                data_batch = Batch.from_data_list(batch_data_list).to(device)

                stn_batch = stn[i:end]
                param_batch = params[i:end, t_idx * parameter_multiplier + 1]

                predictions = decode(model_decoder, HyperParams,
                                     stn_batch, param_batch, data_batch)

                current_loss += loss(predictions, data_batch, HyperParams)
                total_examples += (end - i)

    current_loss /= total_examples
    return current_loss