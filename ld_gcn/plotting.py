import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
from matplotlib import colormaps
from matplotlib import ticker
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.colors as mcolors
from ld_gcn import preprocessing
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.mplot3d import Axes3D
#from scipy import interpolate
import matplotlib.tri as mtri
from matplotlib.colors import Normalize
import torch
import os
from IPython.display import HTML


plt.rcParams.update({
    'axes.labelsize': 22,  #'x-large',
    'legend.fontsize': 19,
    'xtick.labelsize': 17,
    'ytick.labelsize': 17,
    'figure.titlesize': 23
})


# ------------------------------ IMPORTANT ------------------------------ #
# Given the lenght of this file and its low conceptual intricacy, it 
# might be the case that some plotting functions are not commented 
# as thoroughly as the rest of the codebase.
# ----------------------------------------------------------------------- #


# ----- Function to plot the loss -----
def plot_loss(HyperParams):
    """
    Plots the history of losses during the training of the latent net + decoder.

    Parameters:
    HyperParams (object): An object containing the parameters of the architecture.
    """
    history = np.load(HyperParams.net_dir+'history'+HyperParams.net_run+'.npy', allow_pickle=True).item()
    history_test = np.load(HyperParams.net_dir+'history_test'+HyperParams.net_run+'.npy', allow_pickle=True).item()
    ax = plt.figure().gca()
    ax.semilogy(history['train'])
    ax.semilogy(history_test['test'], '--')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.title('Loss over training epochs')
    plt.legend(['Train','Test'])
    plt.savefig(HyperParams.net_dir+'history_losses'+HyperParams.net_run+'.pdf', bbox_inches='tight')#, dpi=500)


# ----- Functions to plot fields and full-order animations -----


def plot_fields(param_idx, SNAP, results, scaler_all, HyperParams, dataset, PARAMS, lid_driven = False, N_DIGITS_TO_PLOT=3):
    """
    Plots the field solution for a given snapshot, the ground truth, and the error field.
    """
    PAD = 10

    TIMES = PARAMS[..., -1][0].to('cpu')
    results=results.to('cpu')
    PARAMS=PARAMS.to('cpu')
    fig = plt.figure(figsize=(15, 5))

    starting_idx = param_idx*len(TIMES)

    if HyperParams.comp == 1:
        z_net = preprocessing.inverse_normalize_input(results[starting_idx+SNAP, :, :], scaler_all)
    elif HyperParams.comp == 2:
        z_net = preprocessing.inverse_normalize_input(results.permute(1,0,2), scaler_all)[:, starting_idx+SNAP, :]

    # If n_comp==2, change z_net into a 1D array by computing the norm of each row
    z_net = z_net.detach().numpy()

    if HyperParams.comp == 1:
        z_net = z_net[:, 0]
        ground_truth = dataset.U[:, starting_idx+SNAP]
    if HyperParams.comp == 2:
        z_net = np.linalg.norm(z_net, axis=1)
        ground_truth = np.linalg.norm(np.column_stack((dataset.VX[:, starting_idx+SNAP], dataset.VY[:, starting_idx+SNAP])), axis=1)
    
    xx = dataset.xx
    yy = dataset.yy
    rel_error_field = abs(ground_truth - z_net) / np.linalg.norm(ground_truth, 2)

    triang = np.asarray(dataset.T - 1)
    cmap = cm.get_cmap(name='jet', lut=None)
    norm1 = mcolors.Normalize(vmin=z_net.min(), vmax=z_net.max())
    gs1 = gridspec.GridSpec(1, 3)  # Change to 2 columns for 2 subplots
    plt.subplots_adjust(wspace=0.4)  

    # Subplot 1
    ax1 = plt.subplot(gs1[0, 0])
    cs1 = ax1.tricontourf(xx[:, starting_idx+SNAP], yy[:, starting_idx+SNAP], triang, z_net, 100, cmap=cmap, norm=norm1)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cbar1 = plt.colorbar(cs1, cax=cax1, format=ticker.FormatStrFormatter('%.1f'))
    cbar1.locator = ticker.MaxNLocator(nbins=N_DIGITS_TO_PLOT)
    cbar1.update_ticks()
    #cbar1.formatter.set_powerlimits((0, 0))
    #cbar1.update_ticks()
    sim_title = r"$\|\boldsymbol{u}_{\text{sim}}\|_{2}$" if lid_driven else r'$\boldsymbol{u}_{\text{sim}}$'
    ax1.set_aspect('equal', 'box')
    ax1.set_title(sim_title, fontsize=22, pad=PAD)# for $\mu$ = {np.around(PARAMS[param_idx][0][:-1].numpy(), 2)}' + f' at t = {TIMES[SNAP].numpy()}')
    

    # Subplot 2
    #average_dataset = torch.mean(dataset.U[:, :90], dim=1)
    norm2 = mcolors.Normalize(vmin=ground_truth.min(), vmax=ground_truth.max())
    ax2 = plt.subplot(gs1[0, 1])  # Add second subplot
    cs2 = ax2.tricontourf(xx[:, starting_idx+SNAP], yy[:, starting_idx+SNAP], triang, ground_truth, 100, cmap=cmap, norm=norm2)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cbar2 = plt.colorbar(cs2, cax=cax2,  format=ticker.FormatStrFormatter('%.1f'))
    #cbar2.formatter.set_powerlimits((0, 0))
    #cbar2.update_ticks()
    cbar2.locator = ticker.MaxNLocator(nbins=N_DIGITS_TO_PLOT) # change for LDNet plots
    cbar2.update_ticks()
    ground_truth_title = r'$\|\boldsymbol{u}_{h}\|_{2}$' if lid_driven else r'$\boldsymbol{u}_{h}$'
    ax2.set_aspect('equal', 'box')
    ax2.set_title(ground_truth_title, fontsize=22, pad=PAD)

    
    class OneDecimalScalarFormatter(ticker.ScalarFormatter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_powerlimits((0, 0))

        def _set_format(self):
            self.format = "%.1f"


    # Subplot 3
    norm3 = mcolors.Normalize(vmin=rel_error_field.min(), vmax=rel_error_field.max())
    ax3 = plt.subplot(gs1[0, 2])
    cs3 = ax3.tricontourf(xx[:, starting_idx+SNAP], yy[:, starting_idx+SNAP], triang, rel_error_field, 100, cmap=cmap, norm=norm3)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.1)
    cbar3 = plt.colorbar(cs3, cax=cax3)
    formatter = OneDecimalScalarFormatter(useMathText=True)
    cbar3.locator = ticker.MaxNLocator(nbins=N_DIGITS_TO_PLOT)
    cbar3.formatter = formatter
    cbar3.update_ticks()
    err_title = r'$\|\boldsymbol{e}_\text{rel}\|_{2}$' if lid_driven else r'$\boldsymbol{e}_\text{rel}$'
    ax3.set_aspect('equal', 'box')
    ax3.set_title(err_title, fontsize=22, pad=PAD)

    # Adjust layout
    plt.tight_layout()
    plt.savefig(HyperParams.net_dir + 'field_solution_SNAP' + str(SNAP) + '.pdf', bbox_inches='tight')#, dpi=500)
    plt.show()


def plot_interp_comparison(param_idx, t_idx, simulated_results, interpolated_results, HyperParams, dataset, times, extrapolation=False, N_DIGITS_TO_PLOT=3):
    """
    Plots the field solution for a given snapshot, the ground truth, and the interpolated field.
    """
    fig = plt.figure(figsize=(15, 5))
    PAD = 10

    starting_idx = param_idx*len(times)
    
    xx = dataset.xx
    yy = dataset.yy
    exact_sol = dataset.U[:, starting_idx+t_idx]

    triang = np.asarray(dataset.T - 1)
    cmap = cm.get_cmap(name='jet', lut=None)
    norm1 = mcolors.Normalize(vmin=simulated_results.min(), vmax=simulated_results.max())
    gs1 = gridspec.GridSpec(1, 3)  # Change to 2 columns for 2 subplots
    plt.subplots_adjust(wspace=0.4)  

    # Subplot 1
    ax1 = plt.subplot(gs1[0, 0])
    cs1 = ax1.tricontourf(xx[:, starting_idx+t_idx], yy[:, starting_idx+t_idx], triang, simulated_results, 100, cmap=cmap, norm=norm1)
    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="5%", pad=0.1)
    cbar1 = plt.colorbar(cs1, cax=cax1, format=ticker.FormatStrFormatter('%.1f'))
    cbar1.locator = ticker.MaxNLocator(nbins=N_DIGITS_TO_PLOT)
    cbar1.update_ticks()
    #cbar1.formatter.set_powerlimits((0, 0))
    #cbar1.update_ticks()
    ax1.set_aspect('equal', 'box')
    ax1.set_title(r'$\boldsymbol{u}_\text{sim}$', fontsize=22, pad=PAD)
    

    # Subplot 2
    norm2 = mcolors.Normalize(vmin=interpolated_results.min(), vmax=interpolated_results.max())
    ax2 = plt.subplot(gs1[0, 1])  # Add second subplot
    cs2 = ax2.tricontourf(xx[:, starting_idx+t_idx], yy[:, starting_idx+t_idx], triang, interpolated_results, 100, cmap=cmap, norm=norm2)
    divider2 = make_axes_locatable(ax2)
    cax2 = divider2.append_axes("right", size="5%", pad=0.1)
    cbar2 = plt.colorbar(cs2, cax=cax2, format=ticker.FormatStrFormatter('%.1f'))
    cbar2.locator = ticker.MaxNLocator(nbins=N_DIGITS_TO_PLOT)
    cbar2.update_ticks()
    #cbar2.formatter.set_powerlimits((0, 0))
    #cbar2.update_ticks()
    ax2.set_aspect('equal', 'box')
    ax2.set_title(r'$\boldsymbol{u}_\text{interp}$', fontsize=22, pad=PAD)

    # Subplot 3
    norm3 = mcolors.Normalize(vmin=exact_sol.min(), vmax=exact_sol.max())
    ax3 = plt.subplot(gs1[0, 2])
    cs3 = ax3.tricontourf(xx[:, starting_idx+t_idx], yy[:, starting_idx+t_idx], triang, exact_sol, 100, cmap=cmap, norm=norm3)
    divider3 = make_axes_locatable(ax3)
    cax3 = divider3.append_axes("right", size="5%", pad=0.1)
    cbar3 = plt.colorbar(cs3, cax=cax3, format=ticker.FormatStrFormatter('%.1f'))
    cbar3.locator = ticker.MaxNLocator(nbins=N_DIGITS_TO_PLOT)
    cbar3.update_ticks()
    #cbar3.formatter.set_powerlimits((0, 0))
    #cbar3.update_ticks()
    ax3.set_aspect('equal', 'box')
    ax3.set_title(r'$\boldsymbol{u}_h$', fontsize=22, pad=PAD)

    # Adjust layout
    plt.tight_layout()
    if not extrapolation:
        plt.savefig(HyperParams.net_dir + 'interp_fields_SNAP' + str(t_idx) + '.pdf', bbox_inches='tight')#, dpi=500)
    else:
        plt.savefig(HyperParams.net_dir + 'extrapolation_interp_fields_SNAP' + str(t_idx) + '.pdf', bbox_inches='tight')
    plt.show()


def create_animation(SAMPLE, Z, HyperParams, dataset, xyz, param_sample, comp="_U", flag='sim'):
    """
    flag can be "sim", "h", or "gca".
    Create animation for time-dependent solutions.
    """
    fig = plt.figure()
    xx = xyz[0]
    yy = xyz[1]
    fmt = ticker.ScalarFormatter(useMathText=True)
    fmt.set_powerlimits((0, 0))
    triang = np.asarray(dataset.T - 1)
    gs1 = gridspec.GridSpec(1, 1)
    ax = plt.subplot(gs1[0, 0])
    sequence_length = Z.shape[1] // param_sample
    start = SAMPLE * sequence_length
    try:
        cs = ax.tricontourf(xx[:, start], yy[:, start], triang, Z[:, start], 100, cmap=colormaps['jet'])
    except:
        Z=Z.detach().numpy()
        cs = ax.tricontourf(xx[:, start], yy[:, start], triang, Z[:, start], 100, cmap=colormaps['jet'])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cbar = plt.colorbar(cs, cax=cax, format=fmt)
    tick_locator = MaxNLocator(nbins=5)
    cbar.locator = tick_locator
    cbar.ax.yaxis.set_offset_position('left')
    cbar.update_ticks()


    def update_animation(i):
        if flag=='sim':
            image_title = rf'$\boldsymbol{{u}}_{{\text{{sim}}}}(t={(i+1) *2* HyperParams.dt:.2f})$'
        elif flag=='h':
            image_title = rf'$\boldsymbol{{u}}_h(t={(i+1) * 2*HyperParams.dt:.2f})$'
        elif flag=='gca':
            image_title = rf'$\boldsymbol{{u}}_{{\text{{GCA}}}}(t={(i+1) * 2*HyperParams.dt:.2f})$'
        else:
            raise ValueError("Invalid flag. Use 'sim', 'h', or 'gca'.")
        cs = ax.tricontourf(xx[:, i+start], yy[:, i+start], triang, Z[:, i+start], 100, cmap=colormaps['jet'])
        # ax.triplot(xx[:, SAMPLE], yy[:, SAMPLE], triang, lw=0.5, color="black")
        plt.tight_layout()
        ax.set_aspect('equal', 'box')
        #ax.set_title('Rollout Solution field for $\mu$ = '+str(np.around(params[SAMPLE][0][:-1].detach().numpy(), 2)))
        ax.set_title(image_title, fontsize=25)
        return cs

    anim = animation.FuncAnimation(fig=fig, func=update_animation, frames=sequence_length, interval=100)
    anim.save(HyperParams.net_dir+'field_'+flag+str(SAMPLE)+''+HyperParams.net_run+comp+'.gif', writer='pillow', fps=10)
    plt.close(fig)
    return HTML(anim.to_jshtml())


# ----- Functions to plot the error


def plot_error(results, dataset, scaler_all, HyperParams, params, PARAMS, time, TIMES, train_trajectories, flag='_all_'):
    """
    This function plots the relative error between the predicted and actual results, for a number of parameters larger than 1
    using time and each of the other n_params-1 parameters for the plot.
    """
    vars = 'vs $\mu$ and $t$'
    Z_net = np.zeros((dataset.VX.shape[1], dataset.VX.shape[0],2))
    ground_truth = np.zeros((dataset.VX.shape[1], dataset.VX.shape[0],2))

    # Scale back the results over the entire dataset
    for i in range(results.shape[0]):
        out = preprocessing.inverse_normalize_input(results[i, :, :], scaler_all, i, HyperParams)
        out = out.detach().numpy()
        Z_net[i, :, :] = out
        ground_truth[i, :, :] = np.column_stack((dataset.VX[:, i], dataset.VY[:, i]))
    # Take only the magnitude of velocity
    Z_net = np.linalg.norm(Z_net, axis=2)
    ground_truth = np.linalg.norm(ground_truth, axis=2)
    # Calculate the relative error
    error = np.linalg.norm(Z_net - ground_truth, axis=1) / np.mean(np.linalg.norm(ground_truth, axis=1))
        
    colors = 0.0
    area = 0.0 
    tr_pt_1 = PARAMS[train_trajectories]
    tr_pt_2 = TIMES[train_trajectories]

    X1, X2 = np.meshgrid(params, time, indexing='ij')
    output = np.reshape(error, (len(params), len(time)))
    fig = plt.figure('Relative error '+vars)
    ax = fig.add_subplot()
    colors = output.flatten()
    area = output.flatten()*500

    sc = plt.scatter(X1.flatten(), X2.flatten(), s=area, c= colors, alpha=0.5, cmap=cm.coolwarm)
    plt.colorbar(sc, format=FuncFormatter(lambda x, pos: f'{x:.1e}'))
    ax.set(xlim = [-10,10], #xlim=tuple([np.min(mu_i_range), np.max(mu_i_range)]
            ylim=[0,2],
            xlabel=f'$\mu$',
            ylabel=f'$t$')
        
    ax.plot(tr_pt_1, tr_pt_2,'*r')
    ax.set_title('Relative Error '+vars)

    plt.tight_layout()
    plt.savefig(HyperParams.net_dir+'relative_error_scatter'+HyperParams.net_run+'_' + flag +'.pdf', transparent=True)#, dpi=500)
    plt.show()


def plot_relative_errors_vs_time(rel_errors, params, HyperParams, final_training_time=None, flag='_all_'):
    """
    Plots relative errors against time for different parameter configurations.

    Parameters:
    - rel_errors (np.ndarray): Array of relative errors, shape (n_sim * n_times,).
    - params (torch.Tensor): Tensor of shape (n_sim, n_times, param_dim), last column = time.
    - final_training_time (float, optional): Time separating training vs extrapolation.

    Returns:
    - None
    """

    params_np = params.cpu().numpy()
    n_sim, n_times, param_dim = params_np.shape
    times = params_np[0, :, -1]  # Assumes same time vector for all simulations

    # Unroll relative errors into shape (n_sim, n_times)
    rel_errors = np.array(rel_errors).reshape(n_sim, n_times)

    plt.figure(figsize=(8, 5))

    for i in range(n_sim):
        plt.semilogy(times, rel_errors[i], linewidth=2)

    if final_training_time is not None:
        plt.axvline(x=final_training_time, color='black', linestyle='--')  # No label here
        plt.text(final_training_time, plt.ylim()[1]*0.95, r'$T_\text{train}$',
                rotation=90, verticalalignment='top', horizontalalignment='right',
                fontsize=19,
                color='black')

    plt.xlabel('$t$')
    plt.ylabel(r'$\varepsilon_\text{rel}$')
    #plt.title('Relative errors vs time', fontsize=15)
    plt.grid(True, which='both', linestyle='--', alpha=0.1)
    plt.tight_layout()
    plt.savefig(HyperParams.net_dir+'rel_errors'+HyperParams.net_run+flag+'.pdf', bbox_inches='tight')#, dpi=500)
    plt.show()


def compare_errors_gca(rel_errors_ld, rel_errors_gca, params, HyperParams, final_training_time=None):
    params_np = params.cpu().numpy()
    n_sim, n_times, param_dim = params_np.shape
    times = params_np[0, :, -1]  # Assumes same time vector for all simulations

    # Unroll relative errors into shape (n_sim, n_times)
    rel_errors_ld = np.array(rel_errors_ld).reshape(n_sim, n_times)
    rel_errors_gca = np.array(rel_errors_gca).reshape(n_sim, n_times)

    plt.figure(figsize=(8, 5))

    for i in range(n_sim):
        if i == 0:
            plt.semilogy(times, rel_errors_ld[i], color='#065895', linewidth=2,label='LD-GCN')
            plt.semilogy(times, rel_errors_gca[i], color='#F79A25', linewidth=2,label='GCA-ROM')
        else:
            plt.semilogy(times, rel_errors_ld[i], color='#065895', linewidth=2)
            plt.semilogy(times, rel_errors_gca[i], color='#F79A25', linewidth=2)

    if final_training_time is not None:
        plt.axvline(x=final_training_time, color='black', linestyle='--')  # No label here
        plt.text(final_training_time, plt.ylim()[1]*0.95, r'$T_\text{train}$',
                rotation=90, verticalalignment='top', horizontalalignment='right',
                fontsize=19, color='black')

    plt.xlabel('$t$')
    plt.ylabel(r'$\varepsilon_\text{rel}$')
    #plt.title('Relative errors vs time', fontsize=15)
    plt.grid(True, which='both', linestyle='--', alpha=0.2)
    plt.legend(loc='upper center')#bbox_to_anchor=(1.05, 1),
    plt.tick_params(axis='both')#, labelsize=15)
    plt.tight_layout()
    plt.savefig(HyperParams.net_dir+'rel_errors_GCA_ROM'+HyperParams.net_run+'.pdf', bbox_inches='tight')#, dpi=500)
    plt.show()


def plot_real_interp_error(times, sim_errors, interp_errors_gpr, interp_sim_errors_gpr, HyperParams, interp_errors_spline=None, interp_sim_errors_spline=None, loc='upper left', bbox_to_anchor=(0.59, 0.02)):
    plt.figure(figsize=(9.5, 5))
    plt.semilogy(times, sim_errors, linewidth=2, color='#37c837', label = r'$\varepsilon_\text{rel}$')
    if interp_errors_spline is not None:
        times_spline = times[:len(interp_errors_spline)]
        plt.semilogy(times_spline, interp_errors_spline, color='#F79A25', linewidth=2, label = r'$\varepsilon_\text{spline}$')
    if interp_sim_errors_spline is not None:
        plt.semilogy(times_spline, interp_sim_errors_spline, color='#F79A25', linestyle='--',linewidth=2, label = r'$\varepsilon_\text{spline, sim}$', alpha=0.4)
    plt.semilogy(times, interp_errors_gpr, linewidth=2, color='#065895',label = r'$\varepsilon_\text{GPR}$')
    plt.semilogy(times, interp_sim_errors_gpr, linestyle='--', color='#065895', linewidth=2, label = r'$\varepsilon_\text{GPR, sim}$', alpha=0.5)
    plt.xlabel('$t$')#, fontsize=18)
    #plt.ylabel('Relative error')
    #plt.legend(
    #    loc=loc,
    #    #bbox_to_anchor=bbox_to_anchor,
    #    ncol=3,
    #    frameon=True,
    #    columnspacing=1.5,
    #    handletextpad=0.5,
    #    #fontsize=15
    #)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.5))
    plt.grid(True, which='both', linestyle='--', alpha=0.2)
    plt.tight_layout()
    if interp_sim_errors_spline is None:
        plt.savefig(HyperParams.net_dir+'interp_errors_extrapolation'+HyperParams.net_run +'.pdf', bbox_inches='tight')
    else:
        plt.savefig(HyperParams.net_dir+'interp_errors'+HyperParams.net_run +'.pdf', bbox_inches='tight')#, dpi=500)
    plt.show()


# ----- Functions to plot the latent trajectories
def plot_latent_time(HyperParams, SAMPLE, latents, params, param_sample):
    """
    Function that plots the evolution of the latent state corresponding to the SAMPLE-th parameter.
    Here SAMPLE refers to the parameter index and param_sample refers to the number of parameters in the dataset.
    """

    plt.figure()
    sequence_length = latents.shape[0] // param_sample # basically the length of the time integration
    start = SAMPLE * sequence_length # SAMPLE refers to the SAMPLE-th parameter in the sequence of param_sample parameters
    end = start + sequence_length # end of the integration

    for i in range(HyperParams.bottleneck_dim):
        stn_evolution = latents[start:end, i]
        plt.plot(np.array(np.arange(sequence_length))*HyperParams.dt+ params[...,-1][0][0].item(), stn_evolution.detach().numpy(), label=rf'$s_{i+1}$')

    plt.xlabel('$t$') # 18 added for Coanda
    plt.legend(loc='upper left')
    #plt.title('Latent state evolution for $\mu = $'+ str(np.around(params[SAMPLE][0][:-1].detach().numpy(), 2)))
    plt.grid(True, which="both", ls="--", color='gray', alpha=0.1)  
    plt.tight_layout()
    plt.savefig(HyperParams.net_dir+'latent_evolution_'+HyperParams.net_run+str(SAMPLE)+'.pdf', bbox_inches='tight')#, dpi=500)
    plt.show()


def plot_latent_component(HyperParams, component, latents, params, param_sample):
    """
    This function plots the evolution of latent states over time and saves the plot as a .png file.

    Parameters:
    latents (np.ndarray): The latent states.
    params (list): The parameters.
    HyperParams (object): The hyperparameters.
    param_sample (int): The number of simulations.

    Returns:
    None
    """

    plt.figure()
    sequence_length = latents.shape[0] // param_sample # basically the length of the time integration
    
    for SAMPLE in range(params.shape[0]):
        start = SAMPLE * sequence_length # SAMPLE probably refers to the SAMPLE-th parameter in the sequence
        end = start + sequence_length # end of the integration

        stn_evolution = latents[start:end, component]
        plt.plot(np.array(np.arange(sequence_length))*HyperParams.dt+ params[...,-1][0][0].item(), stn_evolution.detach().numpy())

    plt.xlabel('$t$')
    plt.ylabel(f'$s_{component+1}(t)$')
    #plt.title(f'Latent state evolution for component {component}')
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig(HyperParams.net_dir+f'latent_component_{component}'+HyperParams.net_run+str(SAMPLE)+'.pdf', bbox_inches='tight')#, dpi=500)
    plt.show()


def plot_latent_by_mu_group(HyperParams, component, latents, params, param_sample, mu_index=0, single_mu_index=None):
    """
    Plots latent trajectories grouped by one mu component.

    Parameters:
    - HyperParams: Contains dt, net_dir, net_run
    - component: latent dimension index
    - latents: shape [n_samples * sequence_length, latent_dim]
    - params: shape [n_samples, sequence_length, parameter_dim]
    - param_sample: total number of parameter samples
    - mu_index: 0 or 1 (mu1 or mu2)
    - single_mu_index: int or None — index of the mu value to emphasize
    """
    plt.figure(figsize=(10, 6))
    sequence_length = latents.shape[0] // param_sample
    time = np.arange(sequence_length) * HyperParams.dt + params[...,-1][0][0].item()

    all_mu_vals = sorted(set(params[:, 0, mu_index].cpu().numpy()))

    if single_mu_index is not None:
        if single_mu_index >= len(all_mu_vals):
            raise IndexError(f"single_mu_index {single_mu_index} out of range for mu{mu_index+1}")

        target_val = all_mu_vals[single_mu_index]
        varying_mu_dim = 1 - mu_index
        selected = []

        for idx in range(params.shape[0]):
            mu_val = params[idx, 0, mu_index].item()
            other_val = params[idx, 0, varying_mu_dim].item()

            start = idx * sequence_length
            end = start + sequence_length
            stn_evolution = latents[start:end, component]

            if mu_val == target_val:
                selected.append((idx, other_val))
            else:
                plt.plot(
                    time,
                    stn_evolution.detach().numpy(),
                    color='gray',
                    alpha=0.15,
                    linewidth=0.8
                )

        selected.sort(key=lambda x: x[1])
        colors = cm.get_cmap('tab10', len(selected))

        for i, (idx, other_val) in enumerate(selected):
            start = idx * sequence_length
            end = start + sequence_length
            stn_evolution = latents[start:end, component]
            plt.plot(
                time,
                stn_evolution.detach().numpy(),
                color=colors(i),
                linewidth=2.0,
                label=f"$\\mu_{{{1 - mu_index + 1}}} = {other_val:.2f}$"
            )

        plt.legend(loc='upper right') # lower left
        #plt.title(f"Latent state evolution for $\\mu_{{{mu_index+1}}} = {target_val:.2f}$")

    else:
        grouped_indices = defaultdict(list)
        for idx in range(params.shape[0]):
            mu_val = params[idx, 0, mu_index].item()
            grouped_indices[mu_val].append(idx)

        unique_mu_vals = sorted(grouped_indices.keys())
        colors = cm.get_cmap('tab10', len(unique_mu_vals))

        for group_idx, (mu_val, sample_indices) in enumerate(grouped_indices.items()):
            color = colors(group_idx)
            for idx in sample_indices:
                start = idx * sequence_length
                end = start + sequence_length
                stn_evolution = latents[start:end, component]
                plt.plot(
                    time,
                    stn_evolution.detach().numpy(),
                    color=color,
                    label=f"$\\mu_{{{2-mu_index}}}={mu_val:.1f}$" if idx == sample_indices[0] else None
                )

        #plt.legend(loc='upper left', fontsize=18, ncol=3, frameon=True, columnspacing=1.5, handletextpad=0.5,)
        plt.legend(loc='upper left')
    plt.xlabel('$t$')
    plt.ylabel(f'$s_{{{component+1}}}(t)$')
    plt.grid(True, which="both", ls="--", color='gray', alpha=0.1)  # light solid grid
    suffix = f"groupby_mu{mu_index+1}" if single_mu_index is None else f"fixed_mu{mu_index+1}_{single_mu_index}"
    plt.tick_params(axis='both', labelsize=15)
    plt.tight_layout()
    folder_name = HyperParams.net_dir+"latent_plots/"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    plt.savefig(folder_name + f'latent_component_{component}_{suffix}' + HyperParams.net_run + '.pdf')#, dpi=500)
    plt.show()


def plot_interpolated_latent(HyperParams, component, interpolated_latents_gpr, interpolated_latents_spline, real_latents, mu_index, params, plot_zoom=True, extrapolation='True', loc='upper right'):
    sequence_length = real_latents.shape[1]
    time = np.arange(sequence_length) * HyperParams.dt + params[...,-1][0][0].item()

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(time, interpolated_latents_gpr[component,:], linewidth=2.0, label = fr'$s_{{{component+1}}}^{{\text{{GPR}}}}$', color='#065895')
    if interpolated_latents_spline is not None:
        ax.plot(time, interpolated_latents_spline[component,:], linewidth=2.0, label = fr'$s_{{{component+1}}}^{{\text{{spline}}}}$', color='#F79A25')

    for i in range(real_latents.shape[2]):
        real_component = real_latents[component, :, i]
        if i == mu_index:
            ax.plot(time, real_component, linewidth=2.0, label = fr'$s_{{{component+1}}}$', color='#37c837')
        else:
            ax.plot(time, real_component, color='gray', alpha=0.15, linewidth=0.8)
    
    ax.set_xlabel('$t$')
    ax.set_ylabel(f'$s_{{{component+1}}}(t)$')
    ax.grid(True, which="both", ls="--", color='gray', alpha=0.05)
    ax.legend(loc=loc)

    if plot_zoom:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
    # --- Inset zoom box ---
        axins = inset_axes(ax, width="40%", height="40%", loc='lower left')
        axins.plot(time, interpolated_latents_gpr[component,:], linewidth=2.0, color=ax.lines[0].get_color())
        if interpolated_latents_spline is not None:
            axins.plot(time, interpolated_latents_spline[component,:], linewidth=2.0, color=ax.lines[1].get_color())

        color_starting_idx = 1 if interpolated_latents_spline is None else 2
        for i in range(real_latents.shape[2]):
            real_component = real_latents[component, :, i]
            if i == mu_index:
                axins.plot(time, real_component, linewidth=2.0, color=ax.lines[color_starting_idx+mu_index].get_color())
            else:
                axins.plot(time, real_component, color='gray', alpha=0.1, linewidth=0.8)

        # Set zoom limits
        time1, time2 = 142, 152
        window_size = 0
        x1, x2 = time[time1], time[time2]
        y1, y2 = np.min(
            interpolated_latents_gpr[component, time1-window_size:time2+window_size+1]), np.max(interpolated_latents_gpr[component, time1-window_size:time2+window_size]
        )
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)
        axins.tick_params(labelleft=False, labelbottom=False)
        mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    plt.tight_layout()
    folder_name = HyperParams.net_dir + "interp_plots/" if not extrapolation else HyperParams.net_dir+"interp_plots_extrapol/"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    plt.savefig(folder_name + f'interpolated_latent_{component}' + HyperParams.net_run + '.pdf')
    plt.show()


def animate_latent_evolution(latent_state, params, component, n_sim, n_times, HyperParams, three_d=True):
    """
    Animate the evolution of a specific latent component over time as a scatterplot in (mu1, mu2) space.

    If `three_d` is True, use 3D plotting with z-axis = latent value.
    If False, use color-coded 2D plot.

    Arguments:
        latent_state: Tensor of shape [n_sim * n_times, n_components]
        params: Tensor of shape [n_sim, _, param_dim] (we use only params[:, 0, :])
        component: which latent component to visualize
        n_sim: number of parameter samples
        n_times: time steps per simulation
        HyperParams: object with .dt, net_dir, net_run, etc.
        three_d: whether to use 3D plotting
    """
    three_dim_flag = 3 if three_d else 2
    # Extract (mu1, mu2)
    param_vals = params[:, 0, :]  # shape: [n_sim, param_dim]
    mu1_vals = param_vals[:, 0]
    mu2_vals = param_vals[:, 1]

    # Reshape latent_state to [n_sim, n_times, n_components]
    latent_state = latent_state.reshape(n_sim, n_times, -1)
    latent_comp = latent_state[:, :, component]  # [n_sim, n_times]

    # Common normalization for consistent animation
    norm = Normalize(vmin=latent_comp.min(), vmax=latent_comp.max())

    if three_d:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim(mu1_vals.min(), mu1_vals.max())
        ax.set_ylim(mu2_vals.min(), mu2_vals.max())
        ax.set_zlim(latent_comp.min(), latent_comp.max())
        sc = ax.scatter(mu1_vals, mu2_vals, latent_comp[:, 0], c=latent_comp[:, 0], cmap='viridis',
                        norm=norm, s=20, alpha=0.8)
        ax.set_xlabel('$\mu_1$', labelpad=16)
        ax.set_ylabel('$\mu_2$', labelpad=16)
        #ax.set_zlabel(f'Latent component {component}')
        #ax.set_title(f'Latent component {component} over time (3D)')
        ax.zaxis.set_tick_params(labelsize=15, pad=13)
        ax.xaxis.set_tick_params(labelsize=15, pad=13)
        ax.yaxis.set_tick_params(labelsize=15, pad=13)

        def update(frame):
            for coll in ax.collections:
                coll.remove()
            ax.scatter(mu1_vals, mu2_vals, latent_comp[:, frame],
                       c=latent_comp[:, frame], cmap='viridis',
                       norm=norm, s=20, alpha=0.8)
            ax.set_title(rf'$s_{{{component}}}(t = {(frame+1) * HyperParams.dt:.2f})$')
            return ax,

    else:
        fig, ax = plt.subplots()
        ax.set_xlim(mu1_vals.min(), mu1_vals.max())
        ax.set_ylim(mu2_vals.min(), mu2_vals.max())
        sc = ax.scatter(mu1_vals, mu2_vals, c=latent_comp[:, 0], cmap='viridis', norm=norm, s=25)
        plt.colorbar(sc, ax=ax)
        plt.xlabel('$\mu_1$')
        plt.ylabel('$\mu_2$')
        plt.title(f'Latent component {component} over time')

        def update(frame):
            sc.set_array(latent_comp[:, frame])
            ax.set_title(f'Latent component {component} at t = {frame * HyperParams.dt:.2f}')
            return sc,

    ani = animation.FuncAnimation(fig, update, frames=n_times, interval=100, blit=False)

    # Save and return inline animation
    ani.save(HyperParams.net_dir + f'{three_dim_flag}d_animation{HyperParams.net_run}.gif', writer='pillow', fps=10)
    plt.close(fig)
    return HTML(ani.to_jshtml())


def plot_latent_3d_by_mu(HyperParams, component, latents, params, param_sample, mu_varying=0, elev=40, azim=45):
    """
    3D plot of latent trajectories grouped by one mu component.
    
    Parameters:
    - mu_varying: int, 0 or 1. The mu dimension used for z-axis (varying)
                  The other mu dimension is used to slice (call function with single_mu_index)
    """
    sequence_length = latents.shape[0] // param_sample
    time = np.arange(sequence_length) * HyperParams.dt + params[...,-1][0][0].item()

    all_mu_vals = sorted(set(params[:, 0, mu_varying].cpu().numpy()))
    mu_slice_dim = 1 - mu_varying  # The mu we slice over

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # We will plot each slice (fixed mu_slice_dim value) on its own y-level,
    # and z axis will correspond to mu_varying values.

    # Group indices by mu_slice_dim value (slices)
    grouped_by_slice = defaultdict(list)
    for idx in range(params.shape[0]):
        mu_slice_val = params[idx, 0, mu_slice_dim].item()
        grouped_by_slice[mu_slice_val].append(idx)

    # Sorted slice values
    slice_vals = sorted(grouped_by_slice.keys())
    colors = cm.get_cmap('tab10', len(slice_vals))

    for slice_i, slice_val in enumerate(slice_vals):
        # For this slice (fixed mu_slice_dim), plot lines over time (x-axis)
        sample_indices = grouped_by_slice[slice_val]
        
        # For each sample, plot latent trajectory as a line with z-values corresponding to mu_varying for that sample
        for idx in sample_indices:
            mu_vary_val = params[idx, 0, mu_varying].item()
            start = idx * sequence_length
            end = start + sequence_length
            stn_evolution = latents[start:end, component].detach().numpy()

            # x = time, y = latent value, z = mu_varying value for that sample
            xs = time
            ys = stn_evolution
            zs = np.full_like(xs, mu_vary_val)

            # Color by slice_val grouping
            ax.plot(xs, ys, zs, color=colors(slice_i), alpha=0.9, label=rf"$\mu_{mu_slice_dim+1}$=${slice_val:.1f}$" if idx == sample_indices[0] else None)

    ax.set_xlabel('$t$', labelpad=18)
    ax.set_ylabel(f'$s_{{{component+1}}}(t)$', labelpad=17)
    ax.set_zlabel(f'$\\mu_{{{mu_varying + 1}}}$', labelpad=15)

    ax.legend(loc='upper center', ncol=3,
        frameon=True,
        columnspacing=1.5,
        handletextpad=0.5,
        bbox_to_anchor=(0.5, 0.95),
        )
    ax.grid(True, which="both", ls="--", color='gray', alpha=0.1)
    ax.zaxis.set_tick_params(pad=8)
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=3))
    ax.xaxis.set_major_locator(plt.MaxNLocator(nbins=5))
    ax.zaxis.set_major_locator(plt.MaxNLocator(nbins=4))
    #plt.title(f'3D Latent Trajectories: varying $\\mu_{{{mu_varying + 1}}}$ on z-axis, slices by $\\mu_{{{mu_slice_dim + 1}}}$', fontsize=16)
   

    folder_name = HyperParams.net_dir + "latent_plots/"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    
    ax.view_init(elev=elev, azim=azim)  
    ax.set_box_aspect(aspect=None, zoom=0.85)
    plt.tight_layout()

    plt.savefig(folder_name + f'latent_component_{component}_3d_mu{mu_varying+1}_by_mu{mu_slice_dim+1}' + HyperParams.net_run + '.pdf')
    plt.show()


def plot_phase_space(HyperParams, latents, mu_indices, n_params, elev=20, azim=-70):
    """
    Plots 3D phase-space trajectories for latent states when latent dimension is 3.

    Parameters:
    - HyperParams: object containing net_dir, net_run, etc.
    - latents: torch.Tensor of shape (n_times*n_simulations, latent_dim)
    - mu_indices: list of parameter/sample indices to plot
    - params: optional, not used here
    """
    latent_dim = latents.shape[1]
    
    if latent_dim != 3:
        raise ValueError(f"Latent dimension must be 3 for phase-space plotting. Got {latent_dim}.")

    # Figure setup
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    for idx in mu_indices:
        # Compute sequence length
        sequence_length = latents.shape[0] // n_params
        start = idx * sequence_length
        end = start + sequence_length
        
        trajectory = latents[start:end, :].detach().cpu().numpy()
        
        # Plot trajectory in 3D
        ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], linewidth=2.0, label=f"Sample {idx}")
    
    ax.set_xlabel("$s_1$")
    ax.set_ylabel("$s_2$")
    ax.set_zlabel("$s_3$")
    #ax.legend(loc='upper left')
    ax.grid(True, which="both", ls="--", color='gray', alpha=0.1)
    ax.view_init(elev=elev, azim=azim)  
    plt.tight_layout()
    folder_name = HyperParams.net_dir + "phase_space_plots/"
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    
    plt.savefig(folder_name + f'phase_space_3d_{HyperParams.net_run}.pdf')
    plt.show()