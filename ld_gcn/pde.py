import numpy as np


def problem(argument):
    """
    Dispatcher for PDE problem configurations and parameter space definitions.

    This function acts as a central registry for various physical problems, defining 
    the problem name, the primary field variable, the parameter discretization 
    (mu_space), and the dimensionality of the domain and solution.

    Note on Library Compatibility:
    - **Problems 1-10:** These cases are legacy configurations inherited from the 
      GCA-ROM (Graph Convolutional Autoencoder - Reduced Order Model) framework. 
      They are not optimized for the current LD-GCN implementation.
    - **Problems 11 onwards:** These are the primary benchmarks natively supported 
      and utilized in this library, focusing on time-dependent advection and 
      complex fluid dynamics (e.g., Coanda effect, Lid-driven cavity).

    Args:
        argument (int): Case selector index (1 through 14).

    Returns:
        tuple: (problem_name, variable, mu_space, n_param, dim_pde, n_comp)
            - **problem_name** (str): Lowercase identifier for the PDE case.
            - **variable** (str): Field identifier (e.g., 'U', 'VX', 'VX_VY').
            - **mu_space** (list[np.ndarray]): Discretized parameter grids for each 
              dimension. If the problem is time-dependent, the last element is 
              usually the time vector.
            - **n_param** (int): Number of independent parameters (excluding time).
            - **dim_pde** (int): Spatial dimensionality of the PDE domain (e.g., 2 for 2D).
            - **n_comp** (int): Number of components in the solution field (e.g., 1 for scalar, 
              2 for velocity vectors).
    """

    match argument:
        case 1:
            problem_name = "poisson"
            variable = 'U'
            mu1 = np.linspace(0.01, 10., 10)
            mu2 = np.linspace(0.01, 10., 10)
            mu_space = [mu1, mu2]
            n_param = 2
        case 2:
            problem_name = "advection"
            variable = 'U'
            mu1 = np.linspace(0., 6., 10)
            mu2 = np.linspace(-1.0, 1.0, 10)
            mu_space = [mu1, mu2]
            n_param = 2
        case 3:
            problem_name = "graetz"
            variable = 'U'
            mu1 = np.linspace(1., 3., 10)
            mu2 = np.linspace(0.01, 0.1, 20)
            mu_space = [mu1, mu2]
            n_param = 2
        case 4:
            problem_name = "navier_stokes"
            variable = 'VX'
            mu1 = np.linspace(0.5, 2., 21)[::2]
            mu2 = np.linspace(2., 0.5, 151)[::5]
            mu_space = [mu1, mu2]
            n_param = 2
        case 5:
            problem_name = "navier_stokes"
            variable = 'VY'
            mu1 = np.linspace(0.5, 2., 21)[::2]
            mu2 = np.linspace(2., 0.5, 151)[::5]
            mu_space = [mu1, mu2]
            n_param = 2
        case 6:
            problem_name = "navier_stokes"
            variable = 'P'
            mu1 = np.linspace(0.5, 2., 21)[::2]
            mu2 = np.linspace(2., 0.5, 151)[::5]
            mu_space = [mu1, mu2]
            n_param = 2
        case 7:
            problem_name = "diffusion"
            variable = 'U'
            mu1 = np.linspace(0.2, 4., 20)
            mu2 = np.linspace(0., 1., 20)
            mu_space = [mu1, mu2]
            n_param = 2
        case 8:
            problem_name = "poiseuille"
            variable = 'U'
            mu1 = np.linspace(0.5, 10., 20)
            mu2 = np.linspace(0., 1., 50)
            mu_space = [mu1, mu2]
            n_param = 2
        case 9:
            problem_name = "elasticity"
            variable = 'U'
            mu1 = np.linspace(2., 20., 11)
            mu2 = np.linspace(2., 200., 11)
            mu_space = [mu1, mu2]
            n_param = 2
        case 10:
            problem_name = "stokes_u"
            variable = 'U'
            mu_range = [(0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (0.5, 1.5), (-np.pi/6, np.pi/6), (-10, 10)]
            mu_space = []
            n_pts = [2]*(len(mu_range)-1)+[11]
            for i in range(len(mu_range)):
                mu_space.append(np.linspace(mu_range[i][0], mu_range[i][1], n_pts[i]))
            n_param = 7
        case 11:
            problem_name = "moving_hole_advection"
            variable = 'U'
            mu1 = np.linspace(0.2, 0.5, 5)
            mu2 = np.linspace(0.2, 0.5, 5)
            step_size = 0.02
            times = np.arange(0.0, 2.0 + step_size, step_size)  
            mu_space = [mu1, mu2, times]
            n_param = 2 # n_param refers to just the fixed parameters and time-dependent signals, and not to time, which is supposed to be passed anyway
            dim_pde = 2
            n_comp = 1
        case 12:
            problem_name = "square_advection"
            variable = 'U'
            mu1 = np.linspace(-1., 1., 5)
            mu2 = np.linspace(-1., 1., 5)
            step_size = 0.02
            times = np.arange(0.0, 2.0 + step_size, step_size)  
            mu_space = [mu1, mu2, times]
            n_param = 2
            dim_pde = 2
            n_comp = 1
        case 13:
            problem_name = "lid_driven_cavity"
            variable = 'VX_VY'
            u_t_matrix = np.load('../dataset/lid_driven_cavity/u_t_matrix.npy') # shape (n_sims, n_times)
            mu1 = u_t_matrix
            n_times = u_t_matrix.shape[1]
            mu2 = np.linspace(0.0, 2.0, n_times, endpoint=False) # times
            mu_space = [mu1, mu2] 
            n_param = 1
            dim_pde = 2
            n_comp = 2 # horizontal and vertical velocity, not dealing with pressure in this case
        case 14:
            problem_name = "coanda"
            variable = "U1"
            mu_space = None # to be defined in the notebook
            n_param = 1
            dim_pde = 2
            n_comp = 2

    return problem_name, variable, mu_space, n_param, dim_pde, n_comp