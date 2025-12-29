import fenics as fe
from tqdm import tqdm 
import matplotlib.pyplot as plt
import numpy as np
from mshr import *
from gaussian_process import eval_u_t_np
from itertools import product
import scipy.stats


# fe.PETScOptions.set("snes_linesearch_monitor", "")
fe.PETScOptions.set("snes_linesearch_type", "bt")


# Define the necessary parameters
N_SIM = 80 #16   # number of simulations
N_F = 2          # number of frequencies
mu = 0.0   
sigma = 3.0 #5.0
N_POINTS_P_AXIS = 80 # 100
TIME_STEP_LENGTH = 0.1 # 0.2
TOTAL_TIME = 2 # 10
N_TIME_STEPS = np.floor(TOTAL_TIME/TIME_STEP_LENGTH).astype(int)
KINEMATIC_VISCOSITY = fe.Constant(0.01)
time_vector = np.arange(0.0, TOTAL_TIME, TIME_STEP_LENGTH)


# Generate the gaussian BCs
np.random.seed(22)
truncate = False

if truncate:
    lower = -10
    upper = 10
    N_samples = N_SIM*N_F
    coeff = np.array(scipy.stats.truncnorm.rvs((lower-mu)/sigma,(upper-mu)/sigma,loc=mu,scale=sigma,size=N_samples))
    alphaM = coeff.reshape((N_SIM, N_F))
    np.save('alphaM.npy', alphaM)
else:
    alphaM = np.random.normal(mu, sigma, (N_SIM, N_F))
    np.save('alphaM.npy', alphaM)


# Create a matrix that stores, in each row, the evaluations of u_i(t_j) for the i-th simulation
u_t_matrix = np.zeros((N_SIM, N_TIME_STEPS))
for i in range(N_SIM):
    u_t_matrix[i, :] = eval_u_t_np(time_vector, alphaM[i, :], TOTAL_TIME)
np.save('u_t_matrix', u_t_matrix)


# Create XDMF files (one for all the simulations) -> current focus only on velocity
xdmffile_u = fe.XDMFFile('lid_cavity_u_nl.xdmf')
xdmffile_p = fe.XDMFFile('lid_cavity_p_nl.xdmf')
xdmffile_u.parameters["flush_output"] = True
xdmffile_u.parameters["functions_share_mesh"] = True
xdmffile_p.parameters["flush_output"] = True
xdmffile_p.parameters["functions_share_mesh"] = True


# Define an expression for the BC u_t
class u_t_Expression(fe.UserExpression):
    def __init__(self, t, alphav, **kwargs):
        super().__init__(**kwargs)
        self.t = t
        self.alphav = alphav
    def eval(self, value, x):
        value[0] = eval_u_t_np(self.t, self.alphav, TOTAL_TIME)
        value[1] = 0.0
    def value_shape(self):
        return (2,) 
    

class BottomVertex(fe.SubDomain):
    def inside(self, x, on_boundary):
        return (abs(x[0]) < fe.DOLFIN_EPS and abs(x[1]) < fe.DOLFIN_EPS)


def solve_problem(alphav, num_sim):
    plot = False
    if plot:
        # Plot the u_t used for the i-th simulation over 2 periods
        t_v = np.linspace(0, 2*TOTAL_TIME, 1000) 
        u_plot = eval_u_t_np(t_v, alphav, TOTAL_TIME)
        plt.figure(figsize=(10, 5))
        plt.plot(t_v, u_plot)
        plt.xlabel('Time')
        plt.ylabel('u(t)')
        plt.title('u(t) current simulation')
        plt.grid(True)
        plt.show()
    
    # mesh = fe.UnitSquareMesh(N_POINTS_P_AXIS, N_POINTS_P_AXIS, "crossed")
    domain = Rectangle(fe.Point(0., 0.), fe.Point(1., 1.))
    mesh = generate_mesh(domain, N_POINTS_P_AXIS)
   
    # Taylor-Hood Elements. 
    element_v = fe.VectorElement("Lagrange", mesh.ufl_cell(), 2)
    element_p = fe.FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    W = fe.FunctionSpace(mesh, fe.MixedElement(element_v, element_p))
    V = fe.FunctionSpace(mesh, element_v)
    Q = fe.FunctionSpace(mesh, element_p)

    # Define trial functions
    vq = fe.TestFunction(W)
    delta_up = fe.TrialFunction(W)
    (v, q) = fe.split(vq)

    up = fe.Function(W)
    (u, p) = fe.split(up)

    up_prev = fe.Function(W)
    (u_prev, _) = fe.split(up_prev)

    # Define boundary conditions
    g = u_t_Expression(0.0, alphav) 
    noslip = fe.DirichletBC(W.sub(0), (0, 0), "x[0] < DOLFIN_EPS || x[0] > 1.0 - DOLFIN_EPS || x[1] < DOLFIN_EPS")
    lid = fe.DirichletBC(W.sub(0), g, "x[1] > 1.0 - DOLFIN_EPS")
    pref = fe.DirichletBC(W.sub(1), 0, "x[0] < DOLFIN_EPS && x[1] < DOLFIN_EPS", "pointwise")

    bc = [noslip, lid, pref]

    # Stabilization if needed
    # h = fe.CellDiameter(mesh)
    # alpha_u = fe.Constant(1.)
    # delta_u = alpha_u*(h**2)
    # alpha_p = fe.Constant(1.)
    # delta_p = alpha_p*(h**2)
    # rho = fe.Constant(0.)
    # rho =   0 -> SUPG (Streamline updwind Petrov Galerkin)
    # rho =   1 -> GALS (Galerkin least squares)
    # rho = - 1 -> Douglas-Wang

    # Tentative velocity step
    F = fe.inner(u, v)/fe.Constant(TIME_STEP_LENGTH)*fe.dx \
        - fe.inner(u_prev, v)/fe.Constant(TIME_STEP_LENGTH)*fe.dx \
        + fe.inner(fe.grad(u) * u, v) * fe.dx \
        + KINEMATIC_VISCOSITY * fe.inner(fe.grad(u), fe.grad(v)) * fe.dx \
        - fe.div(v) * p * fe.dx \
        - q * fe.div(u) * fe.dx 
        # + fe.inner(- KINEMATIC_VISCOSITY*fe.div(fe.grad(u)) + fe.grad(p),
        #            -  rho*fe.delta_u*KINEMATIC_VISCOSITY*fe.div(fe.grad(v)) + delta_p*fe.grad(q))*fe.dx  # Stabilization term 

    J = fe.derivative(F, up, delta_up)

    snes_solver_parameters = {"nonlinear_solver": "snes",
                            "snes_solver": {"linear_solver": "mumps",
                                            "maximum_iterations": 20,
                                            "report": False,
                                            "error_on_nonconvergence": True}}

    (u, p) = up.split()
    u.rename("u", "")
    p.rename("p","")
    xdmffile_u.write(u, num_sim*TOTAL_TIME)
    xdmffile_p.write(p, num_sim*TOTAL_TIME)
    
    for i in tqdm(range(1, N_TIME_STEPS)):
        # Go back to "physical" time in order to correctly update the BC
        t = i*TIME_STEP_LENGTH
        g.t = t

        # KINEMATIC_VISCOSITY.assign(1.)
        problem = fe.NonlinearVariationalProblem(F, up, bc, J)
        solver  = fe.NonlinearVariationalSolver(problem)
        solver.parameters.update(snes_solver_parameters)
        solver.solve()

        # Store the solution in up_prev
        fe.assign(up_prev, up)

        # Save
        (u, p) = up.split()
        u.rename("u", "")
        p.rename("p","")
        xdmffile_u.write(u, t+num_sim*TOTAL_TIME)
        xdmffile_p.write(p, t+num_sim*TOTAL_TIME)
        

def main():
    for i in range(1, N_SIM+1):
        alphav = alphaM[i-1, :]
        solve_problem(alphav, i-1)


if __name__ == "__main__":
    main()
