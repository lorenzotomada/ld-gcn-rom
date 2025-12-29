# Copyright (C) 2015-2020 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from rbnics import *
from problems import *
from reduction_methods import *
import numpy as np
from rbnics.backends import export

@PullBackFormsToReferenceDomain()
@AffineShapeParametrization("data/hole_vertices_mapping.vmp")

class AdvectionDiffusion(ParabolicCoerciveProblem):

    # Default initialization of members
    def __init__(self, V, **kwargs):
        # Call the standard initialization
        ParabolicCoerciveProblem.__init__(self, V, **kwargs)
        # ... and also store FEniCS data structures for assembly
        assert "subdomains" in kwargs
        assert "boundaries" in kwargs
        self.subdomains, self.boundaries = kwargs["subdomains"], kwargs["boundaries"]
        self.u = TrialFunction(V)
        self.v = TestFunction(V)
        self.dx = Measure("dx")(subdomain_data=self.subdomains)
        self.ds = Measure("ds")(subdomain_data=self.boundaries)
        # Store advection and forcing expressions
        self.beta = Constant((1.0, 1.0))
        self.inlet = Expression(("(x[0]-1)*(x[0]-1) + (x[1]-1)*(x[1]-1)"), degree=2)
        self.ic = Expression(("(x[0]-1)*(x[0]-1) + (x[1]-1)*(x[1]-1)"), degree=2)
        self.f = Constant(1.0)

    # Return custom problem name
    def name(self):
        return "AdvectionDiffusion"

    # Return theta multiplicative terms of the affine expansion of the problem.
    def compute_theta(self, term):
        if term == "m":
            theta_m0 = 1.0
            return (theta_m0, )
        elif term == "a":
            theta_a0 = 0.1
            theta_a1 = 1.0 - self.t
            return (theta_a0, theta_a1)
        elif term == "f":
            theta_f0 = 0.0
            return (theta_f0, )
        elif term == "dirichlet_bc":
            theta_bc0 = 1.0
            return (theta_bc0, )
        else:
            raise ValueError("Invalid term for compute_theta().")

    # Return forms resulting from the discretization of the affine expansion of the problem operators.
    def assemble_operator(self, term):
        v = self.v
        dx = self.dx
        if term == "m":
            u = self.u
            m0 = u * v * dx
            return (m0, )
        elif term == "a":
            u = self.u
            beta = self.beta
            a0 = inner(grad(u), grad(v)) * dx
            a1 = inner(beta, grad(u)) * v * dx
            return (a0, a1)
        elif term == "f":
            f = self.f
            f0 = f * v * dx
            return (f0, )
        elif term == "dirichlet_bc":
            bc0 = [DirichletBC(self.V, self.inlet, self.boundaries, 1),]
            return (bc0, )
        elif term == "inner_product":
            u = self.u
            x0 = inner(grad(u), grad(v)) * dx
            return (x0, )
        elif term == "projection_inner_product":
            u = self.u
            x0 = u * v * dx
            return (x0, )
        else:
            raise ValueError("Invalid term for assemble_operator().")


# 1. Read the mesh for this problem
mesh = Mesh("data/hole.xml")
subdomains = MeshFunction("size_t", mesh, "data/hole_physical_region.xml")
boundaries = MeshFunction("size_t", mesh, "data/hole_facet_region.xml")

# 2. Create Finite Element space (Lagrange P1, two components)
V = FunctionSpace(mesh, "Lagrange", 1)

# 3. Allocate an object of the AdvectionDiffusion class
advection_diffusion_problem = AdvectionDiffusion(V, subdomains=subdomains, boundaries=boundaries)
mu_range = [(0.5, 0.5), (0.5, 0.5)]
advection_diffusion_problem.set_mu_range(mu_range)
advection_diffusion_problem.set_time_step_size(0.02)
advection_diffusion_problem.set_final_time(2.) # try with 2.

advection_diffusion_problem.init()

# 6. Perform an online solve
mu0_range = np.linspace(0.2, 0.5, 5)
mu1_range = np.linspace(0.2, 0.5, 5)

l = 0
for i, mu0 in enumerate(mu0_range):
    for j, mu1 in enumerate(mu1_range):
        online_mu = (mu0, mu1)
        print("Parameter mu = ", online_mu)
        advection_diffusion_problem.set_mu(online_mu)
        solution_over_time = advection_diffusion_problem.solve()
        for (k, solution) in enumerate(solution_over_time):
            advection_diffusion_problem.mesh_motion.move_mesh()
            export(solution, "AdvectionDiffusion", "moving_hole_advection", suffix = l)
            advection_diffusion_problem.mesh_motion.reset_reference()
            l += 1
