# Copyright (C) 2015-2020 by the RBniCS authors
#
# This file is part of RBniCS.
#
# SPDX-License-Identifier: LGPL-3.0-or-later

from dolfin import *
from mshr import *
from rbnics.backends.dolfin.wrapping import counterclockwise
from rbnics.shape_parametrization.utils.symbolic import VerticesMappingIO

# Define domain
domain = Rectangle(Point(0., 0.), Point(1., 1.))
mesh = generate_mesh(domain, 30)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, 2, mesh.domains())

class LeftOuter(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0]) < DOLFIN_EPS


class RightOuter(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[0] - 1.) < DOLFIN_EPS


class BottomOuter(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1]) < DOLFIN_EPS


class TopOuter(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and abs(x[1] - 1.) < DOLFIN_EPS


boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
boundaries.set_all(0)
bottomOuter = BottomOuter()
bottomOuter.mark(boundaries, 1)
leftOuter = LeftOuter()
leftOuter.mark(boundaries, 1)
topOuter = TopOuter()
topOuter.mark(boundaries, 1)
rightOuter = RightOuter()
rightOuter.mark(boundaries, 1)

# Save
File("square_adv.xml") << mesh
File("square_adv_physical_region.xml") << subdomains
File("square_adv_facet_region.xml") << boundaries
XDMFFile("square_adv.xdmf").write(mesh)
XDMFFile("square_adv_physical_region.xdmf").write(subdomains)
XDMFFile("square_adv_facet_region.xdmf").write(boundaries)
