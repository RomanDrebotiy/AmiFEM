import numpy as np

from finite_elements.hermite_cubic_element import HermiteCubicElement
from form_language.expressions import TrialFunction, TestFunction, Function, grad
from form_language.integrals import integrate, Measure
from solver.mesh import Mesh
from solver.plot_solution import Plotter
from solver.solve import Solver

u = TrialFunction()
v = TestFunction()

mu = 1
sigma = 1
beta = Function(lambda x, y: np.array([5, 5]))
f = Function(lambda x, y: np.array([1]))

alpha = 1
u_out = 0.01

a = (
    integrate(mu * grad(u) * grad(v) + v * beta * grad(u) + sigma * u * v, Measure.DX)
    + integrate(alpha * u * v, Measure.DS)
)

L = integrate(f * v, Measure.DX) + integrate(alpha * u_out * v, Measure.DS)

mesh = Mesh([(0, 0), (1, 0), (1, 1), (0, 1)], area=0.001)

dirichlet_zero_marker = lambda x, y, on_bnd: on_bnd and (x<0.001 or y<0.001 or y>0.999)

solver = Solver(a, L, mesh, HermiteCubicElement, dirichlet_zero_marker)

sol, tri, vdofs, _, _, actual_dofs, total_dofs = solver.assemble_and_solve()

Plotter(sol, tri, vdofs, actual_dofs, total_dofs).plot()
