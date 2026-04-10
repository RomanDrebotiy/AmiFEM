import numpy as np
from finite_elements.lagrange_quadratic_element import LagrangeQuadraticElement
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

a = integrate(mu * grad(u) * grad(v) + v * beta * grad(u) + sigma * u * v, Measure.DX)
L = integrate(f * v, Measure.DX)

mesh = Mesh([(0, 0), (1, 0), (1, 1), (0, 1)], area=0.01)

solver = Solver(a, L, mesh, LagrangeQuadraticElement, lambda x, y, on_bnd: on_bnd)

sol, tri, vdofs, _, _, actual_dofs, total_dofs = solver.assemble_and_solve()

Plotter(sol, tri, vdofs, actual_dofs, total_dofs).plot()
