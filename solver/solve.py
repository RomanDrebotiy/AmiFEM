from typing import Callable

import numpy as np

from finite_elements.base_element import BaseElement, Triangle
from form_language.integrals import Integral
from solver.mesh import Mesh


class Solver:
    def __init__(self, a: Integral, L: Integral, mesh: Mesh, fe: Callable[[Triangle], BaseElement], dirichlet_zero_marker: Callable[[float, float, bool], bool]):
        self.a = a
        self.L = L
        self.tri = mesh.get_triangulation()
        self.v, self.e, self.t, self.v_markers = self.tri['vertices'], self.tri['edges'], self.tri['triangles'], self.tri['vertex_markers']
        self.fe = fe
        sample_fe = fe(((0, 0), (1, 0), (0, 1)))
        self.dofs_per_vertex = len(sample_fe.node_dof_idx_per_node[0])
        self.dofs_per_edge = len(sample_fe.edge_dof_idx_per_edge[0])
        self.internal_dofs_per_triangle = len(sample_fe.internal_dof_idx)
        self.total_dofs = (
            self.dofs_per_vertex * len(self.v)
            + self.dofs_per_edge * len(self.e)
            + self.internal_dofs_per_triangle * len(self.t)
       )
        self.lhs = np.zeros((self.total_dofs, self.total_dofs))
        self.rhs = np.zeros(self.total_dofs)
        self.global_dofs_v = self.__generate_dofs_v(0)
        self.global_dofs_e = self.__generate_dofs_e(self.global_dofs_v[-1][-1] + 1)
        self.global_dofs_t = self.__generate_dofs_t(self.global_dofs_e[-1][-1] + 1)
        self.reversed_edge_idx = {f"{edge.tolist()}": index for index, edge in enumerate(self.e)}
        self.dirichlet_zero_marker = dirichlet_zero_marker
        self.actual_global_dofs = list(range(self.total_dofs))

    def __generate_dofs_per_geometry(self, geometry_lst: list, per_geometry_cnt: int, start: int) -> list[list[int]]:
        res = []
        for j in range(len(geometry_lst)):
            res.append([])
            for i in range(per_geometry_cnt):
                res[-1].append(start + j * per_geometry_cnt + i)
        return res

    def __generate_dofs_v(self, start: int) -> list[list[int]]:
        return self.__generate_dofs_per_geometry(self.v, self.dofs_per_vertex, start)

    def __generate_dofs_e(self, start: int) -> list[list[int]]:
        return self.__generate_dofs_per_geometry(self.e, self.dofs_per_edge, start)

    def __generate_dofs_t(self, start: int) -> list[list[int]]:
        return self.__generate_dofs_per_geometry(self.t, self.internal_dofs_per_triangle, start)

    def __assemble_aux(self):
        for t_idx, triangle in enumerate(self.t):
            gdofs = []
            for i in range(3):
                gdofs += self.global_dofs_v[triangle[i]]
            for lstart, lend in [
                [0, 1],
                [1, 2],
                [2, 0]
            ]:
                gstart, gend = triangle[lstart], triangle[lend]
                edge_global_idx = self.reversed_edge_idx.get(f"{[gstart, gend]}")
                reverse = False
                if edge_global_idx is None:
                    edge_global_idx = self.reversed_edge_idx[f"{[gend, gstart]}"]
                    reverse = True
                global_edge_dofs = self.global_dofs_e[edge_global_idx]
                if reverse:
                    global_edge_dofs.reverse()
                gdofs += global_edge_dofs
            gdofs += self.global_dofs_t[t_idx]

            v1 = self.v[triangle[0]]
            v2 = self.v[triangle[1]]
            v3 = self.v[triangle[2]]
            fe = self.fe((v1, v2, v3))
            ldofs = fe.ordered_dofs

            if not (len(gdofs) == len(ldofs) == len(fe.basis)):
                raise Exception("Global, local d.o.f. and basis counts not matched")

            lg_pairs = list(zip(ldofs, gdofs))

            for il, ig in lg_pairs:
                phi_i = lambda x, y: np.array([fe.basis[il](x, y)])
                for jl, jg in lg_pairs:
                    phi_j = lambda x, y: np.array([fe.basis[jl](x, y)])
                    self.lhs[ig, jg] = self.a.eval(
                        np.array(v1),
                        np.array(v2),
                        np.array(v3),
                        phi_j,
                        phi_i
                    )
                self.rhs[ig] = self.L.eval(
                    np.array(v1),
                    np.array(v2),
                    np.array(v3),
                    None,
                    phi_i
                )

    def __apply_dirichlet(self):
        """
        This method is limited to the case when we have only 1 dof per vertex for now
        """
        dofs = []
        for i in range(len(self.v)):
            vertex = self.v[i]
            on_boundary = self.v_markers[i][0] == 1
            if self.dirichlet_zero_marker(vertex[0], vertex[1], on_boundary):
                dofs += self.global_dofs_v[i]

        for i in range(len(self.e)):
            edge = self.e[i]
            v1 = self.v[edge[0]]
            v2 = self.v[edge[1]]
            on_boundary1 = self.v_markers[edge[0]][0] == 1
            on_boundary2 = self.v_markers[edge[1]][0] == 1
            if self.dirichlet_zero_marker(v1[0], v1[1], on_boundary1) and self.dirichlet_zero_marker(v2[0], v2[1], on_boundary2):
                dofs += self.global_dofs_e[i]

        self.lhs = np.delete(self.lhs, dofs, axis=0)
        self.lhs = np.delete(self.lhs, dofs, axis=1)
        self.rhs = np.delete(self.rhs, dofs)

        self.actual_global_dofs = np.delete(np.array(self.actual_global_dofs), dofs, axis=0)

    def __assemble(self):
        self.__assemble_aux()
        self.__apply_dirichlet()

    def __solve(self) -> np.ndarray:
        return np.linalg.solve(self.lhs, self.rhs)

    def assemble_and_solve(self) -> tuple[np.ndarray, dict, list[list[int]], list[list[int]], list[list[int]], list[int], int]:
        self.__assemble()
        sol = self.__solve()
        return sol, self.tri, self.global_dofs_v, self.global_dofs_e, self.global_dofs_t, self.actual_global_dofs, self.total_dofs
