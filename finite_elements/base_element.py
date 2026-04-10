from typing import Callable, TypeAlias

import numpy as np

ScalarFunc: TypeAlias = Callable[[float, float], float]
Operator: TypeAlias = Callable[[ScalarFunc], ScalarFunc]
Point: TypeAlias = tuple[float, float]
Triangle: TypeAlias = tuple[Point, Point, Point]

eps = 1e-3

class BaseElement:
    def __init__(self, domain: Triangle):
        self.domain = domain
        # counterclockwise orientation and origin is a starting point
        self.__vertices = [(0, 0), (1, 0), (0, 1)]
        self.basis = self.__generate_basis()
        # next three call should be strictly in this order
        self.node_dof_idx_per_node = self.__get_node_dof_idx_per_node()
        self.edge_dof_idx_per_edge = self.__get_edge_dof_idx_per_edge()
        self.internal_dof_idx = self.__get_internal_dof_idx()
        self.ordered_dofs = (
            np.array(self.node_dof_idx_per_node).flatten().tolist()
            + np.array(self.edge_dof_idx_per_edge).flatten().tolist()
            + self.internal_dof_idx
        )

    def _canonical_basis(self) -> list[ScalarFunc]:
        """
        Canonical basis on real triangle (not reference) !
        """
        raise NotImplementedError()

    def _adjoint_basis_per_point(self) -> list[Operator]:
        """
        Operators defining linear functionals as pointwise values for points defined in def_points
        """
        raise NotImplementedError()

    def dof_points(self) -> list[Point]:
        """
        Set on reference triangle !
        """
        raise NotImplementedError()

    def __map_reference_to_real_vertex(self, r: Point) -> Point:
        t = self.domain
        return (
            t[0][0] + r[0] * (t[1][0] - t[0][0]) + r[1] * (t[2][0] - t[0][0]),
            t[0][1] + r[0] * (t[1][1] - t[0][1]) + r[1] * (t[2][1] - t[0][1])
        )

    def __generate_basis(self) -> list[ScalarFunc]:
        cb: list[ScalarFunc] = self._canonical_basis()
        ab: list[Operator] = self._adjoint_basis_per_point()
        dp: list[Point] = self.dof_points()
        if not (len(cb) == len(ab) == len(dp)):
            raise Exception("Adjoint and canonical basis count should match")
        n = len(cb)
        def matr(i: int, j: int) -> float:
            real_pt = self.__map_reference_to_real_vertex(dp[j])
            # calculating functionals on real point to avoid using Jacobians later in certain places
            res = ab[j](cb[i])(*real_pt)
            return res
        matr_vect = np.vectorize(matr, otypes=[float])
        adjoint_basis_matr = np.fromfunction(matr_vect, (n, n), dtype=int)
        basis_mapping = np.linalg.inv(adjoint_basis_matr)
        def basis(i: int, x: float, y: float) -> float:
            return sum(basis_mapping[i, j] * cb[j](x,y) for j in range(n))
        return [lambda x, y, i=i: basis(i, x, y) for i in range(n)]

    def __check_consistency(self, res: list[list[int]]) -> None:
        if not(np.array(res[0]).shape == np.array(res[1]).shape == np.array(res[2]).shape):
            raise Exception("dof counts per geometry element are not consistent")

    def __get_node_dof_idx_per_node(self) -> list[list[int]]:
        dp = self.dof_points()
        res = [
            [i for i in range(len(dp)) if np.allclose(dp[i], self.__vertices[j], atol=eps)]
            for j in range(3)
        ]
        self.__check_consistency(res)
        return res

    def __point_line_distance(self, p, a, b):
        p, a, b = map(np.asarray, (p, a, b))
        return abs(np.cross(b - a, p - a)) / np.linalg.norm(b - a)

    def __on_edge(self, p: Point, begin: Point, end: Point) -> bool:
        return self.__point_line_distance(p, begin, end) < eps

    def __get_edge_dof_idx_per_edge(self) -> list[list[int]]:
        dp = self.dof_points()
        nodal_dofs = np.array(self.node_dof_idx_per_node).flatten()
        enriched_vertices = self.__vertices + [self.__vertices[0]]
        res = []
        for i in range(3):
            begin, end = enriched_vertices[i:i+2]
            res.append([
                j for j in range(len(dp))
                if not j in nodal_dofs
                   and self.__on_edge(dp[j], begin, end)
            ])
        self.__check_consistency(res)
        return res

    def __get_internal_dof_idx(self) -> list[int]:
        dp = self.dof_points()
        nodal_dofs = np.array(self.node_dof_idx_per_node).flatten()
        edge_dofs = np.array(self.edge_dof_idx_per_edge).flatten()
        return [i for i in range(len(dp)) if i not in nodal_dofs and i not in edge_dofs]
