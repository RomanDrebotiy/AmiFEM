from finite_elements.base_element import Triangle, Operator, Point
from finite_elements.polynomial_element import PolynomialElement


class LagrangeLinearElement(PolynomialElement):
    def __init__(self, domain: Triangle):
        super().__init__(domain, 1)

    def _adjoint_basis_per_point(self) -> list[Operator]:
        point_eval = lambda u: u
        return [point_eval] * 3

    def dof_points(self) -> list[Point]:
        return [
            (0, 0),
            (1, 0),
            (0, 1)
        ]

    def edge_orientation_dof_change(self, gdofs: list[int]) -> list[int]:
        return []
