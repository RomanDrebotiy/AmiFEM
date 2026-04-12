from finite_elements.base_element import Triangle, Operator, Point, eps
from finite_elements.polynomial_element import PolynomialElement


class HermiteCubicElement(PolynomialElement):
    def __init__(self, domain: Triangle):
        super().__init__(domain, 3)

    def _adjoint_basis_per_point(self) -> list[Operator]:
        point_eval = lambda u: u
        der_x_eval = lambda u: (lambda x, y: (u(x + eps, y) - u(x, y)) / eps)
        der_y_eval = lambda u: (lambda x, y: (u(x, y + eps) - u(x, y)) / eps)
        return [
            # vertex 1
            point_eval,
            der_x_eval,
            der_y_eval,

            # vertex 2
            point_eval,
            der_x_eval,
            der_y_eval,

            # vertex 3
            point_eval,
            der_x_eval,
            der_y_eval,

            # barycenter
            point_eval
        ]

    def dof_points(self) -> list[Point]:
        return [
            (0, 0),
            (0, 0),
            (0, 0),

            (1, 0),
            (1, 0),
            (1, 0),

            (0, 1),
            (0, 1),
            (0, 1),

            (1/3, 1/3)
        ]

    def edge_orientation_dof_change(self, gdofs: list[int]) -> list[int]:
        return []
