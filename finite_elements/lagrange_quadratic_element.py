from finite_elements.base_element import Triangle, Operator, Point
from finite_elements.polynomial_element import PolynomialElement


class LagrangeQuadraticElement(PolynomialElement):
    def __init__(self, domain: Triangle):
        super().__init__(domain, 2)

    def _adjoint_basis_per_point(self) -> list[Operator]:
        point_eval = lambda u: u
        return [point_eval] * 6

    def dof_points(self) -> list[Point]:
        return [
            (0, 0),
            (0.5, 0),
            (1, 0),
            (0.5, 0.5),
            (0, 1),
            (0, 0.5)
        ]

    def edge_orientation_dof_change(self, gdofs: list[int]) -> list[int]:
        # added just for reference. For quadratic element there is only
        # one dof per single edge, so it is not affected by orientation change
        gdofs.reverse()
        return gdofs
