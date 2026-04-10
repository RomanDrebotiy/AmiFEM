from finite_elements.base_element import BaseElement, Triangle, ScalarFunc


class PolynomialElement(BaseElement):
    def __init__(self, domain: Triangle, order: int):
        self.order = order
        super().__init__(domain)

    def _canonical_basis(self) -> list[ScalarFunc]:
        res = []
        for p in range(self.order + 1):
            for i in range(p + 1):
                res.append(lambda x, y, i=i, p=p: x**i * y**(p-i))
        return res