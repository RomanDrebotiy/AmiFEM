import matplotlib.pyplot as plt
import numpy as np
import triangle as tr
from finite_elements.base_element import Point


class Mesh:
    def __init__(self, domain_boundary_vertices: list[Point], area: float = 1):
        self.domain_boundary_vertices = dict(vertices=np.array(domain_boundary_vertices))
        self.__tri = tr.triangulate(self.domain_boundary_vertices, f"eqa{area}")

    def get_triangulation(self):
        return self.__tri

    def plot(self):
        tr.compare(plt, self.domain_boundary_vertices, self.__tri)
        plt.show()
