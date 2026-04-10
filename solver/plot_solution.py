import numpy as np
import matplotlib.pyplot as plt


class Plotter:
    """
    Limited to vertex dofs for now
    """
    def __init__(self, sol: np.ndarray, tri: dict, vdofs: list[list[int]], actual_dofs: list[int], total_dofs: int):
        self.sol = sol
        self.vertices = tri['vertices']
        self.triangles = tri['triangles']
        dofs_total_sol = np.zeros(total_dofs)
        sol_map = list(zip(actual_dofs, sol))
        for a, s in sol_map:
            dofs_total_sol[a] = s
        self.sol_in_vtx = []
        for vidx in range(len(self.vertices)):
            dof_idx = vdofs[vidx][0]
            self.sol_in_vtx.append(dofs_total_sol[dof_idx])

    def plot(self):
        x = self.vertices[:, 0]
        y = self.vertices[:, 1]
        z = self.sol_in_vtx

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_trisurf(x, y, z, triangles=self.triangles, cmap='viridis')
        plt.show()

