from typing import Any
import triangle as tr
import numpy as np

class Triangulator:
    # triangulation: dict[str, Any]
    vectices: np.array(object)
    def __init__(self, domain):
        self.vertices = domain.vertices

    def triangulate(self, n, angle = 30, area = 0.2):
        segments = [[i, i + 1] for i in range(n - 1)]
        segments.append([n - 1, 0])
        triang_string = 'pq'+str(angle)+'a'+str(area)
        return tr.triangulate(dict(vertices = self.vertices, segments=segments), triang_string)


    def plot(self, triangulation):
        import matplotlib.pyplot as plt
        points = triangulation['vertices']
        triangles = triangulation['triangles']

        plt.triplot(points[:, 0], points[:, 1], triangles)
        plt.plot(points[:, 0], points[:, 1], 'o')
        plt.gca().set_aspect('equal')
        plt.show()