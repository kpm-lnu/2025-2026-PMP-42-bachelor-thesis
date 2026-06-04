from numpy.polynomial.legendre import leggauss

class Quadrature:
    def __init__(self, n_points: int, boundary_points: int):
        self.n_points = n_points
        self.boundary_points = boundary_points
    
    def gauss_points_boundary(self):
        xi, wi_xi = leggauss(self.boundary_points)
        return xi, wi_xi

    def gauss_points_2D(self, n_points: int = None):
        n_of_points = n_points or self.n_points
        xi, wi_xi = leggauss(n_of_points)
        eta, w_eta = leggauss(n_of_points)
        
        points = []

        for i in range(n_of_points):
            for j in range(n_of_points):
                points.append({
                    "xi": xi[i],
                    "eta": eta[j],
                    "weight": wi_xi[i] * w_eta[j] 
                })
        return points
    