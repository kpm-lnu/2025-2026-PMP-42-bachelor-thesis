import numpy as np
import math

class ShapeFunction1D:
    def evaluate(self, xi: float):
        raise NotImplementedError("ShapeFunction.evaluate must be implemented by subclass.")

class ShapeFunction2D:
    def evaluate(self, xi: float, eta: float):
        raise NotImplementedError("ShapeFunction.evaluate must be implemented by subclass.")

class Linear2DQuadrilateralShapeFunction(ShapeFunction2D):
    def __init__(self):
        self.nodes_count = 4

    def evaluate(self, xi: float, eta: float):
        N = np.array([
            0.25 * (1 - xi) * (1 - eta),  
            0.25 * (1 + xi) * (1 - eta),  
            0.25 * (1 + xi) * (1 + eta),  
            0.25 * (1 - xi) * (1 + eta)   
        ])

        dN_dxi = np.array([
            -0.25 * (1 - eta),  
            0.25 * (1 - eta),  
            0.25 * (1 + eta),  
            -0.25 * (1 + eta)   
        ])

        dN_deta = np.array([
            -0.25 * (1 - xi),  
            -0.25 * (1 + xi),  
            0.25 * (1 + xi),  
            0.25 * (1 - xi)   
        ])

        return (N, dN_dxi, dN_deta)

class Quadratic2D8ShapeFunction(ShapeFunction2D):
    def __init__(self):
        self.nodes_count = 8

    def evaluate(self, xi: float, eta: float):
        N = np.array([
            0.25 * (1 - xi) * (1 - eta) * (-xi - eta - 1), # left bottom
            0.25 * (1 + xi) * (1 - eta) * (xi - eta - 1), # right bottom
            0.25 * (1 + xi) * (1 + eta) * (xi + eta - 1), # right top
            0.25 * (1 - xi) * (1 + eta) * (-xi + eta - 1), # left top
            0.5 * (1 - math.pow(xi, 2)) * (1 - eta), # bottom middle
            0.5 * (1 + xi) * (1 - math.pow(eta, 2)), # right middle
            0.5 * (1 - math.pow(xi, 2)) * (1 + eta), # top middle
            0.5 * (1 - xi) * (1 - math.pow(eta, 2)) # left middle
        ])

        dN_dxi = np.array([
            0.25 * (1 - eta) * (2 * xi + eta),
            0.25 * (1 - eta) * (2 * xi - eta),
            0.25 * (1 + eta) * (2 * xi + eta),
            0.25 * (1 + eta) * (2 * xi - eta),
            -xi * (1 - eta),
            0.5 * (1 - eta**2),
            -xi * (1 + eta),
            -0.5 * (1 - eta**2)
        ])

        dN_deta = np.array([
            0.25 * (1 - xi) * (xi + 2 * eta),
            0.25 * (1 + xi) * (-xi + 2 * eta),
            0.25 * (1 + xi) * (xi + 2 * eta),
            0.25 * (1 - xi) * (-xi + 2 * eta),
            -0.5 * (1 - xi**2),
            -(1 + xi) * eta,
            0.5 * (1 - xi**2),
            -eta * (1 - xi)
        ])

        return (N, dN_dxi, dN_deta)
    
class Quadratic1D2ShapeFunction(ShapeFunction1D):
    def __init__(self):
        self.nodes_count = 2
    
    def evaluate(self, xi: float):
        N = np.array([
            0.5 * (1 - xi),   
            0.5 * (1 + xi)
        ])

        dN_dxi = np.array([
            -0.5,  
            0.5  
        ])

        return (N, dN_dxi)

class Quadratic1D3ShapeFunction(ShapeFunction1D):
    def __init__(self):
        self.nodes_count = 3
    
    def evaluate(self, xi: float):
        N = np.array([
            0.5 * xi * (xi - 1),  
            1 - xi**2,
            0.5 * xi * (xi + 1)
        ])

        dN_dxi = np.array([
            xi - 0.5,  
            -2 * xi,  
            xi + 0.5  
        ])

        return (N, dN_dxi)
