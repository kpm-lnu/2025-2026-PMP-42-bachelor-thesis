from .shapeFunction import Linear2DQuadrilateralShapeFunction, Quadratic2D8ShapeFunction, Quadratic1D3ShapeFunction, Quadratic1D2ShapeFunction
from .quadrature import Quadrature
from .mesh import Mesh4Nodes, Mesh8Nodes

from enum import Enum

class ElementType(Enum):
    LINEAR = 'linear'
    QUADRATIC = 'quadratic'

class FEMFactory:
    def __init__(self, r_min, r_max, z_min, z_max, rN, zN, material, node_dof):
        self.r_min = r_min
        self.r_max = r_max
        self.z_min = z_min
        self.z_max = z_max
        self.rN = rN
        self.zN = zN
        self.material = material
        self.node_dof = node_dof
    
    def init(self, type):
        FEMFactoryClass = None
        if type == ElementType.LINEAR:
            FEMFactoryClass = FEMFactoryLinear
        elif type == ElementType.QUADRATIC:
            FEMFactoryClass = FEMFactoryQuadratic
        return FEMFactoryClass(self.r_min, self.r_max, self.z_min, self.z_max, self.rN, self.zN, self.material, self.node_dof)
class FEMFactoryLinear(FEMFactory):
    def create(self, n_points=2, n_boundary_points=2):
        shape_func = Linear2DQuadrilateralShapeFunction()
        shape_func_boundary = Quadratic1D2ShapeFunction()

        quad_rule = Quadrature(n_points=n_points, boundary_points=n_boundary_points)
        mesh = Mesh4Nodes(
            material=self.material,
            shape_func=shape_func,
            shape_func_boundary=shape_func_boundary,
            quadrature=quad_rule,
            node_dof=self.node_dof
            )
        
        return shape_func, mesh

class FEMFactoryQuadratic(FEMFactory):
    def create(self, n_points=2, n_boundary_points=3):
        shape_func = Quadratic2D8ShapeFunction()
        shape_func_boundary = Quadratic1D3ShapeFunction()
        quad_rule = Quadrature(n_points=n_points, boundary_points=n_boundary_points)
        mesh = Mesh8Nodes(
            material=self.material,
            shape_func=shape_func,
            shape_func_boundary=shape_func_boundary,
            quadrature=quad_rule,
            node_dof=self.node_dof
            )
        return shape_func, mesh