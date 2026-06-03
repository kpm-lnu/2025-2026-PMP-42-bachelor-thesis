import numpy as np

class Material:
    def __init__(self, name: str, E: float, nu: float):
        self.name = name
        self.E = E
        self.nu = nu

    def __repr__(self):
        return f"Material(name={self.name}, E={self.E}, nu={self.nu})"
    
    def get_elastic_matrix(self) -> np.ndarray:
        E = self.E
        nu = self.nu

        D = (1 - nu) * E / ((1 + nu) * (1 - 2 * nu))

        D11 = D22 = D44 = D
        D12 = D14 = D24 = (nu / (1 - nu)) * D
        D33 = ((1 - 2 * nu) / (2 * (1 - nu))) * D

        D_matrix = np.array([
            [D11, D12,   0,   D14],  # σ_rr
            [D12, D22,   0,   D24],  # σ_zz
            [0,    0,  D33,     0],  # σ_rz
            [D14, D24,   0,   D44]   # σ_φφ
        ])

        return D_matrix
