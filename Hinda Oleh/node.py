class Node:
    def __init__(self, node_id: int, r: float, z: float):
        self.node_id = node_id
        self.r = r
        self.z = z
        
        self.dof_indices = []
        self.displacements = [0.0, 0.0]  # ur, uz

    def __repr__(self):
        return f"Node({self.node_id}, r={self.r:.3f}, z={self.z:.3f})"
