"""
Генератор навчальних даних для нейронних мереж
Зберігає результати МСЕ-розрахунків у форматі JSON
"""

import json
import numpy as np
import csv

from .material import Material
from .factory import FEMFactory, ElementType
from .boundaryConditions import DirichletBC, NeumannBC
from .axisymmetricFEMSolver import AxisymmetricFEMSolver
from .postprocessors.stress_recovery import recover_all_nodal_stresses


def run_single_simulation(pressure, E, nu, r_min=1.0, r_max=2.0, z_min=0.0, z_max=1.0, verbose=False):
    """Виконує один МСЕ-розрахунок для заданих параметрів."""
    
    if verbose:
        print(f"Розрахунок: p={pressure}, E={E}, nu={nu}")
    
    mat = Material("Material", E=E, nu=nu)
    node_dof = 2
    
    fem_factory = FEMFactory(r_min, r_max, z_min, z_max, 0, 0, material=mat, node_dof=node_dof).init(ElementType.LINEAR)
    shape_func, mesh = fem_factory.create(n_points=2)
    
    rN, zN = 2, 2
    r_func = lambda z: 0
    mesh.generate(r_min, r_max, z_min, z_max, rN, zN, r_func, z_forced=None)
    
    bcs = []
    left_nodes = mesh.leftBoundaryNodes
    bottom_nodes = mesh.bottomBoundaryNodes
    top_nodes = mesh.topBoundaryNodes
    
    bcs.append(NeumannBC(edge_nodes=left_nodes, dof=0, traction_value=pressure))
    
    for node_id in bottom_nodes:
        bcs.append(DirichletBC(node_id=node_id, dof=1, value=0.0))
    for node_id in top_nodes:
        bcs.append(DirichletBC(node_id=node_id, dof=1, value=0.0))
    
    solver = AxisymmetricFEMSolver(mesh, bcs)
    solver.run(custom_n_points=2, element_type=ElementType.LINEAR)
    
    # Відновлення напружень у вузлах
    nodal_stresses = recover_all_nodal_stresses(mesh, mat, mesh.shape_func)
    
    nodes_data = []
    for nid, node in mesh.nodes.items():
        stresses = nodal_stresses.get(nid, {})
        nodes_data.append({
            "id": nid,
            "r": float(node.r),
            "z": float(node.z),
            "u_r": float(node.displacements[0]),
            "u_z": float(node.displacements[1]),
            "sigma_rr": float(stresses.get("sigma_rr", 0.0)),
            "sigma_zz": float(stresses.get("sigma_zz", 0.0)),
            "sigma_rz": float(stresses.get("sigma_rz", 0.0)),
            "sigma_tt": float(stresses.get("sigma_tt", 0.0))
        })
    
    elements_data = []
    for eid, elem in mesh.elements.items():
        elements_data.append({
            "id": eid,
            "node_ids": elem.node_ids
        })
    
    return {
        "nodes": nodes_data,
        "elements": elements_data,
        "num_nodes": len(nodes_data),
        "num_elements": len(elements_data)
    }


def generate_dataset(output_file="training_data.json", num_samples=20, 
                     pressure_range=(10, 100), E_range=(1.0, 2.0), nu=0.3,
                     verbose=True):
    """Генерує набір даних для навчання нейромережі."""
    
    print("=" * 60)
    print("ГЕНЕРАТОР НАВЧАЛЬНИХ ДАНИХ ДЛЯ НЕЙРОННОЇ МЕРЕЖІ")
    print("=" * 60)
    print(f"Кількість розрахунків: {num_samples}")
    print(f"Діапазон тиску: {pressure_range[0]} - {pressure_range[1]}")
    print(f"Діапазон модуля Юнга: {E_range[0]} - {E_range[1]}")
    print("=" * 60 + "\n")
    
    np.random.seed(42)
    pressures = np.random.uniform(pressure_range[0], pressure_range[1], num_samples)
    E_values = np.random.uniform(E_range[0], E_range[1], num_samples)
    
    dataset = []
    
    for i, (p, E) in enumerate(zip(pressures, E_values)):
        if verbose:
            print(f"[{i+1}/{num_samples}] Розрахунок: p={p:.2f}, E={E:.3f}")
        
        try:
            result = run_single_simulation(pressure=p, E=E, nu=nu, verbose=False)
            
            sample = {
                "sample_id": i,
                "parameters": {"pressure": float(p), "E": float(E), "nu": float(nu)},
                "result": result
            }
            dataset.append(sample)
            
            if verbose:
                print(f"  ✓ Вузлів: {result['num_nodes']}, Елементів: {result['num_elements']}")
        except Exception as e:
            print(f"  ✗ Помилка: {e}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 60)
    print(f"ГОТОВО! Дані збережено у файл: {output_file}")
    print(f"Кількість успішних розрахунків: {len(dataset)}/{num_samples}")
    print("=" * 60)
    
    return dataset


def save_dataset_as_csv(dataset, output_file="training_data.csv"):
    """Зберігає набір даних у CSV форматі."""
    if not dataset:
        print("Немає даних для збереження")
        return
    
    rows = []
    for sample in dataset:
        params = sample["parameters"]
        for node in sample["result"]["nodes"]:
            rows.append({
                "sample_id": sample["sample_id"],
                "pressure": params["pressure"],
                "E": params["E"],
                "nu": params["nu"],
                "node_id": node["id"],
                "r": node["r"],
                "z": node["z"],
                "u_r": node["u_r"],
                "u_z": node["u_z"],
                "sigma_rr": node["sigma_rr"],
                "sigma_zz": node["sigma_zz"],
                "sigma_rz": node["sigma_rz"],
                "sigma_tt": node["sigma_tt"]
            })
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"CSV файл збережено: {output_file}")


if __name__ == "__main__":
    dataset = generate_dataset(num_samples=20)
    save_dataset_as_csv(dataset, "training_data.csv")