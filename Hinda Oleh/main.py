from .material import Material
from .shapeFunction import (
    Quadratic1D2ShapeFunction,
    Quadratic1D3ShapeFunction,
)
from .factory import FEMFactory, ElementType
from .boundaryConditions import DirichletBC, NeumannBC
from .axisymmetricFEMSolver import AxisymmetricFEMSolver
import numpy as np

from .services.io_service import IOService, NullIOService

from .FindElementsForRefinement import (
    FindElementsForRefinementIBEM,
)


class ExperimentConfig:
    def __init__(self,
                 r_min, r_max, z_min, z_max,
                 r_func,
                 mesh_resolutions,
                 p, nu, mu,
                 fixed_z, fixed_r,
                 compare_eps,
                 element_types,
                 n_points_range,
                 mortar_n_gp: int = 2,
                 mortar_normal_dof: int | None = None,
                 mortar_tol: float = 1e-9,
                 adaptive_enabled=False,
                 adaptive_max_cycles=4,
                 adaptive_refine_fraction_threshold=1,
                 adaptive_min_cycles=4,
                 refinement_mode='eta',
                 do_plots=True,
                 do_console_prints=False,
                 center_load_only=False,
                 center_load_length=None,
                 ibem_static_boundary=False,
                 ibem_static_segments_per_side=None):
        self.r_min = r_min
        self.r_max = r_max
        self.z_min = z_min
        self.z_max = z_max
        self.r_func = r_func
        self.mesh_resolutions = mesh_resolutions
        self.p = p
        self.nu = nu
        self.mu = mu
        self.fixed_z = fixed_z
        self.fixed_r = fixed_r
        self.compare_eps = compare_eps
        self.element_types = element_types
        self.n_points_range = n_points_range
        # Mortar coupling
        self.mortar_n_gp = mortar_n_gp
        self.mortar_normal_dof = mortar_normal_dof
        self.mortar_tol = mortar_tol
        # Adaptive
        self.adaptive_enabled = adaptive_enabled
        self.adaptive_max_cycles = adaptive_max_cycles
        self.adaptive_refine_fraction_threshold = adaptive_refine_fraction_threshold
        self.adaptive_min_cycles = adaptive_min_cycles
        self.do_plots = do_plots
        self.do_console_prints = do_console_prints
        self.refinement_mode = refinement_mode
        self.center_load_only = center_load_only
        self.center_load_length = center_load_length
        # IBEM static boundary
        self.ibem_static_boundary = ibem_static_boundary
        self.ibem_static_segments_per_side = ibem_static_segments_per_side or {
            'left': 8, 'bottom': 8, 'right': 4, 'top': 4,
            'inner_vertical': 4, 'inner_horizontal': 4,
        }

def select_elements_for_refinement(mesh, method='IBEM', threshold=0.3, bcs=None, ibem_strategy=None, mode='eta'):
    strategies = {
        'IBEM': ibem_strategy if ibem_strategy is not None else FindElementsForRefinementIBEM(),
    }
    strategy = strategies.get(method)
    if strategy is None:
        raise ValueError(f"Unknown refinement method: {method}")
    return strategy.find(mesh, threshold, bcs=bcs, mode=mode)

def adaptive_refine_and_solve(
    factory,
    config,
    rN,
    zN,
    elem_type,
    custom_n_points,
    do_plots: bool = True,
    refine_method: str = 'IBEM',
    io: IOService | None = None,
):
    """Run adaptive refinement cycles for a single mesh resolution & element type."""
    if io is None:
        io = NullIOService()
    shape_func, mesh = factory.create(n_points=custom_n_points)
    # Compute forced z-lines for center load boundaries
    z_forced = None
    if config.center_load_only and config.center_load_length is not None:
        z_mid = (config.z_min + config.z_max) / 2
        half = config.center_load_length / 2
        z_forced = [z_mid - half, z_mid + half]
    mesh.generate(config.r_min, config.r_max, config.z_min, config.z_max, rN, zN, config.r_func, z_forced=z_forced)
    mesh.refinement_pressure = config.p

    # Create a persistent IBEM strategy instance for static boundary caching
    ibem_strategy = FindElementsForRefinementIBEM(
        static_boundary=getattr(config, 'ibem_static_boundary', False),
        static_segments_per_side=getattr(config, 'ibem_static_segments_per_side', None),
        io=io,
    )

    cycle_results = []  # each entry: {'cycle': int, 'num_elements': int, 'snapshot': {...}}
    for cycle in range(config.adaptive_max_cycles):
        # Regenerate boundary conditions each cycle to include new boundary mid-edge nodes
        bcs = create_boundary_conditions(mesh, config.p, center_load_only=config.center_load_only, center_load_length=config.center_load_length)
        solver = AxisymmetricFEMSolver(
            mesh,
            bcs,
            mortar_interfaces=[],
        )

        mesh.build_and_attach_mortar_interfaces(
            solver,
            n_gp=getattr(config, 'mortar_n_gp', 2),
            normal_dof=getattr(config, 'mortar_normal_dof', None),
            tol=getattr(config, 'mortar_tol', 1e-9),
        )
        solver.run(custom_n_points, elem_type)
        # Plot u and sigma fields for this cycle (pre-refinement)
        if do_plots:
            try:
                io.plot_ur_heatmap(mesh)
                io.plot_grid_node_values(mesh, field='u_r', annotate=True)
                io.plot_grid_node_values(mesh, field='u_z', annotate=True)
                for comp in ['sigma_rr', 'sigma_zz', 'sigma_rz', 'sigma_tt', 'sigma_eff']:
                    io.plot_sigma_heatmap_dense_per_element(mesh, comp, material=mesh.material)
                    io.plot_grid_node_values(mesh, field=comp, annotate=True, max_annotate=250, fmt='{:.3e}')
            except Exception as exc:
                print(f"Cycle {cycle} field plots skipped: {exc}")

        snapshot = {
            'cycle': cycle,
            'nodes': [(nid, n.r, n.z, n.displacements[0], n.displacements[1]) for nid, n in mesh.nodes.items()],
            'elements': [{'id': e_id, 'node_ids': mesh.elements[e_id].node_ids} for e_id in mesh.elements],
            'num_elements': len(mesh.elements)
        }
        cycle_results.append({'cycle': cycle, 'num_elements': len(mesh.elements), 'snapshot': snapshot})

        refine_ids = select_elements_for_refinement(mesh, method=refine_method, threshold=config.adaptive_refine_fraction_threshold, bcs=bcs, ibem_strategy=ibem_strategy, mode=config.refinement_mode)
        io.print_adaptive_cycle(cycle, len(refine_ids), len(mesh.elements))
        if config.adaptive_max_cycles != cycle + 1:
            mesh.refine_elements(refine_ids, auto_plot=do_plots)
            if do_plots:
                io.plot_mesh_connectivity(mesh, elem_type == ElementType.QUADRATIC, Quadratic1D3ShapeFunction if elem_type == ElementType.QUADRATIC else Quadratic1D2ShapeFunction)
    return shape_func, mesh, cycle_results

def create_boundary_conditions(mesh, p, center_load_only=False, center_load_length=None):
    bcs = []

    bn = getattr(mesh, "boundary_nodes", {}) or {}

    # Outer boundaries (keep your existing behavior + backward compatibility)
    left_nodes  = bn.get("left",  getattr(mesh, "leftBoundaryNodes",  []))
    right_nodes  = bn.get("right",  getattr(mesh, "rightBoundaryNodes",  []))
    bottom_nodes = bn.get("bottom", getattr(mesh, "bottomBoundaryNodes", []))
    top_nodes    = bn.get("top",    getattr(mesh, "topBoundaryNodes",    []))

    # Internal pressure (traction = -p*n). With this NeumannBC signature we supply the component.
    if center_load_only:
        sorted_right = sorted(right_nodes, key=lambda nid: mesh.nodes[nid].z)
        z_vals = [mesh.nodes[nid].z for nid in sorted_right]
        z_mid = (min(z_vals) + max(z_vals)) / 2
        if center_load_length is not None:
            half = center_load_length / 2
            z_lo = z_mid - half
            z_hi = z_mid + half
        else:
            # Default: middle third
            z_lo = min(z_vals) + (max(z_vals) - min(z_vals)) / 3
            z_hi = min(z_vals) + 2 * (max(z_vals) - min(z_vals)) / 3
        loaded = [nid for nid in sorted_right if z_lo - 1e-12 <= mesh.nodes[nid].z <= z_hi + 1e-12]
        bcs.append(NeumannBC(edge_nodes=loaded, dof=0, traction_value=p))
        bcs.append(NeumannBC(edge_nodes=left_nodes, dof=0, traction_value=0))
    else:
        bcs.append(NeumannBC(edge_nodes=left_nodes, dof=0, traction_value=p))
        bcs.append(NeumannBC(edge_nodes=right_nodes, dof=0, traction_value=0))

    # Outer surface traction-free (usually you can omit this; keep only if your solver expects explicit zero BCs)
    # bcs.append(NeumannBC(edge_nodes=right_nodes, dof=1, pressure=0.0))

    # Axial constraints (as you had)
    for node_id in bottom_nodes:
        bcs.append(DirichletBC(node_id=node_id, dof=1, value=0.0))
    for node_id in top_nodes:
        bcs.append(DirichletBC(node_id=node_id, dof=1, value=0.0))


    return bcs

def run_experiment(factory, config, rN, zN, custom_n_points=None, io=None):
    if io is None:
        io = NullIOService()
    io.print_new_mesh(rN, zN)
    shape_func, mesh = factory.create(n_points=custom_n_points, n_boundary_points=30)
    z_forced = None
    if config.center_load_only and config.center_load_length is not None:
        z_mid = (config.z_min + config.z_max) / 2
        half = config.center_load_length / 2
        z_forced = [z_mid - half, z_mid + half]
    mesh.generate(config.r_min, config.r_max, config.z_min, config.z_max, rN, zN, config.r_func, z_forced=z_forced)
    bcs = create_boundary_conditions(mesh, config.p, center_load_only=config.center_load_only, center_load_length=config.center_load_length)
    solver = AxisymmetricFEMSolver(mesh, bcs)
    solver.run(custom_n_points, config.element_types[0])
    return shape_func, mesh, solver.local_stiffnesses

def main():
    r_min = 1.0#22
    r_max = 2.0#33.5
    z_min = 0.0
    z_max = 1.0#27
    nu = 0.3#0.225
    mu = 0.7
    p = 100#-6550
    fixed_z = 0.5
    fixed_r = 1
    r_func = lambda z: 0#0.5*(-math.fabs(z-0.5) + 0.5)/(config.r_max-config.r_min)
    compare_eps = 1e-6
    # E = 2e6#2.10e6
    E = 2 * mu * (1 + nu)
    mu = E / (2 * (1 + nu))
    node_dof = 2

    element_type = ElementType.LINEAR # adaptive algorithm currently do not work with ElementType.QUADRATIC

    config = ExperimentConfig(
        r_min, r_max, z_min, z_max,
        r_func,
        mesh_resolutions=[(2, 2)],
        p=p,
        nu=nu,
        mu=mu,
        fixed_z=fixed_z,
        fixed_r=fixed_r,
        compare_eps=compare_eps,
        element_types=[element_type],
        n_points_range=[3 if element_type == ElementType.QUADRATIC else 2],
        adaptive_enabled=True,
        refinement_mode='eta', # eta, eta_e_t, theta_tilde_B, theta_tilde_T
        do_plots=True,
        do_console_prints=True,
        adaptive_refine_fraction_threshold=0.3,
        center_load_only=False,
        center_load_length=9.16,  # length of center load region on left boundary
        ibem_static_boundary=False,
        ibem_static_segments_per_side={
            'left': 8, 
            'bottom': 8, 
            'right': 8, 
            'top': 8,
            'inner_vertical': 4, 
            'inner_horizontal': 4,
        },
    )

    mat = Material("Test", E=E, nu=nu)

    io = IOService(enabled=config.do_plots)

    experiment_results = []
    
    for elem_type in config.element_types:
        fem_factory = FEMFactory(r_min, r_max, z_min, z_max, 0, 0, material=mat, node_dof=node_dof).init(elem_type)
        for (rN, zN) in config.mesh_resolutions:
            for n_points in config.n_points_range:
                io.print_experiment_header(rN, zN, elem_type, n_points)
                if config.adaptive_enabled:
                    shape_func, mesh, adaptive_cycles = adaptive_refine_and_solve(
                        fem_factory,
                        config,
                        rN,
                        zN,
                        elem_type,
                        custom_n_points=n_points,
                        do_plots=config.do_plots,
                        io=io,
                    )
                    if config.do_plots:
                        io.plot_adaptive_overlay(adaptive_cycles, fixed_z, compare_eps, r_min, r_max, p, mu, nu)
                        io.plot_adaptive_overlay_slice(adaptive_cycles, fixed_z, r_min, r_max, p, mu, nu)
                    result = {
                        'rN': rN,
                        'zN': zN,
                        'elem_type': elem_type,
                        'n_points': n_points,
                        'mesh': mesh,
                        'shape_func': shape_func,
                        'adaptive_cycles': adaptive_cycles
                    }
                else:
                    shape_func, mesh, _ = run_experiment(
                        fem_factory,
                        config,
                        rN,
                        zN,
                        custom_n_points=n_points
                    )
                    result = {
                        'rN': rN,
                        'zN': zN,
                        'elem_type': elem_type,
                        'n_points': n_points,
                        'mesh': mesh,
                        'shape_func': shape_func
                    }
                experiment_results.append(result)
            if config.do_plots:
                io.plot_mesh_connectivity(mesh, elem_type == ElementType.QUADRATIC, Quadratic1D3ShapeFunction if elem_type == ElementType.QUADRATIC else Quadratic1D2ShapeFunction)

    mesh = experiment_results[-1]['mesh']

    if config.do_plots:
        io.plot_u_sigma(
            [res['mesh'] for res in experiment_results],
            r_min, r_max, p, mu, nu, fixed_z, fixed_r, mat, compare_eps,
            experiment_results,
            True
        )
        
        io.plot_grid_node_indices_and_ur(mesh)
        io.plot_ur_heatmap(mesh)

        for comp in ['sigma_rr', 'sigma_zz', 'sigma_rz', 'sigma_tt']:
            io.plot_grid_node_indices_and_sigma(mesh, comp, material=mat)
            io.plot_sigma_heatmap_dense_per_element(mesh, comp, material=mat)

        io.print_node_table_ur(mesh, a=r_min, b=r_max, p=p, mu=mu, nu=nu)
        for comp in ['sigma_rr', 'sigma_zz', 'sigma_rz', 'sigma_tt']:
            io.print_node_table_sigma(mesh, mat, component=comp, a=r_min, b=r_max, p=p, mu=mu, nu=nu)

if __name__ == "__main__":
    main()

