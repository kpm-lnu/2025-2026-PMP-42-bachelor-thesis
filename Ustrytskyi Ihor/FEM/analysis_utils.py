from __future__ import annotations
import numpy as np


def analytical_solution(r, a, b, p, mu, nu):
    ur = (1 / (2 * mu * (b**2 - a**2))) * ((1 - 2 * nu) * a**2 * p * r + p * a**2 * b**2 / r)
    uz = 0
    sigma_rr = (p / (b**2 - a**2)) * (a**2 - a**2 * b**2 / r**2)
    sigma_zz = (2 * nu * a**2 * p) / (b**2 - a**2)
    sigma_rz = 0
    sigma_phi_phi = (p / (b**2 - a**2)) * (a**2 + a**2 * b**2 / r**2)
    return ur, uz, sigma_rr, sigma_zz, sigma_phi_phi, sigma_rz


def compute_errors(
    r_values,
    sigma_rr, sigma_zz, sigma_rz, sigma_phi_phi,
    sigma_rr_analytical, sigma_zz_analytical, sigma_rz_analytical, sigma_phi_phi_analytical,
    ur_values, uz_values, ur_analytical, uz_analytical,
    fixed_z
):
    r_values = np.array(r_values)
    sigma_rr = np.array(sigma_rr)
    sigma_zz = np.array(sigma_zz)
    sigma_rz = np.array(sigma_rz)
    sigma_phi_phi = np.array(sigma_phi_phi)

    sigma_rr_analytical = np.array(sigma_rr_analytical)
    sigma_zz_analytical = np.array(sigma_zz_analytical)
    sigma_rz_analytical = np.array(sigma_rz_analytical)
    sigma_phi_phi_analytical = np.array(sigma_phi_phi_analytical)

    ur_values = np.array(ur_values)
    uz_values = np.array(uz_values)
    ur_analytical = np.array(ur_analytical)
    uz_analytical = np.array(uz_analytical)

    abs_error_ur = np.abs(ur_values - ur_analytical)
    abs_error_uz = np.abs(uz_values - uz_analytical)
    abs_error_sigma_rr = np.abs(sigma_rr - sigma_rr_analytical)
    abs_error_sigma_zz = np.abs(sigma_zz - sigma_zz_analytical)
    abs_error_sigma_rz = np.abs(sigma_rz - sigma_rz_analytical)
    abs_error_sigma_phi_phi = np.abs(sigma_phi_phi - sigma_phi_phi_analytical)

    tol = 1e-12

    with np.errstate(divide='ignore', invalid='ignore'):
        rel_error_ur = np.divide(
            abs_error_ur, np.abs(ur_analytical), out=np.zeros_like(abs_error_ur),
            where=np.abs(ur_analytical) > tol
        )
        rel_error_uz = np.divide(
            abs_error_uz, np.abs(uz_analytical), out=np.zeros_like(abs_error_uz),
            where=np.abs(uz_analytical) > tol
        )
        rel_error_sigma_rr = np.divide(
            abs_error_sigma_rr, np.abs(sigma_rr_analytical), out=np.zeros_like(abs_error_sigma_rr),
            where=np.abs(sigma_rr_analytical) > tol
        )
        rel_error_sigma_zz = np.divide(
            abs_error_sigma_zz, np.abs(sigma_zz_analytical), out=np.zeros_like(abs_error_sigma_zz),
            where=np.abs(sigma_zz_analytical) > tol
        )
        rel_error_sigma_rz = np.divide(
            abs_error_sigma_rz, np.abs(sigma_rz_analytical), out=np.zeros_like(abs_error_sigma_rz),
            where=np.abs(sigma_rz_analytical) > tol
        )
        rel_error_sigma_phi_phi = np.divide(
            abs_error_sigma_phi_phi, np.abs(sigma_phi_phi_analytical), out=np.zeros_like(abs_error_sigma_phi_phi),
            where=np.abs(sigma_phi_phi_analytical) > tol
        )

    errors = {
        'absolute_errors': {
            'ur': abs_error_ur,
            'uz': abs_error_uz,
            'sigma_rr': abs_error_sigma_rr,
            'sigma_zz': abs_error_sigma_zz,
            'sigma_rz': abs_error_sigma_rz,
            'sigma_phi_phi': abs_error_sigma_phi_phi,
        },
        'relative_errors': {
            'ur': rel_error_ur,
            'uz': rel_error_uz,
            'sigma_rr': rel_error_sigma_rr,
            'sigma_zz': rel_error_sigma_zz,
            'sigma_rz': rel_error_sigma_rz,
            'sigma_phi_phi': rel_error_sigma_phi_phi,
        }
    }

    analytical = {
        'ur': ur_analytical,
        'uz': uz_analytical,
        'sigma_rr': sigma_rr_analytical,
        'sigma_zz': sigma_zz_analytical,
        'sigma_rz': sigma_rz_analytical,
        'sigma_phi_phi': sigma_phi_phi_analytical
    }

    var_names = ['ur', 'uz', 'sigma_rr', 'sigma_zz', 'sigma_rz', 'sigma_phi_phi']

    print_error_tables(errors, analytical, var_names)

    return errors


def print_error_tables(errors, analytical, var_names, tol=1e-12):
    print("\nAbsolute Errors:")
    header = "{:15} {:15} {:15} {:15}".format("Variable", "Mean", "Max", "Min")
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for var in var_names:
        abs_err = errors['absolute_errors'][var]
        mean_err = np.mean(abs_err)
        max_err = np.max(abs_err)
        min_err = np.min(abs_err)
        print("{:15} {:15.5e} {:15.5e} {:15.5e}".format(var, mean_err, max_err, min_err))
    print("-" * len(header))
    
    print("\nRelative Errors (in percentage, only for variables with nonzero analytical solution):")
    header_perc = "{:15} {:15} {:15} {:15}".format("Variable", "Mean (%)", "Max (%)", "Min (%)")
    print("-" * len(header_perc))
    print(header_perc)
    print("-" * len(header_perc))
    for var in var_names:
        if np.any(np.abs(analytical[var]) > tol):
            rel_err = errors['relative_errors'][var]
            mean_rel = np.mean(rel_err) * 100.0
            max_rel = np.max(rel_err) * 100.0
            min_rel = np.min(rel_err) * 100.0
            print("{:15} {:15.5e} {:15.5e} {:15.5e}".format(var, mean_rel, max_rel, min_rel))
    print("-" * len(header_perc))
