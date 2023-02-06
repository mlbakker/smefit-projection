# -*- coding: utf-8 -*-

import json
import pathlib
from collections import namedtuple

import numpy as np
import pandas as pd
import scipy.linalg as la
import yaml

# from .basis_rotation import rotate_to_fit_basis
# from .covmat import construct_covmat, covmat_from_systematics
# from .log import logging


def load_theory(
    file_name,
    order,
    use_quad = False,
    use_theory_covmat = True,
    # Now I set the multiplicative prescription to False, because otherwise it does not
    # work with the calculation in compute_theory_values
    # Not sure if that is correct with the way the coefficients are given
    use_multiplicative_prescription = False,
    rotation_matrix = None,
    theory_folder = "/data/theorie/maaikeb/smefit_database/theory"
    ):
    """
    Load theory predictions

    Parameters
    ----------
        operators_to_keep: list
            list of operators to keep
        order: "LO", "NLO"
            EFT perturbative order
        use_quad: bool
            if True returns also |HO| corrections
        use_theory_covmat: bool
            if True add the theory covariance matrix to the experimental one
        rotation_matrix: numpy.ndarray
            rotation matrix from tables basis to fitting basis

    Returns
    -------
        sm: numpy.ndarray
            |SM| predictions
        lin_dict: dict
            dictionary with |NHO| corrections
        quad_dict: dict
            dictionary with |HO| corrections, empty if not use_quad
    """
    theory_file = f"{theory_folder}/{file_name}.json"
    # check_file(theory_file)
    # load theory predictions
    with open(theory_file, encoding="utf-8") as f:
        raw_th_data = json.load(f)

    quad_dict = {}
    lin_dict = {}

    # save sm prediction at the chosen perturbative order
    sm = np.array(raw_th_data[order]["SM"])

    # split corrections into a linear and quadratic dict
    for key, value in raw_th_data[order].items():

        # quadratic terms
        if "*" in key and use_quad:
            quad_dict[key] = np.array(value)
            if use_multiplicative_prescription:
                quad_dict[key] = np.divide(quad_dict[key], sm)

        # linear terms
        elif "SM" not in key and "*" not in key:
            lin_dict[key] = np.array(value)
            if use_multiplicative_prescription:
                lin_dict[key] = np.divide(lin_dict[key], sm)

    # rotate corrections to fitting basis
    if rotation_matrix is not None:
        lin_dict_final, quad_dict_final = rotate_to_fit_basis(
            lin_dict, quad_dict, rotation_matrix
        )
    else:
        lin_dict_final = lin_dict
        quad_dict_final = quad_dict

    best_sm = np.array(raw_th_data["best_sm"])
    th_cov = np.zeros((best_sm.size, best_sm.size))
    if use_theory_covmat:
        th_cov = raw_th_data["theory_cov"]
    # import pdb; pdb.set_trace()
    return raw_th_data["best_sm"], th_cov, lin_dict_final, quad_dict_final


def load_fit_results(path_to_fit, operators_to_keep):
    """
    Load the fit results corresponding to the operators that we want to keep.
    Results are taken from the file specified in path_to_fit.
    NOTE: for quadratic operators, a fit result with quadratic terms is needed.

    Returns dictionary with the fit results for each operator.
    """

    with open(path_to_fit, encoding="utf-8") as f:
        fit_results = json.load(f)

    fit_results_selection = {}

    for key in fit_results:
        if f"{key}" in operators_to_keep:
            fit_results_selection.update({f"{key}": fit_results[key]})

    return fit_results_selection


def compute_theory_values(theory_file_name, path_to_fit, order):
    """
    Compute Di with the theory data and fit results.
    """

    theory_data = load_theory(theory_file_name, order)
    sm_values = theory_data[0]
    smeft_operators = theory_data[2] | theory_data[3]

    fit_data = load_fit_results(path_to_fit, smeft_operators.keys())

    # calculate the mean of the fit results for every coefficient
    smeft_coefficients = {k: np.mean(i) for k, i in fit_data.items()}

    D = []


    for i in range(len(sm_values)):

        d_i = sm_values[i]

        # calculate the product of the coefficient with the operator for every linear correction term
        for operator in smeft_operators:

            if operator not in smeft_coefficients:
                print(f"Warning: operator {operator} in theory file but not in fit results")

            else:
                smeft_corr = smeft_operators[operator][i] * smeft_coefficients[operator]
                d_i += smeft_corr

        D.append(d_i)

    return D

theory_file = "ATLAS_tt_13TeV_ljets_2016_Mtt"
fit_path = "/data/theorie/maaikeb/smefit_release/results/MC_GLOBAL_NLO_NHO/posterior.json"
order = "NLO"


print(compute_theory_values(theory_file, fit_path, order))

