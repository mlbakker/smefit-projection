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
    operators_to_keep,
    order,
    use_quad = True,
    use_theory_covmat = True,
    use_multiplicative_prescription = True,
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

    # select corrections to keep
    def is_to_keep(op1, op2=None):
        if op2 is None:
            return op1 in operators_to_keep
        return op1 in operators_to_keep and op2 in operators_to_keep

    # rotate corrections to fitting basis
    if rotation_matrix is not None:
        lin_dict_to_keep, quad_dict_to_keep = rotate_to_fit_basis(
            lin_dict, quad_dict, rotation_matrix
        )
    else:
        lin_dict_to_keep = {k: val for k, val in lin_dict.items() if is_to_keep(k)}
        quad_dict_to_keep = {
            k: val
            for k, val in quad_dict.items()
            if is_to_keep(k.split("*")[0], k.split("*")[1])
        }

    best_sm = np.array(raw_th_data["best_sm"])
    th_cov = np.zeros((best_sm.size, best_sm.size))
    if use_theory_covmat:
        th_cov = raw_th_data["theory_cov"]
    # import pdb; pdb.set_trace()
    return raw_th_data["best_sm"], th_cov, lin_dict_to_keep, quad_dict_to_keep

print(load_theory("ATLAS_tt_13TeV_ljets_2016_Mtt", ["OtG", "OtG*OtG"], "LO"))
