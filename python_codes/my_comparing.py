import sys
import os

import numpy as np
import math

# Project specific imports

import myShapes_adapted

from my_biot_savart import get_my_tv
from lmfit import Parameters, fit_report, Minimizer


def my_field_comp_fun(params,
                      real_pos: np.ndarray,
                      real_field: np.ndarray,
                      field_err: np.ndarray) -> np.ndarray:
    """
    The parameters of this function. All non direct fun parameters are stored in params
    :param params: contains the following variables:
        sol_R: Radius large coil
        sol_N: Winding number large coil
        sol_L: Length large coil
        sol_step: step size per segment in z-direction
        sol_start: start on z-axes, offset in x and y-axes
        sol_I: Current large coil
        sol_Nfactor: Multiplication factor to account for real amount of windings of large coil
        end_r: Radius small coil
        end_i: current small coil
        end_l: length small coil
        end_off: offset small coil from the start
        end_nfactor: Multiplication factor to account for real amount of windings of endcoil
    :param real_field: Data of measured field
    :param real_pos: Measurement positions
    :param field_err:
    :param coil_type: Sets the type of the coil to simulate. Empty corresponds to simple solenoid
    :return: Returns the residue vector for all field vectors
    """

    myShape = myShapes_adapted.Wire()
    myShape.create_my_wire_config(params)

    # This line should probably leave but enough rewriting for today
    sim_real_pos = np.transpose(real_pos[:, np.newaxis, np.newaxis, :],
                                axes=(1, 2, 3, 0))

    sim_fields, sim_positions = get_my_tv(myShape.coordz, sim_real_pos, save_filename='')  # Sim pos is same as real pos
    sim_fields = np.squeeze(sim_fields)
    sim_fields = sim_fields.T

    res = np.zeros(sim_fields.shape[1])
    checking = np.zeros(sim_fields.shape[1])

    for i, _ in enumerate(res):
        res[i] = np.linalg.norm((sim_fields[:, i] - real_field[:, i]) / field_err[:, i])
        checking[i] = np.sum(np.square((sim_fields[:, i] - real_field[:, i])) / np.square(field_err[:, i]))

        if math.isnan(checking[i]):
            checking[i] = 0

    if params['const_print'].value:
        print("Res avg check:", np.average(checking))
        print("Quadratic average of residuals (std. devs): ",
              np.sqrt(sum(checking ** 2) / len(checking)))
        # params.pretty_print()

    return res
