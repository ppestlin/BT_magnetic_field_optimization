# Basic imports

import sys
import os
import numpy as np

# Project specific imports

from my_plotting import plot_after_opt
from lmfit import Parameters, fit_report, Minimizer
from my_comparing import my_field_comp_fun
from my_handle_data import get_data, inside_data, order_and_err

# Begin code


def opt_my_coil(params, data_filename, coil_savefilename: str = '',
                fmap_savefilename: str = '', my_range=np.array([-1e8, 1e8]),
                **kwargs_minimize):
    '''
        Starts the numerical optimization of the function creating the coil with respect to
        the data given in data_filename

        The coil and the fieldmap can be saved using the respective savefilenames

        The optimization can be limited to an interval on the z-axis using my_range

        The parameters use common letters and can be found in my_userinterface.py

        After the optimization is finished. The result is plotted next to the data

        **kwargs can contain all arguments for the minimize function

    '''
    # Import recorded (real) data
    real_field, real_pos = get_data(data_filename, 1)

    # Apply my_range (adjust at which part of the z-axes you actually want to look)
    real_field_inside, real_pos_inside = inside_data(real_field, real_pos, my_range)

    # Order the field and get the field error
    ordered_real_field, ordered_real_pos, field_err = order_and_err(real_field_inside, real_pos_inside)

    minim = Minimizer(my_field_comp_fun,
                      params,
                      fcn_args=(ordered_real_pos, ordered_real_field, field_err),
                      nan_policy='omit')

    out = minim.minimize(**kwargs_minimize)
    print(fit_report(out))

    plot_after_opt(out.params, ordered_real_pos, ordered_real_field, field_err,
                   coil_savefilename, fmap_savefilename)

    return
