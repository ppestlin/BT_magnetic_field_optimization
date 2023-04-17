# Basic imports

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

# Project specific imports

import myShapes_adapted
from my_plotting import plot_simfields_2d, plot_fields_data_2d, plot_z_field, coil_plot
from my_biot_savart import get_my_tv
from my_handle_data import get_data, inside_data, order_and_err


# Begin code
def test_solenoid_plus_endcoils(params, data_filename, coil_savefilename: str = '',
                                fmap_savefilename: str = '', my_range=np.array([-10e7, 10e7])):

    real_field, real_pos = get_data(data_filename, 1)

    # Apply my_range (adjust at which part of the z-axes you actually want to look)
    real_field_inside, real_pos_inside = inside_data(real_field, real_pos, my_range)
    ordered_real_field, ordered_real_pos, field_err = order_and_err(real_field_inside, real_pos_inside)

    myShape = myShapes_adapted.Wire()
    myShape.create_my_wire_config(params)

    # Declare if you want to save the coil to a txt file
    if not(coil_savefilename == ''):
        with open(coil_savefilename + '.txt', 'w') as f:
            np.savetxt(f, myShape.coordz, fmt='%1.4f', delimiter=",")

    sim_real_pos = np.transpose(ordered_real_pos[:, np.newaxis, np.newaxis, :], axes=(1, 2, 3, 0))
    fields, positions = get_my_tv(myShape.coordz, sim_real_pos, fmap_savefilename)
    ordered_sim_fields = np.squeeze(fields)
    ordered_sim_fields = ordered_sim_fields.T
    # 2D Plots:

    # fig, axes = plt.subplots(nrows=1, ncols=2)

    # plot_fields_data_2d(fields, real_pos, which_plane='x', level=0, shared_plt=[fig, axes], measurement=False)
    # shared_plt = None -> single plot

    # plot_fields_data_2d(real_field, real_pos, which_plane='x', level=0, shared_plt=[fig, axes], measurement=True)
    # shared_plt = None -> single plot

    #1D Plot of z-field
    plot_z_field(ordered_sim_fields, ordered_real_field, ordered_real_pos, field_err, 15)

    return
