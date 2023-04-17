# Basic imports

import sys
import os

import numpy as np
import matplotlib.pyplot as plt

# Project specific imports

from lmfit import Parameters
from Test_bs import test_solenoid_plus_endcoils
from start_num_opt import opt_my_coil
from my_biot_savart import get_my_grid, calc_my_fields
from my_plotting import my_contour_plot, coil_plot
import myShapes_adapted


# get points on a straight line in ND space
def get_points(a, b, d):
    # Calculate the distance between the two points
    distance = np.linalg.norm(np.array(b) - np.array(a))
    # Calculate the number of points needed
    num_points = int(distance / d) + 1
    # Create an array of equally spaced points between a and b
    points = np.linspace(a, b, num_points)
    return points


def get_psc_params():

    '''
    params = Parameters()
    # Parameters for simple solenoid
    params.add('sol_N', value=27_000, vary=False)  # [#]
    params.add('sol_R', value=11.9535655, min=10, max=14, vary=False)  # [cm]
    params.add('sol_L', value=79.7852332, min=75, max=95, vary=False)  # [cm]
    params.add('sol_L_norm', value=85, vary=False)  # [cm]
    params.add('sol_I', value=73.4626978, min=70, max=95.1, vary=True)  # [A]
    params.add('sol_step', value=0.2, vary=False)  # [rad]
    params.add('sol_start_z', value=1.67296252, min=0, max=5, vary=True)  # [cm]
    params.add('sol_Nfactor', value=0.01, vary=False)
    params.add('sol_layers', value=4,
               vary=False)  # number of layers of wire in the solenoid
    # x and y offset of the solenoid
    params.add('x_offset', value=-0.16380282, min=-5, max=5, vary=True)  # [cm]
    params.add('y_offset', value=0.14135580, min=-5, max=5, vary=True)  # [cm]
    # Additional parameters for end coils
    params.add('end_n', value=730, min=100, max=2000, vary=False)  # [#]
    params.add('end_i', value=435.1, min=50, max=500.1, vary=True)  # [A]
    params.add('end_r', value=16.0000000, min=10, max=16, vary=True)  # [cm]
    params.add('end_l', value=5.00000000, min=5, max=100, vary=True)  # [cm]
    params.add('end_l_norm', value=11, vary=False)  # [cm]
    params.add('end_off', value=0, min=-5, max=5, vary=False)  # [cm]
    params.add('end_nfactor', value=0.1, vary=False)
    params.add('end_layers', value= 2, vary=False)
    # Additional coils (solenoid shape)
    params.add('extra_coils', value=0, vary=False)  # number of extra coils you are adding
    # Example, increase 'name number' when adding new coils. Start with 0
    params.add('n_0', value=100, vary=False)  # [#]
    params.add('nfactor_0', value=1, vary=False)  # [#]
    params.add('i_0', value=75, vary=False)  # [A]
    params.add('r_0', value=20, vary=False)  # [cm]
    params.add('l_0', value=15, vary=False)  # [cm]
    params.add('l_norm_0', value=11, vary=False)  # [cm]
    params.add('x_off_0', value=0,
               vary=False)  # [cm]  x offset to center of main solenoid
    params.add('y_off_0', value=0,
               vary=False)  # [cm]  y offset to center of main solenoid
    params.add('z_off_0', value=0,
               vary=False)  # [cm]  z offset to center of main solenoid
    params.add('layers_0', value=5, vary=False)
    '''
    params = Parameters()
    params.add('const_print', value=0, min=0, max=1, vary=False)
    # Parameters for simple solenoid
    params.add('sol_N', value=26_852.6388, min=25_000, max=29_000, vary=True)  # [#]
    params.add('sol_R', value=12.7739153, min=11, max=14, vary=True)  # [cm]
    params.add('sol_L', value=82.5498994, min=75, max=88, vary=True)  # [cm]
    params.add('sol_L_norm', value=85, vary=False)  # [cm]
    params.add('sol_I', value=75, min=70, max=95.1, vary=False)  # [A]
    params.add('sol_step', value=0.1*np.pi, vary=False)  # [rad]
    params.add('sol_start_z', value=1.57185953, min=0, max=3, vary=True)  # [cm]
    # params.add('sol_start_z', value=0, min=0, max=5, vary=True)  # [cm]
    params.add('sol_Nfactor', value=0.1, vary=False)
    params.add('sol_layers', value=4,
               vary=False)  # number of layers of wire in the solenoid
    # x and y offset of the solenoid
    params.add('x_offset', value=-0.04009486, min=-1, max=1, vary=True)  # [cm]
    # params.add('x_offset', value=0, min=-2, max=2, vary=True)  # [cm]
    params.add('y_offset', value=0.21412684, min=-1, max=1, vary=True)  # [cm]
    # params.add('y_offset', value=0, min=-2, max=2, vary=True)  # [cm]
    # Additional parameters for end coils
    params.add('end_n', value=1203.92864, min=800, max=1400, vary=True)  # [#]
    params.add('end_i', value=75, min=50, max=500.1, vary=False)  # [A]
    params.add('end_r', value=13.7472243, min=12, max=15, vary=True)  # [cm]
    params.add('end_l', value=12.2649487, min=8, max=15, vary=True)  # [cm]
    params.add('end_l_norm', value=11, vary=False)  # [cm]
    params.add('end_off', value=0, min=-5, max=5, vary=False)  # [cm]
    params.add('end_nfactor', value=0.1, vary=False)
    params.add('end_layers', value=2, vary=False)
    # Additional coils (solenoid shape)

    params.add('extra_coils', value=0, vary=False)  # number of extra coils you are adding
    # Example, increase 'name number' when adding new coils. Start with 0
    params.add('n_0', value=100, vary=False)  # [#]
    params.add('nfactor_0', value=1, vary=False)  # [#]
    params.add('i_0', value=75, vary=False)  # [A]
    params.add('r_0', value=20, vary=False)  # [cm]
    params.add('l_0', value=15, vary=False)  # [cm]
    params.add('l_norm_0', value=11, vary=False)  # [cm]
    params.add('x_off_0', value=0,
               vary=False)  # [cm]  x offset to center of main solenoid
    params.add('y_off_0', value=0,
               vary=False)  # [cm]  y offset to center of main solenoid
    params.add('z_off_0', value=0,
               vary=False)  # [cm]  z offset to center of main solenoid
    params.add('layers_0', value=5, vary=False)

    return params


def get_ben_params():


    params = Parameters()
    params.add('const_print', value=1, vary=False)
    # Parameters for simple solenoid
    params.add('sol_N', value=16223.8391, min=15_900, max=16_500, vary=True)  # [#]  16224.9526
    params.add('sol_R', value=17.4264706, min=17, max=18, vary=True)  # [cm]
    params.add('sol_L', value=52.6361633, min=50, max=55, vary=True)  # [cm]
    params.add('sol_L_norm', value=50, vary=False)  # [cm]
    params.add('sol_I', value=75, vary=False)  # [A]
    params.add('sol_step', value=0.03*np.pi, vary=False)  # [rad]
    params.add('sol_start_z', value=-0.5, min=-1, max=0, vary=True)  # [cm]
    # params.add('sol_start_z', value=0, min=-1, max=0, vary=True)  # [cm]
    params.add('sol_Nfactor', value=0.1, vary=False)
    params.add('sol_layers', value=4, vary=False)  # number of layers of wire in the solenoid
    # x and y offset of the solenoid
    params.add('x_offset', value=0, min=-0.1, max=0.1, vary=False)  # [cm]
    params.add('y_offset', value=-0.15, min=-1, max=0, vary=True)  # [cm]  -0.55
    # params.add('y_offset', value=0, min=-1, max=0, vary=True)  # [cm]  -0.55
    # Additional parameters for end coils
    params.add('end_n', value=2274, min=2000, max=2500, vary=True)  # [#] 2273.36920
    params.add('end_i', value=75, vary=False)  # [A]
    params.add('end_r', value=19.3035617, min=18, max=21, vary=True)  # [cm]
    params.add('end_l', value=10.3089900, min=8.5, max=12.5, vary=True)  # [cm]
    params.add('end_l_norm', value=10, vary=False)  # [cm]
    params.add('end_off', value=0, min=-0.5, max=0.5, vary=False)  # [cm]
    params.add('end_nfactor', value=0.1, vary=False)
    params.add('end_layers', value=2, vary=False)
    # Additional coils (solenoid shape)
    params.add('extra_coils', value=0, vary=False)  # number of extra coils you are adding
    # Example, increase 'name number' when adding new coils. Start with 0
    params.add('n_0', value=100, vary=False)  # [#]
    params.add('nfactor_0', value=1, vary=False)  # [#]
    params.add('i_0', value=75, vary=False)  # [A]
    params.add('r_0', value=20, vary=False)  # [cm]
    params.add('l_0', value=15, vary=False)  # [cm]
    params.add('l_norm_0', value=11, vary=False)  # [cm]
    params.add('x_off_0', value=0,
               vary=False)  # [cm]  x offset to center of main solenoid
    params.add('y_off_0', value=0,
               vary=False)  # [cm]  y offset to center of main solenoid
    params.add('z_off_0', value=0,
               vary=False)  # [cm]  z offset to center of main solenoid
    params.add('layers_0', value=5, vary=False)
    '''
    params = Parameters()
    params.add('const_print', value=0, vary=False)
    # Parameters for simple solenoid
    params.add('sol_N', value=18336.0962, min=14_000, max=19_000, vary=True)  # [#]
    params.add('sol_R', value=21.7784702, min=15, max=23, vary=True)  # [cm]
    params.add('sol_L', value=43.2076538, min=40, max=64, vary=True)  # [cm]
    params.add('sol_L_norm', value=50, vary=False)  # [cm]
    params.add('sol_I', value=75, min=70, max=75, vary=False)  # [A]
    params.add('sol_step', value=0.1, vary=False)  # [rad]
    params.add('sol_start_z', value=-0.67645899, min=-2, max=2, vary=True)  # [cm]
    params.add('sol_Nfactor', value=0.01, vary=False)
    params.add('sol_layers', value=4,
               vary=False)  # number of layers of wire in the solenoid
    # x and y offset of the solenoid
    params.add('x_offset', value=0, min=-0.1, max=0.1, vary=False)  # [cm]
    params.add('y_offset', value=-0.08645267, min=-1, max=1, vary=False)  # [cm]
    # Additional parameters for end coils
    params.add('end_n', value=2359.65005, min=100, max=3000, vary=True)  # [#]
    params.add('end_i', value=75, min=0, max=125.1, vary=False)  # [A]
    params.add('end_r', value=16.3192119, min=16, max=22, vary=True)  # [cm]
    params.add('end_l', value=9.61389492, min=5, max=15, vary=True)  # [cm]
    params.add('end_l_norm', value=10, vary=False)  # [cm]
    params.add('end_off', value=0, min=-5, max=5, vary=False)  # [cm]
    params.add('end_nfactor', value=0.1, vary=False)
    params.add('end_layers', value=2, vary=False)
    # Additional coils (solenoid shape)
    params.add('extra_coils', value=0, vary=False)  # number of extra coils you are adding
    # Example, increase 'name number' when adding new coils. Start with 0
    params.add('n_0', value=100, vary=False)  # [#]
    params.add('nfactor_0', value=1, vary=False)  # [#]
    params.add('i_0', value=75, vary=False)  # [A]
    params.add('r_0', value=20, vary=False)  # [cm]
    params.add('l_0', value=15, vary=False)  # [cm]
    params.add('l_norm_0', value=11, vary=False)  # [cm]
    params.add('x_off_0', value=0,
               vary=False)  # [cm]  x offset to center of main solenoid
    params.add('y_off_0', value=0,
               vary=False)  # [cm]  y offset to center of main solenoid
    params.add('z_off_0', value=0,
               vary=False)  # [cm]  z offset to center of main solenoid
    params.add('layers_0', value=5, vary=False)
    '''
    return params


if __name__ == "__main__":

    psc = True
    if psc:
        psc_params = get_psc_params()
        data_filename = "Datasets/field_map_data_fixed.dat"  # Include path and filetype (.dat etc)
        # coil_savefilename = ''    # Fill, if want to save file. Don't include filetype (.txt)
        # fmap_savefilename = ''    # Fill, if want to save file. Don't include filetype (.txt)
        # my_range = [,]      # Use to declare range on the z access in which to evaluate the field
        test_solenoid_plus_endcoils(psc_params, data_filename)
        # opt_my_coil(psc_params, data_filename, method='ampgo', disp=True)  # , disp=True)  # , max_nfev=100_000)
        # coil_plot(psc_params)
        plt.show()

    ben = False
    if ben:
        ben_params = get_ben_params()
        data_filename = "Datasets/ben_field_map_fix.dat"
        # coil_savefilename = ''    # Fill, if want to save file. Don't include filetype (.txt)
        # fmap_savefilename = ''    # Fill, if want to save file. Don't include filetype (.txt)
        # my_range      # Use to declare range on the z access in which to evaluate the field
        test_solenoid_plus_endcoils(ben_params, data_filename)
        # opt_my_coil(ben_params, data_filename, method='ampgo')  # , disp=True)
        # coil_plot(ben_params)
        plt.show()

    get_grid = False
    if get_grid:
        psc_params = get_psc_params()
        z_size = float(input('Enter size of box in z-direction: '))
        grid_box_sizes = (22, 22, z_size)  # [cm]
        z_start = float(input('Enter start of box on z-axis: '))
        grid_starting_point = (-11, -11, z_start)  # [cm]
        grid_resolution = float(input('Enter grid resolution in cm: '))  # [cm]
        split_step = float(input('Enter maximum difference between 2 points on the z-axis within one file in cm: '))
        # [cm] the maximum difference between 2 points on the z-axis within one file (div by resolution)
        save_foldername = input('Enter the destination folder name: ')
        save_filename = save_foldername + '/' + input('Enter prefix of the files you will generate: ')
        get_my_grid(params=psc_params,
                    box_size=grid_box_sizes,
                    start_point=grid_starting_point,
                    split_step=split_step,
                    volume_resolution=grid_resolution,
                    save_filename=save_filename,
                    save_foldername=save_foldername)

        plt.show()

    contour = False
    if contour:

        # # Psc magnet
        '''params = get_psc_params()
        # grid_box_sizes = (60, 0, 140)  # [cm]
        # grid_starting_point = (-35, 4.00047029285, -80)  # [cm]
        grid_box_sizes = (60, 0, 120)  # [cm]
        grid_starting_point = (-30, 0, -60)  # [cm]'''

        # Ben magnet
        params = get_ben_params()
        grid_box_sizes = (50, 0, 120)  # [cm]
        grid_starting_point = (-25, 0, -60)  # [cm]

        # [cm] the maximum difference between 2 points on the z-axis within one file (div by resolution)
        grid_resolution = 1  # [cm]
        fields, positions = get_my_grid(params=params,
                                        box_size=grid_box_sizes,
                                        start_point=grid_starting_point,
                                        volume_resolution=grid_resolution)

        my_contour_plot(fields, positions, level=grid_starting_point[1], which_plane='y')

        plt.show()

    calc_field = False
    if calc_field:
        # params = get_psc_params()
        params = get_ben_params()
        myShape = myShapes_adapted.Wire()
        myShape.create_my_wire_config(params)
        coil = myShape.coordz[:, :3]
        current = myShape.coordz[:, 3]

        # a = [4.00047029285, 4.00047029285, -43.5]
        # b = [-29.7212409083, 4.00047029285, -70.7032538226]
        a = [4.00047029285, 4.00047029285, -43.5]
        b = [-29.7212409083, 4.00047029285, -70.7032538226]
        positions = get_points(a, b, 0.2)

        fields = calc_my_fields(coil, current, positions)
        fields = np.linalg.norm(fields, axis=1)
        pos_and_field = np.concatenate((positions, fields[:, np.newaxis]), axis=1)

        # with open('field_along_lines/injection_line_psc.txt', 'w') as f:
        #     np.savetxt(f, pos_and_field, delimiter=', ')
        plt.rcParams['font.size'] = 14
        fig, ax = plt.subplots()
        ax.plot(np.linspace(0, 217/5, 217), fields)
        ax.set_xlabel('Distance [cm]')
        ax.set_ylabel('Magnetic field [T]')
        # ax.set_title('Magnetic field along injection line')
        plt.show()

    plot_along_lines = False
    if plot_along_lines:
        plt.rcParams['font.size'] = 14
        fig, ax = plt.subplots(figsize=(9, 6))
        for i in range(2):
            if i == 0:
                params = get_ben_params()
            else:
                params = get_psc_params()
            myShape = myShapes_adapted.Wire()
            myShape.create_my_wire_config(params)
            coil = myShape.coordz[:, :3]
            current = myShape.coordz[:, 3]

            a_0 = [0, 0, -50]
            b_0 = [0, 0, 50]
            a_32 = [0, 3.2, -50]
            b_32 = [0, 3.2, 50]
            positions_0 = get_points(a_0, b_0, 0.2)
            positions_32 = get_points(a_32, b_32, 0.2)
            fields_0 = calc_my_fields(coil, current, positions_0)
            fields_32 = calc_my_fields(coil, current, positions_32)
            fields_0 = np.linalg.norm(fields_0, axis=1)
            fields_32 = np.linalg.norm(fields_32, axis=1)
            if i == 0:
                col = 'green'
            else:
                col = 'blue'

            ax.plot(positions_0[:, 2], fields_0, color=col, linestyle='-')
            ax.plot(positions_32[:, 2], fields_32, color=col, linestyle='--')

        ax.set_xlabel('Z-position [cm]')
        ax.set_ylabel('Magnetic field [T]')
        # ax.set_title('')
        # ax.legend(['Ben, r=32mm', 'PSC, r=32mm'])
        ax.legend(['Ben, r=0mm', 'Ben, r=32mm', 'PSC, r=0mm', 'PSC, r=32mm'])
        # ax.set_xlim([-40, 40])
        ax.set_ylim([2.85, 2.95])
        plt.show()
