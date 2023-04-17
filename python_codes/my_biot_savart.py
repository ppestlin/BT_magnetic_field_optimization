import os
import numpy as np

from scipy.integrate import simpson
import biot_savart as bs
import myShapes_adapted
import math


# This code is mostly adapted from the package biot_savart.
# With this adaptation we added several new features.
#
# 1 Positions can be given as a grid (np.meshgrid()) or as
#   a 2D array to allow field calculations measured positions
#
# 2 This field of every position is calculated separately
#   in a for loop to avoid memory overflow (see calc_my_fields())
#
# 3 Files are only written if demanded by the user and not
#   read on every call. In general np.arrays of positions
#   and fields are passed
#
# 4 With the function get_my_grid(), one is able to generate
#   large grids, even spread across multiple files


def point_line_vector(point, line_start, line_end):
    point = np.asarray(point)
    line_start = np.asarray(line_start)
    line_end = np.asarray(line_end)
    line_vector = line_end - line_start
    hyp_vector_1 = point - line_start
    hyp_vector_2 = point - line_end

    # Left overs to calculate smallest vector between point and vector. Probably not needed again
    # unit_line_vector = line_vector / np.linalg.norm(line_vector, axis=1)[:, np.newaxis]
    # hyp_vector_1_scaled = hyp_vector_1 / np.linalg.norm(line_vector, axis=1)[:, np.newaxis]
    # t = np.clip(np.einsum('ij,ij->i', unit_line_vector, hyp_vector_1_scaled), 0, 1)
    # nearest = line_vector * t[:, np.newaxis] + line_start
    # vector_connection = nearest - point

    projection = np.sum(hyp_vector_1 * line_vector, axis=-1) / np.sum(line_vector ** 2, axis=-1)
    projection_point = line_start + projection[..., np.newaxis] * line_vector
    vector_rejection = projection_point - point
    return vector_rejection, np.linalg.norm(hyp_vector_1, axis=-1), np.linalg.norm(hyp_vector_2, axis=-1)


def calc_my_fields(coil: np.ndarray, current: np.ndarray,
                    positions: np.ndarray) -> np.ndarray:
    '''

    :param: coil: endpoints of coil segments. Of shape (N, 3) -> N-1 segments
    :param: current: Current of the respective segment. Shape (N,), last entry is ignored
    :param: positions: Positions for which the field should be calculated. Can have shape (X, Y, Z, 3) or (X*Y*Z, 3)
    :return: B-Field vector of shape (X*Y*Z, 3) in units of T

    Calculate the magnetic field for a given coil at given positions.
    positions should be given in units of cm
    '''
    FACTOR = 1e-5  # mu_0 / 4pi when lengths are in cm, and B-field is in T, current in A
    midpoints = (coil[1:] + coil[:-1])/2  # midpoints of each coil position (needed for V1)
    dl = np.diff(coil, axis=0)  # dl row vectors for each segment
    dl_norm = np.linalg.norm(dl, axis=-1)

    positions = positions.reshape(-1, 3)
    results = np.empty((np.shape(positions)[0], 3))

    for i in range(np.shape(positions)[0]):
        # V1:
        '''
        # Here, for the new axis, remember we have a difference to all midpoints for every point.
        # This means we have all differences for one point stored in the 'new axis dimension'
        R_Rdash = np.subtract(positions[i, np.newaxis, :], midpoints)
        mags = np.linalg.norm(R_Rdash[:, :], axis=-1)
        # This new axis we need for multiplication. Have not found a better explenation yet
        elemental_integrands = FACTOR * (current[:-1] / mags[:]**3)[:, np.newaxis] * np.cross(dl, R_Rdash[:, :])
        # Evaluate the integrand using BSL
        # BSL is current * mu/4pi * dl x (R-R') / |R-R'|^3
        # The "area" underneath each rectangle
        results[i, :] = np.sum(elemental_integrands[:, :], axis=-2)
        '''

        # V2:

        # This version was only implemented by the end of the project. It uses the analytical
        # solution of a finite straight wire. Approximations therefore only arise from the
        # step size

        a, hyp_start, hyp_end = point_line_vector(positions[i, :], coil[:-1], coil[1:])

        a_norm = np.linalg.norm(a, axis=-1)
        field_vec = np.cross(a, dl, axis=1)
        field_dir = field_vec / np.linalg.norm(field_vec, axis=-1)[:, np.newaxis]
        cos_1 = (- hyp_end**2 + hyp_start**2 + dl_norm**2) / (2 * hyp_start * dl_norm)
        cos_2 = (- hyp_start**2 + hyp_end**2 + dl_norm**2) / (2 * hyp_end * dl_norm)
        elemental_integrands = FACTOR * (current[:-1] / a_norm) * (cos_1 + cos_2)
        results[i, :] = np.sum(field_dir * elemental_integrands[:, np.newaxis], axis=0)

    return results


def produce_my_tv(coil: np.ndarray, current: np.ndarray,
                          sim_real_pos: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":

    '''
    :parameter: coil: endpoints of coil segments. Of shape (N, 3) -> N-1 segments
    :param: current: Current of the respective segment. Shape (N,), last entry is ignored
    :param: positions: Positions for which the field should be calculated. Can have shape (X, Y, Z, 3) or (X*Y*Z, 3)
    :return: tuple of [fields, positions] with fields in units of T and positions in units of cm
    '''

    positions = sim_real_pos
    return calc_my_fields(coil, current, positions), positions


def get_my_tv(coil_and_current, sim_real_pos: np.ndarray,
              save_filename: str = '') -> "tuple[np.ndarray, np.ndarray]":
    '''
    Takes a coil specified in coil_and_current, generates a target volume, and saves the generated target volume to
    save_filename if needed.
    '''

    coil = coil_and_current[:, :3]
    current = coil_and_current[:, 3]
    fields, positions = produce_my_tv(
        coil, current, sim_real_pos)

    positions = positions.reshape(-1, 3)

    pos_and_field = np.concatenate((positions, fields), axis=1)

    if not (save_filename == ''):
        with open(save_filename + '.txt', 'w') as f:
            np.savetxt(f, pos_and_field, delimiter='  ')

    return fields, positions


def get_my_grid(params, box_size: tuple, start_point: tuple,
                volume_resolution: float = 1,
                split_step: float = -1,
                save_filename: str = '',
                save_foldername: str = 'my_grid_files') -> tuple[np.ndarray, np.ndarray]:
    '''
    Takes a coil specified in coil_and_current, generates a grid like target volume, and saves the generated
    target volume to save_filename if needed.

    box_size: (x, y, z) dimensions of the box in cm
    start_point: (x, y, z) = (0, 0, 0) = bottom left corner position of the box AKA the offset
    volume_resolution: Division of volumetric meshgrid (generate a point every volume_resolution cm)
    '''

    if split_step == -1:
        split_step = box_size[2]

    os.makedirs(save_foldername, exist_ok=True)

    myShape = myShapes_adapted.Wire()
    myShape.create_my_wire_config(params)

    coil = myShape.coordz[:, :3]
    current = myShape.coordz[:, 3]

    res_mm = volume_resolution * 10

    for i in range(math.ceil(box_size[2]/split_step)):

        this_box = (box_size[0], box_size[1], split_step)
        this_startpoint = (start_point[0], start_point[1],
                           round(start_point[2] + i * split_step, 10))

        header = f'Grid Output Min: [{this_startpoint[0]*10}mm {this_startpoint[1]*10}mm {this_startpoint[2]*10}mm]' \
                 f' Max: [{(this_startpoint[0]+this_box[0])*10}mm {(this_startpoint[1]+this_box[1])*10}mm ' \
                 f'{(this_startpoint[2]+this_box[2])*10}mm] Grid Size: [{res_mm}mm {res_mm}mm {res_mm}mm] \n' \
                 f'X, Y, Z, Vector data "Smooth(B_Vector)”'

        print('Started calculations for slice ' + str(i + 1) + ' of ' + str(math.ceil(box_size[2] / split_step)))

        positions = bs.generate_positions(this_box, this_startpoint, volume_resolution)

        fields, positions = produce_my_tv(coil, current, positions)

        fields = fields.reshape(-1, 3)

        positions = positions.reshape(-1, 3) / 100  # convert from cm to m for output

        pos_and_field = np.concatenate((positions, fields), axis=1)

        if not(save_filename == ''):
            print('Writing slice number ' + str(i+1) + ' of ' + str(math.ceil(box_size[2]/split_step)))
            with open(save_filename + '_' + str(round(start_point[2] + i * split_step, 4)) + '.txt', 'w') as f:
                np.savetxt(f, pos_and_field, header=header, delimiter='  ', comments='')

    return fields, positions * 100
