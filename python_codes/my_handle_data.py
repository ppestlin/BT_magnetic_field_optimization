# Standard imports
import numpy as np


# We try to keep most if not all data management within this file.
# At least tasks that need their own function

def read_data(fname: str, type=0) -> "tuple[np.ndarray, np.ndarray]":

    '''
    Read data, relatively specific to the project.
    '''

    x = np.empty(0)
    y = np.empty(0)
    z = np.empty(0)
    bx = np.empty(0)
    by = np.empty(0)
    bz = np.empty(0)

    if type == 1:
        loc_arr = np.array([2, 3, 4, 5, 6, 7])
    else:
        loc_arr = np.array([0, 1, 2, 3, 4, 5])

    with open(fname) as f:
        for i, line in enumerate(f):
            s = line.split()
            if s[0] == 'X,mm':
                continue
            x = np.append(x, s[loc_arr[0]])
            y = np.append(y, s[loc_arr[1]])
            z = np.append(z, s[loc_arr[2]])

            bx = np.append(bx, s[loc_arr[3]])
            by = np.append(by, s[loc_arr[4]])
            bz = np.append(bz, s[loc_arr[5]])

    x = x.astype(float)
    y = y.astype(float)
    z = z.astype(float)
    bx = bx.astype(float)
    by = by.astype(float)
    bz = bz.astype(float)
    positions = np.array([x, y, z])
    fields = np.array([bx, by, bz])

    return fields, positions


# This function only exists so we keep the data read out separate from the conversion to cm
def get_data(fname: str, type=0) -> "tuple[np.ndarray, np.ndarray]":

    """

    :param fname: Name of the file
    :param type: gives the structure of the with regard to fixed structures. There are:

    1:          Columns 0-1 are radius and angle, 2-4 are pos, 5-7 are field. We only read 2-7

    else:  Columns 0-2 are pos, 3-5 are pos

    :return: Field vectors in Tesla and positions in cm
    """

    data_field, data_pos = read_data(fname, type)

    return data_field, data_pos/10   # Convert positions from [mm] to [cm]


def inside_data(real_field, real_pos, pos_range) -> "tuple[np.ndarray, np.ndarray]":

    '''
    Returns data that is within a  certain interval on the x-axis

    :param real_field: magnetic field vector components
    :param real_pos: positions of the vector components
    :param pos_range: interval on z-axis. Data within this interval is considered and returned
    :return: field and position within the new interval
    '''

    x = np.empty(0)
    y = np.empty(0)
    z = np.empty(0)
    bx = np.empty(0)
    by = np.empty(0)
    bz = np.empty(0)

    for i, row in enumerate(real_pos.T):

        if pos_range[0] < row[2] < pos_range[1]:

            x = np.append(x, real_pos[0, i])
            y = np.append(y, real_pos[1, i])
            z = np.append(z, real_pos[2, i])

            bx = np.append(bx, real_field[0, i])
            by = np.append(by, real_field[1, i])
            bz = np.append(bz, real_field[2, i])

    real_pos_inside = np.array([x, y, z])
    real_field_inside = np.array([bx, by, bz])

    return real_field_inside, real_pos_inside


def slice_data(fields, real_field, real_pos) -> "tuple[list, list, list]":

    '''
    Slice up the data into lists of measurements with same z-coordinate. To do this for measured data only
    you can hand over the field twice (I know, its trivial)

    :param fields: Magnetic field Nr. 1 (for example simulated data)
    :param real_field: Magnetic field Nr. 2 (for example measured data)
    :param real_pos: Positions of the both magnetic field vectors
    :return: Lists of slices which share the same x- and y-coordinate and where measured consecutively
    '''

    slice_pos = []
    slice_real = []
    slice_sim = []
    current_pos = np.empty(1)
    current_real = np.empty(1)
    current_sim = np.empty(1)

    for i in range(np.shape(real_pos)[1]):

        if i == 0:
            current_pos = np.array(real_pos[:, 0], ndmin=2).T
            current_real = np.array(real_field[:, 0], ndmin=2).T
            current_sim = np.array(fields[:, 0], ndmin=2).T
        else:
            if abs(current_pos[0, 0] - real_pos[0, i]) > 0.01 or abs(current_pos[1, 0] - real_pos[1, i]) > 0.01:

                order = current_pos[2, :].argsort()
                current_pos = current_pos[:, order]
                current_real = current_real[:, order]
                current_sim = current_sim[:, order]

                slice_pos.append(current_pos)
                slice_real.append(current_real)
                slice_sim.append(current_sim)

                current_pos = np.array(real_pos[:, i], ndmin=2).T
                current_real = np.array(real_field[:, i], ndmin=2).T
                current_sim = np.array(fields[:, i], ndmin=2).T

            else:
                current_pos = np.append(current_pos, np.array(real_pos[:, i], ndmin=2).T, axis=1)
                current_real = np.append(current_real, np.array(real_field[:, i], ndmin=2).T, axis=1)
                current_sim = np.append(current_sim, np.array(fields[:, i], ndmin=2).T, axis=1)

    # Add last slice

    order = current_pos[2, :].argsort()
    current_pos = current_pos[:, order]
    current_real = current_real[:, order]
    current_sim = current_sim[:, order]

    slice_pos.append(current_pos)
    slice_real.append(current_real)
    slice_sim.append(current_sim)

    return slice_sim, slice_real, slice_pos


def order_and_err(real_field, real_pos) -> "tuple[np.ndarray, np.ndarray, np.ndarray]":

    '''
    The data is divided into parts with the same x- and y- coordinates and then ordered along
    the z-axis.
    We also calculate the field error using the gradient within the provided data.

    :param real_field: magnetic field vector components
    :param real_pos: positions of the vector components
    :return:
    '''
    # Orders the data for us
    _, sliced_fields, sliced_pos = slice_data(real_field, real_field, real_pos)

    x_gradient = np.empty(0)
    y_gradient = np.empty(0)
    z_gradient = np.empty(0)
    ordered_field = np.empty(0)
    ordered_pos = np.empty(0)

    for i, _ in enumerate(sliced_fields):
        temp_grad_x = np.gradient(sliced_fields[i][0, :],
                                  sliced_pos[i][2, :])
        temp_grad_y = np.gradient(sliced_fields[i][1, :],
                                  sliced_pos[i][2, :])
        temp_grad_z = np.gradient(sliced_fields[i][2, :],
                                  sliced_pos[i][2, :])
        if i == 0:
            x_gradient = temp_grad_x
            y_gradient = temp_grad_y
            z_gradient = temp_grad_z
            ordered_field = sliced_fields[i]
            ordered_pos = sliced_pos[i]
        else:
            x_gradient = np.append(x_gradient, temp_grad_x)
            y_gradient = np.append(y_gradient, temp_grad_y)
            z_gradient = np.append(z_gradient, temp_grad_z)
            ordered_field = np.append(ordered_field,
                                      sliced_fields[i],
                                      axis=1)
            ordered_pos = np.append(ordered_pos,
                                    sliced_pos[i],
                                    axis=1)

    field_err = np.empty((3, np.shape(z_gradient)[0]))

    # Currently hardcoded error parameters. This is very much experiment specific and needs to be adjusted on a case
    # to case basis. Systematic error commented out as it influences all measurements the same way

    field_err[0, :] = np.sqrt(np.square(np.abs(x_gradient) * 0.2) +
                              np.square(np.ones(np.shape(y_gradient)[0]) * 0.0006) / 12 +
                              np.square(np.ones(np.shape(y_gradient)[0]) * 0.0001) / 12)
    field_err[1, :] = np.sqrt(np.square(np.abs(y_gradient) * 0.2) +
                              np.square(np.ones(np.shape(y_gradient)[0]) * 0.0006) / 12 +
                              np.square(np.ones(np.shape(y_gradient)[0]) * 0.0001) / 12)
    field_err[2, :] = np.sqrt(np.square(np.abs(z_gradient) * 0.2) +
                              np.square(np.ones(np.shape(y_gradient)[0]) * 0.0006) / 12 +
                              np.square(np.ones(np.shape(y_gradient)[0]) * 0.0001) / 12)

    return ordered_field, ordered_pos, field_err
