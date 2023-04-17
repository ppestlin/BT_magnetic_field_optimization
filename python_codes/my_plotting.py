# Standard import
import numpy as np

# Plotting imports
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.widgets import Slider, Button
import matplotlib.patches as patches


# Project imports
import biot_savart as bs
import myShapes_adapted
from my_biot_savart import get_my_tv


# Plotting extension for biot_savart plotting.py


def update_errorbar(errobj, x, y, xerr=None, yerr=None):
    ln, caps, bars = errobj

    if len(bars) == 2:
        assert xerr is not None and yerr is not None, \
            "Your errorbar object has 2 dimension of error bars defined. You must provide xerr and yerr."
        barsx, barsy = bars  # bars always exist (?)
        try:  # caps are optional
            errx_top, errx_bot, erry_top, erry_bot = caps
        except ValueError:  # in case there is no caps
            pass

    elif len(bars) == 1:
        assert (xerr is     None and yerr is not None) or\
               (xerr is not None and yerr is     None),  \
               "Your errorbar object has 1 dimension of error bars defined. You must provide xerr or yerr."

        if xerr is not None:
            barsx, = bars  # bars always exist (?)
            try:
                errx_top, errx_bot = caps
            except ValueError:  # in case there is no caps
                pass
        else:
            barsy, = bars  # bars always exist (?)
            try:
                erry_top, erry_bot = caps
            except ValueError:  # in case there is no caps
                pass

    ln.set_data(x, y)

    try:
        errx_top.set_xdata(x + xerr)
        errx_bot.set_xdata(x - xerr)
        errx_top.set_ydata(y)
        errx_bot.set_ydata(y)
    except NameError:
        pass
    try:
        barsx.set_segments([
            np.array([[xt, y], [xb, y]])
            for xt, xb, y in zip(x + xerr, x - xerr, y)
        ])
    except NameError:
        pass

    try:
        erry_top.set_xdata(x)
        erry_bot.set_xdata(x)
        erry_top.set_ydata(y + yerr)
        erry_bot.set_ydata(y - yerr)
    except NameError:
        pass
    try:
        barsy.set_segments([
            np.array([[x, yt], [x, yb]])
            for x, yt, yb in zip(x, y + yerr, y - yerr)
        ])
    except NameError:
        pass


def coil_plot(params):
    '''
    Plots one or more coils in space.

    input_filenames: Name of the files containing the coils.
    Should be formatted appropriately.
    '''
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("$x$ (cm)")
    ax.set_ylabel("$y$ (cm)")
    ax.set_zlabel("$z$ (cm)")

    myShape = myShapes_adapted.Wire()
    myShape.create_my_wire_config(params)

    coil = myShape.coordz[:, :3]
    c1 = 0
    c2 = 0
    c3 = 0
    for i in range(np.shape(coil)[0]):
        if np.sqrt(np.square(coil[i, 0] - params['x_offset'].value) + np.square(coil[i, 1]-params['y_offset'].value)) < 13.5:
            if c1 == 0:
                coil_1 = np.array(coil[i, :], ndmin=2).T
                c1 = 1
            else:
                coil_1 = np.append(coil_1, np.array(coil[i, :], ndmin=2).T, axis=1)
        elif coil[i, 2] < 0:
            if c2 == 0:
                coil_2 = np.array(coil[i, :], ndmin=2).T
                c2 = 1
            else:
                coil_2 = np.append(coil_2, np.array(coil[i, :], ndmin=2).T, axis=1)
        else:
            if c3 == 0:
                coil_3 = np.array(coil[i, :], ndmin=2).T
                c3 = 1
            else:
                coil_3 = np.append(coil_3, np.array(coil[i, :], ndmin=2).T, axis=1)

    ax.plot3D(coil_1[0, :], coil_1[1, :],
              coil_1[2, :], lw=2, color='blue')
    ax.plot3D(coil_2[0, :], coil_2[1, :],
              coil_2[2, :], lw=2, color='red')
    ax.plot3D(coil_3[0, :], coil_3[1, :],
              coil_3[2, :], lw=2, color='red')

    return


def my_contour_plot(fields: np.ndarray,
                    positions: np.ndarray,
                    what_to_plot: str='abs',
                    which_plane='x',
                    level=0) -> None:

    if np.shape(positions)[1] != 3:
        positions = positions.T
        fields = fields.T

    # Choose coordinates and field vector components with respect to the normal access of the plane
    if which_plane == 'x':

        w_id = np.where(abs(positions[:, 0] - level) < 0.1)

        coord_1 = positions[w_id, 1]
        coord_2 = positions[w_id, 2]
        B_abs = np.linalg.norm(fields, axis=1)[w_id]
        B_z = fields[:, 2][w_id]

        coord_1_unique, id_1 = np.unique(coord_1, return_index=True)
        coord_2_unique, id_2 = np.unique(coord_2, return_index=True)

        M = coord_1_unique.size
        N = coord_2_unique.size

        B_abs = np.reshape(B_abs, (M, N))
        B_z = np.reshape(B_z, (M, N))

        for i in range(M):
            for j in range(N):

                if B_abs[i, j] > 3.2:
                    B_abs[i, j] = 3.2

                if np.abs(B_z[i, j]) > 3.2:
                    B_z[i, j] = 3.2

        label_1, label_2 = "y", "z"

    elif which_plane == 'y':
        w_id = np.where(abs(positions[:, 1] - level) < 0.1)

        coord_1 = positions[w_id, 0]
        coord_2 = positions[w_id, 2]
        B_abs = np.linalg.norm(fields, axis=1)[w_id]
        B_z = fields[:, 2][w_id]

        coord_1_unique, id_1 = np.unique(coord_1, return_index=True)
        coord_2_unique, id_2 = np.unique(coord_2, return_index=True)

        M = coord_1_unique.size
        N = coord_2_unique.size

        B_abs = np.reshape(B_abs, (M, N))
        B_z = np.reshape(B_z, (M, N))

        for i in range(M):
            for j in range(N):

                if B_abs[i, j] > 3.2:
                    B_abs[i, j] = 3.2

                if np.abs(B_z[i, j]) > 3.2:
                    B_z[i, j] = 3.2

        label_1, label_2 = "x", "z"

    else:
        w_id = np.where(abs(positions[:, 2] - level) < 0.1)

        coord_1 = positions[w_id, 0]
        coord_2 = positions[w_id, 1]
        B_abs = np.linalg.norm(fields, axis=1)[w_id]
        B_z = fields[:, 2][w_id]

        coord_1_unique, id_1 = np.unique(coord_1, return_index=True)
        coord_2_unique, id_2 = np.unique(coord_2, return_index=True)

        M = coord_1_unique.size
        N = coord_2_unique.size

        B_abs = np.reshape(B_abs, (M, N))
        B_z = np.reshape(B_z, (M, N))

        for i in range(M):
            for j in range(N):

                if B_abs[i, j] > 3.2:
                    B_abs[i, j] = 3.2

                if np.abs(B_z[i, j]) > 3.2:
                    B_z[i, j] = 3.2

        label_1, label_2 = "x", "y"

    plt.rcParams['font.size'] = 14
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9, 4))
    axes.set_xlabel(label_2 + " [cm]")
    axes.set_ylabel(label_1 + " [cm]")
    axes.set_aspect('equal', adjustable='box', anchor='C')

    # actual 2D plot
    if what_to_plot == 'abs':
        cp = axes.contourf(coord_2_unique, coord_1_unique, B_abs, levels=32)
        cb = plt.colorbar(cp)
        cb.set_label('B [T]')
        # fig.tight_layout()
    else:   # plot field in z-direction
        cp = axes.contourf(coord_1_unique, coord_2_unique, B_z.T, levels=32)
        cb = plt.colorbar(cp)
        cb.set_label('B [T]')
        plt.tight_layout()

    rect1_psc = patches.Rectangle((-82.5505734/2, -12.7550955), 82.5505734, 2*12.7550955, fill=False, color='white')
    rect2_psc = patches.Rectangle((-50, -10), 100, 20, fill=False, color='red')

    rect1_ben = patches.Rectangle((-52.3361633/2, -17.4264706), 52.3361633, 2*17.4264706, fill=False, color='white')
    rect2_ben = patches.Rectangle((-32.7, -15), 65.4, 30, fill=False, color='red')

    # my_line = np.array([[4.00047029285, -43.5], [-29.7212409083, -70.7032538226]])

    # axes.plot(my_line[:, 0], my_line[:, 1], color='orange')
    # axes.plot([3.2, 3.2], [-50, 50], color='white', linestyle='--')
    # axes.plot([0, 0], [-50, 50], color='white', linestyle='-')
    # Add the box to the plot
    # axes.add_patch(rect1_psc)
    # axes.add_patch(rect2_psc)

    # axes.add_patch(rect1_ben)
    # axes.add_patch(rect2_ben)

    return


def plot_simfields_2d(fields: np.ndarray,
                      positions: np.ndarray,
                      which_plane='x',
                      level=0,
                      shared_plt=None) -> None:
    # Ryan's Magic Code
    '''
    Plots the set of Bfields in the given region, at the specified resolutions.

    Bfields: A 4D array of the Bfield.
    box_size: (x, y, z) dimensions of the box in cm
    start_point: (x, y, z) = (0, 0, 0) = bottom left corner position of the box AKA the offset
    vol_resolution: Division of volumetric meshgrid (generate a point every volume_resolution cm)
    which_plane: Plane to plot on, can be "x", "y" or "z"
    level : The "height" of the plane. For instance the Z = 5 plane would have a level of 5
    num_contours: THe amount of contours on the contour plot.

    '''
    # filled contour plot of Bx, By, and Bz on a chosen slice plane

    # This function is used to only plot simulated data with a grid of positions
    X = positions[:, 0, 0, 0]
    Y = positions[0, :, 0, 1]
    Z = positions[0, 0, :, 2]

    if which_plane == 'x':

        converted_level = np.where(X >= level)

        B_sliced = [fields[converted_level[0][0], :, :, i].T for i in range(3)]
        B_1 = fields[converted_level[0][0], :, :, 1].T
        B_2 = fields[converted_level[0][0], :, :, 2].T
        x_label, y_label = "y", "z"
        x_array, y_array = Y, Z

    elif which_plane == 'y':
        converted_level = np.where(Y >= level)
        B_sliced = [fields[:, converted_level[0][0], :, i].T for i in range(3)]
        B_1 = fields[:, converted_level[0][0], :, 0].T
        B_2 = fields[:, converted_level[0][0], :, 2].T
        x_label, y_label = "x", "z"
        x_array, y_array = X, Z
    else:
        converted_level = np.where(Z >= level)
        B_sliced = [fields[:, :, converted_level[0][0], i].T for i in range(3)]
        B_1 = fields[:, :, converted_level[0][0], 0].T
        B_2 = fields[:, :, converted_level[0][0], 1].T
        x_label, y_label = "x", "y"
        x_array, y_array = X, Y

    Bmin, Bmax = np.amin(B_sliced), np.amax(B_sliced)

    component_labels = ['x', 'y', 'z']
    normalizer = Normalize(Bmin, Bmax)

    i = 0
    len_1 = np.zeros_like(B_1)
    len_2 = np.zeros_like(B_1)

    for line in B_1:
        len_1[i] = line - x_array
        len_2[i] = B_2[i, :] - y_array[i]
        i = i + 1

    color = np.sqrt(np.square(len_1) + np.square(len_2))

    norm = matplotlib.colors.Normalize()
    norm.autoscale(color)
    cm1 = matplotlib.cm.YlGnBu
    sm = matplotlib.cm.ScalarMappable(cmap=cm1, norm=norm)
    sm.set_array([])

    if shared_plt is None:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 7))
        axes.set_ylabel(y_label + " (cm)")
        axes.set_xlabel(x_label + " (cm)")
        col = axes.quiver(x_array, y_array, B_1, B_2, color)
        # plt.colorbar(sm, ax=0)
        plt.tight_layout()
        plt.show()

    else:
        shared_plt[1][0].set_ylabel(y_label + " (cm)")
        shared_plt[1][0].set_xlabel(x_label + " (cm)")
        col = shared_plt[1][0].quiver(x_array, y_array, B_1, B_2, color)
        # plt.colorbar(sm, ax=2)
        shared_plt[0].colorbar(col, ax=shared_plt[1][0])


def plot_fields_data_2d(fields: np.ndarray,
                        positions: np.ndarray,
                        which_plane='x',
                        level=0,
                        shared_plt=None,
                        measurement=False) -> None:
    # Used to plot non grid like data. Fields and positions are therefore 2D arrays.

    if np.shape(positions)[0] != 3:
        positions = positions.T
        fields = fields.T

    X = positions[0, :]
    Y = positions[1, :]
    Z = positions[2, :]

    # Initiate variables
    B_1 = np.empty(0)
    B_2 = np.empty(0)

    coord_1 = np.empty(0)
    coord_2 = np.empty(0)

    # Check the fields shape and adjust if it doesn't match (generated fields vs. data fields)
    if fields.ndim == 4:
        if fields.shape == (1, 1, fields.shape[2], 3):
            fields = np.squeeze(fields)
            fields = fields.T

    # Choose coordinates and field vector components with respect to the normal access of the plane
    if which_plane == 'x':
        for i, x in enumerate(X):

            if abs(x - level) < 1:
                B_1 = np.append(B_1, fields[1, i])  # Field in the y direction
                B_2 = np.append(B_2, fields[2, i])  # Field in the z direction

                coord_1 = np.append(coord_1, Y[i])  # Coordinates in y
                coord_2 = np.append(coord_2, Z[i])  # Coordinates in z

        label_1, label_2 = "y", "z"

    elif which_plane == 'y':
        for i, y in enumerate(Y):

            if abs(y - level) < 1:
                B_1 = np.append(B_1, fields[0, i])  # Field in the x direction
                B_2 = np.append(B_2, fields[2, i])  # Field in the z direction

                coord_1 = np.append(coord_1, X[i])  # Coordinates in x
                coord_2 = np.append(coord_2, Z[i])  # Coordinates in z

        label_1, label_2 = "x", "z"

    else:
        for i, z in enumerate(Z):

            if abs(z - level) < 10:
                B_1 = np.append(B_1, fields[0, i])  # Field in the x direction
                B_2 = np.append(B_2, fields[1, i])  # Field in the z direction

                coord_1 = np.append(coord_1, X[i])  # Coordinates in x
                coord_2 = np.append(coord_2, Y[i])  # Coordinates in z

        label_1, label_2 = "x", "y"

    # Color for each arrow according to the length (strength of the field)
    color = np.sqrt(np.square(B_1) + np.square(B_2))

    # Plot if No shared plot is handed over
    if shared_plt is None:
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(12, 7))
        axes.set_xlabel(label_1 + " (cm)")
        axes.set_ylabel(label_2 + " (cm)")
        # actual 2D plot
        axes.quiver(coord_1, coord_2, B_1, B_2, color)
        plt.tight_layout()
        # plt.colorbar(sm, ax=0)
        plt.show()

    # Plot if a shared plot layout is handed over. Plot layout needs to be 1x2
    else:
        shared_plt[1][measurement].set_xlabel(label_1 + " (cm)")
        shared_plt[1][measurement].set_ylabel(label_2 + " (cm)")
        col = shared_plt[1][measurement].quiver(coord_1, coord_2, B_1, B_2, color)
        if measurement == 0:
            shared_plt[1][measurement].title.set_text('Simulated Data')
        else:
            shared_plt[1][measurement].title.set_text('Recorded Data')

        shared_plt[0].colorbar(col, ax=shared_plt[1][measurement])

        # plt.colorbar(sm, ax=2)


def plot_z_field(fields, real_field, real_pos, field_err, Radius) -> None:

    slice_pos = []
    slice_real = []
    slice_sim = []
    slice_real_err = []
    current_pos = np.empty(1)
    current_real = np.empty(1)
    current_sim = np.empty(1)
    current_real_err = np.empty(1)

    for i in range(np.shape(real_pos)[1]):

        if i == 0:
            current_pos = np.array(real_pos[:, 0], ndmin=2).T
            current_real = np.array(real_field[:, 0], ndmin=2).T
            current_sim = np.array(fields[:, 0], ndmin=2).T
            current_real_err = np.array(field_err[:, 0], ndmin=2).T
        else:
            if abs(current_pos[0, 0] -
                   real_pos[0, i]) > 0.01 or abs(current_pos[1, 0] -
                                                 real_pos[1, i]) > 0.01:

                order = current_pos[2, :].argsort()
                current_pos = current_pos[:, order]
                current_real = current_real[:, order]
                current_sim = current_sim[:, order]
                current_real_err = current_real_err[:, order]

                slice_pos.append(current_pos)
                slice_real.append(current_real)
                slice_sim.append(current_sim)
                slice_real_err.append(current_real_err)

                current_pos = np.array(real_pos[:, i], ndmin=2).T
                current_real = np.array(real_field[:, i], ndmin=2).T
                current_sim = np.array(fields[:, i], ndmin=2).T
                current_real_err = np.array(field_err[:, i], ndmin=2).T

            else:
                current_pos = np.append(current_pos,
                                        np.array(real_pos[:, i], ndmin=2).T,
                                        axis=1)
                current_real = np.append(current_real,
                                         np.array(real_field[:, i], ndmin=2).T,
                                         axis=1)
                current_sim = np.append(current_sim,
                                        np.array(fields[:, i], ndmin=2).T,
                                        axis=1)
                current_real_err = np.append(current_real_err,
                                             np.array(field_err[:, i],
                                                      ndmin=2).T,
                                             axis=1)

    # Add last slice

    order = current_pos[2, :].argsort()
    current_pos = current_pos[:, order]
    current_real = current_real[:, order]
    current_sim = current_sim[:, order]
    current_real_err = current_real_err[:, order]

    slice_pos.append(current_pos)
    slice_real.append(current_real)
    slice_sim.append(current_sim)
    slice_real_err.append(current_real_err)

    init_slice = 0

    fig, ax = plt.subplots()
    real_plt = ax.errorbar(slice_pos[init_slice][2, :],
                           slice_real[init_slice][2, :],
                           yerr=slice_real_err[init_slice][2, :],
                           fmt='o',
                           c='orange',
                           markersize=3,
                           label='Measured data')
    sim_plt, = ax.plot(slice_pos[init_slice][2, :],
                       slice_sim[init_slice][2, :],
                       marker='o',
                       label='Simulated data')

    ax.set_xlabel('z [cm]')
    ax.set_ylabel('Field [T]')
    ax.legend(handles=[real_plt, sim_plt])
    ax.set_title('Z - Field')

    fig.subplots_adjust(left=0.25)
    fig.subplots_adjust(bottom=0.2)

    # Make a vertically oriented slider to control the amplitude
    ax_slice = fig.add_axes([0.1, 0.225, 0.05, 0.63])
    ax_ax = fig.add_axes([0.325, 0.05, 0.5, 0.05])
    slice_slider = Slider(ax=ax_slice,
                          label="slice",
                          valmin=0,
                          valmax=len(slice_pos) - 1,
                          valstep=1,
                          valinit=init_slice,
                          orientation="vertical",
                          initcolor='none')

    axis_slider = Slider(ax=ax_ax,
                         label="Axis",
                         valmin=0,
                         valmax=2,
                         valstep=1,
                         valinit=2,
                         orientation="horizontal",
                         initcolor='none')

    # Make a small plot which shows the current position in x and y
    ax_info = fig.add_axes([0.4, 0.3, 0.18, 0.18], box_aspect=1)
    ax_info.grid(visible=True)

    theta = np.linspace(0, 2 * np.pi, 150)

    a = Radius * np.cos(theta)
    b = Radius * np.sin(theta)

    circle_plt, = ax_info.plot(a, b)
    info_plt, = ax_info.plot(slice_pos[init_slice][0, 0],
                             slice_pos[init_slice][1, 0],
                             marker='o',
                             markersize=3)

    # The function to be called anytime a slider's value changes
    def update(slice, axis):
        if axis == 2:
            ax.set_title('Z - Field')
            # ax.relim()
            # ax.autoscale_view()
            ax.set_ylim([0, 3.2])
        elif axis == 1:
            ax.set_title('Y - Field')
            ax.set_ylim([-1, 1])
        else:
            ax.set_title('X - Field')
            ax.set_ylim([-1, 1])

        update_errorbar(real_plt,
                        slice_pos[slice][2, :],
                        slice_real[slice][axis, :],
                        yerr=slice_real_err[slice][axis, :])
        # real_plt.set_data(slice_pos[slice_slider.val][2, :],
        #                   slice_real[slice_slider.val][2, :])
        sim_plt.set_data(slice_pos[slice][2, :],
                         slice_sim[slice][axis, :])
        info_plt.set_data(slice_pos[slice][0, 0],
                          slice_pos[slice][1, 0])
        fig.canvas.draw_idle()

    # register the update function with each slider
    slice_slider.on_changed(lambda val: update(val, axis_slider.val))
    axis_slider.on_changed(lambda val: update(slice_slider.val, val))

    # Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
    reset_ax = fig.add_axes([0.075, 0.05, 0.1, 0.04])
    button = Button(reset_ax, 'Reset', hovercolor='0.975')

    def reset(event):
        slice_slider.reset()
        axis_slider.reset()

    button.on_clicked(reset)


def plot_after_opt(params, real_pos, real_field, field_err,
                   coil_savefilename: str = '',
                   fmap_savefilename: str = ''):

    myShape = myShapes_adapted.Wire()
    myShape.create_my_wire_config(params)

    if not (coil_savefilename == ''):
        with open(coil_savefilename + '.txt', 'w') as f:
            np.savetxt(f, myShape.coordz, fmt='%1.4f', delimiter=",")

    sim_real_pos = np.transpose(real_pos[:, np.newaxis, np.newaxis, :],
                                axes=(1, 2, 3, 0))
    # bs.plot_coil("actual_coil.txt")

    fields, positions = get_my_tv(myShape.coordz, sim_real_pos, fmap_savefilename)
    sim_fields = np.squeeze(fields)
    sim_fields = sim_fields.T

    # plot_fields_data_2d(fields,
    #                     real_pos,
    #                     which_plane='x',
    #                     level=0,
    #                     shared_plt=[fig, axes])
    # # shared_plt = None -> single plot

    # plot_fields_data_2d(real_field,
    #                     real_pos,
    #                     which_plane='x',
    #                     level=0,
    #                     shared_plt=[fig, axes])
    # # shared_plt = None -> single plot

    # plt.show()

    plot_z_field(sim_fields, real_field, real_pos, field_err, Radius=10)
