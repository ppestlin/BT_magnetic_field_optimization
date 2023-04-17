'''
Simple Biot Savart Solver for arbitrarily shaped wires
Here a simple Wire builder
Copyright (C) 2012  Antonio Franco (antonio_franco@live.it)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.

ppestlin:
This is an adapted form of the initial file, especially adding
solenoids as well as solenoids with shimming coils and the ability
to add an arbitrary amount of solenoid like shaped wires.
'''

from numpy import array, pi, cos, sin, r_, linspace, zeros, concatenate, append, full
import numpy as np
import cmath
from lmfit import Parameters


class Wire:
    """
    Implements an arbitrary shaped wire
    """
    coordz = []
    '''Coordinates of the vertex of the wire in the form [X,Y,Z]
    I = complex(1, 0) Complex current carried by the wire'''

    def __init__(self):
        """By default initited as a toroidal coil with
            R1 = 10
            R2 = 1
            N = 100
            step = 0.001
            and current 1A with 0 phase
        """
        R1 = 10
        R2 = 1
        N = 100
        step = 0.001
        self.Create_Toroidal_Coil(R1, R2, N, step)
        self.Set_Current(1, 0)
        return


    def Set_Current(self, modulus, angle):
        """Sets current with absolute value modulus and phase angle (in radians)"""
        self.I = cmath.rect(modulus, angle)
        return

    def Create_Toroidal_Coil(self, R1, R2, N, step):
        """
        Create_Toroidal_Coil( R1 , R2 , N , step )
        Creates a toroidal coil of major radius R1, minor radius R2 with N
         turns and a step step
         Initiates coordz
        """
        a = R1
        b = R2
        c = N

        t = r_[0:2 * pi:step]

        X = (a + b * sin(c * t)) * cos(t);
        Y = (a + b * sin(c * t)) * sin(t);
        Z = b * cos(c * t);

        self.coordz = [X, Y, Z]

        return

    # Not used
    def Create_Solenoid(self, R, N, L, step, start, I):
        """
        Create_Solenoid(self, R , N , l , step )
        Creates a solenoid whose length is l with radius R, N turns with step
        steps along the z axis and a current I
        """
        a = R;
        b = L / (2 * pi * N);
        T = L / b;

        t = r_[0:T:step]

        X = a * cos(t) + start[0]
        Y = a * sin(t) + start[1]
        Z = b * t + start[2] - L/2
        C = full(np.shape(X), I)

        self.coordz = np.array([X, Y, Z, C]).transpose()
        return

    def create_my_wire_config(self, params):
        """
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
        :return:
        """

        # Initiate the variables for our
        N = params['sol_N'].value * params['sol_Nfactor'].value / params['sol_layers'].value
        L = params['sol_L'].value
        step = params['sol_step'].value
        start = np.array([params['x_offset'].value, params['y_offset'].value, params['sol_start_z'].value])
        Layers = params['sol_layers'].value
        I = params['sol_I'].value / params['sol_Nfactor'].value * (params['sol_L'].value / params['sol_L_norm'].value)
        n = params['end_n'].value * params['end_nfactor'].value / params['end_layers'].value
        l = params['end_l'].value
        i = params['end_i'].value / params['end_nfactor'].value * (params['end_l'].value / params['end_l_norm'].value)
        off = params['end_off'].value
        layers = params['end_layers'].value

        # Large solenoid coil
        large = np.empty(1)
        for j in range(Layers):
            R = params['sol_R'].value + j * 0.1
            A = L / (2 * pi * N)
            B = L / A
            T = r_[0:B:step]
            X = R * cos(T + 2 * pi * Layers / (j + 1)) + start[0]
            Y = R * sin(T + 2 * pi * Layers / (j + 1)) + start[1]
            Z = A * T + start[2] - L/2
            C = full(np.shape(X), I)
            C[-1] = 0
            if j == 0:
                large = np.array([X, Y, Z, C]).transpose()
            else:
                large = np.concatenate((large, np.array([X, Y, Z, C]).transpose()))

        # Shiming (end) coils
        end_1 = np.empty(1)
        end_2 = np.empty(1)
        for j in range(layers):
            r = params['end_r'].value + j * 0.1
            a = l / (2 * pi * n)
            b = l / a
            t = r_[0:b:step]
            x = r * cos(t + 2 * pi * layers / (j + 1)) + start[0]
            y = r * sin(t + 2 * pi * layers / (j + 1)) + start[1]
            z1 = a * t + (start[2] - L/2) + off
            z2 = a * t + (start[2] - L/2) + (L - l) - off
            c = full(np.shape(x), i)
            c[-1] = 0
            if j == 0:
                end_1 = np.array([x, y, z1, c]).transpose()
                end_2 = np.array([x, y, z2, c]).transpose()
            else:
                end_1 = np.concatenate((end_1, np.array([x, y, z1, c]).transpose()))
                end_2 = np.concatenate((end_2, np.array([x, y, z2, c]).transpose()))

        # Combine the solenoid and the shimming coils
        myCoordz = np.concatenate((end_1, large))
        myCoordz = np.concatenate((myCoordz, end_2))

        # Extra coils
        for j in range(round(params['extra_coils'].value)):
            # Read
            j_n = params['n_' + str(j)].value * params['nfactor_' + str(j)].value / params['layers_' + str(j)].value
            j_l = params['l_' + str(j)].value
            j_layers = params['layers_' + str(j)].value
            j_i = params['i_' + str(j)].value / params['nfactor_' + str(j)].value * (
                        params['l_' + str(j)].value / params['l_norm_' + str(j)].value)
            j_x_off = params['x_off_' + str(j)].value
            j_y_off = params['y_off_' + str(j)].value
            j_z_off = params['z_off_' + str(j)].value
            # Convert
            j_a = j_l / (2 * pi * j_n)
            j_b = j_l / j_a
            j_t = r_[0:j_b:step]

            j_extra = np.empty(1)
            # Layer per layer
            for jj in range(j_layers):
                j_r = params['r_' + str(j)].value + jj * 0.1   # Adding 1mm every layer, might want to adjust
                j_x = j_r * cos(j_t + 2 * pi * layers / (jj + 1)) + start[0] + j_x_off
                j_y = j_r * sin(j_t + 2 * pi * layers / (jj + 1)) + start[1] + j_y_off
                j_z = j_a * j_t + start[2] + j_z_off
                j_c = full(np.shape(j_x), j_i)
                j_c[-1] = 0
                if jj == 0:
                    j_extra = np.array([j_x, j_y, j_z, j_c]).transpose()
                else:
                    j_extra = np.concatenate((j_extra, np.array([j_x, j_y, j_z, j_c]).transpose()))

            # Add the new wire to the rest
            if j_layers > 0:
                myCoordz = np.concatenate((myCoordz, j_extra))


        self.coordz = myCoordz

        return


    # Not used
    def Create_Loop(self, center, radius, NOP, Orientation='xy'):
        """
        Create_Loop(self,center,radius,NOP)
        a circle with center defined as
        a vector CENTER, radius as a scaler RADIS. NOP is 
        the number of points on the circle.
        """
        t = linspace(0, 2 * pi, NOP)

        if Orientation == 'xy':
            X = center[0] + radius * sin(t)
            Y = center[1] + radius * cos(t)
            Z = zeros(NOP)
        elif Orientation == 'xz':
            X = center[0] + radius * sin(t)
            Z = center[1] + radius * cos(t)
            Y = zeros(NOP)
        elif Orientation == 'yz':
            Y = center[0] + radius * sin(t)
            Z = center[1] + radius * cos(t)
            X = zeros(NOP)

        self.coordz = [X, Y, Z]
        return

    # Not used
    def AugmentWire(self, Theta, Phi, length, Origin=None):
        """
        AugmentWire(self,Theta,Phi,length,Origin=None)
        augments the existing wire by a segment lenght long, starting from point
        Origin, with inclination Theta and Azimuth Phi. If origin = None then the last
        calculated vertex is used
        """

        # If an origin is not specified, the last vertex is assumed as origin
        if not Origin is None:
            newWire = self.__Create_Wire(Origin, Theta, Phi, length)
        else:
            temp = array(self.coordz)
            newOrigin = temp[:, 1]
            newWire = self.__Create_Wire(newOrigin, Theta, Phi, length)

        # If no coordinates are present, then we simply put the new vertices in the list
        if len(self.coordz) == 0:
            self.coordz = newWire
        elif Origin != None:
            X = concatenate((self.coordz[0], newWire[0]), axis=1)
            Y = concatenate((self.coordz[1], newWire[1]), axis=1)
            Z = concatenate((self.coordz[2], newWire[2]), axis=1)
            self.coordz = [X, Y, Z]
        else:
            X = append(self.coordz[0], newWire[0][1])
            Y = append(self.coordz[1], newWire[1][1])
            Z = append(self.coordz[2], newWire[2][1])
            self.coordz = [X, Y, Z]

        return

    def __Create_Wire(self, Origin, Theta, Phi, length):
        """
        Create_Wire(self,Origin,Theta,Phi,length)
        creates a single wire lenght long, starting from point
        Origin, with inclination Theta and Azimuth Phi and
        returns its coordinates in the form [X,Y,Z]
        """

        # Computes the unit vector
        ux = cos(Phi) * sin(Theta)
        uy = sin(Phi) * sin(Theta)
        uz = cos(Theta)

        u = array([ux, uy, uz])

        # Computes the second vertex
        P2 = Origin + length * u

        X = array([Origin[0], P2[0]])
        Y = array([Origin[1], P2[1]])
        Z = array([Origin[2], P2[2]])

        return [X, Y, Z]

    # Not used
    def plotme(self, ax=None):
        """Plots itself. Optional axis argument, otherwise new axes are created
        inactive until ShowPlots is called"""
        import pylab as p
        import mpl_toolkits.mplot3d.axes3d as p3

        X = self.coordz[0]
        Y = self.coordz[1]
        Z = self.coordz[2]

        if ax is None:
            fig = p.figure(None)
            ax1 = p3.Axes3D(fig)
        else:
            ax1 = ax

        ax1.plot(X, Y, Z)
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_zlabel('Z')

        self.axis_equal(ax1)

        p.draw()

        return ax1

    # Not used
    def axis_equal(self, ax):
        """Makes axis the same size"""
        X = self.coordz[0]
        Y = self.coordz[1]
        Z = self.coordz[2]

        Xmax = max(X)
        Ymax = max(Y)
        Zmax = max(Z)

        maxx = max(Xmax, Ymax, Zmax)

        ax.set_xlim3d(min(X), maxx)
        ax.set_ylim3d(min(Y), maxx)
        ax.set_zlim3d(min(Z), maxx)
        return

    # Not used
    def ShowPlots(self):
        """Triggers pylab.show()"""
        import pylab as p

        p.show()
        return
