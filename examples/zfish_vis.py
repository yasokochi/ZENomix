# This code is used to plot the zfish data in 3D.
# The code is adapted from the spaOTsc package "Cang, Z., Nie, Q. 
# Inferring spatial and signaling relationships between cells from single cell transcriptomic data. 
# Nat Commun 11, 2084 (2020). https://doi.org/10.1038/s41467-020-15968-5".

import numpy as np
from scipy import interpolate
from scipy.interpolate import griddata
from matplotlib import cm
import matplotlib.pyplot as plt


def plot_zf_plain(pts, f, title, vmin, vmax, cmap='viridis', elev=3, azim=0):
    f = np.array(f)
    n_theta = 100
    n_phi = 50
    r = 1
    theta, phi = np.mgrid[-0.5*np.pi:0.5 *
                          np.pi:n_theta*1j, 0.0:0.5*np.pi:n_phi*1j]
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)
    phi_data = np.arccos(pts[:, 2])
    theta_data = np.arcsin(pts[:, 0]/np.sin(phi_data))
    pts_angle = np.concatenate(
        (theta_data.reshape(-1, 1), phi_data.reshape(-1, 1)), axis=1)
    grid_c_nearest = griddata(pts_angle, f, (theta, phi), method='nearest')
    colors = (grid_c_nearest - vmin)/(vmax - vmin)
    colors = np.where(colors < 0, 0, colors)
    colors = np.where(colors > 1, 1, colors)
    x = np.vstack([x, -x])
    y = np.vstack([y, y])
    z = np.vstack([z, z])
    colors = np.vstack([colors, colors])
    fig = plt.figure(figsize=(plt.figaspect(1.)))
    ax = fig.add_subplot(111, projection='3d')
    eval('ax.plot_surface(x, y, z, facecolors=cm.' + cmap +
         '(colors), shade=False, rstride=1, cstride=1)')
    ax.set_box_aspect((4, 4, 2))
    ax.set_title(title, fontsize=30, fontname='arial', y=0.9)
    ax.axis("off")
    ax.view_init(elev=elev, azim=azim)


def get_grid_color(pts, f, vmin, vmax):
    f = np.array(f)
    n_theta = 100
    n_phi = 50
    r = 1
    theta, phi = np.mgrid[-0.5*np.pi:0.5 *
                          np.pi:n_theta*1j, 0.0:0.5*np.pi:n_phi*1j]
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)
    phi_data = np.arccos(pts[:, 2])
    theta_data = np.arcsin(pts[:, 0]/np.sin(phi_data))
    pts_angle = np.concatenate(
        (theta_data.reshape(-1, 1), phi_data.reshape(-1, 1)), axis=1)
    grid_c_nearest = griddata(pts_angle, f, (theta, phi), method='nearest')
    colors = (grid_c_nearest - vmin)/(vmax - vmin)
    colors = np.where(colors < 0, 0, colors)
    colors = np.where(colors > 1, 1, colors)
    x = np.vstack([x, -x])
    y = np.vstack([y, y])
    z = np.vstack([z, z])
    colors = np.vstack([colors, colors])
    return (x, y, z), colors


def plot_zf_interpolate(pts, f, title_text, vmin, vmax, cmap='viridis', elev=3, azim=0):
    f = np.array(f)
    n_theta = 100
    n_phi = 50
    r = 1
    theta, phi = np.mgrid[-0.5*np.pi:0.5 *
                          np.pi:n_theta*1j, 0.0:0.5*np.pi:n_phi*1j]
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)

    phi_data = np.arccos(pts[:, 2])
    theta_data = np.arcsin(pts[:, 0]/np.sin(phi_data))
    pts_angle = np.concatenate(
        (theta_data.reshape(-1, 1), phi_data.reshape(-1, 1)), axis=1)
    phi_grid = phi_data.reshape(8, 8)
    theta_grid = theta_data.reshape(8, 8)
    f_grid = f.reshape(8, 8)
    f_grid = (f_grid - vmin)/(vmax - vmin)
    tck = interpolate.bisplrep(theta_grid, phi_grid, f_grid)
    fnew = interpolate.bisplev(theta[:, 0], phi[0, :], tck)

    x = np.vstack([x, -x])
    y = np.vstack([y, y])
    z = np.vstack([z, z])
    fnew = np.vstack([fnew, fnew])

    fig = plt.figure(figsize=plt.figaspect(0.4))
    ax = fig.add_subplot(111, projection='3d')
    eval('ax.plot_surface(x,y,z,facecolors=cm.' + cmap + '(fnew), shade =False)')
    ax.set_title(title_text, fontsize=36, y=0.9, x=0.5, fontname='arial')
    ax.set_box_aspect((4, 4, 2))
    ax.view_init(elev=elev, azim=azim)
    plt.axis('off')


def get_grid_color_interpolate(pts, f, vmin, vmax):
    f = np.array(f)
    n_theta = 100
    n_phi = 50
    r = 1
    theta, phi = np.mgrid[-0.5*np.pi:0.5 *
                          np.pi:n_theta*1j, 0.0:0.5*np.pi:n_phi*1j]
    x = r*np.sin(phi)*np.cos(theta)
    y = r*np.sin(phi)*np.sin(theta)
    z = r*np.cos(phi)

    phi_data = np.arccos(pts[:, 2])
    theta_data = np.arcsin(pts[:, 0]/np.sin(phi_data))
    pts_angle = np.concatenate(
        (theta_data.reshape(-1, 1), phi_data.reshape(-1, 1)), axis=1)
    phi_grid = phi_data.reshape(8, 8)
    theta_grid = theta_data.reshape(8, 8)
    f_grid = f.reshape(8, 8)
    f_grid = (f_grid - vmin)/(vmax - vmin)
    tck = interpolate.bisplrep(theta_grid, phi_grid, f_grid)
    fnew = interpolate.bisplev(theta[:, 0], phi[0, :], tck)

    x = np.vstack([x, -x])
    y = np.vstack([y, y])
    z = np.vstack([z, z])
    fnew = np.vstack([fnew, fnew])

    return (x, y, z), fnew
