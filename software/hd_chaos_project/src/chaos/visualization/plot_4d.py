#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
4D chaotic system visualization with 3D glass-sphere rendering.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

plt.style.use("default")
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["savefig.facecolor"] = "white"


def _setup_glass_axes(ax):
    """Configure semi-transparent 'frosted glass' axes for 3D plots."""
    fig = ax.get_figure()
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for axis_line in (ax.w_xaxis.line, ax.w_yaxis.line, ax.w_zaxis.line):
        axis_line.set_color((0.2, 0.2, 0.2, 1.0))
        axis_line.set_linewidth(4)

    ax.grid(False)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    ax.set_xlabel(r'$x_{1}$', color='#333333', fontsize=12, labelpad=10)
    ax.set_ylabel(r'$x_{2}$', color='#333333', fontsize=12, labelpad=10)
    ax.set_zlabel(r'$x_{3}$', color='#333333', fontsize=12, labelpad=10)

    for axis in ['x', 'y', 'z']:
        ax.tick_params(axis=axis, colors='#333333', labelsize=10)

    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = True
        pane.set_facecolor((0.9, 0.9, 0.95, 0.28))
        pane.set_edgecolor((1.0, 1.0, 1.0, 0.6))
        pane.set_linewidth(1.5)

    ax.w_xaxis.line.set_color("white")
    ax.w_yaxis.line.set_color("white")
    ax.w_zaxis.line.set_color("white")
    for line in (ax.w_xaxis.line, ax.w_yaxis.line, ax.w_zaxis.line):
        line.set_linewidth(1.5)


def _generate_sphere_surface(radius, quality=20):
    u = np.linspace(0, 2 * np.pi, quality)
    v = np.linspace(0, np.pi, quality)
    x = radius * np.outer(np.cos(u), np.sin(v))
    y = radius * np.outer(np.sin(u), np.sin(v))
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v))
    return x, y, z


def plot_4d_chaos_glass_spheres(data_4d, title="4D Chaos Visualization",
                                color_map='viridis', alpha=0.8,
                                figsize=(14, 12), elev=25, azim=45,
                                normalization_method='auto',
                                fixed_range=None, sphere_quality=20):
    """High-quality 3D sphere rendering of 4D chaotic data.

    First 3 dimensions are spatial coordinates, 4th dimension is color.
    """
    if data_4d.shape[1] != 4:
        raise ValueError("Input data must be 4D, shape (N, 4)")

    xyz = data_4d[:, :3]
    color_values = data_4d[:, 3]

    if normalization_method == 'fixed':
        if fixed_range is None:
            fixed_range = (0, 65535)
        xyz_normalized = (xyz - fixed_range[0]) / (fixed_range[1] - fixed_range[0])
        xyz_normalized = np.clip(xyz_normalized, 0, 1)
    elif normalization_method == 'percentile':
        def _percentile_normalize(data, low=1, high=99):
            p_low = np.percentile(data, low)
            p_high = np.percentile(data, high)
            return np.clip((data - p_low) / (p_high - p_low), 0, 1)
        xyz_normalized = np.column_stack([
            _percentile_normalize(xyz[:, 0]),
            _percentile_normalize(xyz[:, 1]),
            _percentile_normalize(xyz[:, 2])
        ])
    else:
        xyz_normalized = (xyz - xyz.min(axis=0)) / (xyz.max(axis=0) - xyz.min(axis=0))

    n_points = len(data_4d)
    if n_points <= 100:
        sphere_size = 0.03
    elif n_points <= 500:
        sphere_size = 0.02
    elif n_points <= 1000:
        sphere_size = 0.015
    else:
        sphere_size = 0.01

    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    ax.view_init(elev=elev, azim=azim)

    cmap = plt.get_cmap(color_map)
    norm = Normalize(vmin=color_values.min(), vmax=color_values.max())

    print("Rendering 3D spheres...")
    for i in range(len(data_4d)):
        if i % 50 == 0:
            print(f"  progress: {i}/{len(data_4d)}")

        x, y, z = xyz_normalized[i]
        color_rgba = cmap(norm(color_values[i]))

        sphere_x, sphere_y, sphere_z = _generate_sphere_surface(sphere_size, sphere_quality)
        sphere_x += x
        sphere_y += y
        sphere_z += z

        ax.plot_surface(sphere_x, sphere_y, sphere_z,
                        color=color_rgba, alpha=alpha,
                        shade=True, antialiased=True,
                        linewidth=0, rstride=1, cstride=1)

    _setup_glass_axes(ax)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.1, location='right')
    cbar.set_label(r'$x_4$', rotation=0, labelpad=20,
                   color=(0.2, 0.2, 0.2, 1.0), fontsize=12)
    cbar.ax.yaxis.set_tick_params(color=(0.2, 0.2, 0.2, 1.0),
                                  labelcolor=(0.2, 0.2, 0.2, 1.0))
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
    ax.set_title(title, fontsize=16, pad=20, color='white', fontweight='bold')

    plt.tight_layout()
    return fig, ax


def plot_4d_chaos_simple_spheres(data_4d, title="",
                                 color_map='viridis', alpha=0.8,
                                 figsize=(14, 12), elev=25, azim=45,
                                 sphere_size_factor=1.0):
    """Simplified scatter-based 3D visualization (faster)."""
    if data_4d.shape[1] != 4:
        raise ValueError("Input data must be 4D, shape (N, 4)")

    xyz = data_4d[:, :3]
    color_values = data_4d[:, 3]
    xyz_normalized = (xyz - xyz.min(axis=0)) / (xyz.max(axis=0) - xyz.min(axis=0))

    n_points = len(data_4d)
    base_size = 50 * sphere_size_factor
    if n_points <= 100:
        marker_size = base_size * 3
    elif n_points <= 500:
        marker_size = base_size * 2
    elif n_points <= 1000:
        marker_size = base_size
    else:
        marker_size = base_size * 0.5

    fig = plt.figure(figsize=figsize, facecolor='black')
    ax = fig.add_subplot(111, projection='3d', facecolor='black')
    ax.view_init(elev=elev, azim=azim)

    scatter = ax.scatter(
        xyz_normalized[:, 0], xyz_normalized[:, 1], xyz_normalized[:, 2],
        c=color_values, cmap=color_map, s=marker_size,
        alpha=alpha, edgecolors='white', linewidths=0.5, depthshade=True)

    _setup_glass_axes(ax)

    cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label(r'$x_4$', rotation=270, labelpad=20, color='white', fontsize=12)
    cbar.ax.yaxis.set_tick_params(color='white', labelcolor='white')

    ax.set_title(title, fontsize=16, pad=20, color='white', fontweight='bold')
    info_text = f": {n_points}"
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes,
              bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
              fontsize=10, color='black')

    plt.tight_layout()
    return fig, ax
