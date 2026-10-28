#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Maintainer: zhaowenbo
"""
Visualize last 200 points of 4D chaotic trajectory with glass-sphere 3D rendering.
"""

import numpy as np
import matplotlib.pyplot as plt

from chaos.core.base_map import HighDimChaosMap
from chaos.visualization.plot_4d import (
    plot_4d_chaos_glass_spheres,
    plot_4d_chaos_simple_spheres,
)


def generate_last200_4d_points(params, word_size: int = 16,
                               total_steps: int = 5000,
                               last_points: int = 200, seed: int = 123):
    chaos_map = HighDimChaosMap(params=params, word_size=word_size, seed=seed)
    n = chaos_map.n
    if n < 4:
        raise ValueError(f"System dimension n={n} < 4, cannot create 4D visualization")

    traj = chaos_map.iterate(total_steps)
    if last_points > total_steps:
        raise ValueError("last_points cannot exceed total_steps")
    data_last = traj[-last_points:, :]
    data_4d = data_last[:, :4].copy()
    return data_4d, chaos_map.mod


def main():
    params = {
        "a_j1": np.array([371, 373, 375], dtype=np.int64),
        "a_jn": np.array([50, 48, 46], dtype=np.int64),
        "b": 111,
    }

    total_steps = 5000
    last_points = 200

    data_4d, mod = generate_last200_4d_points(
        params=params, word_size=16, total_steps=total_steps,
        last_points=last_points, seed=123)

    print(f"data_4d shape = {data_4d.shape}")
    print(f"x1 range: {data_4d[:, 0].min()} ~ {data_4d[:, 0].max()}")
    print(f"x2 range: {data_4d[:, 1].min()} ~ {data_4d[:, 1].max()}")
    print(f"x3 range: {data_4d[:, 2].min()} ~ {data_4d[:, 2].max()}")
    print(f"x4 range: {data_4d[:, 3].min()} ~ {data_4d[:, 3].max()}")

    fig1, ax1 = plot_4d_chaos_glass_spheres(
        data_4d=data_4d, title="", color_map="plasma", alpha=0.85,
        figsize=(10, 8), elev=25, azim=45,
        normalization_method="fixed", fixed_range=(0, mod - 1),
        sphere_quality=15)
    fig1.savefig("chaos_4d_glass_last200.eps", bbox_inches="tight", dpi=300)

    fig2, ax2 = plot_4d_chaos_simple_spheres(
        data_4d=data_4d, title="4D Chaos (scatter, last 200 of 5000)",
        color_map="viridis", alpha=0.85, figsize=(10, 8),
        elev=25, azim=45, sphere_size_factor=1.5)
    fig2.savefig("chaos_4d_scatter_last200.eps", bbox_inches="tight", dpi=300)

    plt.show()
    print("Done. Figures saved: chaos_4d_glass_last200.eps, chaos_4d_scatter_last200.eps")


if __name__ == "__main__":
    main()
