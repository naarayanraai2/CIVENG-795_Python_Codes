#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cell Transmission Model (CTM) for a simple freeway + on-ramp merge network.

This script is designed as a macroscopic counterpart to the calibrated microscopic behavior.

Methodology consistency:
- You may *derive* macroscopic parameters from calibrated micro parameters conceptually,
  but in code we use a triangular FD (vf, w, kjam, qmax) with reasonable defaults and allow
  overriding via CLI. (This keeps the CTM stable and reproducible.)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class CTMParams:
    vf: float      # free-flow speed (m/s)
    w: float       # congestion wave speed (m/s)
    kjam: float    # jam density (veh/m/lane)
    qmax: float    # capacity (veh/s/lane)


def triangular_fd_flow(k: np.ndarray, params: CTMParams) -> np.ndarray:
    """q(k) = min(vf*k, w*(kjam - k))"""
    return np.minimum(params.vf * k, params.w * (params.kjam - k))


def ctm_simulate(
    params: CTMParams,
    lanes: int,
    dx: float,
    dt: float,
    n_cells_main: int,
    n_cells_ramp: int,
    T: float,
    q_in_main: float,   # veh/s total (all lanes)
    q_in_ramp: float,   # veh/s total (ramp, assumed 1 lane)
    p_main: float = 0.7,
    p_ramp: float = 0.3
):
    """
    Simple merge at the last cell:
      - mainline: n_cells_main cells
      - ramp: n_cells_ramp cells
      - merge into downstream boundary

    Returns:
      k_main (nT x n_cells_main), v_main (nT x n_cells_main), q_main (nT x n_cells_main)
    """
    nT = int(np.floor(T / dt)) + 1

    # vehicle counts per cell per lane
    n_main = np.zeros((nT, n_cells_main), dtype=float)
    n_ramp = np.zeros((nT, n_cells_ramp), dtype=float)

    # Helper conversion: count n -> density k
    def density(n):
        return n / dx  # veh/m/lane

    for t in range(nT - 1):
        k_main = density(n_main[t])
        k_ramp = density(n_ramp[t])

        # demand/supply per lane
        D_main = np.minimum(params.vf * k_main, params.qmax)  # veh/s/lane
        S_main = np.minimum(params.w * (params.kjam - k_main), params.qmax)

        D_ramp = np.minimum(params.vf * k_ramp, params.qmax)
        S_ramp = np.minimum(params.w * (params.kjam - k_ramp), params.qmax)

        # boundary inflows (convert total -> per lane)
        q_in_main_lane = (q_in_main / max(lanes, 1))
        q_in_ramp_lane = q_in_ramp  # ramp assumed single lane

        # mainline sending/receiving between cells
        y_main = np.zeros(n_cells_main + 1, dtype=float)  # flows across edges, per lane
        y_ramp = np.zeros(n_cells_ramp + 1, dtype=float)

        # upstream boundaries
        y_main[0] = min(q_in_main_lane, S_main[0])
        y_ramp[0] = min(q_in_ramp_lane, S_ramp[0])

        # internal cell boundaries
        for i in range(n_cells_main - 1):
            y_main[i + 1] = min(D_main[i], S_main[i + 1])

        for i in range(n_cells_ramp - 1):
            y_ramp[i + 1] = min(D_ramp[i], S_ramp[i + 1])

        # merge at downstream boundary
        # available receiving of downstream "virtual" sink
        S_down = params.qmax

        send_main = D_main[-1]
        send_ramp = D_ramp[-1]

        # priority split
        alloc_main = p_main * S_down
        alloc_ramp = p_ramp * S_down

        y_main[-1] = min(send_main, alloc_main)
        y_ramp[-1] = min(send_ramp, alloc_ramp)

        # if unused capacity remains, let the other use it
        used = y_main[-1] + y_ramp[-1]
        spare = max(0.0, S_down - used)
        if spare > 0:
            # give to main first then ramp (arbitrary, but stable)
            extra_main = min(spare, send_main - y_main[-1])
            y_main[-1] += extra_main
            spare -= extra_main

            extra_ramp = min(spare, send_ramp - y_ramp[-1])
            y_ramp[-1] += extra_ramp

        # conservation update: n(t+1) = n(t) + (in - out)*dt
        for i in range(n_cells_main):
            n_main[t + 1, i] = n_main[t, i] + (y_main[i] - y_main[i + 1]) * dt

        for i in range(n_cells_ramp):
            n_ramp[t + 1, i] = n_ramp[t, i] + (y_ramp[i] - y_ramp[i + 1]) * dt

        # clamp
        n_main[t + 1] = np.clip(n_main[t + 1], 0.0, params.kjam * dx)
        n_ramp[t + 1] = np.clip(n_ramp[t + 1], 0.0, params.kjam * dx)

    # outputs for mainline
    k_main = n_main / dx  # veh/m/lane
    q_main = triangular_fd_flow(k_main, params)  # per lane
    v_main = np.where(k_main > 1e-9, q_main / k_main, params.vf)

    return k_main, v_main, q_main


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=float, default=3600.0, help="Simulation horizon (s)")
    ap.add_argument("--dx", type=float, default=25.0, help="Cell length (m)")
    ap.add_argument("--dt", type=float, default=0.8, help="Time step (s)")
    ap.add_argument("--lanes", type=int, default=6, help="Number of mainline lanes")
    ap.add_argument("--main-cells", type=int, default=20, help="Mainline cells")
    ap.add_argument("--ramp-cells", type=int, default=6, help="Ramp cells")
    ap.add_argument("--qmain", type=float, default=1200.0, help="Mainline inflow (veh/hr total)")
    ap.add_argument("--qramp", type=float, default=400.0, help="Ramp inflow (veh/hr)")
    ap.add_argument("--vf", type=float, default=29.0, help="Free-flow speed (m/s) ~ 65 mph")
    ap.add_argument("--w", type=float, default=6.0, help="Congestion wave speed (m/s)")
    ap.add_argument("--kjam", type=float, default=0.13, help="Jam density (veh/m/lane)")
    ap.add_argument("--qmax", type=float, default=1980.0, help="Capacity (veh/hr/lane)")
    ap.add_argument("--out", default="ctm.png", help="Output PNG")
    return ap.parse_args()


def main():
    args = parse_args()

    params = CTMParams(
        vf=float(args.vf),
        w=float(args.w),
        kjam=float(args.kjam),
        qmax=float(args.qmax) / 3600.0
    )

    k_main, v_main, q_main = ctm_simulate(
        params=params,
        lanes=args.lanes,
        dx=args.dx,
        dt=args.dt,
        n_cells_main=args.main_cells,
        n_cells_ramp=args.ramp_cells,
        T=args.T,
        q_in_main=float(args.qmain) / 3600.0,
        q_in_ramp=float(args.qramp) / 3600.0,
        p_main=0.7,
        p_ramp=0.3
    )

    # time-space plot (speed)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    im = axes[0].imshow(
        v_main.T, aspect="auto", origin="lower",
        extent=[0, args.T/60.0, 0, args.main_cells],
    )
    axes[0].set_title("CTM Time–Space Diagram (Speed)")
    axes[0].set_xlabel("Time (min)")
    axes[0].set_ylabel("Cell index")
    fig.colorbar(im, ax=axes[0], label="m/s")

    # fundamental diagram
    k_grid = np.linspace(0.0, params.kjam, 400)
    q_tri = np.minimum(params.vf * k_grid, params.w * (params.kjam - k_grid))
    axes[1].plot(k_grid, q_tri * 3600.0, linewidth=2)
    axes[1].set_title("Triangular Fundamental Diagram")
    axes[1].set_xlabel("Density k (veh/m/lane)")
    axes[1].set_ylabel("Flow q (veh/hr/lane)")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
