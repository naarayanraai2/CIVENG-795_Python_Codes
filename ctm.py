#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# Unit helpers
# -------------------------
FTPS_TO_MPH = 0.681818  # 1 ft/s = 0.681818 mph
FT_TO_M = 0.3048

# ============================================================
# 1) CTM core
# ============================================================

@dataclass
class CTMParams:
    vf: float       # ft/s
    w: float        # ft/s
    kjam: float     # veh/ft/lane
    qmax: float     # veh/s/lane
    dx: float       # ft
    dt: float       # s
    n_cells: int


def triangular_from_cc(cc0_ft: float, cc1_s: float, vf: float, veh_len_ft: float):
    """
    Very simple mapping:
      jam spacing ~ CC0 + vehicle length
      qmax ~ 1/CC1 (veh/s) as a crude headway->capacity mapping
    """
    s_jam = max(cc0_ft + veh_len_ft, 1.0)
    kjam = 1.0 / s_jam
    qmax = 1.0 / max(cc1_s, 0.1)       # veh/s
    kc = qmax / max(vf, 1e-6)
    w = qmax / max((kjam - kc), 1e-9)
    return kjam, qmax, kc, w


def ctm_simulate(params: CTMParams, T_end: float, q_in_func, q_out_cap=None):
    nT = int(np.floor(T_end / params.dt)) + 1
    ts = np.arange(0.0, nT * params.dt, params.dt)

    k = np.zeros((params.n_cells, nT), dtype=float)          # veh/ft
    q = np.zeros((params.n_cells + 1, nT), dtype=float)      # veh/s at boundaries

    for t in range(nT - 1):
        S = np.minimum(params.vf * k[:, t], params.qmax)                  # veh/s
        R = np.minimum(params.qmax, params.w * (params.kjam - k[:, t]))   # veh/s

        q_in = float(q_in_func(ts[t]))
        q[0, t] = min(q_in, R[0])

        for i in range(params.n_cells - 1):
            q[i + 1, t] = min(S[i], R[i + 1])

        out_cap = params.qmax if q_out_cap is None else float(q_out_cap)
        q[params.n_cells, t] = min(S[params.n_cells - 1], out_cap)

        for i in range(params.n_cells):
            k[i, t + 1] = k[i, t] + (params.dt / params.dx) * (q[i, t] - q[i + 1, t])
            k[i, t + 1] = float(np.clip(k[i, t + 1], 0.0, params.kjam))

    return ts, k, q


# ============================================================
# 2) Derived fields (speed) + plots
# ============================================================

def ctm_speed_from_kq(k: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    v(i,t) = q_out(i,t) / k(i,t)
    returns v_fts: (n_cells, nT-1) ft/s
    """
    n_cells, nT = k.shape
    q_out = q[1:n_cells + 1, :-1]       # (n_cells, nT-1)
    k_mid = k[:, :-1]                  # (n_cells, nT-1)
    v_fts = q_out / np.maximum(k_mid, eps)
    return v_fts


def plot_timespace_speed(ax, ts: np.ndarray, v_fts: np.ndarray, params: CTMParams,
                         vmax_mph: float = 70.0):
    """
    Time–space heatmap of speed (mph).
    FIXED pcolormesh dimensions using time edges and space edges.
    """
    v_mph = v_fts * FTPS_TO_MPH                  # (n_cells, nT-1)
    v_plot = v_mph.T                             # (nT-1, n_cells)

    # space edges (n_cells+1)
    x_edges_m = (np.arange(params.n_cells + 1) * params.dx) * FT_TO_M

    # time edges (nT) so that (nT-1, n_cells) is valid for pcolormesh
    # Example: ts is length nT, and v_plot has nT-1 rows -> perfect
    t_edges = ts  # len = nT

    # Build 2D edge grids: (nT, n_cells+1)
    X, T = np.meshgrid(x_edges_m, t_edges)

    pcm = ax.pcolormesh(X, T, v_plot, shading="flat", vmin=0.0, vmax=vmax_mph)
    cb = plt.colorbar(pcm, ax=ax)
    cb.set_label("Speed (mph)")

    ax.set_xlabel("Space x (m)")
    ax.set_ylabel("Time (s)")
    ax.set_title("Time–Space Diagram: Speed (mph)")


def plot_fundamental_diagram(ax, k: np.ndarray, q: np.ndarray, params: CTMParams,
                            n_bins: int = 25, sample: int = 80000,
                            density_units: str = "veh/mi/lane",
                            flow_units: str = "veh/hr/lane",
                            overlay_triangular: bool = True):
    """
    Fundamental diagram: scatter + binned mean curve.
    """
    n_cells, nT = k.shape
    K = k[:, :-1].ravel()
    Q = q[1:n_cells + 1, :-1].ravel()

    if sample is not None and len(K) > sample:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(K), size=sample, replace=False)
        K = K[idx]
        Q = Q[idx]

    if density_units == "veh/mi/lane":
        Kp = K * 5280.0
        k_label = "Density k (veh/mi/lane)"
    else:
        Kp = K
        k_label = "Density k (veh/ft/lane)"

    if flow_units == "veh/hr/lane":
        Qp = Q * 3600.0
        q_label = "Flow q (veh/hr/lane)"
    else:
        Qp = Q
        q_label = "Flow q (veh/s/lane)"

    # scatter
    ax.scatter(Kp, Qp, s=4, alpha=0.15)

    # binned mean line (red like your screenshot)
    bins = np.linspace(Kp.min(), Kp.max(), n_bins + 1)
    bin_idx = np.digitize(Kp, bins) - 1

    k_cent, q_mean = [], []
    for b in range(n_bins):
        mask = (bin_idx == b)
        if mask.sum() < 60:
            continue
        k_cent.append(0.5 * (bins[b] + bins[b + 1]))
        q_mean.append(Qp[mask].mean())

    if len(k_cent) > 0:
        ax.plot(k_cent, q_mean, color="red", linewidth=2)

    # optional triangular overlay
    if overlay_triangular:
        k_grid = np.linspace(0.0, params.kjam, 400)  # veh/ft
        q_tri = np.minimum(params.vf * k_grid, params.w * (params.kjam - k_grid))  # veh/s

        if density_units == "veh/mi/lane":
            k_grid_p = k_grid * 5280.0
        else:
            k_grid_p = k_grid

        if flow_units == "veh/hr/lane":
            q_tri_p = q_tri * 3600.0
        else:
            q_tri_p = q_tri

        ax.plot(k_grid_p, q_tri_p, linewidth=2)

    ax.set_xlabel(k_label)
    ax.set_ylabel(q_label)
    ax.set_title("Fundamental Diagram: Flow vs Density")
    ax.grid(True, alpha=0.3)


def plot_ctm_two_panel(ts: np.ndarray, k: np.ndarray, q: np.ndarray, params: CTMParams,
                       vmax_mph: float = 70.0):
    """
    One figure with the two plots (like your screenshot).
    """
    v_fts = ctm_speed_from_kq(k, q)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    plot_timespace_speed(ax1, ts, v_fts, params, vmax_mph=vmax_mph)
    plot_fundamental_diagram(ax2, k, q, params, n_bins=25, sample=80000,
                             density_units="veh/mi/lane", flow_units="veh/hr/lane",
                             overlay_triangular=True)

    fig.suptitle("Cellular Transmission Model Development", fontsize=22, y=1.03)
    plt.tight_layout()
    plt.show()


# ============================================================
# 3) Main
# ============================================================

def main():
    cf = pd.read_csv("cf_best_params.csv")
    CC0 = float(cf["CC0"].iloc[0])
    CC1 = float(cf["CC1"].iloc[0])

    vf = 95.0  # ft/s (~65 mph) - replace with calibrated free speed if you have it
    veh_len_ft = 15.0

    kjam, qmax, kc, w = triangular_from_cc(CC0, CC1, vf, veh_len_ft)

    print("Derived FD params:")
    print(f"  kjam={kjam:.6f} veh/ft ({kjam*5280:.1f} veh/mi)")
    print(f"  qmax={qmax:.3f} veh/s  ({qmax*3600:.0f} veh/hr)")
    print(f"  kc={kc:.6f} veh/ft    ({kc*5280:.1f} veh/mi)")
    print(f"  w={w:.2f} ft/s")

    # stability-ish requirement: dt <= dx/vf
    p = CTMParams(vf=vf, w=w, kjam=kjam, qmax=qmax, dx=264.0, dt=1.0, n_cells=20)

    def q_in(t):
        # veh/s
        if t < 300:
            return 1800 / 3600
        if t < 600:
            return 2400 / 3600
        return 2000 / 3600

    ts, k, q = ctm_simulate(p, T_end=900.0, q_in_func=q_in, q_out_cap=None)
    plot_ctm_two_panel(ts, k, q, p, vmax_mph=70.0)


if __name__ == "__main__":
    main()
