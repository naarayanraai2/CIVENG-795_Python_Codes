#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stability analysis for calibrated Wiedemann-99 car-following.

Provides:
- Equilibrium gap solving (optional)
- Finite-difference linearization around (s*, v*)
- Local stability (trace/determinant + eigenvalues)
- String stability:
    - Frequency-domain magnitude of transfer function |G(jw)|
    - Time-domain platoon simulation pulse response
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.optimize import fsolve

from models import CCParams, VehicleParams, accel_from_state, simulate_episode_wiedemann


def find_equilibrium_gap(v_target: float, cc: CCParams, veh: VehicleParams, L: float) -> float:
    """
    Solve for gap s* such that acceleration at (gap=s*, vF=vL=v_target) is ~0.
    """
    def func(g):
        return accel_from_state(cc, veh, gap=float(g[0]), vF=v_target, vL=v_target, L_lead=L)

    s_guess = 60.0
    return float(fsolve(func, [s_guess])[0])


def finite_diff_partials(
    cc: CCParams,
    veh: VehicleParams,
    s_star: float,
    v_star: float,
    L: float,
    eps_s: float = 0.5,
    eps_v: float = 0.25
):
    """
    Compute a0, a_s, a_v, a_vl at equilibrium by central finite differences.

      a_s  = ∂a/∂s
      a_v  = ∂a/∂vF
      a_vl = ∂a/∂vL
    """
    def a(gap, vF, vL):
        return accel_from_state(cc, veh, gap=gap, vF=vF, vL=vL, L_lead=L)

    a0 = a(s_star, v_star, v_star)
    a_s = (a(s_star + eps_s, v_star, v_star) - a(s_star - eps_s, v_star, v_star)) / (2.0 * eps_s)
    a_v = (a(s_star, v_star + eps_v, v_star) - a(s_star, v_star - eps_v, v_star)) / (2.0 * eps_v)
    a_vl = (a(s_star, v_star, v_star + eps_v) - a(s_star, v_star, v_star - eps_v)) / (2.0 * eps_v)

    return float(a0), float(a_s), float(a_v), float(a_vl)


def local_stability(a_s: float, a_v: float):
    """
    A = [[0, -1],
         [a_s, a_v]]

    Stable iff tr(A)<0 and det(A)>0 for this 2D linear system.
    """
    tr = a_v
    det = a_s
    return tr, det, (tr < 0 and det > 0)


def bode_string_stability(a_s: float, a_v: float, a_vl: float, w: np.ndarray) -> np.ndarray:
    """
    Transfer function magnitude |G(jw)| from linearized model.
    """
    s = 1j * w
    # One reasonable linearized form:
    # G(s) = (a_vl * s + a_s) / (s^2 - a_v*s + a_s)
    num = a_vl * s + a_s
    den = s**2 - a_v * s + a_s
    G = num / den
    return np.abs(G)


def platoon_simulation(
    cc: CCParams,
    veh: VehicleParams,
    n_veh: int = 30,
    T: float = 120.0,
    dt: float = 0.1,
    v_star: float = 25.0,
    gap_star: float = 60.0,
    L: float = 15.0,
    pulse_dv: float = 10.0,
    pulse_t0: float = 10.0,
    pulse_t1: float = 12.0
):
    """
    Time-domain platoon simulation with a speed pulse applied to the lead vehicle.
    Returns time grid and speeds for all vehicles.
    """
    t = np.arange(0.0, T + 1e-9, dt)
    nT = len(t)

    x = np.zeros((n_veh, nT), dtype=float)
    v = np.zeros((n_veh, nT), dtype=float)

    # init: evenly spaced
    v[:, 0] = v_star
    x[0, 0] = 0.0
    for i in range(1, n_veh):
        x[i, 0] = x[i - 1, 0] - (gap_star + L)

    for k in range(nT - 1):
        # leader speed pulse
        v_lead = v[0, k]
        if pulse_t0 <= t[k] <= pulse_t1:
            v_lead = v_star + pulse_dv

        # leader: simple kinematics (prescribed speed)
        v[0, k] = v_lead
        x[0, k + 1] = x[0, k] + v[0, k] * dt
        v[0, k + 1] = v_lead

        # followers
        for i in range(1, n_veh):
            gap = (x[i - 1, k] - x[i, k]) - L
            a_i = accel_from_state(cc, veh, gap=gap, vF=v[i, k], vL=v[i - 1, k], L_lead=L)
            v[i, k + 1] = max(0.0, v[i, k] + a_i * dt)
            x[i, k + 1] = x[i, k] + v[i, k] * dt

    return t, v, x


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cf-best", default="cf_best_params.csv", help="Best CF params CSV")
    ap.add_argument("--vstar", type=float, default=24.5, help="Equilibrium speed ft/s")
    ap.add_argument("--gapstar", type=float, default=60.0, help="Equilibrium gap ft (if not solving)")
    ap.add_argument("--solve-gap", action="store_true", help="Solve equilibrium gap so a≈0 at (s*, v*)")
    ap.add_argument("--out", default="stability.png", help="Output PNG")
    return ap.parse_args()


def main():
    args = parse_args()

    best_df = pd.read_csv(args.cf_best)
    CC0, CC1, CC2 = float(best_df["CC0"].iloc[0]), float(best_df["CC1"].iloc[0]), float(best_df["CC2"].iloc[0])
    cc = CCParams(CC0=CC0, CC1=CC1, CC2=CC2, CC3=8.0, CC4=-0.35, CC5=0.35, CC6=11.44, CC7=0.82, CC8=11.48, CC9=None)
    veh = VehicleParams(v_free=110.0, bmin=-15.0, tau=1.0, alpha=0.4)

    L = 15.3
    v_star = args.vstar

    if args.solve_gap:
        gap_star = find_equilibrium_gap(v_star, cc, veh, L)
        print(f"Solved equilibrium gap: {gap_star:.3f} ft")
    else:
        gap_star = args.gapstar

    a0, a_s, a_v, a_vl = finite_diff_partials(cc, veh, gap_star, v_star, L)
    tr, det, is_stable = local_stability(a_s, a_v)

    print("\n=== Local stability (linearized) ===")
    print(f"a0   = {a0:.6f}")
    print(f"a_s  = {a_s:.6f}")
    print(f"a_v  = {a_v:.6f}")
    print(f"a_vl = {a_vl:.6f}")
    print(f"trace(A)={tr:.6f}, det(A)={det:.6f}, stable={is_stable}")

    # Frequency-domain string stability
    w = np.logspace(-3, 1, 300)
    mag = bode_string_stability(a_s, a_v, a_vl, w)

    # Time-domain simulation
    t, v, x = platoon_simulation(cc, veh, n_veh=30, T=120.0, dt=0.1, v_star=v_star, gap_star=gap_star, L=L)

    # Simple plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    axes[0, 0].semilogx(w, mag)
    axes[0, 0].axhline(1.0, linestyle="--")
    axes[0, 0].set_title("String Stability: |G(jω)|")
    axes[0, 0].set_xlabel("ω (rad/s)")
    axes[0, 0].set_ylabel("|G|")
    axes[0, 0].grid(True, which="both", alpha=0.3)

    # Speed amplitude down platoon (max deviation)
    amp = np.max(np.abs(v - v_star), axis=1)
    axes[0, 1].plot(np.arange(len(amp)), amp, marker="o")
    axes[0, 1].set_title("Speed Amplitude Decay Down Platoon")
    axes[0, 1].set_xlabel("Vehicle index (0=leader)")
    axes[0, 1].set_ylabel("max |v - v*| (ft/s)")
    axes[0, 1].grid(True, alpha=0.3)

    # Show a few vehicle speed deviations
    for i in [1, 5, 10, 20, 29]:
        axes[1, 0].plot(t, v[i] - v_star, label=f"veh {i}")
    axes[1, 0].set_title("Velocity Deviations (selected vehicles)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("δv (ft/s)")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Trajectory lines (space-time)
    for i in range(v.shape[0]):
        axes[1, 1].plot(t, x[i], linewidth=0.8)
    axes[1, 1].set_title("Vehicle Trajectories (space-time)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Position (ft)")
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(args.out, dpi=200, bbox_inches="tight")
    print(f"Saved {args.out}")
    plt.show()


if __name__ == "__main__":
    main()
