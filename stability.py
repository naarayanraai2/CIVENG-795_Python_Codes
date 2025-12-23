#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from gettext import find
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import CCParams, VehicleParams, accel_from_wiedemann_state


# ============================================================
# Helpers: numerical linearization around equilibrium
# ============================================================

from scipy.optimize import fsolve
def find_equilibrium_gap(v_target, cc, veh, L):
    func = lambda gap: accel_from_wiedemann_state(cc, veh, gap[0], v_target, v_target, L)
    s_guess = 100.0
    return fsolve(func, s_guess)[0]

def finite_diff_partials(cc: CCParams, veh: VehicleParams,
                         s_star: float, v_star: float, L: float,
                         eps_s: float = 0.5,
                         eps_v: float = 0.25):
    """
    Compute (a_s, a_v, a_vl) at equilibrium by central finite differences:

      a_s  = ∂a/∂s
      a_v  = ∂a/∂vF
      a_vl = ∂a/∂vL

    At equilibrium: vF = vL = v_star, gap = s_star.
    """
    def a(gap, vF, vL):
        return accel_from_wiedemann_state(cc, veh, gap=gap, vF=vF, vL=vL, L_lead=L)

    a0 = a(s_star, v_star, v_star)
    if abs(a0) > 1e-2:
        print("\nWARNING: Not at equilibrium (a0 not near 0).")
        print("Your chosen (s*, v*) is not a fixed point for the model in its current regime.")
        print("Stability results will be misleading until you pick/solve an equilibrium.\n")


    # central differences
    a_s  = (a(s_star + eps_s, v_star, v_star) - a(s_star - eps_s, v_star, v_star)) / (2.0 * eps_s)
    a_v  = (a(s_star, v_star + eps_v, v_star) - a(s_star, v_star - eps_v, v_star)) / (2.0 * eps_v)
    a_vl = (a(s_star, v_star, v_star + eps_v) - a(s_star, v_star, v_star - eps_v)) / (2.0 * eps_v)

    return float(a0), float(a_s), float(a_v), float(a_vl)


def local_stability_report(a_s: float, a_v: float):
    """
    Lecture result:
      A = [[0, -1],
           [a_s, a_v]]

    For 2x2 real matrix: stable iff tr(A)<0 and det(A)>0
      tr(A)=a_v
      det(A)=a_s
    :contentReference[oaicite:4]{index=4}
    """
    A = np.array([[0.0, -1.0],
                  [a_s,  a_v]], dtype=float)
    eig = np.linalg.eigvals(A)

    tr = float(np.trace(A))
    det = float(np.linalg.det(A))

    local_ok = (tr < 0.0) and (det > 0.0)

    # more restrictive (monotonic decay): a_v^2 >= 4 a_s, plus local conditions
    # :contentReference[oaicite:5]{index=5}
    monotonic_ok = local_ok and ((a_v * a_v) >= (4.0 * a_s))

    return A, eig, tr, det, local_ok, monotonic_ok


# ============================================================
# String stability: lecture transfer function + Bode
# ============================================================

def gain_mag_G(a_s: float, a_v: float, a_vl: float, w: np.ndarray) -> np.ndarray:
    """
    Lecture transfer function:
      G(s) = (a_s + a_vl s) / (s^2 - a_v s + a_s)
    Substitute s = j w and compute |G(jw)|
    :contentReference[oaicite:6]{index=6}
    """
    jw = 1j * w
    num = a_s + a_vl * jw
    den = (jw**2) - a_v * jw + a_s
    G = num / den
    return np.abs(G)


def string_stability_checks(a_s: float, a_v: float, a_vl: float):
    """
    Lecture condition (sufficient form shown in slides):
      a_v^2 - a_vl^2 >= 2 a_s
    :contentReference[oaicite:7]{index=7}
    """
    lhs = (a_v * a_v) - (a_vl * a_vl)
    rhs = 2.0 * a_s
    return float(lhs), float(rhs), (lhs >= rhs)


def plot_bode_magnitude(a_s: float, a_v: float, a_vl: float,
                        w_min: float = 1e-3, w_max: float = 10.0, n: int = 800):
    """
    Bode magnitude plot: MdB = 20 log10 |G(jw)|
    0 dB line corresponds to |G|=1
    :contentReference[oaicite:8]{index=8}
    """
    w = np.logspace(np.log10(w_min), np.log10(w_max), n)
    mag = gain_mag_G(a_s, a_v, a_vl, w)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))

    plt.figure(figsize=(8.5, 4.5))
    plt.semilogx(w, mag_db)
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Frequency ω (rad/s)")
    plt.ylabel("Magnitude (dB)")
    plt.title("String stability (Bode magnitude): 20 log10 |G(jω)|")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.show()

    return float(np.max(mag)), float(np.max(mag_db))


# ============================================================
# Numerical platoon simulation (your original, but returns more)
# ============================================================
# 1. Modify simulate_platoon to return x as well
def simulate_platoon(N: int, T_end: float, dt: float, cc: CCParams, veh: VehicleParams,
                     v_star: float, s_star: float, L: float,
                     pulse_amp: float = 1.0, pulse_t0: float = 10.0, pulse_dur: float = 2.0):
    ts = np.arange(0.0, T_end + 1e-9, dt)
    T = len(ts)
    v = np.zeros((N, T), dtype=float)
    x = np.zeros((N, T), dtype=float)
    s = np.zeros((N, T), dtype=float)

    # 1. Initialization
    v[:, 0] = v_star
    x[0, 0] = 0.0
    s[0, :] = np.nan

    for n in range(1, N):
        # Spacing: x[n] = x[n-1] - length - gap
        x[n, 0] = x[n - 1, 0] - (L + s_star)

    # 2. Main Simulation Loop
    for k in range(T - 1):
        t = ts[k]
        
        # --- Leader Perturbation Logic ---
        # Accelerate then decelerate to return to v_star
        if pulse_t0 <= t < (pulse_t0 + pulse_dur):
            aL = pulse_amp
        elif (pulse_t0 + pulse_dur) <= t < (pulse_t0 + 2 * pulse_dur):
            aL = -pulse_amp
        else:
            aL = 0.0
            
        v[0, k + 1] = max(0.0, v[0, k] + aL * dt)
        
        # Hard-reset v_star to stop integration drift after pulses
        if t > (pulse_t0 + 2 * pulse_dur + dt):
            v[0, k+1] = v_star
            
        x[0, k + 1] = x[0, k] + 0.5 * (v[0, k] + v[0, k + 1]) * dt

        # --- Followers Movement Logic (Indented inside time loop k) ---
        for n in range(1, N):
            # Calculate gap based on front bumper of follower and rear bumper of leader
            gap = (x[n - 1, k] - x[n, k]) - L
            gap = max(0.0, float(gap))
            s[n, k] = gap

            vF = float(v[n, k])
            vL = float(v[n - 1, k])
            
            # Compute acceleration from your Wiedemann logic
            a = accel_from_wiedemann_state(cc, veh, gap=gap, vF=vF, vL=vL, L_lead=L)

            # Numerical integration (Forward Euler / Trapezoidal hybrid)
            v_next = max(0.0, v[n, k] + a * dt)
            x_next = x[n, k] + 0.5 * (v[n, k] + v_next) * dt
            
            # Physical safety guard: prevent crossing the leader
            max_pos = x[n - 1, k + 1] - L - 0.1 # 0.1 ft buffer
            if x_next > max_pos:
                x_next = max_pos
                v_next = min(v_next, v[n - 1, k + 1])
                
            v[n, k + 1] = v_next
            x[n, k + 1] = x_next

    # Fill final gap values
    for n in range(1, N):
        s[n, -1] = (x[n - 1, -1] - x[n, -1]) - L

    return ts, v, s, x

# 2. Add the Trajectory Plotting function
def plot_trajectories(ts: np.ndarray, x: np.ndarray, vehicles_to_show=None):
    """
    Plots the position of vehicles over time.
    """
    plt.figure(figsize=(10, 6))
    N = x.shape[0]
    if vehicles_to_show is None:
        vehicles_to_show = range(N)
    
    for i in vehicles_to_show:
        plt.plot(ts, x[i], label=f"Veh {i}" if i % 5 == 0 or i == N-1 else "")
    
    plt.xlabel("Time (s)")
    plt.ylabel("Position (ft)")
    plt.title("Vehicle Trajectories (Space-Time Diagram)")
    plt.grid(True, alpha=0.3)
    if len(vehicles_to_show) < 15:
        plt.legend()
    plt.tight_layout()
    plt.show()


def speed_amplitude(v: np.ndarray, t0_idx: int) -> np.ndarray:
    return np.array([v[i, t0_idx:].max() - v[i, t0_idx:].min()
                     for i in range(v.shape[0])], dtype=float)


# ============================================================
# Plots that match “lecture” intuition
# ============================================================

def plot_string_amplitude(amps: np.ndarray):
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(amps)), amps, marker="o")
    plt.xlabel("Vehicle index (0=leader)")
    plt.ylabel("Speed amplitude (ft/s)")
    plt.title("String stability (numerical): amplitude down the platoon")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_timespace_speed(ts: np.ndarray, v: np.ndarray):
    plt.figure(figsize=(10, 5))
    plt.imshow(v, aspect="auto", origin="lower",
               extent=[ts[0], ts[-1], 0, v.shape[0] - 1])
    plt.xlabel("Time (s)")
    plt.ylabel("Vehicle index")
    plt.title("Time–space diagram (speed)")
    plt.colorbar(label="Speed (ft/s)")
    plt.tight_layout()
    plt.show()


def plot_deviation_curves(ts: np.ndarray, v: np.ndarray, s: np.ndarray,
                          v_star: float, s_star: float,
                          vehicles_to_show=(0, 1, 5, 10, 20, 29)):
    """
    Much easier to interpret than raw speed:
      plot δv_n(t) and δs_n(t) relative to equilibrium.
    This directly matches the linearization story in the slides. :contentReference[oaicite:9]{index=9}
    """
    vehicles_to_show = [i for i in vehicles_to_show if i < v.shape[0]]

    # δv
    plt.figure(figsize=(9, 4.5))
    for i in vehicles_to_show:
        plt.plot(ts, v[i] - v_star, label=f"veh {i}")
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("δv(t) = v - v* (ft/s)")
    plt.title("Velocity deviations from equilibrium (δv)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.show()

    # δs (followers only)
    plt.figure(figsize=(9, 4.5))
    for i in vehicles_to_show:
        if i == 0:
            continue
        plt.plot(ts, s[i] - s_star, label=f"veh {i}")
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("Time (s)")
    plt.ylabel("δs(t) = s - s* (ft)")
    plt.title("Spacing deviations from equilibrium (δs)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=3, fontsize=9)
    plt.tight_layout()
    plt.show()

def gain_mag_G(a_s: float, a_v: float, a_vl: float, w: np.ndarray) -> np.ndarray:
    jw = 1j * w
    num = a_s + a_vl * jw
    den = (jw**2) - a_v * jw + a_s
    G = num / den
    return np.abs(G)


# ============================================================
# Main
# ============================================================

def main():
    # 1. Load Parameters
    # Attempt to load from CSV, otherwise fallback to your calibrated values
    try:
        cf = pd.read_csv("cf_best_params.csv")
        CC0 = float(cf["CC0"].iloc[0])
        CC1 = float(cf["CC1"].iloc[0])
        CC2 = float(cf["CC2"].iloc[0])
    except (FileNotFoundError, KeyError):
        # Fallback to the values provided in the prompt
        CC0, CC1, CC2 = 4.325544441709977, 1.0723277149690673, 38.78868862631028

    cc = CCParams(CC0=CC0, CC1=CC1, CC2=CC2, CC3=8.0, CC4=-0.35, CC5=0.35,
                  CC6=11.44, CC7=0.82, CC8=11.48, CC9=None)
    veh = VehicleParams(v_free=110.0, bmin=-15.0, tau=1.0, alpha=0.4)

    # 2. Define Operating Point
    # Note: s_star is the equilibrium gap. 
    # v_star is the steady-state speed.
    s_star = 20.0  # Adjusted equilibrium gap (ft)
    v_star = 24.50 # Steady-state speed (ft/s)
    L = 15.30      # Vehicle length (ft)

    print("\n=== Operating point ===")
    print(f"s*={s_star:.2f} ft, v*={v_star:.2f} ft/s, L={L:.2f} ft")

    # 3. Linearization Coefficients (Analytical Stability)
    a0, a_s, a_v, a_vl = finite_diff_partials(cc, veh, s_star=s_star, v_star=v_star, L=L)
    print("\n=== Linearization (finite differences) ===")
    print(f"a(s*,v*,v*) = {a0:.6f} ft/s^2 (should be near 0 at equilibrium)")
    print(f"a_s  = ∂a/∂s      = {a_s:.6f} 1/s^2")
    print(f"a_v  = ∂a/∂v      = {a_v:.6f} 1/s")
    print(f"a_vl = ∂a/∂v_lead = {a_vl:.6f} 1/s")

    # 4. Local Stability Check
    A, eig, tr, det, local_ok, monotonic_ok = local_stability_report(a_s, a_v)
    print("\n=== Local stability (lecture test) ===")
    print(f"trace(A) = {tr:.6f} (need < 0), det(A) = {det:.6f} (need > 0)")
    print(f"Local stable? {bool(local_ok)} | Monotonic decay? {bool(monotonic_ok)}")

    # 5. Numerical Platoon Simulation (Perturbation Test)
    print("\n=== Numerical platoon test ===")
    # Using the double-pulse logic: accelerate then decelerate back to v_star
    ts, v, s, x_pos = simulate_platoon(N=30, T_end=120.0, dt=0.1, cc=cc, veh=veh,
                                       v_star=v_star, s_star=s_star, L=L,
                                       pulse_amp=1.0, pulse_t0=10.0, pulse_dur=2.0)

    # ============================================================
    # 6. CONSOLIDATED DASHBOARD PLOTTING (Two-Column Format)
    # ============================================================
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    plt.subplots_adjust(hspace=0.4, wspace=0.25)

    # --- [0,0] Bode Magnitude Plot (Analytical String Stability) ---
    w_vec = np.logspace(-3, 1, 800)
    mag = gain_mag_G(a_s, a_v, a_vl, w_vec)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))
    axes[0, 0].semilogx(w_vec, mag_db)
    axes[0, 0].axhline(0.0, color='red', linestyle="--", alpha=0.7)
    axes[0, 0].set_title("String Stability (Bode Magnitude)")
    axes[0, 0].set_xlabel("Frequency ω (rad/s)")
    axes[0, 0].set_ylabel("Magnitude (dB)")
    axes[0, 0].grid(True, which="both", alpha=0.3)

    # --- [0,1] Speed Amplitude Plot (Numerical String Stability) ---
    ignore_s = 20.0
    t0_idx = int(ignore_s / 0.1)
    amps = speed_amplitude(v, t0_idx)
    axes[0, 1].plot(np.arange(len(amps)), amps, marker="o", color='tab:blue')
    axes[0, 1].set_title("Speed Amplitude Decay Down Platoon")
    axes[0, 1].set_xlabel("Vehicle Index (0=leader)")
    axes[0, 1].set_ylabel("Speed Amplitude (ft/s)")
    axes[0, 1].grid(True, alpha=0.3)

    # --- [1,0] Velocity Deviations (δv) ---
    veh_indices = [0, 1, 5, 10, 20, 29]
    for i in veh_indices:
        if i < v.shape[0]:
            axes[1, 0].plot(ts, v[i] - v_star, label=f"Veh {i}")
    axes[1, 0].set_title("Velocity Deviations (δv = v - v*)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Deviation (ft/s)")
    axes[1, 0].legend(ncol=2, fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    # --- [1,1] Spacing Deviations (δs) ---
    for i in veh_indices:
        if 0 < i < v.shape[0]:
            axes[1, 1].plot(ts, s[i] - s_star, label=f"Veh {i}")
    axes[1, 1].set_title("Spacing Deviations (δs = s - s*)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Deviation (ft)")
    axes[1, 1].legend(ncol=2, fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # --- [2,0] Time-Space Speed Heatmap ---
    im = axes[2, 0].imshow(v, aspect="auto", origin="lower",
                          extent=[ts[0], ts[-1], 0, v.shape[0] - 1], cmap='viridis')
    axes[2, 0].set_title("Speed Propagation Heatmap")
    axes[2, 0].set_xlabel("Time (s)")
    axes[2, 0].set_ylabel("Vehicle Index")
    fig.colorbar(im, ax=axes[2, 0], label="Speed (ft/s)")

    # --- [2,1] Space-Time Trajectories ---
    for i in range(x_pos.shape[0]):
        # Label only every 5th vehicle for clarity
        label = f"Veh {i}" if i % 5 == 0 or i == x_pos.shape[0]-1 else ""
        axes[2, 1].plot(ts, x_pos[i], label=label, alpha=0.6)
    axes[2, 1].set_title("Vehicle Trajectories (Space-Time Diagram)")
    axes[2, 1].set_xlabel("Time (s)")
    axes[2, 1].set_ylabel("Position (ft)")
    axes[2, 1].grid(True, alpha=0.3)

    plt.suptitle(f"Wiedemann-99 Stability Analysis Dashboard\nOperating Point: v*={v_star} ft/s, s*={s_star} ft", fontsize=16)
    plt.show()

    print("\nDONE.")

if __name__ == "__main__":
    main()
