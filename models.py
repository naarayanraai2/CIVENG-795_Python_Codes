#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal
import numpy as np

Regime = Literal["free", "closing", "following", "emergency"]

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.sqrt(np.mean((a - b) ** 2)))

def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))

def logloss_binary(y: np.ndarray, p: np.ndarray, eps: float = 1e-9) -> float:
    y = np.asarray(y, dtype=float)
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

@dataclass
class CCParams:
    CC0: float
    CC1: float
    CC2: float
    CC3: float
    CC4: float
    CC5: float
    CC6: float
    CC7: float
    CC8: float
    CC9: float

@dataclass
class VehicleParams:
    v_free: float     # ft/s
    bmin: float       # negative ft/s^2
    tau: float = 1.0
    alpha: float = 0.4

def thresholds_w99(cc: CCParams, DX: float, Vlead: float) -> Dict[str, float]:
    AX  = cc.CC0
    ABX = AX + cc.CC1 * Vlead
    SDX = ABX + cc.CC2
    CLDV = cc.CC5 + (cc.CC6 / 17000.0) * (DX ** 2)
    OPDV = cc.CC4 - (cc.CC6 / 17000.0) * (DX ** 2)
    SDV  = cc.CC5 + (DX - SDX) / max(cc.CC3, 1e-6)
    return {"AX": AX, "ABX": ABX, "SDX": SDX, "CLDV": CLDV, "OPDV": OPDV, "SDV": SDV}

def classify_regime(DX: float, DV: float, th: Dict[str, float]) -> Regime:
    AX, ABX, SDX, CLDV, OPDV, SDV = th["AX"], th["ABX"], th["SDX"], th["CLDV"], th["OPDV"], th["SDV"]

    if (DX >= SDX and DV <= SDV) or (DV < OPDV):
        return "free"
    if ((DX > ABX and DX < SDX and DV > CLDV) or (DX >= SDX and DV > SDV)):
        return "closing"
    if (DX > ABX and DX < SDX and DV > OPDV and DV <= CLDV):
        return "following"
    if (DV >= OPDV and DX <= ABX and DX > AX):
        return "emergency"
    return "following"

def bmax_free_flow(cc: CCParams, veh: VehicleParams, v_f: float, DX: float, ABX: float) -> float:
    if DX <= ABX:
        return 0.0
    return cc.CC8 * (1.0 - veh.alpha * (v_f / max(veh.v_free, 1e-6)))

def accel_closing(cc: CCParams, veh: VehicleParams, DV: float, DX: float) -> float:
    denom = max(DX - cc.CC0, 0.01)
    return max(-0.5 * (DV ** 2) / denom, veh.bmin)

def accel_following(cc: CCParams, Bmax: float, DV: float) -> float:
    if DV < 0:
        return min(cc.CC7, Bmax)
    return -cc.CC7

# def accel_emergency(cc: CCParams, veh: VehicleParams, DV: float, DX: float, ABX: float, b_prev: float) -> float:
#     if DV < 0:
#         return 0.0
#     denom = max(DX - cc.CC0, 0.01)
#     base = (-0.5 * (DV ** 2) / denom) + b_prev
#     out = max(base, veh.bmin)
#     if out > 0:
#         extra_denom = max(ABX - cc.CC0, 0.01)
#         extra = veh.bmin * (ABX - DX) / extra_denom
#         out = max(base + extra, veh.bmin)
#     return out

def accel_emergency(cc: CCParams, veh: VehicleParams, DV: float, DX: float, ABX: float, b_prev: float) -> float:
    # If we are already closer than the minimum safety distance, force maximum braking
    if DX <= cc.CC0:
        return veh.bmin

    # If follower is slower than leader (DV < 0), we don't necessarily need to brake harder
    # UNLESS we are already inside the ABX threshold.
    if DV < 0:
        return 0.0 if DX > cc.CC0 else veh.bmin

    denom = max(DX - cc.CC0, 0.01)
    # The core Wiedemann emergency formula:
    base = (-0.5 * (DV ** 2) / denom) + b_prev
    
    # Ensure we don't exceed the vehicle's physical braking limit
    out = max(base, veh.bmin)
    
    # If the gap is closing and we are inside ABX, apply extra braking pressure
    if DX < ABX:
        extra_denom = max(ABX - cc.CC0, 0.01)
        extra = veh.bmin * (ABX - DX) / extra_denom
        out = max(out + extra, veh.bmin)
        
    return out

def integrate_beeman(x_t: float, v_t: float, b_t: float, b_tm1: float, b_tp1: float, dt: float) -> Tuple[float, float]:
    v_next = v_t + (1.0 / 6.0) * (2.0 * b_tp1 + 5.0 * b_t - b_tm1) * dt
    x_next = x_t + v_t * dt + (1.0 / 6.0) * (4.0 * b_t - b_tm1) * (dt ** 2)
    return x_next, v_next

def simulate_episode_paper_wiedemann(
    t: np.ndarray,
    xL: np.ndarray,
    vL: np.ndarray,
    L_lead: np.ndarray,
    v0: float,
    gap0_obs: float,
    cc: CCParams,
    veh: VehicleParams,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(t)
    dt = float(np.median(np.diff(t)))

    xF = np.zeros(n, dtype=float)
    vF = np.zeros(n, dtype=float)
    bF = np.zeros(n, dtype=float)

    xF[0] = float(xL[0]) - float(L_lead[0]) - float(gap0_obs)
    vF[0] = float(v0)
    bF[0] = 0.0
    b_tm1 = 0.0

    # for k in range(n - 1):
    #     DX = (xL[k] - xF[k]) - L_lead[k]
    #     DX = float(max(DX, 0.0))
    #     DV = float(vF[k] - vL[k])

    #     th = thresholds_w99(cc, DX=DX, Vlead=float(vL[k]))
    #     reg = classify_regime(DX=DX, DV=DV, th=th)
    #     Bmax = bmax_free_flow(cc, veh, v_f=float(vF[k]), DX=DX, ABX=th["ABX"])

    #     if reg == "free":
    #         b_next = Bmax
    #     elif reg == "closing":
    #         b_next = accel_closing(cc, veh, DV=DV, DX=DX)
    #     elif reg == "following":
    #         b_next = accel_following(cc, Bmax=Bmax, DV=DV)
    #     else:
    #         b_next = accel_emergency(cc, veh, DV=DV, DX=DX, ABX=th["ABX"], b_prev=float(bF[k]))

    #     x_next, v_next = integrate_beeman(xF[k], vF[k], bF[k], b_tm1, b_next, dt)
    #     xF[k + 1] = x_next
    #     vF[k + 1] = max(0.0, v_next)
    #     b_tm1 = bF[k]
    #     bF[k + 1] = b_next

    for k in range(n - 1):
        DX = (xL[k] - xF[k]) - L_lead[k]
        DX = float(max(DX, 0.0))
        DV = float(vF[k] - vL[k])

        th = thresholds_w99(cc, DX=DX, Vlead=float(vL[k]))
        reg = classify_regime(DX=DX, DV=DV, th=th)
        Bmax = bmax_free_flow(cc, veh, v_f=float(vF[k]), DX=DX, ABX=th["ABX"])

        # Regime Selection Logic
        if reg == "free":
            b_next = Bmax
        elif reg == "closing":
            b_next = accel_closing(cc, veh, DV=DV, DX=DX)
        elif reg == "following":
            b_next = accel_following(cc, Bmax=Bmax, DV=DV)
        else:
            b_next = accel_emergency(cc, veh, DV=DV, DX=DX, ABX=th["ABX"], b_prev=float(bF[k]))

        # --- INTEGRATION & SAFETY GUARD ---
        x_next, v_next = integrate_beeman(xF[k], vF[k], bF[k], b_tm1, b_next, dt)
        
        # Hard Constraint: Prevent the front bumper from passing the leader's rear bumper
        # We leave a tiny margin (e.g., CC0) to prevent overlapping floating point errors
        max_allowed_pos = xL[k+1] - L_lead[k+1] - 0.1 
        
        if x_next >= max_allowed_pos:
            xF[k+1] = max_allowed_pos
            vF[k+1] = min(v_next, vL[k+1]) # Don't go faster than the leader if touching
            b_next = veh.bmin              # Force maximum braking for the next state
        else:
            xF[k+1] = x_next
            vF[k+1] = max(0.0, v_next)
            
        b_tm1 = bF[k]
        bF[k + 1] = b_next

    gap = np.maximum((xL - xF) - L_lead, 0.0)
    return vF, gap, bF

def accel_from_wiedemann_state(
    cc: CCParams, veh: VehicleParams,
    gap: float, vF: float, vL: float, L_lead: float
) -> float:
    DX = max(float(gap), 0.0)
    DV = float(vF - vL)
    th = thresholds_w99(cc, DX=DX, Vlead=float(vL))
    reg = classify_regime(DX=DX, DV=DV, th=th)
    Bmax = bmax_free_flow(cc, veh, v_f=float(vF), DX=DX, ABX=th["ABX"])
    if reg == "free":
        return float(Bmax)
    if reg == "closing":
        return float(accel_closing(cc, veh, DV=DV, DX=DX))
    if reg == "following":
        return float(accel_following(cc, Bmax=Bmax, DV=DV))
    return float(accel_emergency(cc, veh, DV=DV, DX=DX, ABX=th["ABX"], b_prev=0.0))
