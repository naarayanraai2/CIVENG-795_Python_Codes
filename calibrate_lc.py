#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lane-change calibration (MOBIL-inspired + probabilistic choice).

Correct methodology (consistent with corrected report + implementation):
- MOBIL incentive computed from expected acceleration changes:
    inc = (aE_new - aE_old) + p * (a_nf_new - a_nf_old) + bias_left(if target is left)
- Safety constraint enforced:
    a_nf_new >= -b_safe  (otherwise lane change infeasible)
- Decision is probabilistic:
    P(left) = sigmoid( inc_left - inc_right )
- Calibration uses bounded random search minimizing binary log-loss.

Important:
- NO hard threshold a_thr is used (removed for consistency with corrected methodology).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import argparse

import numpy as np
import pandas as pd

from models import CCParams, VehicleParams, sigmoid, logloss_binary, accel_from_state


@dataclass(frozen=True)
class MOBILParams:
    p: float
    b_safe: float
    bias_left: float


def load_ngsim(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = ["Vehicle_ID", "Frame_ID", "Local_Y", "Lane_ID", "v_Vel", "Preceding", "Space_Headway", "v_Length", "v_Class"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")

    df = df[df["v_Class"] == 2].copy()
    df = df[df["Local_Y"].notna()].copy()
    df = df[df["v_Vel"].notna() & (df["v_Vel"] >= 0)].copy()
    return df


def build_frame_index(df: pd.DataFrame) -> Dict[int, pd.DataFrame]:
    out: Dict[int, pd.DataFrame] = {}
    for fid, g in df.groupby("Frame_ID"):
        out[int(fid)] = g.sort_values(["Lane_ID", "Local_Y"]).copy()
    return out


def neighbor_in_lane(frame_df: pd.DataFrame, lane: int, y: float):
    g = frame_df[frame_df["Lane_ID"] == lane].sort_values("Local_Y")
    if len(g) == 0:
        return None, None
    ahead = g[g["Local_Y"] > y]
    behind = g[g["Local_Y"] < y]
    leader = ahead.iloc[0] if len(ahead) else None
    follower = behind.iloc[-1] if len(behind) else None
    return leader, follower


def mobil_incentive(cc: CCParams, veh: VehicleParams, frame_df: pd.DataFrame, ego: pd.Series, target_lane: int, mobil: MOBILParams) -> Tuple[float, bool]:
    """
    Returns: (incentive_value, safe_ok)
    """
    y = float(ego["Local_Y"])
    vE = float(ego["v_Vel"])
    cur_lane = int(ego["Lane_ID"])

    cur_lead, _ = neighbor_in_lane(frame_df, cur_lane, y)
    new_lead, new_fol = neighbor_in_lane(frame_df, target_lane, y)

    # Require both leader and follower in the target lane for incentive eval (consistent with prior code)
    if new_lead is None or new_fol is None:
        return -1e9, False

    def gap_between(lead: pd.Series, fol: pd.Series) -> float:
        return float(lead["Local_Y"] - fol["Local_Y"] - lead["v_Length"])

    # ego old accel
    if cur_lead is None:
        aE_old = 0.0
    else:
        gap_old = gap_between(cur_lead, ego)
        aE_old = accel_from_state(cc, veh, gap_old, vE, float(cur_lead["v_Vel"]), float(cur_lead["v_Length"]))

    # ego new accel (if moved to target lane)
    gap_new = gap_between(new_lead, ego)
    aE_new = accel_from_state(cc, veh, gap_new, vE, float(new_lead["v_Vel"]), float(new_lead["v_Length"]))

    # new follower effect (before vs after insertion)
    gap_nf_old = gap_between(new_lead, new_fol)
    a_nf_old = accel_from_state(
        cc, veh, gap_nf_old, float(new_fol["v_Vel"]), float(new_lead["v_Vel"]), float(new_lead["v_Length"])
    )

    gap_nf_new = float(ego["Local_Y"] - new_fol["Local_Y"] - float(ego["v_Length"]))
    a_nf_new = accel_from_state(
        cc, veh, gap_nf_new, float(new_fol["v_Vel"]), vE, float(ego["v_Length"])
    )

    safe_ok = (a_nf_new >= -mobil.b_safe)

    inc = (aE_new - aE_old) + mobil.p * (a_nf_new - a_nf_old)

    # assumption: smaller lane id = left
    if target_lane < cur_lane:
        inc += mobil.bias_left

    return float(inc), bool(safe_ok)


def build_lane_change_events(df: pd.DataFrame, max_events: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Detect lane-change moments by comparing Lane_ID at frame t and t+1 (contiguous only).
    lc_dir = -1 for left, +1 for right.
    """
    df = df.sort_values(["Vehicle_ID", "Frame_ID"]).copy()
    df["Lane_next"] = df.groupby("Vehicle_ID")["Lane_ID"].shift(-1)
    df["Frame_next"] = df.groupby("Vehicle_ID")["Frame_ID"].shift(-1)

    is_contig = (df["Frame_next"] - df["Frame_ID"] == 1)
    is_lc = is_contig & df["Lane_next"].notna() & (df["Lane_next"] != df["Lane_ID"])

    events = df[is_lc].copy()
    if len(events) == 0:
        raise RuntimeError("No lane-change events found.")

    if len(events) > max_events:
        events = events.sample(max_events, random_state=seed)

    events["lc_dir"] = np.sign(events["Lane_next"].astype(float) - events["Lane_ID"].astype(float)).astype(int)
    return events[["Vehicle_ID", "Frame_ID", "Lane_ID", "Lane_next", "lc_dir"]].copy()


def mobil_objective_logloss(events: pd.DataFrame, frame_map: Dict[int, pd.DataFrame], cc: CCParams, veh: VehicleParams, mobil: MOBILParams) -> float:
    y_true = []
    z = []

    for _, row in events.iterrows():
        fid = int(row["Frame_ID"])
        fr = frame_map.get(fid)
        if fr is None:
            continue

        ego_df = fr[fr["Vehicle_ID"] == row["Vehicle_ID"]]
        if len(ego_df) == 0:
            continue

        ego = ego_df.iloc[0]
        cur_lane = int(row["Lane_ID"])
        left_lane = cur_lane - 1
        right_lane = cur_lane + 1

        incL, safeL = mobil_incentive(cc, veh, fr, ego, left_lane, mobil) if left_lane >= 1 else (-1e9, False)
        incR, safeR = mobil_incentive(cc, veh, fr, ego, right_lane, mobil)

        if not safeL:
            incL = -1e9
        if not safeR:
            incR = -1e9

        # label: 1 = left, 0 = right
        y = 1.0 if int(row["lc_dir"]) == -1 else 0.0
        y_true.append(y)
        z.append(incL - incR)

    if len(y_true) < 50:
        return 1e9

    p_left = sigmoid(np.array(z, dtype=float))
    return logloss_binary(np.array(y_true, dtype=float), p_left)


def random_search_mobil(bounds: Dict[str, Tuple[float, float]], n_samples: int, seed: int, obj_kwargs: dict):
    rng = np.random.default_rng(seed)
    best_val = np.inf
    best = None

    for i in range(n_samples):
        cand = MOBILParams(
            p=float(rng.uniform(*bounds["p"])),
            b_safe=float(rng.uniform(*bounds["b_safe"])),
            bias_left=float(rng.uniform(*bounds["bias_left"]))
        )
        val = mobil_objective_logloss(mobil=cand, **obj_kwargs)
        if val < best_val:
            best_val = float(val)
            best = cand

        if (i + 1) % max(1, n_samples // 10) == 0:
            print(f"[LC] {i+1:5d}/{n_samples} best_logloss={best_val:.6f} best={best}")

    return best, best_val


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/trajectories-0500-0515.csv", help="Path to NGSIM CSV")
    ap.add_argument("--cf-best", default="cf_best_params.csv", help="CF params CSV from calibrate_cf.py")
    ap.add_argument("--events", type=int, default=2000, help="Max lane-change events to sample")
    ap.add_argument("--samples", type=int, default=400, help="Random search samples")
    ap.add_argument("--seed", type=int, default=123, help="RNG seed")
    ap.add_argument("--out", default="lc_best_params.csv", help="Output CSV")
    return ap.parse_args()


def main():
    args = parse_args()

    # Load best CF params
    best_df = pd.read_csv(args.cf_best)
    CC0, CC1, CC2 = float(best_df["CC0"].iloc[0]), float(best_df["CC1"].iloc[0]), float(best_df["CC2"].iloc[0])

    cc = CCParams(CC0=CC0, CC1=CC1, CC2=CC2, CC3=8.0, CC4=-0.35, CC5=0.35, CC6=11.44, CC7=0.82, CC8=11.48, CC9=None)
    veh = VehicleParams(v_free=110.0, bmin=-15.0, tau=1.0, alpha=0.4)

    print("Loading data...")
    df = load_ngsim(args.csv)
    frame_map = build_frame_index(df)
    events = build_lane_change_events(df, max_events=args.events, seed=args.seed)
    print("Events:", len(events))

    bounds = {
        "p": (0.0, 1.0),
        "b_safe": (2.0, 12.0),
        "bias_left": (-0.5, 0.5),
    }

    best, best_val = random_search_mobil(
        bounds=bounds,
        n_samples=args.samples,
        seed=args.seed,
        obj_kwargs=dict(events=events, frame_map=frame_map, cc=cc, veh=veh)
    )

    out = pd.DataFrame([{
        "p": best.p,
        "b_safe": best.b_safe,
        "bias_left": best.bias_left,
        "logloss": best_val
    }])
    out.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
