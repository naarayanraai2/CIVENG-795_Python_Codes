#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from models import CCParams, VehicleParams, rmse, simulate_episode_paper_wiedemann

@dataclass
class EpisodeArrays:
    t: np.ndarray
    xL: np.ndarray
    vL: np.ndarray
    L_lead: np.ndarray
    v0: float
    gap0_obs: float
    v_obs: np.ndarray
    gap_obs: np.ndarray
    meta: Tuple[int,int,int]

def load_ngsim(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    needed = ["Vehicle_ID","Frame_ID","Local_Y","Lane_ID","v_Vel","v_Acc","Preceding","Space_Headway","v_Length","v_Class"]
    miss = [c for c in needed if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns: {miss}")
    df = df[df["v_Class"] == 2].copy()
    df = df[df["Local_Y"].notna()].copy()
    df = df[df["v_Vel"].notna() & (df["v_Vel"] >= 0)].copy()
    df = df[df["Space_Headway"].notna() & (df["Space_Headway"] > 0)].copy()
    df = df[df["Preceding"] > 0].copy()
    return df

def build_cf_with_leader(df: pd.DataFrame) -> pd.DataFrame:
    leaders = df[["Frame_ID","Vehicle_ID","Lane_ID","Local_Y","v_Vel","v_Length"]].copy()
    leaders = leaders.rename(columns={
        "Vehicle_ID":"Leader_ID",
        "Lane_ID":"Leader_Lane",
        "Local_Y":"Leader_Local_Y",
        "v_Vel":"Leader_v_Vel",
        "v_Length":"Leader_v_Length",
    })
    fol = df.copy()
    fol["Leader_ID"] = fol["Preceding"]
    cf = fol.merge(leaders, how="inner", on=["Frame_ID","Leader_ID"])
    cf = cf[cf["Lane_ID"] == cf["Leader_Lane"]].copy()
    cf = cf.sort_values(["Vehicle_ID","Leader_ID","Lane_ID","Frame_ID"]).copy()
    return cf

def build_all_contiguous_episodes(cf: pd.DataFrame, fps: float, min_seconds: float = 8.0) -> List[pd.DataFrame]:
    min_frames = int(round(min_seconds * fps))
    eps = []
    for _, g in cf.groupby(["Vehicle_ID","Leader_ID","Lane_ID"]):
        g = g.sort_values("Frame_ID").copy()
        if len(g) < min_frames:
            continue
        if (g["Frame_ID"].diff().fillna(1) > 1).any():
            continue
        eps.append(g)
    return eps

def stratified_select_episodes(cf: pd.DataFrame, fps: float, n_total: int = 50, min_seconds: float = 8.0, seed: int = 42) -> List[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    candidates = build_all_contiguous_episodes(cf, fps=fps, min_seconds=min_seconds)
    if not candidates:
        raise RuntimeError("No contiguous episodes found.")
    catA, catB, catC = [], [], []
    for ep in candidates:
        gap = ep["Space_Headway"].to_numpy(float)
        dv  = (ep["v_Vel"] - ep["Leader_v_Vel"]).to_numpy(float)
        v   = ep["v_Vel"].to_numpy(float)
        if gap.min() < 10 and v.mean() < 30:
            catA.append(ep)
        if np.median(gap) < 45 and np.std(dv) < 3:
            catB.append(ep)
        if np.std(gap) > 8 and np.std(dv) > 3:
            catC.append(ep)

    def take(lst, k):
        if len(lst) == 0 or k <= 0:
            return []
        k = min(k, len(lst))
        idx = rng.choice(len(lst), size=k, replace=False)
        return [lst[i] for i in idx]

    pick = []
    pick += take(catA, 15)
    pick += take(catB, 20)
    pick += take(catC, 15)

    if len(pick) < n_total:
        remaining = [ep for ep in candidates if ep not in pick]
        idx = rng.choice(len(remaining), size=min(n_total-len(pick), len(remaining)), replace=False)
        pick += [remaining[i] for i in idx]

    return pick[:n_total]

def precompute_episode_arrays(episodes: List[pd.DataFrame], fps: float) -> List[EpisodeArrays]:
    out = []
    for ep in episodes:
        ep = ep.sort_values("Frame_ID").copy()
        t = (ep["Frame_ID"].to_numpy(float) - float(ep["Frame_ID"].iloc[0])) / fps
        xL = ep["Leader_Local_Y"].to_numpy(float)
        vL = ep["Leader_v_Vel"].to_numpy(float)
        L_lead = ep["Leader_v_Length"].to_numpy(float)
        v_obs = ep["v_Vel"].to_numpy(float)
        gap_obs = ep["Space_Headway"].to_numpy(float)
        meta = (int(ep["Vehicle_ID"].iloc[0]), int(ep["Leader_ID"].iloc[0]), int(ep["Lane_ID"].iloc[0]))
        out.append(EpisodeArrays(
            t=t, xL=xL, vL=vL, L_lead=L_lead,
            v0=float(v_obs[0]),
            gap0_obs=float(gap_obs[0]),
            v_obs=v_obs, gap_obs=gap_obs, meta=meta
        ))
    return out

def objective_cc0_cc1_cc2(theta: np.ndarray, template_cc: CCParams, veh: VehicleParams, episodes: List[EpisodeArrays],
                          w_gap=1.0, w_v=1.0, gap_scale=10.0, v_scale=5.0) -> float:
    CC0, CC1, CC2 = map(float, theta)
    cc = CCParams(CC0=CC0, CC1=CC1, CC2=CC2,
                 CC3=template_cc.CC3, CC4=template_cc.CC4, CC5=template_cc.CC5,
                 CC6=template_cc.CC6, CC7=template_cc.CC7, CC8=template_cc.CC8, CC9=template_cc.CC9)

    gap_errs, v_errs = [], []
    for ep in episodes:
        v_sim, gap_sim, _ = simulate_episode_paper_wiedemann(
            t=ep.t, xL=ep.xL, vL=ep.vL, L_lead=ep.L_lead,
            v0=ep.v0, gap0_obs=ep.gap0_obs, cc=cc, veh=veh
        )
        gap_errs.append(rmse(ep.gap_obs, gap_sim))
        v_errs.append(rmse(ep.v_obs, v_sim))
    return w_gap*(float(np.mean(gap_errs))/gap_scale) + w_v*(float(np.mean(v_errs))/v_scale)

def random_search(bounds: np.ndarray, n_samples: int, seed: int, obj_kwargs: dict):
    rng = np.random.default_rng(seed)
    best_val = np.inf
    best_theta = None
    for i in range(n_samples):
        theta = np.array([rng.uniform(lo, hi) for lo,hi in bounds], dtype=float)
        val = objective_cc0_cc1_cc2(theta, **obj_kwargs)
        if val < best_val:
            best_val = val
            best_theta = theta.copy()
        if (i+1) % max(1, n_samples//10) == 0:
            print(f"[CF] {i+1:5d}/{n_samples} best={best_val:.6f} theta={best_theta}")
    return best_theta, float(best_val)

def plot_best_fit(episodes: List[EpisodeArrays], cc_best: CCParams, veh: VehicleParams, n_show: int = 4):
    show = episodes[:min(n_show, len(episodes))]
    nrows = len(show)
    fig, axes = plt.subplots(nrows=nrows, ncols=2, figsize=(16, 3.2*nrows))
    if nrows == 1:
        axes = np.array([axes])
    for i, ep in enumerate(show):
        v_sim, gap_sim, _ = simulate_episode_paper_wiedemann(
            t=ep.t, xL=ep.xL, vL=ep.vL, L_lead=ep.L_lead,
            v0=ep.v0, gap0_obs=ep.gap0_obs, cc=cc_best, veh=veh
        )
        fol, lead, lane = ep.meta
        axg = axes[i,0]
        axg.plot(ep.t, ep.gap_obs, label="gap_obs")
        axg.plot(ep.t, gap_sim, "--", label="gap_sim")
        axg.set_title(f"Ep {i+1}: GAP | fol={fol} lead={lead} lane={lane}")
        axg.set_xlabel("Time (s)"); axg.set_ylabel("Gap (ft)")
        axg.grid(True, alpha=0.3)
        if i == 0: axg.legend()

        axs = axes[i,1]
        axs.plot(ep.t, ep.v_obs, label="v_obs")
        axs.plot(ep.t, v_sim, "--", label="v_sim")
        axs.plot(ep.t, ep.vL, ":", label="v_leader")
        axs.set_title(f"Ep {i+1}: SPEED")
        axs.set_xlabel("Time (s)"); axs.set_ylabel("Speed (ft/s)")
        axs.grid(True, alpha=0.3)
        if i == 0: axs.legend(ncol=3)

    plt.tight_layout()
    plt.show()

def main():
    CSV_PATH = "data/trajectories-0500-0515.csv"
    FPS = 10.0

    template_cc = CCParams(
        CC0=4.92, CC1=0.9, CC2=13.12,
        CC3=8.0, CC4=-0.35, CC5=0.35,
        CC6=11.44, CC7=0.82, CC8=11.48, CC9=None
    )
    veh = VehicleParams(v_free=110.0, bmin=-15.0, tau=1.0, alpha=0.4)

    print("Loading + building CF...")
    df = load_ngsim(CSV_PATH)
    cf = build_cf_with_leader(df)

    print("Selecting episodes...")
    eps_df = stratified_select_episodes(cf, fps=FPS, n_total=50, min_seconds=8.0, seed=42)
    eps = precompute_episode_arrays(eps_df, fps=FPS)
    print(f"Episodes: {len(eps)}")

    bounds = np.array([
        [3.0, 10.0],
        [0.3, 3.0],
        [6, 40.0],
    ], dtype=float)

    best_theta, best_obj = random_search(
        bounds=bounds,
        n_samples=800,
        seed=42,
        obj_kwargs=dict(template_cc=template_cc, veh=veh, episodes=eps,
                        w_gap=1.0, w_v=1.0, gap_scale=10.0, v_scale=5.0)
    )

    print("\n=== CF BEST ===")
    print("Best objective:", best_obj)
    print("Best theta:", best_theta)

    cc_best = CCParams(
        CC0=float(best_theta[0]), CC1=float(best_theta[1]), CC2=float(best_theta[2]),
        CC3=template_cc.CC3, CC4=template_cc.CC4, CC5=template_cc.CC5,
        CC6=template_cc.CC6, CC7=template_cc.CC7, CC8=template_cc.CC8, CC9=template_cc.CC9
    )

    plot_best_fit(eps, cc_best, veh, n_show=4)

    # save to file for other scripts
    out = pd.DataFrame([{
        "CC0": cc_best.CC0, "CC1": cc_best.CC1, "CC2": cc_best.CC2,
        "objective": best_obj
    }])
    out.to_csv("cf_best_params.csv", index=False)
    print("Wrote cf_best_params.csv")

if __name__ == "__main__":
    main()
