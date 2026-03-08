"""
Evaluation Layer - Computes multi-objective performance indicators,
sensitivity analysis, and scalability benchmarks.
Updated to accept optional candidate_df for venue-aware problem construction.
"""
import time
import numpy as np
import pandas as pd
from modules.optimization_layer import EVStationProblem, run_nsga2, run_mopso


# ---------------------------------------------------------------------------
# Performance Indicators
# ---------------------------------------------------------------------------

def hypervolume(pareto_obj: np.ndarray, reference_point: np.ndarray) -> float:
    n, m = pareto_obj.shape
    if n == 0: return 0.0
    dominated = np.all(pareto_obj <= reference_point, axis=1)
    pts = pareto_obj[dominated]
    if len(pts) == 0: return 0.0
    if m == 2:
        order = np.argsort(pts[:, 0]); pts = pts[order]
        hv = 0.0; y_ref = reference_point[1]
        for i in range(len(pts)):
            hv += (reference_point[0] - pts[i, 0]) * (y_ref - pts[i, 1])
            if i + 1 < len(pts): y_ref = min(y_ref, pts[i + 1, 1])
        return max(hv, 0.0)
    else:
        n_s = 50000; lb = pts.min(axis=0)
        sample = lb + np.random.rand(n_s, m) * (reference_point - lb)
        dc = sum(np.any(np.all(pts <= s, axis=1)) for s in sample)
        return np.prod(reference_point - lb) * dc / n_s


def generational_distance(pareto_obj, true_pareto_obj) -> float:
    if len(pareto_obj) == 0 or len(true_pareto_obj) == 0: return np.inf
    return float(np.mean([
        np.min(np.linalg.norm(true_pareto_obj - sol, axis=1))
        for sol in pareto_obj
    ]))


def spread_metric(pareto_obj) -> float:
    if len(pareto_obj) < 2: return 0.0
    pts = pareto_obj[np.argsort(pareto_obj[:, 0])]
    dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    d_bar = np.mean(dists)
    return float((dists[0] + dists[-1] + np.sum(np.abs(dists - d_bar))) /
                 (dists[0] + dists[-1] + len(dists) * d_bar + 1e-9))


def compute_all_indicators(nsga2_obj: np.ndarray, mopso_obj: np.ndarray) -> dict:
    def clean(obj):
        mask = np.all(np.isfinite(obj), axis=1)
        return obj[mask] if mask.any() else np.zeros((1, obj.shape[1]))
    n2 = clean(nsga2_obj); mo = clean(mopso_obj)
    combined = np.vstack([n2, mo])
    ref = combined.max(axis=0) * 1.1 + 1.0
    from modules.optimization_layer import _fast_nondominated_sort
    true_pareto = combined[_fast_nondominated_sort(combined)[0]]
    indicators = {}
    for name, obj in [("NSGA-II", n2), ("MOPSO", mo)]:
        hv = hypervolume(obj, ref)
        gd = generational_distance(obj, true_pareto)
        sp = spread_metric(obj)
        indicators[name] = {
            "Hypervolume": round(hv, 4) if np.isfinite(hv) else 0.0,
            "GD":          round(gd, 4) if np.isfinite(gd) else 0.0,
            "Spread":      round(sp, 4) if np.isfinite(sp) else 0.0,
        }
        print(f"  [{name}] HV={indicators[name]['Hypervolume']:.4f}, "
              f"GD={indicators[name]['GD']:.4f}, Spread={indicators[name]['Spread']:.4f}")
    return indicators


# ---------------------------------------------------------------------------
# Sensitivity Analysis
# ---------------------------------------------------------------------------

def _make_problem(cfg, future_df, mc_scenarios, candidate_df=None):
    """Helper: build EVStationProblem with optional candidate_df."""
    if candidate_df is not None:
        return EVStationProblem(cfg, future_df, mc_scenarios, candidate_df)
    # Fallback: build a minimal synthetic candidate_df from wards
    import pandas as pd, numpy as np
    from modules.data_layer import build_candidate_demand
    # Create a dummy locations frame matching n_stations from config
    n = cfg.get("optimization", {}).get("n_candidate_stations", 20)
    df25 = future_df[future_df["year"] == 2025].reset_index(drop=True)
    idx = np.array_split(np.arange(len(df25)), n)
    rows = []
    for i, grp in enumerate(idx):
        r = df25.iloc[grp[0]]
        demand = df25.iloc[grp]["daily_demand_sessions"].sum()
        rows.append({"loc_id": i+1, "name": f"Site {i+1}", "type": "parking",
                     "ward_no": int(r["ward_no"]), "lat": 22.72, "lon": 75.86,
                     "size": "medium", "chargers": 4,
                     "opening_cost": 500000.0, "grid_upgrade_cost": 200000.0,
                     "total_capex": 700000.0, "zone": r["zone"],
                     "daily_demand": round(float(demand), 2)})
    return EVStationProblem(cfg, future_df, mc_scenarios, pd.DataFrame(rows))


def sensitivity_analysis(cfg: dict, future_df: pd.DataFrame,
                          mc_scenarios: np.ndarray,
                          candidate_df=None) -> dict:
    sa = cfg["sensitivity"]
    results = {}

    # 1. EV growth rate
    print("[Sensitivity] Varying EV growth rates...")
    ev_profits, ev_wq, ev_n_stations, ev_coverages = [], [], [], []
    for rate in sa["ev_growth_rates"]:
        temp_df = future_df.copy()
        temp_df["daily_demand_sessions"] *= (rate / 0.3)
        temp_mc = mc_scenarios * (rate / 0.3)
        cdf = candidate_df.copy() if candidate_df is not None else None
        if cdf is not None:
            cdf["daily_demand"] = cdf["daily_demand"] * (rate / 0.3)
        prob = _make_problem(cfg, temp_df, temp_mc, cdf)
        x = np.zeros(prob.n_stations); x[:prob.n_stations // 2] = 1
        f = prob.evaluate(x)
        ev_profits.append(-f[2]); ev_wq.append(f[3])
        ev_n_stations.append(int(x.sum())); ev_coverages.append(-f[1])
    results["ev_growth_rate"] = {"x": sa["ev_growth_rates"], "profit": ev_profits,
                                  "wq_min": ev_wq, "n_stations": ev_n_stations,
                                  "coverage": ev_coverages}

    # 2. Electricity cost (modeled as revenue reduction)
    print("[Sensitivity] Varying electricity costs...")
    el_profits, el_wq = [], []
    for ecost in sa["electricity_costs"]:
        t_pr = dict(cfg.get("pricing", {}))
        reduction = ecost * 5.0   # rough: 1 kWh per session avg × ecost
        t_pr["peak_rate_per_session"]   = max(1.0, cfg.get("pricing", {}).get("peak_rate_per_session",   210) - reduction)
        t_pr["normal_rate_per_session"] = max(1.0, cfg.get("pricing", {}).get("normal_rate_per_session", 150) - reduction)
        t_pr["idle_rate_per_session"]   = max(1.0, cfg.get("pricing", {}).get("idle_rate_per_session",   112) - reduction)
        t_cfg = {**cfg, "pricing": t_pr}
        prob = _make_problem(t_cfg, future_df, mc_scenarios, candidate_df)
        x = np.zeros(prob.n_stations); x[:prob.n_stations // 2] = 1
        f = prob.evaluate(x)
        el_profits.append(-f[2]); el_wq.append(f[3])
    results["electricity_cost"] = {"x": sa["electricity_costs"],
                                    "profit": el_profits, "wq_min": el_wq}

    # 3. Installation cost factor
    print("[Sensitivity] Varying installation cost factors...")
    ic_profits, ic_n = [], []
    for factor in sa["installation_cost_factors"]:
        cdf = candidate_df.copy() if candidate_df is not None else None
        if cdf is not None:
            cdf["total_capex"]       = cdf["total_capex"]       * factor
            cdf["opening_cost"]      = cdf["opening_cost"]      * factor
            cdf["grid_upgrade_cost"] = cdf["grid_upgrade_cost"] * factor
        prob = _make_problem(cfg, future_df, mc_scenarios, cdf)
        x = np.zeros(prob.n_stations); x[:prob.n_stations // 2] = 1
        f = prob.evaluate(x)
        ic_profits.append(-f[2]); ic_n.append(int(x.sum()))
    results["installation_cost"] = {"x": sa["installation_cost_factors"],
                                     "profit": ic_profits, "n_stations": ic_n}

    # 4. Service rate (charging time)
    print("[Sensitivity] Varying service rates (charging times)...")
    sr_wq, sr_profits = [], []
    for t_min in sa["service_rates_min"]:
        t_q = {**cfg["queue"], "avg_charging_time_min": t_min}
        t_cfg = {**cfg, "queue": t_q}
        prob = _make_problem(t_cfg, future_df, mc_scenarios, candidate_df)
        x = np.zeros(prob.n_stations); x[:prob.n_stations // 2] = 1
        f = prob.evaluate(x)
        sr_wq.append(f[3]); sr_profits.append(-f[2])
    results["service_rate"] = {"x": sa["service_rates_min"],
                                "wq_min": sr_wq, "profit": sr_profits}
    return results


# ---------------------------------------------------------------------------
# Scalability Benchmarks
# ---------------------------------------------------------------------------

def scalability_benchmark(cfg: dict, future_df: pd.DataFrame,
                           mc_scenarios: np.ndarray,
                           candidate_df=None) -> dict:
    sizes = [5, 10, 15, 20]
    nsga2_times, mopso_times = [], []
    print("[Scalability] Running benchmarks...")
    for sz in sizes:
        cdf = candidate_df.head(min(sz, len(candidate_df))).reset_index(drop=True) \
              if candidate_df is not None else None
        t_opt = {**cfg["optimization"],
                 "nsga2": {**cfg["optimization"]["nsga2"], "n_generations": 10, "population_size": 20},
                 "mopso": {**cfg["optimization"]["mopso"], "n_iterations": 10, "n_particles": 20},
                 "n_candidate_stations": sz}
        t_cfg = {**cfg, "optimization": t_opt}
        prob = _make_problem(t_cfg, future_df, mc_scenarios, cdf)
        t0 = time.time(); run_nsga2(prob, t_cfg); nsga2_times.append(time.time() - t0)
        t0 = time.time(); run_mopso(prob, t_cfg); mopso_times.append(time.time() - t0)
        print(f"  Size={sz}: NSGA-II={nsga2_times[-1]:.2f}s, MOPSO={mopso_times[-1]:.2f}s")
    return {"sizes": sizes, "nsga2_times": nsga2_times, "mopso_times": mopso_times}
