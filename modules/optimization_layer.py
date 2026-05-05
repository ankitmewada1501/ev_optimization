"""
Optimization Layer - NSGA-II and MOPSO with venue-aware, dynamic station selection.

Enhancements over previous version:
  - n_stations is dynamic: derived from len(candidate_locations)
  - f1 (Cost) = opening_cost + grid_upgrade_cost per selected location (not uniform)
  - f3 (ROI)  = (annual_revenue - annual_opex) / total_capex  [replaces raw profit]
    ROI is orthogonal to coverage: equal-coverage solutions differ in cost/ROI.
  - Queue uses per-location charger count from CSV + M/M/c/K with waiting bays
  - Per-station metrics (EVs served, Wq, profit share) stored for interactive report

Objectives (all minimized internally):
  f1: minimize Total Capex (opening_cost + grid_upgrade_cost)
  f2: maximize Coverage             (-> minimize -coverage)
  f3: maximize Net ROI              (-> minimize -roi)  [was: profit, r=0.9986 w/ coverage]
  f4: minimize Weighted Mean Wq_min (time-of-use M/M/c/K queue model)
  Constraint: stations with P_block > MAX_BLOCKING (10%) receive a heavy penalty on f4.
"""
import time
import numpy as np
import pandas as pd
from modules.queue_simulation_layer import evaluate_fleet_queues


# ---------------------------------------------------------------------------
# Problem Definition
# ---------------------------------------------------------------------------

class EVStationProblem:
    """
    Multi-objective problem for EV station placement.
    Decision variable x: binary vector of length n_candidates.
    x[i] = 1 → select venue i from candidate_locations.
    """

    # Indore centroid for wards with no candidate anchor.
    _INDORE_LAT = 22.7196
    _INDORE_LON = 75.8577

    def __init__(self, cfg: dict, future_df: pd.DataFrame,
                 mc_scenarios: np.ndarray, candidate_df: pd.DataFrame):
        self.cfg          = cfg
        self.future_df    = future_df
        self.mc_scenarios = mc_scenarios
        self.candidate_df = candidate_df   # per-location demand + cost info

        # --- Queue / ToU config ---
        q = cfg["queue"]
        self.avg_charging_time = q["avg_charging_time_min"]
        self.wait_threshold    = q["wait_time_threshold_min"]
        self.tou_cfg           = q.get("time_of_use", None)

        # --- Per-location arrays (aligned with candidate_df index) ---
        self.n_stations        = len(candidate_df)
        self.chargers          = candidate_df["chargers"].values.astype(int)
        self.opening_cost      = candidate_df["opening_cost"].values.astype(float)
        self.grid_upgrade_cost = candidate_df["grid_upgrade_cost"].values.astype(float)
        self.total_capex       = candidate_df["total_capex"].values.astype(float)
        self.station_zones     = candidate_df["zone"].values
        self.base_demand       = candidate_df["daily_demand"].values.astype(float)
        self.location_type     = candidate_df["type"].values

        # --- Per-location Monte Carlo (INDEPENDENT noise per station) ---
        # Previously used one scalar noise factor per scenario, giving perfectly
        # correlated noise across stations (a fleet-wide shift, not true MC).
        # Draw independent N(1, σ) multipliers per (station, scenario).
        n_scen = mc_scenarios.shape[0] if mc_scenarios is not None else 100
        noise_frac = cfg.get("monte_carlo", {}).get("demand_noise_fraction", 0.15)
        rng = np.random.default_rng(seed=42)  # reproducible
        mults = rng.normal(loc=1.0, scale=noise_frac,
                           size=(self.n_stations, n_scen))
        self.mc_station = np.clip(self.base_demand[:, None] * mults, 0, None)

        # --- ToU pricing ---
        pr = cfg.get("pricing", {})
        self.peak_rate   = pr.get("peak_rate_per_session",   210.0)
        self.normal_rate = pr.get("normal_rate_per_session", 150.0)
        self.idle_rate   = pr.get("idle_rate_per_session",   112.0)

        # ToU hours
        tou = self.tou_cfg or {}
        self.peak_h   = tou.get("peak_hours",   3)
        self.normal_h = tou.get("normal_hours", 13)
        self.idle_h   = tou.get("idle_hours",   8)
        pk_mul = tou.get("peak_demand_multiplier",   2.8)
        nm_mul = tou.get("normal_demand_multiplier", 1.0)
        id_mul = tou.get("idle_demand_multiplier",   0.15)
        total_weight = self.peak_h * pk_mul + self.normal_h * nm_mul + self.idle_h * id_mul
        self.peak_frac   = (self.peak_h   * pk_mul) / total_weight
        self.normal_frac = (self.normal_h * nm_mul) / total_weight
        self.idle_frac   = (self.idle_h   * id_mul) / total_weight

        # --- Grid capacity ---
        self.grid_cap = cfg["economics"]["grid_capacity_kw"]
        unique_zones = list(set(self.station_zones))
        self.zone_station_map = {
            z: [j for j, sz in enumerate(self.station_zones) if sz == z]
            for z in unique_zones
        }

        # Fix 2: Zone-adjusted coverage radii (MoRTH guideline adapted for density)
        # Central (dense): 1.5 km   |  Outer zones (sparse): 2.5 km
        zone_radii = cfg.get("coverage", {}).get("zone_radii", {})
        default_r  = cfg["coverage"].get("radius_km", 2.0)
        self.cov_radius_per_station = np.array([
            zone_radii.get(str(self.candidate_df.iloc[j]["zone"]).strip(), default_r)
            for j in range(self.n_stations)
        ])
        self.cov_radius = default_r          # kept for backward compat
        self.op_cost_yr = cfg["economics"]["operating_cost_per_station_per_year"]

        # --- Improvement A: Radius-based coverage matrix (ward × station) ---
        # Ward centroid = mean lat/lon of candidates in ward (fallback: Indore CBD).
        ward_rows = (
            future_df[future_df["year"] == 2025]
            [["ward_no", "zone", "daily_demand_sessions"]]
            .drop_duplicates("ward_no")
            .reset_index(drop=True)
        )
        self.ward_nos       = ward_rows["ward_no"].values
        self.ward_zones     = ward_rows["zone"].values
        self.ward_demand    = ward_rows["daily_demand_sessions"].values.astype(float)
        self.total_ward_demand = float(self.ward_demand.sum())

        cand_lat = candidate_df["lat"].values.astype(float)
        cand_lon = candidate_df["lon"].values.astype(float)
        cand_ward = candidate_df["ward_no"].values.astype(int)

        ward_centroid = {}
        for w in self.ward_nos:
            mask = cand_ward == w
            if mask.any():
                ward_centroid[w] = (float(cand_lat[mask].mean()),
                                    float(cand_lon[mask].mean()))
            else:
                ward_centroid[w] = (self._INDORE_LAT, self._INDORE_LON)

        # Planar approximation around Indore (~22.7°N): 1°lat≈111km, 1°lon≈102km.
        KM_PER_DEG_LAT = 111.0
        KM_PER_DEG_LON = 111.0 * np.cos(np.radians(self._INDORE_LAT))

        ward_lat = np.array([ward_centroid[w][0] for w in self.ward_nos])
        ward_lon = np.array([ward_centroid[w][1] for w in self.ward_nos])

        dlat = (ward_lat[:, None] - cand_lat[None, :]) * KM_PER_DEG_LAT
        dlon = (ward_lon[:, None] - cand_lon[None, :]) * KM_PER_DEG_LON
        dist_km = np.sqrt(dlat * dlat + dlon * dlon)  # (n_wards, n_stations)

        # Use the ward's zone radius for its row.
        ward_radius = np.array([
            zone_radii.get(str(z).strip(), default_r) for z in self.ward_zones
        ])
        self.cover_mat = (dist_km <= ward_radius[:, None])  # bool (n_wards, n_stations)

        # --- Improvement C: Per-zone grid capacity (pre-build zone × station idx) ---
        self.zones_list = list({z for z in self.station_zones})

    def _tou_revenue(self, daily_sessions: float) -> float:
        """Compute daily revenue using per-period pricing."""
        peak_sessions   = daily_sessions * self.peak_frac
        normal_sessions = daily_sessions * self.normal_frac
        idle_sessions   = daily_sessions * self.idle_frac
        return (peak_sessions   * self.peak_rate
                + normal_sessions * self.normal_rate
                + idle_sessions   * self.idle_rate)

    MAX_BLOCKING = 0.10   # max acceptable blocking probability per station

    def evaluate(self, x: np.ndarray) -> tuple:
        """
        Evaluate 4 objectives for solution x (binary vector).
        Returns (f1_cost, f2_neg_coverage, f3_neg_roi, f4_mean_wq_min)
        Lower is better in all dimensions.
        """
        selected = np.where(x > 0.5)[0]
        n_sel = len(selected)

        if n_sel == 0:
            self._last_raw_wq = 0.0
            self._last_n_overloaded = 0
            return (0.0, 0.0, 0.0, 0.0)

        # --- f1: Total Capital Cost (opening + grid upgrade, varies by location) ---
        f1 = self.total_capex[selected].sum()

        # --- Improvement C: Grid capacity penalty (per-zone kW ≤ grid_cap) ---
        # 7.4 kW per AC charger × sum(selected chargers in zone) ≤ grid_cap.
        grid_penalty = 0.0
        for z, idx_list in self.zone_station_map.items():
            sel_in_zone = [j for j in idx_list if x[j] > 0.5]
            if not sel_in_zone:
                continue
            zone_kw = 7.4 * self.chargers[sel_in_zone].sum()
            if zone_kw > self.grid_cap:
                grid_penalty += (zone_kw - self.grid_cap) * 50000.0  # ₹/kW over
        f1 += grid_penalty

        # --- f2: Coverage (fraction of total ward demand within service radius) ---
        # Uses pre-built ward×station boolean cover matrix — a ward is served iff
        # any selected station sits within its zone-specific radius.
        if self.total_ward_demand > 0:
            covered_wards = self.cover_mat[:, selected].any(axis=1)
            coverage = float((self.ward_demand * covered_wards).sum()
                             / self.total_ward_demand)
        else:
            coverage = 0.0
        f2 = -coverage

        # --- f3: Net Annual ROI = (annual_profit) / total_capex  [Bug #2 fix] ---
        # ROI is genuinely orthogonal to coverage: two solutions with identical
        # coverage but different costs have different ROI, giving real trade-off.
        mc_sel = self.mc_station[selected, :]        # (n_sel, n_scenarios)
        mean_sessions_per_station = mc_sel.mean(axis=1)
        daily_rev = np.array([self._tou_revenue(s) for s in mean_sessions_per_station])
        annual_rev  = daily_rev.sum() * 365
        annual_opex = n_sel * self.op_cost_yr        # operating costs
        annual_profit = annual_rev - annual_opex
        roi = annual_profit / f1 if f1 > 0 else 0.0  # dimensionless ratio
        f3 = -roi

        # --- f4: Mean queue waiting time in minutes (M/M/c/K with waiting bays) ---
        mean_demand = mc_sel.mean(axis=1)
        q_metrics = evaluate_fleet_queues(
            mean_demand,
            self.avg_charging_time,
            self.chargers[selected],       # per-station charger counts!
            self.wait_threshold,
            tou_cfg=self.tou_cfg,
        )
        # Real Wq_min values from M/M/c/K model (no longer always 0)
        wq_mins = q_metrics["waiting_times"]   # list of Wq_min per station
        raw_wq = q_metrics["mean_Wq_min"]      # physical mean Wq, no penalties
        f4 = raw_wq

        # Blocking probability constraint: P_block > MAX_BLOCKING → heavy penalty
        # This raises f4 sharply for solutions where stations turn away >10% of EVs
        p_blocks = q_metrics["p_blocks"]
        n_overloaded = sum(1 for pb in p_blocks if pb > self.MAX_BLOCKING)
        if n_overloaded > 0:
            f4 += n_overloaded * 50.0   # 50-min penalty per overloaded station

        # Extra penalty for truly unstable stations
        f4 += q_metrics["n_unstable"] * 20.0
        self._last_n_infeasible = q_metrics["n_unstable"]
        self._last_raw_wq = raw_wq
        self._last_n_overloaded = n_overloaded

        return (f1, f2, f3, f4)

    def evaluate_detailed(self, x: np.ndarray) -> dict:
        """
        Full per-station breakdown for the interactive report.
        Returns dict with per-station metrics needed for map popups.
        """
        selected = np.where(x > 0.5)[0]
        if len(selected) == 0:
            return {}

        results = []
        tou = self.tou_cfg or {}
        for i in selected:
            loc      = self.candidate_df.iloc[i]
            c        = int(self.chargers[i])
            demand   = float(self.mc_station[i].mean())
            daily_rev = self._tou_revenue(demand)

            # Per-period session split
            pk_s  = demand * self.peak_frac
            nm_s  = demand * self.normal_frac
            id_s  = demand * self.idle_frac

            # M/M/c queue for this station
            from modules.queue_simulation_layer import station_queue_analysis_tou
            if tou:
                q_res = station_queue_analysis_tou(
                    demand, self.avg_charging_time, c, tou)
                wq = q_res.get("weighted_Wq_min", 0.0)
                pk_wq = q_res.get("peak_Wq_min", 0.0)
            else:
                wq = 0.0; pk_wq = 0.0

            # EVs served per day = demand sessions (M/M/c stable throughput ≈ demand)
            # Vehicle type split
            vt = self.cfg.get("vehicle_types", {})
            tw_frac = vt.get("two_wheeler_fraction", 0.45)
            fw_frac = vt.get("four_wheeler_fraction", 0.55)
            evs_2w = round(demand * tw_frac, 1)
            evs_4w = round(demand * fw_frac, 1)

            # Max simultaneous EVs (charger capacity limit)
            max_sim = c   # one EV per charger simultaneously

            results.append({
                "loc_id":            int(loc["loc_id"]),
                "name":              str(loc["name"]),
                "type":              str(loc["type"]),
                "ward_no":           int(loc["ward_no"]),
                "lat":               float(loc["lat"]),
                "lon":               float(loc["lon"]),
                "zone":              str(loc["zone"]),
                "chargers":          c,
                "max_simultaneous":  max_sim,
                "daily_sessions":    round(demand, 1),
                "evs_2wheeler":      evs_2w,
                "evs_4wheeler":      evs_4w,
                "peak_sessions":     round(pk_s, 1),
                "normal_sessions":   round(nm_s, 1),
                "idle_sessions":     round(id_s, 1),
                "peak_rate":         self.peak_rate,
                "normal_rate":       self.normal_rate,
                "idle_rate":         self.idle_rate,
                "daily_revenue":     round(daily_rev, 0),
                "annual_revenue":    round(daily_rev * 365, 0),
                "opening_cost":      float(loc["opening_cost"]),
                "grid_upgrade_cost": float(loc["grid_upgrade_cost"]),
                "total_capex":       float(loc["total_capex"]),
                "annual_opex":       self.op_cost_yr,
                "net_annual_profit": round(daily_rev * 365 - self.op_cost_yr - loc["total_capex"], 0),
                "coverage_fraction": round(demand / max(self.base_demand.sum(), 1), 4),
                "coverage_radius_km":float(self.cov_radius_per_station[i]),
                "mean_wq_min":       round(wq, 2),
                "peak_wq_min":       round(pk_wq, 2),
                "stable":            (wq < self.wait_threshold),
                "feasible":          (wq < np.inf),
                "utilisation_pct":   round(q_res.get("peak_rho", float("nan")) * 100, 1) if tou else 0.0,
            })
        return results

    def grid_feasible(self, x: np.ndarray) -> bool:
        """Check per-zone grid capacity (kW)."""
        for zone, indices in self.zone_station_map.items():
            zone_kw = sum(
                self.chargers[i] * 7.4   # 7.4 kW per AC charger
                for i in indices if x[i] > 0.5
            )
            if zone_kw > self.grid_cap:
                return False
        return True


# ---------------------------------------------------------------------------
# NSGA-II helpers (unchanged)
# ---------------------------------------------------------------------------

def _crowding_distance(front_objs: np.ndarray) -> np.ndarray:
    n, m = front_objs.shape
    dist = np.zeros(n)
    for k in range(m):
        order = np.argsort(front_objs[:, k])
        rng = front_objs[order[-1], k] - front_objs[order[0], k]
        dist[order[0]] = dist[order[-1]] = np.inf
        if rng <= 0:           # all values identical in this objective → skip
            continue
        for idx in range(1, n - 1):
            val = (front_objs[order[idx + 1], k]
                   - front_objs[order[idx - 1], k]) / rng
            if np.isfinite(val):
                dist[order[idx]] += val
    return np.nan_to_num(dist, nan=0.0, posinf=1e9)



def _hypervolume_2d(pareto_obj2d: np.ndarray, ref_cost: float, ref_cov: float = 1.0) -> float:
    """
    2D hypervolume on normalised objectives.
    Input:  pareto_obj2d[:, 0] = f1 (cost, minimise, positive)
            pareto_obj2d[:, 1] = f2 (-coverage, minimise, negative, range -1..0)
    Steps:
      1. Flip f2 to coverage = -f2 in [0, 1] (higher = better = lower is worse)
      2. Normalise f1 to [0, 1] using ref_cost
      3. Build 2D front: cost_norm (minimise) vs uncov = 1 - coverage (minimise)
      4. Sweep: ref = (1, 1)  => maximise area dominated within unit square
    """
    if len(pareto_obj2d) == 0:
        return 0.0
    cost_norm = pareto_obj2d[:, 0] / (ref_cost + 1e-9)       # in [0, ~1]
    coverage  = -pareto_obj2d[:, 1]                           # in [0, 1]
    uncov     = 1.0 - coverage                                # lower = better
    # Stack normalised minimisation objectives: cost_norm, uncov
    pts2d = np.column_stack([cost_norm, uncov])
    # Keep only Pareto-non-dominated (already sorted, but re-filter)
    # Reference point: (1, 1) + small margin
    rx, ry = 1.05, 1.05
    # Only include pts dominated by reference
    valid = (pts2d[:, 0] < rx) & (pts2d[:, 1] < ry)
    pts2d = pts2d[valid]
    if len(pts2d) == 0:
        return 0.0
    # Sort by f1 (cost_norm) ascending
    order = np.argsort(pts2d[:, 0])
    pts2d = pts2d[order]
    hv = 0.0
    y_prev = ry
    for p in pts2d:
        hv += (rx - p[0]) * (y_prev - p[1])
        y_prev = p[1]
    return max(hv, 0.0)

def _fast_nondominated_sort(objs: np.ndarray) -> list:
    n = len(objs)
    domination_count = np.zeros(n, dtype=int)
    dominated_sets   = [[] for _ in range(n)]
    fronts           = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if np.all(objs[p] <= objs[q]) and np.any(objs[p] < objs[q]):
                dominated_sets[p].append(q)
            elif np.all(objs[q] <= objs[p]) and np.any(objs[q] < objs[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_sets[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)
    return [f for f in fronts if f]


def _tournament_select(pop, objs, ranks, crowd):
    i, j = np.random.randint(0, len(pop), 2)
    if ranks[i] < ranks[j]:   return pop[i]
    elif ranks[j] < ranks[i]: return pop[j]
    elif crowd[i] > crowd[j]: return pop[i]
    else:                      return pop[j]


# ---------------------------------------------------------------------------
# NSGA-II
# ---------------------------------------------------------------------------

def run_nsga2(problem: EVStationProblem, cfg: dict) -> tuple:
    """Run NSGA-II with convergence early-stop."""
    opt        = cfg["optimization"]["nsga2"]
    pop_size   = opt["population_size"]
    n_gen      = opt["n_generations"]
    cx_prob    = opt["crossover_prob"]
    mut_prob   = opt["mutation_prob"]
    conv_win   = opt.get("convergence_window", 30)
    conv_tol   = opt.get("convergence_tol", 0.001)
    n          = problem.n_stations
    t_start    = time.time()

    pop  = (np.random.rand(pop_size, n) > 0.5).astype(float)
    objs = np.array([problem.evaluate(ind) for ind in pop])
    history = []

    for gen in range(n_gen):
        fronts = _fast_nondominated_sort(objs)
        ranks  = np.zeros(len(pop), dtype=int)
        crowd  = np.zeros(len(pop))
        for rank, front in enumerate(fronts):
            for idx in front: ranks[idx] = rank
            if len(front) > 1:
                cd = _crowding_distance(objs[front])
                for k, idx in enumerate(front): crowd[idx] = cd[k]

        offspring = []
        while len(offspring) < pop_size:
            p1 = _tournament_select(pop, objs, ranks, crowd)
            p2 = _tournament_select(pop, objs, ranks, crowd)
            mask  = np.random.rand(n) < 0.5
            child = np.where(mask, p1, p2) if np.random.rand() < cx_prob else p1.copy()
            flip  = np.random.rand(n) < mut_prob
            child = np.where(flip, 1 - child, child)
            offspring.append(child)

        offspring  = np.array(offspring)
        off_objs   = np.array([problem.evaluate(ind) for ind in offspring])
        comb_pop   = np.vstack([pop, offspring])
        comb_objs  = np.vstack([objs, off_objs])
        all_fronts = _fast_nondominated_sort(comb_objs)
        all_ranks  = np.zeros(len(comb_pop), dtype=int)
        all_crowd  = np.zeros(len(comb_pop))
        for rank, front in enumerate(all_fronts):
            for idx in front: all_ranks[idx] = rank
            if len(front) > 1:
                cd = _crowding_distance(comb_objs[front])
                for k, idx in enumerate(front): all_crowd[idx] = cd[k]
        order    = np.lexsort((-all_crowd, all_ranks))
        selected = order[:pop_size]
        pop      = comb_pop[selected]
        objs     = comb_objs[selected]

        pf_size  = len(all_fronts[0])
        pf_objs  = comb_objs[all_fronts[0]]
        fin_mask = np.all(np.isfinite(pf_objs[:, :2]), axis=1)
        pf2      = pf_objs[fin_mask, :2] if fin_mask.any() else pf_objs[:1, :2]
        # Reference point: 10% beyond worst observed values in first 2 objectives
        if gen == 0:
            _ns2_ref_cost = float(pf2[:, 0].max() * 1.1 + 1.0)
        hv_val = _hypervolume_2d(pf2, _ns2_ref_cost)
        history.append(hv_val)
        if (gen + 1) % 50 == 0 or gen == 0:
            print(f"    NSGA-II gen {gen+1:>4}/{n_gen}  |  Pareto: {pf_size:3d}"
                  f"  |  HV: {hv_val:.4e}")
        # Convergence: HV improvement < conv_tol over conv_win gens; min 100 gens
        min_gens = max(conv_win + 1, 100)
        if gen >= min_gens:
            recent = history[-conv_win:]
            best   = max(recent)
            delta  = (best - min(recent)) / (abs(best) + 1e-12)
            if delta < conv_tol:
                print(f"    NSGA-II converged at gen {gen+1} (HV_delta={delta:.6f})")
                break

    runtime      = time.time() - t_start
    final_fronts = _fast_nondominated_sort(objs)
    pareto_idx   = final_fronts[0]
    # history contains centroid-norms; expose as-is (visualizer will plot them)
    return pop[pareto_idx], objs[pareto_idx], history, runtime


# ---------------------------------------------------------------------------
# MOPSO archive
# ---------------------------------------------------------------------------

class _Archive:
    def __init__(self, max_size: int = 100):
        self.max_size   = max_size
        self.solutions  = []
        self.objectives = []

    def update(self, positions, objectives):
        all_pos = np.array(list(self.solutions) + list(positions))
        all_obj = np.array(list(self.objectives) + list(objectives))
        fronts  = _fast_nondominated_sort(all_obj)
        nd_idx  = fronts[0]
        nd_pos  = all_pos[nd_idx]
        nd_obj  = all_obj[nd_idx]
        if len(nd_idx) > self.max_size:
            cd   = _crowding_distance(nd_obj)
            keep = np.argsort(cd)[-self.max_size:]
            nd_pos = nd_pos[keep]; nd_obj = nd_obj[keep]
        self.solutions  = list(nd_pos)
        self.objectives = list(nd_obj)

    def select_leader(self):
        if not self.solutions: return None
        cd    = _crowding_distance(np.array(self.objectives))
        cd    = np.nan_to_num(cd, nan=1.0, posinf=1e9, neginf=0.0)
        total = cd.sum()
        if total <= 0 or not np.isfinite(total):
            return np.array(self.solutions[np.random.randint(len(self.solutions))])
        probs = cd / total
        probs = probs / probs.sum()
        return np.array(self.solutions[np.random.choice(len(self.solutions), p=probs)])


def run_mopso(problem: EVStationProblem, cfg: dict) -> tuple:
    """Run MOPSO with convergence early-stop."""
    opt        = cfg["optimization"]["mopso"]
    n_part     = opt["n_particles"]
    n_iter     = opt["n_iterations"]
    w          = opt["inertia"]
    c1, c2     = opt["c1"], opt["c2"]
    conv_win   = opt.get("convergence_window", 30)
    conv_tol   = opt.get("convergence_tol", 0.001)
    n          = problem.n_stations
    t_start    = time.time()

    pos   = np.random.rand(n_part, n)
    vel   = np.random.uniform(-0.5, 0.5, (n_part, n))
    pbest_pos = pos.copy()
    pbest_obj = np.array([problem.evaluate((p > 0.5).astype(float)) for p in pos])

    archive = _Archive(max_size=100)
    archive.update(pos, pbest_obj)
    history = []

    for it in range(n_iter):
        leader = archive.select_leader()
        if leader is None: leader = pos[0]
        r1, r2  = np.random.rand(n_part, n), np.random.rand(n_part, n)
        vel     = w * vel + c1 * r1 * (pbest_pos - pos) + c2 * r2 * (leader - pos)
        vel     = np.clip(vel, -4, 4)
        pos     = np.clip(pos + vel, 0, 1)
        cur_obj = np.array([problem.evaluate((p > 0.5).astype(float)) for p in pos])

        for i in range(n_part):
            p_obj, c_obj = pbest_obj[i], cur_obj[i]
            c_dom = np.all(c_obj <= p_obj) and np.any(c_obj < p_obj)
            p_dom = np.all(p_obj <= c_obj) and np.any(p_obj < c_obj)
            if c_dom:
                pbest_pos[i] = pos[i]; pbest_obj[i] = c_obj
            elif not p_dom and np.random.rand() < 0.5:
                pbest_pos[i] = pos[i]; pbest_obj[i] = c_obj

        archive.update(pos, cur_obj)
        arc_size = len(archive.solutions)
        if archive.objectives:
            arc_obj  = np.array(archive.objectives)
            fin_mask = np.all(np.isfinite(arc_obj[:, :2]), axis=1)
            arc2     = arc_obj[fin_mask, :2] if fin_mask.any() else arc_obj[:1, :2]
            if it == 0:
                _mo_ref_cost = float(arc2[:, 0].max() * 1.1 + 1.0)
            hv_val = _hypervolume_2d(arc2, _mo_ref_cost)
        else:
            hv_val = 0.0
        history.append(hv_val)
        if (it + 1) % 50 == 0 or it == 0:
            print(f"    MOPSO iter {it+1:>4}/{n_iter}  |  Archive: {arc_size:3d}"
                  f"  |  HV: {hv_val:.4e}")
        min_iters = max(conv_win + 1, 100)
        if it >= min_iters:
            recent = history[-conv_win:]
            best   = max(recent)
            delta  = (best - min(recent)) / (abs(best) + 1e-12)
            if delta < conv_tol:
                print(f"    MOPSO converged at iter {it+1} (HV_delta={delta:.6f})")
                break

    runtime    = time.time() - t_start
    pareto_pos = np.array([(p > 0.5).astype(float) for p in archive.solutions])
    pareto_obj = np.array(archive.objectives)
    return pareto_pos, pareto_obj, history, runtime


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_optimization_layer(cfg: dict, future_df: pd.DataFrame,
                            mc_scenarios: np.ndarray,
                            candidate_df: pd.DataFrame) -> dict:
    """Run both optimizers. Returns results dict."""
    problem = EVStationProblem(cfg, future_df, mc_scenarios, candidate_df)
    print(f"[Optimization] {problem.n_stations} candidate locations loaded.")

    print("[Optimization] Running NSGA-II...")
    ns2_sol, ns2_obj, ns2_hist, ns2_time = run_nsga2(problem, cfg)
    print(f"  NSGA-II: {len(ns2_sol)} Pareto solutions in {ns2_time:.1f}s")

    print("[Optimization] Running MOPSO...")
    mo_sol, mo_obj, mo_hist, mo_time = run_mopso(problem, cfg)
    print(f"  MOPSO: {len(mo_sol)} solutions in {mo_time:.1f}s")

    def _save(solutions, objectives, path, algo):
        n_cands = solutions.shape[1]
        cols = [f"station_{i}" for i in range(n_cands)]
        df = pd.DataFrame(solutions, columns=cols)
        df["cost"]        = objectives[:, 0]
        df["coverage"]    = -objectives[:, 1]
        df["roi"]         = -objectives[:, 2]   # Net ROI (replaces profit)
        df["mean_wq_min"] = objectives[:, 3]    # Real Wq in minutes from M/M/c/K
        df["algorithm"]   = algo
        df.to_csv(path, index=False)
        print(f"  Saved {path}")

    _save(ns2_sol, ns2_obj, cfg["data"]["optimal_nsga2_csv"], "NSGA-II")
    _save(mo_sol,  mo_obj,  cfg["data"]["optimal_mopso_csv"], "MOPSO")

    # Recompute raw (unpenalised) wait times for every Pareto solution so the
    # visualization layer can show the physical Wq separately from the f4
    # penalty used during search.
    def _raw_wq(solutions):
        out = np.zeros(len(solutions))
        for i, x in enumerate(solutions):
            problem.evaluate(x)
            out[i] = getattr(problem, "_last_raw_wq", 0.0)
        return out

    ns2_raw_wq = _raw_wq(ns2_sol)
    mo_raw_wq  = _raw_wq(mo_sol)

    # Compute per-station detail for best NSGA-II solution (highest coverage)
    best_idx_ns2 = int(np.argmax(-ns2_obj[:, 1]))
    best_sol_x   = ns2_sol[best_idx_ns2]
    station_details = problem.evaluate_detailed(best_sol_x)

    return {
        "problem": problem,
        "nsga2":   {"solutions": ns2_sol, "objectives": ns2_obj,
                    "history": ns2_hist, "runtime": ns2_time,
                    "raw_wq_min": ns2_raw_wq},
        "mopso":   {"solutions": mo_sol,  "objectives": mo_obj,
                    "history": mo_hist,  "runtime": mo_time,
                    "raw_wq_min": mo_raw_wq},
        "best_solution_x":     best_sol_x,
        "station_details":     station_details,
    }
