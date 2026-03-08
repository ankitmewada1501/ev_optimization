"""
Queue Simulation Layer — M/M/c/K finite-capacity queuing model (K = c).

Model: M/M/c/K  where K = c  (total system capacity = number of servers)
This is the ERLANG-B LOSS MODEL — no waiting room.
Arriving EVs finding all c chargers occupied are BLOCKED (drive away).

Real-life EV charging demand pattern (24 hours):
  - Peak hours  : 3h  (8-10 AM + 5-7 PM rush)    demand multiplier × 2.8
  - Normal hours: 13h (regular daytime / evening) demand multiplier × 1.0
  - Idle hours  : 8h  (late night / early morning) demand multiplier × 0.15

Outputs (replacing old Wq_min):
  - P_block      : blocking probability  (Erlang-B formula)
  - W_sec        : mean sojourn time in SECONDS for served customers = 1/μ × 3600
  - Wq_sec       : mean queue waiting time in SECONDS = 0 (no waiting room in K=c)
  - throughput   : effective arrival rate λ × (1 - P_block)
  - rho          : per-server utilisation = λ_eff / (c × μ)
"""
import math
import numpy as np


# ---------------------------------------------------------------------------
# Core — Erlang-B (M/M/c/c = M/M/c/K with K=c)
# ---------------------------------------------------------------------------

def erlang_b(c: int, offered_load: float) -> float:
    """
    Erlang-B blocking probability for M/M/c/K with K=c.

    Parameters
    ----------
    c             : number of servers (chargers)
    offered_load  : a = λ/μ  (dimensionless traffic intensity)

    Returns
    -------
    P_block : probability that an arriving EV is blocked (all c chargers busy)
    """
    if c <= 0 or offered_load <= 0:
        return 0.0
    # Recursive Erlang-B (numerically stable, avoids factorial overflow)
    B = 1.0
    for k in range(1, c + 1):
        B = (offered_load * B) / (k + offered_load * B)
    return min(B, 1.0)


def mm_c_k_metrics(lam: float, mu: float, c: int, K: int = None) -> dict:
    """
    M/M/c/K metrics. Default K=c (Erlang-B, no waiting room).

    Parameters
    ----------
    lam  : arrival rate (sessions/hour)
    mu   : service rate (sessions/hour) = 60 / avg_charging_time_min
    c    : number of servers (chargers)
    K    : total system capacity; if None defaults to c (Erlang-B)

    Returns
    -------
    dict with P_block, W_sec, Wq_sec, rho, throughput, stable
    """
    if K is None:
        K = c  # default: Erlang-B

    if lam <= 0 or mu <= 0 or c <= 0:
        return {
            "P_block":    0.0,
            "Wq_sec":     0.0,
            "W_sec":      round(3600.0 / max(mu, 1e-9), 2),
            "rho":        0.0,
            "throughput": 0.0,
            "stable":     True,
            "Lq":         0.0,
        }

    offered_load = lam / mu   # a = λ/μ

    if K == c:
        # ── Erlang-B (loss model) ──────────────────────────────────────────
        P_block = erlang_b(c, offered_load)
        lam_eff = lam * (1.0 - P_block)
        # Utilisation per server based on effective throughput
        rho     = lam_eff / (c * mu) if c * mu > 0 else 0.0
        W_sec   = 3600.0 / mu         # mean service time in seconds (1/μ × 3600)
        Wq_sec  = 0.0                 # no waiting room → Wq = 0
        Lq      = 0.0
        stable  = True                # loss systems are always stable
    else:
        # ── General M/M/c/K (finite buffer) ──────────────────────────────
        # Total waiting room = K - c
        rho_0 = offered_load / c      # per-server utilisation
        # Compute normalisation constant P(0) via steady-state probabilities
        try:
            # Sum for n = 0..c
            sum_sc = sum(offered_load**n / math.factorial(n) for n in range(c + 1))
            # Sum for n = c+1..K
            sum_ov = sum(
                offered_load**n / (math.factorial(c) * c**(n - c))
                for n in range(c + 1, K + 1)
            )
            p0 = 1.0 / (sum_sc + sum_ov) if (sum_sc + sum_ov) > 0 else 0.0
        except (OverflowError, ZeroDivisionError):
            p0 = 0.0

        # P(K) = blocking probability
        if K <= c:
            pK = (offered_load**K / math.factorial(K)) * p0
        else:
            pK = (offered_load**K / (math.factorial(c) * c**(K - c))) * p0
        P_block = min(max(pK, 0.0), 1.0)

        lam_eff = lam * (1.0 - P_block)
        rho     = lam_eff / (c * mu) if c * mu > 0 else 0.0

        # Mean queue length Lq
        if abs(1.0 - rho_0) < 1e-9:
            Lq = 0.0
        else:
            try:
                term = ((offered_load**c * rho_0) /
                        (math.factorial(c) * (1 - rho_0)**2))
                Lq = p0 * term * (
                    1 - rho_0**(K - c) - (K - c) * rho_0**(K - c) * (1 - rho_0)
                )
                Lq = max(Lq, 0.0)
            except (OverflowError, ZeroDivisionError):
                Lq = 0.0

        Wq_sec = (Lq / lam_eff * 3600.0) if lam_eff > 0 else 0.0
        W_sec  = Wq_sec + 3600.0 / mu
        stable = rho < 1.0

    return {
        "P_block":    round(P_block, 6),
        "Wq_sec":     round(Wq_sec, 2),
        "W_sec":      round(W_sec,  2),
        "rho":        round(rho, 4),
        "throughput": round(lam * (1.0 - P_block), 4),
        "stable":     stable,
        "Lq":         round(Lq if K != c else 0.0, 4),
    }


# ---------------------------------------------------------------------------
# Time-of-Use Queue Analysis (M/M/c/K with K=c per period)
# ---------------------------------------------------------------------------

def station_queue_analysis_tou(
    daily_demand_sessions: float,
    avg_charging_time_min: float,
    chargers_per_station: int,
    tou_cfg: dict,
    K: int = None,        # default K=c (Erlang-B)
) -> dict:
    """
    Compute M/M/c/K metrics for peak / normal / idle periods.
    Returns time-weighted blocking probability and sojourn time in seconds.
    """
    pk_h = tou_cfg["peak_hours"]
    nm_h = tou_cfg["normal_hours"]
    id_h = tou_cfg["idle_hours"]
    pk_m = tou_cfg["peak_demand_multiplier"]
    nm_m = tou_cfg["normal_demand_multiplier"]
    id_m = tou_cfg["idle_demand_multiplier"]

    total_w = pk_h * pk_m + nm_h * nm_m + id_h * id_m
    pk_s = daily_demand_sessions * (pk_h * pk_m) / total_w
    nm_s = daily_demand_sessions * (nm_h * nm_m) / total_w
    id_s = daily_demand_sessions * (id_h * id_m) / total_w

    mu = 60.0 / avg_charging_time_min          # service rate (sessions/hr)
    c  = chargers_per_station
    if K is None:
        K = c                                   # Erlang-B by default

    lam_pk = pk_s / pk_h if pk_h > 0 else 0.0
    lam_nm = nm_s / nm_h if nm_h > 0 else 0.0
    lam_id = id_s / id_h if id_h > 0 else 0.0

    m_pk = mm_c_k_metrics(lam_pk, mu, c, K)
    m_nm = mm_c_k_metrics(lam_nm, mu, c, K)
    m_id = mm_c_k_metrics(lam_id, mu, c, K)

    # Time-weighted blocking probability
    total_h  = pk_h + nm_h + id_h
    w_pblock = ((m_pk["P_block"] * pk_h +
                 m_nm["P_block"] * nm_h +
                 m_id["P_block"] * id_h) / total_h) if total_h > 0 else 0.0

    # Time-weighted sojourn (seconds) — weighted by arrivals that are served
    w_W = ((m_pk["W_sec"] * pk_h +
            m_nm["W_sec"] * nm_h +
            m_id["W_sec"] * id_h) / total_h) if total_h > 0 else 0.0

    # Peak Wq in seconds (always 0 for K=c)
    wq_peak_sec = m_pk["Wq_sec"]

    return {
        "peak":             m_pk,
        "normal":           m_nm,
        "idle":             m_id,
        "lam_peak_hr":      round(lam_pk, 4),
        "lam_normal_hr":    round(lam_nm, 4),
        "lam_idle_hr":      round(lam_id, 4),
        "peak_sessions":    round(pk_s, 2),
        "normal_sessions":  round(nm_s, 2),
        "idle_sessions":    round(id_s, 2),
        # Primary metrics (seconds-based)
        "weighted_P_block": round(w_pblock, 6),
        "peak_P_block":     round(m_pk["P_block"], 6),
        "normal_P_block":   round(m_nm["P_block"], 6),
        "idle_P_block":     round(m_id["P_block"], 6),
        "weighted_W_sec":   round(w_W, 2),    # mean sojourn (seconds)
        "peak_Wq_sec":      round(wq_peak_sec, 2),  # always 0 for Erlang-B
        "weighted_Wq_sec":  0.0,              # no queue → Wq=0
        "peak_rho":         round(m_pk["rho"], 4),
        "stable":           True,             # Erlang-B is always stable
        # Legacy aliases so old code doesn't break
        "weighted_Wq_min":  0.0,
        "peak_Wq_min":      0.0,
        "normal_Wq_min":    0.0,
        "idle_Wq_min":      0.0,
    }


# ---------------------------------------------------------------------------
# Fleet evaluation (used by optimizer)
# ---------------------------------------------------------------------------

def evaluate_fleet_queues(
    allocations: np.ndarray,
    avg_charging_time_min: float,
    chargers_per_station,
    wait_threshold_min: float = 20.0,   # kept for back-compat, not used in K=c
    tou_cfg: dict = None,
    K: int = None,                       # default K=c (Erlang-B)
) -> dict:
    """
    Evaluate M/M/c/K (K=c Erlang-B) across a fleet of stations.
    Primary metric: blocking probability (P_block) per station.
    """
    n = len(allocations)
    if np.isscalar(chargers_per_station):
        chargers_arr = np.full(n, int(chargers_per_station), dtype=int)
    else:
        chargers_arr = np.asarray(chargers_per_station, dtype=int)

    p_blocks   = []
    W_secs     = []
    n_unstable = 0       # always 0 for Erlang-B, kept for compat

    for i, demand in enumerate(allocations):
        c  = int(chargers_arr[i])
        Kk = c if K is None else K   # default Erlang-B

        if tou_cfg is not None:
            result   = station_queue_analysis_tou(
                demand, avg_charging_time_min, c, tou_cfg, Kk)
            p_block  = result["weighted_P_block"]
            w_sec    = result["weighted_W_sec"]
        else:
            mu      = 60.0 / avg_charging_time_min
            offered = demand / 18.0 / mu   # rough λ/μ
            m       = mm_c_k_metrics(demand / 18.0, mu, c, Kk)
            p_block = m["P_block"]
            w_sec   = m["W_sec"]

        p_blocks.append(p_block)
        W_secs.append(w_sec)

    mean_pb = float(np.mean(p_blocks)) if p_blocks else 0.0
    max_pb  = float(np.max(p_blocks))  if p_blocks else 0.0
    mean_W  = float(np.mean(W_secs))   if W_secs   else 0.0

    # SLA: blocking probability ≤ 5% (standard teletraffic grade-of-service)
    sla_ok = max_pb <= 0.05

    return {
        # Primary M/M/c/K outputs
        "mean_P_block":      round(mean_pb, 6),
        "max_P_block":       round(max_pb,  6),
        "mean_W_sec":        round(mean_W,  2),
        "p_blocks":          p_blocks,
        "W_secs":            W_secs,
        "n_unstable":        n_unstable,   # always 0 for Erlang-B
        "constraint_ok":     sla_ok,
        # Legacy aliases for backward compat
        "waiting_times":     p_blocks,     # optimizer uses this field
        "mean_Wq_min":       0.0,
        "peak_waiting_times": p_blocks,
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tou = {
        "peak_hours": 3, "normal_hours": 13, "idle_hours": 8,
        "peak_demand_multiplier": 2.8,
        "normal_demand_multiplier": 1.0,
        "idle_demand_multiplier": 0.15,
    }
    r = station_queue_analysis_tou(80, 30, 4, tou)
    print(f"Peak   P_block: {r['peak_P_block']:.4f}  W_sec: {r['weighted_W_sec']:.1f}s")
    print(f"Normal P_block: {r['normal_P_block']:.4f}")
    print(f"Weighted P_block: {r['weighted_P_block']:.4f} | Wq_sec=0 (Erlang-B)")

    allocs   = np.array([60.0, 80.0, 40.0])
    chargers = np.array([4, 8, 2])
    res = evaluate_fleet_queues(allocs, 30, chargers, tou_cfg=tou)
    print(f"\nFleet: mean_P_block={res['mean_P_block']:.4f}  "
          f"max_P_block={res['max_P_block']:.4f}  "
          f"mean_W_sec={res['mean_W_sec']:.1f}s")
