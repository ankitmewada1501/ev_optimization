"""
Data Layer - Loads ward data, candidate venue locations, runs Monte Carlo demand
simulation, and produces future_demand.csv.

New in this version:
  - Loads candidate_locations.csv (30 venue-typed charging locations)
  - Merges ward-level demand into each candidate location
  - Each candidate inherits its ward's daily demand (proportional share by
    number of candidates in that ward)
"""
import os
import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_wards(cfg: dict) -> pd.DataFrame:
    """Load Indore wards CSV."""
    return pd.read_csv(cfg["data"]["wards_csv"])


def load_candidate_locations(cfg: dict) -> pd.DataFrame:
    """Load candidate EV station locations (parking/fuel/restaurant/mall)."""
    path = cfg["data"]["candidate_locations_csv"]
    df = pd.read_csv(path)
    print(f"[Data Layer] Loaded {len(df)} candidate locations "
          f"({df['type'].value_counts().to_dict()})")
    return df


def logistic_ev_rate(t: float, K: float, a: float, t0: float) -> float:
    """EV adoption fraction at year t using logistic growth."""
    return K / (1.0 + np.exp(-a * (t - t0)))


def forecast_ev_demand(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Apply CAGR population growth and logistic EV adoption to forecast ward demand."""
    fc = cfg["forecasting"]
    base_year = fc["base_year"]
    r = fc["cagr_rate"]
    K = fc["ev_adoption"]["K"]
    a = fc["ev_adoption"]["a"]
    t0 = fc["ev_adoption"]["t0"]
    ev_per_1000 = fc["ev_per_1000_people"]

    records = []
    for _, row in df.iterrows():
        pop_base = row["population_2011"]
        for yr in fc["forecast_years"]:
            t = yr - base_year
            pop_future = pop_base * ((1 + r) ** t)
            ev_rate = logistic_ev_rate(yr, K, a, t0)
            n_evs = pop_future * ev_rate * (ev_per_1000 / 1000.0)
            daily_demand = n_evs * 0.20   # 20% of EVs charge daily
            records.append({
                "ward_no":               row["ward_no"],
                "ward_name":             row["ward_name"],
                "lgd_code":              row["lgd_code"],
                "zone":                  row["zone"],
                "year":                  yr,
                "population_future":     round(pop_future, 0),
                "ev_adoption_rate":      round(ev_rate, 4),
                "n_evs":                 round(n_evs, 1),
                "daily_demand_sessions": round(daily_demand, 1),
            })
    return pd.DataFrame(records)


def build_candidate_demand(
    locations_df: pd.DataFrame,
    future_df: pd.DataFrame,
    year: int = 2025,
) -> pd.DataFrame:
    """
    Assign expected daily demand to each candidate location for a given year.
    Demand is split equally among candidates in the same ward.
    """
    ward_demand = (
        future_df[future_df["year"] == year]
        .set_index("ward_no")["daily_demand_sessions"]
    )
    ward_count = locations_df.groupby("ward_no")["loc_id"].count()
    global_avg = ward_demand.mean()

    rows = []
    for _, loc in locations_df.iterrows():
        wno = loc["ward_no"]
        n_in_ward = ward_count.get(wno, 1)
        share = ward_demand.get(wno, global_avg) / n_in_ward
        rows.append({
            "loc_id":            int(loc["loc_id"]),
            "name":              loc["name"],
            "type":              loc["type"],
            "ward_no":           int(loc["ward_no"]),
            "lat":               float(loc["lat"]),
            "lon":               float(loc["lon"]),
            "size":              loc["size"],
            "chargers":          int(loc["chargers"]),
            "opening_cost":      float(loc["opening_cost"]),
            "grid_upgrade_cost": float(loc["grid_upgrade_cost"]),
            "total_capex":       float(loc["opening_cost"]) + float(loc["grid_upgrade_cost"]),
            "zone":              loc["zone"],
            "daily_demand":      round(float(share), 2),
            "auto_generated":    False,
        })
    return pd.DataFrame(rows)


def generate_dynamic_candidates(
    base_candidate_df: pd.DataFrame,
    future_df: pd.DataFrame,
    cfg: dict,
    year: int = 2025,
) -> pd.DataFrame:
    """
    Dynamically add extra candidate stations for high-demand wards.

    Logic:
      - Compute daily demand per ward for the given year
      - Find wards above the 75th percentile (high demand) → up to 3 stations allowed
      - Find wards between 50th-75th percentile (medium demand) → up to 2 stations allowed
      - For wards that need more stations, add synthetic parking candidates
        (small offset in lat/lon to make them distinct on the map)
    """
    dc = cfg.get("dynamic_candidates", {})
    if not dc.get("enable", True):
        return base_candidate_df

    hi_pct = dc.get("high_demand_percentile",   75)
    md_pct = dc.get("medium_demand_percentile", 50)
    max_hi = dc.get("max_stations_high_demand",  3)
    max_md = dc.get("max_stations_medium_demand", 2)
    extra_chargers  = dc.get("default_chargers_extra",  4)
    extra_open_cost = dc.get("extra_opening_cost",     500000)
    extra_grid_cost = dc.get("extra_grid_upgrade_cost", 200000)

    ward_demand_25 = (
        future_df[future_df["year"] == year]
        .set_index("ward_no")["daily_demand_sessions"]
    )
    hi_thresh = ward_demand_25.quantile(hi_pct / 100)
    md_thresh = ward_demand_25.quantile(md_pct / 100)

    # Count existing candidates per ward
    ward_cand_count = base_candidate_df.groupby("ward_no")["loc_id"].count()

    new_rows = []
    next_id  = int(base_candidate_df["loc_id"].max()) + 1
    ward_meta = (
        future_df[future_df["year"] == year]
        [["ward_no", "ward_name", "zone"]]
        .set_index("ward_no")
    )

    for ward_no, demand in ward_demand_25.items():
        if demand >= hi_thresh:
            target = max_hi
        elif demand >= md_thresh:
            target = max_md
        else:
            continue  # no extra stations for low-demand wards

        existing = int(ward_cand_count.get(ward_no, 0))
        needed   = target - existing
        if needed <= 0:
            continue

        # Get representative lat/lon from existing candidate in this ward
        mask = base_candidate_df["ward_no"] == ward_no
        if mask.any():
            ref = base_candidate_df[mask].iloc[0]
            base_lat = float(ref["lat"])
            base_lon = float(ref["lon"])
            zone     = ref["zone"]
        elif ward_no in ward_meta.index:
            # Approximate Indore position with a small ward-indexed offset
            base_lat = 22.72 + (ward_no % 10) * 0.005
            base_lon = 75.86 + (ward_no % 8)  * 0.006
            zone     = ward_meta.loc[ward_no, "zone"] if "zone" in ward_meta.columns else "Unknown"
        else:
            continue

        wname = ward_meta.loc[ward_no, "ward_name"] if ward_no in ward_meta.index else f"Ward {ward_no}"

        for k in range(needed):
            # Small offset so markers don't overlap on the map
            offset_lat = base_lat + (k + 1) * 0.003
            offset_lon = base_lon + (k + 1) * 0.002
            demand_share = demand / (existing + needed)
            new_rows.append({
                "loc_id":            next_id,
                "name":              f"{wname} Extra Parking {k+1}",
                "type":              "parking",
                "ward_no":           int(ward_no),
                "lat":               round(offset_lat, 6),
                "lon":               round(offset_lon, 6),
                "size":              "medium",
                "chargers":          extra_chargers,
                "opening_cost":      float(extra_open_cost),
                "grid_upgrade_cost": float(extra_grid_cost),
                "total_capex":       float(extra_open_cost + extra_grid_cost),
                "zone":              zone,
                "daily_demand":      round(float(demand_share), 2),
                "auto_generated":    True,
            })
            next_id += 1

    if new_rows:
        extra_df = pd.DataFrame(new_rows)
        result = pd.concat([base_candidate_df, extra_df], ignore_index=True)
        auto_count = len(new_rows)
        print(f"[Data Layer] Dynamic candidates: added {auto_count} extra slots "
              f"for high/medium demand wards → total {len(result)} candidates")
        return result
    return base_candidate_df


def monte_carlo_demand(future_df: pd.DataFrame, cfg: dict) -> np.ndarray:
    """
    Generate Monte Carlo scenarios for each ward-year combination.
    Returns array of shape (n_scenarios, n_wards_years).
    D_i ~ Normal(mu_i, noise_fraction * mu_i)
    """
    mc = cfg["monte_carlo"]
    n_scenarios = mc["n_scenarios"]
    noise = mc["demand_noise_fraction"]
    mu = future_df["daily_demand_sessions"].values.astype(float)
    scenarios = np.random.normal(loc=mu, scale=noise * mu,
                                 size=(n_scenarios, len(mu)))
    return np.clip(scenarios, 0, None)


def run_data_layer(config_path: str = "config.yaml") -> tuple:
    """
    Main entry point for the data layer.
    Returns: cfg, wards_df, future_df, scenarios, locations_df, candidate_df
    """
    cfg = load_config(config_path)
    os.makedirs("outputs", exist_ok=True)

    wards_df = load_wards(cfg)
    print(f"[Data Layer] Loaded {len(wards_df)} wards.")

    locations_df = load_candidate_locations(cfg)

    future_df = forecast_ev_demand(wards_df, cfg)
    out_path = cfg["data"]["future_demand_csv"]
    future_df.to_csv(out_path, index=False)
    print(f"[Data Layer] future_demand.csv -> {out_path} ({len(future_df)} rows)")

    # Build per-candidate demand for 2025 (base candidates)
    base_candidate_df = build_candidate_demand(locations_df, future_df, year=2025)

    # Dynamically add extra candidates for high-demand wards
    candidate_df = generate_dynamic_candidates(base_candidate_df, future_df, cfg, year=2025)

    print(f"[Data Layer] Final candidate pool: {len(candidate_df)} venues")
    vc = candidate_df['type'].value_counts().to_dict()
    auto = candidate_df.get('auto_generated', pd.Series([False]*len(candidate_df))).sum()
    print(f"  Types: {vc}  |  Auto-generated: {int(auto)} extra slots")

    scenarios = monte_carlo_demand(future_df, cfg)
    print(f"[Data Layer] Monte Carlo scenarios shape: {scenarios.shape}")

    return cfg, wards_df, future_df, scenarios, locations_df, candidate_df


if __name__ == "__main__":
    run_data_layer()
