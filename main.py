"""
Main Orchestrator - EV Charging Station Optimization Framework
Runs all layers in sequence:
  1. Data Layer      → load wards + candidate locations, forecast demand, Monte Carlo
  2. Optimization    → NSGA-II + MOPSO (venue-aware, dynamic stations, ToU pricing)
  3. Evaluation      → HV, GD, Spread, sensitivity, scalability
  4. Visualization   → PNGs + Interactive HTML report with Leaflet map
"""
import os
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_layer         import run_data_layer
from modules.optimization_layer import run_optimization_layer
from modules.evaluation_layer   import (
    compute_all_indicators, sensitivity_analysis, scalability_benchmark,
)
from modules.visualization_layer import run_visualization_layer


def main():
    config_path = "config.yaml"
    print("=" * 70)
    print("  EV Charging Station Optimization Framework – Indore (85 Wards)")
    print("=" * 70)
    t_total = time.time()

    # ── 1. Data Layer ──────────────────────────────────────────────────────
    print("\n[1/4] DATA LAYER")
    cfg, wards_df, future_df, mc_scenarios, locations_df, candidate_df = \
        run_data_layer(config_path)

    # ── 2. Optimization Layer ──────────────────────────────────────────────
    print("\n[2/4] OPTIMIZATION LAYER")
    opt_results = run_optimization_layer(cfg, future_df, mc_scenarios, candidate_df)

    # ── 3. Evaluation Layer ────────────────────────────────────────────────
    print("\n[3/4] EVALUATION LAYER")
    print("[Evaluation] Computing performance indicators...")
    indicators = compute_all_indicators(
        opt_results["nsga2"]["objectives"],
        opt_results["mopso"]["objectives"],
    )

    print("[Evaluation] Running sensitivity analysis...")
    sensitivity_data = sensitivity_analysis(cfg, future_df, mc_scenarios,
                                             candidate_df=candidate_df)

    print("[Evaluation] Running scalability benchmarks...")
    scalability_data = scalability_benchmark(cfg, future_df, mc_scenarios,
                                              candidate_df=candidate_df)

    # ── 4. Visualization & Report Layer ─────────────────────────────────────
    print("\n[4/4] VISUALIZATION & REPORT LAYER")
    run_visualization_layer(
        cfg, wards_df, future_df,
        opt_results,
        indicators,
        sensitivity_data,
        scalability_data,
        candidate_df=candidate_df,
    )

    elapsed = time.time() - t_total
    print("\n" + "=" * 70)
    print(f"  ✓ All done in {elapsed:.1f} seconds")
    print("=" * 70)
    print("\nOutput files:")
    for key in ["future_demand_csv", "optimal_nsga2_csv", "optimal_mopso_csv"]:
        print(f"  • {cfg['data'][key]}")
    for key in ["pareto_png", "queue_dist_png", "profit_dist_png",
                "scalability_png", "html_report"]:
        print(f"  • {cfg['outputs'][key]}")
    print()


if __name__ == "__main__":
    main()
