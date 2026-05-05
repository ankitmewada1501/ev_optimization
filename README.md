# EV Charging Station Optimization Framework

An open, reproducible Python framework that jointly decides **where** to
place electric-vehicle (EV) charging stations, **how many chargers** to
install at each, and **how to price** the resulting sessions, in a
single multi-objective pipeline. Validated on the 85 wards of the
**Indore Municipal Corporation** with 30 real candidate venues
(parking lots, fuel stations, malls, restaurants), expanded to 118
high-demand slots through a demand-proportional dynamic generator.

The framework couples:

- **Demand forecasting** — ward population CAGR + logistic EV adoption curve.
- **Monte Carlo uncertainty** — `S = 100` independent per-station scenarios.
- **Finite-buffer queueing** — `M/M/c/K` with three-period time-of-use
  (peak / normal / idle) demand multipliers and pricing.
- **Multi-objective optimisation** — both **NSGA-II** and **MOPSO** with
  hypervolume-based early stopping, run on the same fitness function and
  compared on standard performance indicators (HV, GD, Spread).
- **Reproducible reporting** — emits a self-contained interactive HTML
  report with a Leaflet map, station pop-ups, ToU legend, and all
  performance plots base-64 encoded inline.

The four jointly minimised objectives are: total CAPEX, demand-weighted
zone-radius coverage (negated), net annual ROI (negated), and the
ToU-weighted mean queue waiting time, with a soft per-zone grid-capacity
penalty (default 500 kW / zone).

---

## Quick start

```bash
# Clone
git clone https://github.com/ankitmewada1501/ev_optimization.git
cd ev_optimization

# Install dependencies (Python 3.10+)
pip install -r requirements.txt   # see notes below if file is absent

# Run the end-to-end pipeline
python main.py
```

The whole pipeline (data + queue + optimisation + evaluation +
visualisation) completes in **under four minutes** on a commodity
laptop. All artefacts are written to `outputs/`.

If `requirements.txt` is missing, the core dependencies are:

```
numpy
scipy
pandas
matplotlib
pyyaml
folium       # for the Leaflet map in the HTML report
```

---

## Repository layout

```
ev_optimization/
├── main.py                       # Top-level orchestrator (4 layers)
├── config.yaml                   # All hyperparameters, prices, caps
├── data/
│   ├── indore_wards.csv          # 85-ward population baseline
│   └── candidate_locations.csv   # 30 real candidate venues (typed)
├── modules/
│   ├── data_layer.py             # Forecasting + Monte Carlo + dynamic candidates
│   ├── queue_simulation_layer.py # M/M/c/K with ToU
│   ├── optimization_layer.py     # NSGA-II + MOPSO with HV early stopping
│   ├── evaluation_layer.py       # HV / GD / Spread / sensitivity / scalability
│   └── visualization_layer.py    # PNGs + self-contained HTML report
├── outputs/                      # Pareto CSVs, PNG plots, ev_report.html
├── paper/
│   ├── ieee_paper.tex            # IEEE conference manuscript
│   └── figures/                  # Paper-ready figures
├── Report/                       # Split MTP report sources
│   ├── intro.tex
│   ├── model.tex
│   ├── methodology.tex
│   ├── result.tex
│   ├── conclusion.tex
│   └── Liter.tex
└── MTP_newreport.pdf             # Compiled MTP report
```

---

## Configuration

Every modelling assumption lives in [`config.yaml`](config.yaml). The most
commonly tuned knobs:

| Group | Key | Default | Meaning |
|---|---|---|---|
| `forecasting` | `cagr_rate` | `0.018` | Ward population CAGR |
| `forecasting.ev_adoption` | `t0` | `2030` | Logistic inflection year (NEMMP target) |
| `monte_carlo` | `n_scenarios` | `100` | Number of MC demand draws per candidate |
| `monte_carlo` | `demand_noise_fraction` | `0.15` | Sigma on per-station demand multiplier |
| `queue` | `avg_charging_time_min` | `30` | Mean session duration |
| `queue.time_of_use` | `peak_demand_multiplier` | `2.8` | Peak load factor |
| `queue.time_of_use` | `idle_demand_multiplier` | `0.15` | Idle load factor |
| `pricing` | `peak_rate_per_session` | `Rs 210` | Peak ToU price |
| `pricing` | `idle_rate_per_session` | `Rs 112` | Off-peak discount price |
| `optimisation` | `n_generations` | `500` | NSGA-II generations / MOPSO iterations |
| `grid_capacity_kw` | — | `500` | Per-zone grid capacity cap (soft penalty) |

Run the pipeline against an alternative config:

```bash
python main.py --config path/to/your_config.yaml
```

---

## Outputs

After a successful run, the [`outputs/`](outputs/) directory contains:

| File | Contents |
|---|---|
| `optimal_solutions_nsga2.csv` | 80 non-dominated NSGA-II layouts (binary selection + 4 objectives) |
| `optimal_solutions_mopso.csv` | 100 non-dominated MOPSO layouts |
| `nsga2_convergence.png` | NSGA-II hypervolume vs. generation |
| `mopso_convergence.png` | MOPSO hypervolume vs. iteration |
| `mopso_convergence_rate.png` | MOPSO HV improvement-rate windows |
| `pareto_comparison.png` | Two-algorithm overlay across all four objective pairs |
| `nsga2_individual_pareto.png` | NSGA-II Pareto archive (4 panels) |
| `mopso_individual_pareto.png` | MOPSO Pareto archive (4 panels) |
| `queue_waiting_time_distribution.png` | Real (penalty-free) `Wq` histograms |
| `uncertainty_profit_distribution.png` | Annual-profit distribution across Pareto solutions |
| `scalability_runtime_plot.png` | Runtime vs. candidate-pool size |
| `future_demand.csv` | Per-ward, per-year forecasted daily session demand |
| `ev_report.html` | **Self-contained** interactive report with Leaflet station map |

Key headline numbers from the included Indore run:

- NSGA-II: 80 Pareto solutions, cost **Rs 71L – Rs 411L**, coverage
  **46.4% – 100%**, ROI **1.85 – 4.15**, mean wait **0.26 – 4.37 min**
  (median 1.45 min).
- MOPSO: 100 archive solutions, cost **Rs 39L – Rs 344L**, coverage
  **75.8% – 100%**, ROI **2.12 – 4.33**, mean wait **0.13 – 5.55 min**
  (median 1.79 min).
- Final hypervolume: NSGA-II `0.9967`, MOPSO `1.0434`. MOPSO is
  marginally ahead on HV and GD; NSGA-II produces a more uniformly
  spread front.
- Annual profit across the union archive: **Rs 1.2 Cr to Rs 11.6 Cr / year**.
- All Pareto-optimal layouts satisfy the `P_block <= 10%` blocking
  bound and the `rho < 1` stability bound across all three ToU periods.

---

## Pipeline architecture

```
+-------------------+
| Ward + venue CSVs |   data/
+--------+----------+
         |
         v
+-------------------+
|   Data layer      |   forecast (CAGR + logistic), MC noise (S=100),
|                   |   dynamic candidate generation (30 -> 118 slots)
+--------+----------+
         |
         v
+-------------------+
|   Queue layer     |   M/M/c/K + 3-period ToU; per-station Wq, P_K, rho
+--------+----------+
         |
         v
+-------------------+
| Optimisation layer|   NSGA-II + MOPSO, HV-based early stopping
+----+----------+---+
     |          |
     v          v
+--------+ +--------------+
| Eval   | | Visualisation|   HV/GD/Spread + Leaflet HTML report
+---+----+ +------+-------+
    |              |
    v              v
   outputs/  <-  outputs/ev_report.html
```

---

## Reproducing the figures and tables in the paper

A single command regenerates everything referenced in
[`paper/ieee_paper.tex`](paper/ieee_paper.tex) and
[`Report/result.tex`](Report/result.tex):

```bash
python main.py
```

Random seeds are pinned (`numpy` seed `42` on the Monte Carlo matrix and
both optimisers), so two runs with the same `config.yaml` produce
byte-identical CSVs and pixel-identical PNGs.

---

## Citation

If you use this framework or its results, please cite the accompanying
IEEE manuscript:

```bibtex
@inproceedings{mewada2026ev,
  title     = {A Multi-Objective Optimization Framework for Electric
               Vehicle Charging Station Placement and Time-of-Use
               Pricing in Smart Cities},
  author    = {Mewada, Ankit and Trivedi, Aditya and Bhadauria, Saumya},
  booktitle = {Proc. IEEE Conf.},
  year      = {2026},
  note      = {ABV-IIITM Gwalior}
}
```

The full LaTeX source is in [`paper/ieee_paper.tex`](paper/ieee_paper.tex).

---

## Authors

| Name | Department | Email |
|---|---|---|
| **Ankit Mewada** | Dept. of IT, ABV-IIITM Gwalior | `ankit.mewada@iiitm.ac.in` |
| **Prof. Aditya Trivedi** | Dept. of ICT, ABV-IIITM Gwalior | `atrivedi@iiitm.ac.in` |
| **Dr. Saumya Bhadauria** | Dept. of CSE/IT, ABV-IIITM Gwalior | `saumya@iiitm.ac.in` |
# MTP_FINAL_CODE
