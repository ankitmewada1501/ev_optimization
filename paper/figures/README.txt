Drop any additional figures here (dummy.png, architecture diagram, etc.).
The paper already references the pipeline-generated PNGs by name via
\graphicspath, so these resolve automatically:

  nsga2_convergence.png
  mopso_convergence.png
  pareto_comparison.png
  queue_waiting_time_distribution.png
  uncertainty_profit_distribution.png

If a figure is missing, pdflatex will error with "File ... not found".
Either generate it by running `python main.py` from the repo root, or
drop your own PNG into this folder with the same name.
