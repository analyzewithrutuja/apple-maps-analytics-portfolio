# Maps & Analytics Portfolio

Three projects built around the core skills a Maps/routing data science role actually uses:
evaluating routing quality against real driver behavior, running rigorous A/B tests and causal
inference, and processing geospatial data at real scale.

## Projects

| Project | What it does | Key result |
|---|---|---|
| [`routing-quality-evaluation/`](routing-quality-evaluation/) | Matches real GPS trajectories against a live OpenStreetMap road network to evaluate routing accuracy and ETA reliability | Route distance accuracy is strong (mean detour ratio 1.10), but free-flow ETA underestimates real travel time by a median of 98.6% |
| [`ab-testing-causal-inference/`](ab-testing-causal-inference/) | Analyzes a real randomized experiment, then recovers a valid causal estimate from non-randomized observational data via propensity score matching | PSM estimate within $126 of the true experimental effect — 98.8% bias reduction vs. a naive comparison |
| [`spark-scale-geo-analysis/`](spark-scale-geo-analysis/) | Processes ~24.9 million real GPS points via distributed PySpark window functions and spatial aggregation | Full pipeline (ingest → windowed speed calc → 62,013 spatial cells) runs in ~712 seconds |

## Common thread across all three

Each project uses real, public data (not synthetic), and each one includes a genuine bug that was
diagnosed and fixed along the way rather than a clean run reported at face value — a
misleading-metric issue, a recursive-vs-direct estimation bug, a graph-coverage bug, and a
Windows-specific PySpark internals bug. See each project's README for the specifics.

## Setup

Each project folder has its own `requirements.txt`, `README.md` with data-download instructions
(where the raw data is too large to include directly), and results.
