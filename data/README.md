# Data

This folder holds the price panels used by `notebooks/financial_template.ipynb`.

`groups.csv` is small and committed. The six price panels
(`prices_open.csv`, `prices_high.csv`, `prices_low.csv`, `prices_close.csv`,
`prices_volume.csv`, `prices_returns.csv`) are **not committed** — one of them
is 102MB, over GitHub's 100MB hard file-size limit, and together they add
~290MB to the repo for no benefit since they're course-provided, not ours to
redistribute, and the grader already has them.

To run the notebooks locally, copy the six `prices_*.csv` files the course
provided into this folder, alongside `groups.csv`. `finance_toolkit.load_prices`
expects exactly that layout (`data/prices_<field>.csv` for each field in
`open, high, low, close, volume, returns`).
