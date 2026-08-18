# Applied Machine Learning — Final Assignment

Bar-Ilan University, Applied ML. Two independent parts: cross-sectional
alpha research on daily equity panels, and grayscale image colorization.
Full task spec in [`directives.txt`](directives.txt).

## Part I — Cross-Sectional Alpha Research

**Framework:** [`notebooks/finance_toolkit.py`](notebooks/finance_toolkit.py) (fixed, unedited —
scoring must be identical for the whole class). **Notebook:**
[`notebooks/financial_template.ipynb`](notebooks/financial_template.ipynb).

- Panel: 2,669 trading days (2013-01 to 2023-06) x 2,994 US equities.
- Split: everything before 2020-01-01 trains; the rest is a local hold-out.
- 13 hand-built features across five families — momentum (3 horizons +
  sector-neutral), reversal (market/sector/industry-neutral), volatility,
  volume, and technical (range position, gap, price-to-MA).
- A curated 24-config search (not the full 96-combo grid) over
  {feature family x MLP/LSTM x hidden size x learning rate x epochs x
  weight decay}, evaluated on the hold-out.

**Results:** see [`docs/RESULTS.md`](docs/RESULTS.md) (generated from the actual search
output, not hand-typed).

### A real bug found along the way

The course-provided `prices_close.csv` uses ambiguous day-first dates
(`14/01/2013`), which crashes `pandas.to_datetime` under the pandas
version in this environment (it auto-infers `%m/%d/%Y` from the first
row, then fails on any day > 12). Since `finance_toolkit.py` is the
fixed framework and only the notebook is submitted, the fix lives in
data preparation, not the toolkit: the date column was normalized to
ISO-8601 once, upstream of the notebook, and the toolkit file is
byte-identical to what was provided.

## Part II — Grayscale Image Colorization

**Notebook:** [`notebooks/colorization_template.ipynb`](notebooks/colorization_template.ipynb).
A U-Net (~4.4M params) trained on the [Flowers102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
dataset (chosen freely, per the assignment — colorful, diverse natural
images). See [`docs/RESULTS.md`](docs/RESULTS.md) for training details and sample outputs.

## Running locally

```
pip install -r requirements.txt
```

Part I needs the six `prices_*.csv` panels in `data/` — see
[`data/README.md`](data/README.md) (they're course-provided and too large to commit).
Part II downloads its own dataset on first run.

## Repo layout

```
data/            price panels (groups.csv committed; prices_*.csv local-only)
notebooks/       finance_toolkit.py (fixed) + both completed notebooks
output/          Part I submission: model_<i>.pt + params.json
docs/            results and findings
```
