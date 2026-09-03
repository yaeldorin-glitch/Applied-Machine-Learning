"""finance_toolkit.py -- a tiny ML-for-finance framework.

This is the shared, FIXED framework.  You import it and call its
functions; you do NOT edit it.  It gives you the data, the
operators, and the SCORING panel -> a portfolio, a PnL, a Sharpe
and a turnover.  The scoring is the same for the whole class, so
grading is fair.

Between the data and the scoring, the middle is YOURS.  You choose
the features, the model (any shape -- not only the LSTM below), the
loss and the training.  The one thing your model must satisfy is
the MODEL CONTRACT:

    model.predict(feat_t) -> scores
    feat_t : float tensor (days, stocks, features)
    scores : float tensor (days, stocks)

Anything with that method plugs into the scorer.  Two starter
models (MLP, LSTM) are provided; replace or extend them freely.

THE ONE RULE -- everything is a "panel":
  a panel is a pandas DataFrame, one row per trading date and one
  column per stock (shape = days x stocks).  Every operator takes
  panels and returns a panel of the SAME shape, so you can chain
  them freely and the pipeline always lines up.

The sections are:
  1. load the data and build the prediction target,
  2. operators to build features (ts_* over time, cs_* across
     stocks, with optional per-group variants),
  3. the valid mask (which stock is tradable on which day),
  4. features -> tensors -> a per-day torch Dataset,
  5. starter models (MLP, LSTM) and the model contract,
  6. an OPTIONAL modular loss you can shape,
  7. an OPTIONAL training loop,
  8. scores -> portfolio -> PnL -> performance,
  9. correlation, to pick a set of un-alike alphas,
  10. plots and a small alpha search
"""

import os
import numpy
import pandas
import torch


FIELDS = ["open", "high", "low", "close", "volume", "returns"]

# =====================================================================
# 1. Load the data and build the target
# =====================================================================


def load_prices(data_dir):
    """Read the prices_*.csv files into a dict {field: panel}.

    All panels are aligned to the same dates and stocks.  The
    "returns" panel is the clean daily return field (already
    adjusted for splits and dividends).  Always build return-based
    features from "returns", never from close.pct_change(): raw
    close has huge jumps on split days that wreck any statistic.
    """
    panels = {}
    for field in FIELDS:
        path = os.path.join(data_dir, "prices_" + field + ".csv")
        if not os.path.exists(path):
            continue
        frame = pandas.read_csv(path, index_col="Date")
        frame.index = pandas.to_datetime(frame.index)
        panels[field] = frame.sort_index()

    dates = panels["close"].index
    stocks = panels["close"].columns
    for field in panels:
        panels[field] = panels[field].reindex(
            index=dates, columns=stocks)
    return panels


def forward_return(returns, delay=2):
    """The target we predict: the return two days ahead.

    forward_return[t] = returns[t + delay]

    The gap (delay=2) means a signal built from data up to day t is
    traded at the next day's close and earns the day after -- so a
    feature can never peek at the return it is trying to predict.
    Pass the clean "returns" panel here.
    """
    return returns.shift(-delay)


# =====================================================================
# 2. Operators for building features
# =====================================================================
# ts_* work DOWN each column   (one stock, over time).
# cs_* work ACROSS each row    (one day, over all stocks).  Pass a
# `group` (a ticker -> code Series) to a cs_* operator to make it
# work WITHIN each sector/industry instead of over the whole
# market (this is how you neutralise a signal by sector).
# Every operator takes a panel and returns a panel of same shape.


def ts_delta(panel, window=1):
    """Change over `window` days: panel[t] - panel[t - window]."""
    return panel - panel.shift(window)


def ts_mean(panel, window):
    """Rolling average over the last `window` days."""
    return panel.rolling(window).mean()


def ts_std(panel, window):
    """Rolling standard deviation over the last `window` days."""
    return panel.rolling(window).std()


def ts_zscore(panel, window):
    """Rolling z-score: (value - rolling mean) / rolling std."""
    return (panel - ts_mean(panel, window)) / ts_std(panel, window)


def _by_group(panel, group):
    """Group the panel's columns by their sector/industry code.

    Returns a groupby laid out so a cs_* op can be applied per day
    within each group, then transposed back to days x stocks.
    """
    codes = group.reindex(panel.columns)
    return panel.T.groupby(codes)


def cs_groupmean(panel, group=None):
    """Subtract the cross-sectional mean, making it dollar-neutral.

    With a `group`, subtract each group's own mean instead -- this
    neutralises the signal within every sector/industry, so it
    takes no net bet on any group.
    """
    if group is None:
        return panel.sub(panel.mean(axis=1), axis=0)
    return panel - _by_group(panel, group).transform("mean").T


def cs_zscore(panel, group=None):
    """Per-day z-score across stocks: (value - mean) / std.

    With a `group`, standardise within each sector/industry.
    """
    if group is None:
        return cs_groupmean(panel).div(panel.std(axis=1), axis=0)
    grouped = _by_group(panel, group)
    mean = grouped.transform("mean").T
    std = grouped.transform("std").T
    return (panel - mean) / std


def cs_rank(panel, group=None):
    """Per-day rank across stocks, 0..1 (1 = highest).

    With a `group`, rank within each sector/industry.
    """
    if group is None:
        return panel.rank(axis=1, pct=True)
    return _by_group(panel, group).rank(pct=True).T


def load_groups(path):
    """Ticker -> group-code table (ONE row per ticker).

    Columns: ticker,sector_code,industry_code.  Returns (sector,
    industry) Series indexed by ticker for the cs_* `group` arg.
    """
    table = pandas.read_csv(path, index_col="ticker")
    return table["sector_code"], table["industry_code"]


# =====================================================================
# 3. Valid mask  (which stock is tradable on which day)
# =====================================================================


def valid_mask(panels, min_history=60):
    """Boolean panel: True where a stock can be traded that day.

    A stock is valid when it has a price, a clean return, a defined
    target, and at least `min_history` past days so rolling
    features are warmed up.
    """
    has_target = forward_return(panels["returns"]).notna()
    has_price = panels["close"].notna()
    has_return = panels["returns"].notna()
    warmed_up = has_price.cumsum() >= min_history
    return has_price & has_return & has_target & warmed_up


# =====================================================================
# 4. Features -> tensors -> per-day Dataset
# =====================================================================


def build_features(features, panels):
    """Assemble the feature panels the model will use.

    `features` is {name: value} where each value is EITHER a ready
    (days x stocks) panel OR a function panels -> panel.  So a
    feature can be a plain expression (close - open) or a function;
    both work.  Each feature is aligned to the common grid.
    """
    dates = panels["close"].index
    stocks = panels["close"].columns
    out = {}
    for name, value in features.items():
        panel = value(panels) if callable(value) else value
        out[name] = panel.reindex(index=dates, columns=stocks)
    return out


def stack_features(features):
    """Stack {name: panel} into (array, names).

    array shape = (days, stocks, num_features); names is sorted so
    the order is fixed and reproducible.
    """
    names = sorted(features)
    array = numpy.stack(
        [features[name].to_numpy() for name in names], axis=-1)
    return array, names


def make_tensors(feature_array, forward_returns, mask):
    """Turn the numpy panels into the three training tensors.

    feat  : (days, stocks, features)  NaNs -> 0
    label : (days, stocks)            forward returns, NaNs -> 0
    valid : (days, stocks)            1.0 tradable else 0.0
    """
    feat = numpy.nan_to_num(
        feature_array, nan=0.0, posinf=0.0, neginf=0.0)
    feat_t = torch.tensor(feat, dtype=torch.float32)
    label_t = torch.tensor(
        numpy.nan_to_num(forward_returns.to_numpy(), nan=0.0),
        dtype=torch.float32)
    valid_t = torch.tensor(mask.to_numpy().astype("float32"))
    return feat_t, label_t, valid_t


class DayDataset(torch.utils.data.Dataset):
    """One sample = one day.

    Fixed width, so a sequence model sees a constant stock axis:
      x : (stocks, features) that day's features,
      y : (stocks,) forward returns,
      v : (stocks,) 1.0 where tradable else 0.0.
    A DataLoader batches consecutive days into the time axis.
    """

    def __init__(self, feat_t, label_t, valid_t, day_mask):
        keep = torch.as_tensor(
            numpy.asarray(day_mask), dtype=torch.bool)
        self.x = feat_t[keep]
        self.y = label_t[keep]
        self.v = valid_t[keep]

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], self.y[i], self.v[i]


# =====================================================================
# 5. Starter models  (replace or extend -- see the model contract)
# =====================================================================
# THE MODEL CONTRACT (all the scorer needs):
#   model.predict(feat_t) -> scores
#     feat_t : (days, stocks, features) float tensor
#     scores :               (days, stocks) float tensor
# The built-in train() also uses a stateful step interface so it
# can carry memory across day-batches:
#   model.step(x, state) -> (scores, new_state)
# For a memoryless model, state stays None (see MLP).  Any model
# you write has to provide predict(); add step() too if you want
# to reuse the built-in train().


class MLP(torch.nn.Module):
    """A memoryless per-stock MLP: same map applied every day.
    Each (stock, features) row is scored on its own -- there is no
    memory across days.  A good "not an LSTM" baseline.
    """

    def __init__(self, input_size, hidden=(16, 16), dropout=0.0):
        super().__init__()
        dims = [input_size] + list(hidden) + [1]
        layers = []
        for i in range(len(dims) - 1):
            linear = torch.nn.Linear(dims[i], dims[i + 1])
            torch.nn.init.xavier_uniform_(linear.weight)
            layers.append(linear)
            if i != len(dims) - 2:
                layers.append(torch.nn.ReLU())
                layers.append(torch.nn.Dropout(dropout))
        self.net = torch.nn.Sequential(*layers)

    def init_state(self, num_stocks):
        return None

    def step(self, x, state):
        return self.predict(x), None

    def predict(self, x):
        days, stocks, feats = x.shape
        flat = x.reshape(days * stocks, feats)
        return self.net(flat).reshape(days, stocks)


class LSTM(torch.nn.Module):
    """Per-stock LSTM over the day axis (a sequence model).

    x is (days, stocks, features): days are the time steps, stocks
    are the batch.  The LSTM output feeds a small fully-connected
    head that gives one score per stock per day.  `dropout` adds
    regularisation between the LSTM and the head.
    """

    def __init__(self, input_size, hidden_size=5, num_layers=1,
                 fc_dims=(5, 5, 1), dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # Build the FC head before the LSTM so a fixed seed
        # reproduces the same starting weights every run.
        dims = [hidden_size] + list(fc_dims[1:])
        layers = []
        for i in range(len(dims) - 1):
            linear = torch.nn.Linear(dims[i], dims[i + 1])
            torch.nn.init.xavier_uniform_(linear.weight)
            layers.append(linear)
            if i != len(dims) - 2:
                layers.append(torch.nn.ReLU())
        self.fc = torch.nn.Sequential(*layers)
        self.drop = torch.nn.Dropout(dropout)
        self.lstm = torch.nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=False)

    def init_state(self, num_stocks):
        shape = (self.num_layers, num_stocks, self.hidden_size)
        return (torch.zeros(shape), torch.zeros(shape))

    def step(self, x, state):
        # x: (days, stocks, features)
        batch_size = x.size(1)
        if state is None:
            state = self.init_state(batch_size)
        h, c = state[0].type_as(x), state[1].type_as(x)
        if h.size(1) < batch_size:                # pad new stocks
            extra = batch_size - h.size(1)
            pad = torch.zeros(
                self.num_layers, extra,
                self.hidden_size).type_as(x)
            h = torch.cat([h, pad], 1)
            c = torch.cat([c, pad], 1)
        out, state = self.lstm(x, (h, c))
        out = self.drop(out)
        out = self.fc(out).squeeze(-1)             # (days, stocks)
        return out, state

    def predict(self, x):
        out, _ = self.step(x, None)
        return out

    def forward(self, x):
        return self.predict(x)


# =====================================================================
# 5b. Additional models (student addition -- "you may add ... models
#     ... following the examples there")
# =====================================================================
# predict_scores (section 8) runs model.predict(feat_t) on the FULL
# (days, stocks, features) tensor in one un-batched shot. For a
# sequence model that means one un-batched forward pass over the
# whole ~2700-day x ~3000-stock panel -- for the LSTM that needs a
# single ~8GB allocation and reliably crashes on an 8GB machine
# (measured directly). predict_scores is part of the FIXED framework
# above, so it can't be changed; SafeMLP/SafeLSTM below override only
# predict() to process the day axis in bounded chunks (carrying LSTM
# hidden state across chunks for SafeLSTM), giving the exact same
# output with a small, bounded memory footprint. MLP.step() also
# calls self.predict() during training, so SafeMLP must NOT force
# no_grad in predict() or gradients break.


class SafeMLP(MLP):
    def predict(self, x, chunk_days=300):
        outs = [MLP.predict(self, x[start:start + chunk_days])
                for start in range(0, x.shape[0], chunk_days)]
        return torch.cat(outs, dim=0)


class SafeLSTM(LSTM):
    def predict(self, x, chunk_days=150):
        outs, state = [], None
        with torch.no_grad():
            for start in range(0, x.shape[0], chunk_days):
                out, state = self.step(x[start:start + chunk_days], state)
                outs.append(out)
        return torch.cat(outs, dim=0)


class LinearAlpha(torch.nn.Module):
    """score = w . features + b -- no hidden layers, no interactions.
    Added after a correlation diagnostic showed MLP/LSTM alphas across
    different feature families still converge on shared factors; no
    cross-feature interaction and no memory across days makes this
    structurally distinct enough to decorrelate better than more
    MLP/LSTM variants."""

    def __init__(self, input_size, hidden=None):
        super().__init__()
        self.linear = torch.nn.Linear(input_size, 1)
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def init_state(self, n):
        return None

    def step(self, x, state):
        return self.predict(x), None

    def predict(self, x, chunk_days=300):
        outs = []
        for start in range(0, x.shape[0], chunk_days):
            chunk = x[start:start + chunk_days]
            days, stocks, feats = chunk.shape
            outs.append(self.linear(chunk.reshape(days * stocks, feats)).reshape(days, stocks))
        return torch.cat(outs, dim=0)


class SplineAlpha(torch.nn.Module):
    """GAM: score = sum_i spline_i(feature_i) -- a cubic (thin-plate)
    basis per feature, independently, then linearly combined. No
    cross-feature interactions (unlike MLP), no memory (unlike LSTM)."""

    def __init__(self, input_size, hidden=None, n_knots=7, knot_range=4.0):
        super().__init__()
        knots = torch.linspace(-knot_range, knot_range, n_knots)
        self.register_buffer("knots", knots)
        self.coef = torch.nn.Linear(input_size * n_knots, 1)
        torch.nn.init.xavier_uniform_(self.coef.weight)

    def init_state(self, n):
        return None

    def step(self, x, state):
        return self.predict(x), None

    def _basis(self, flat):
        diffs = flat.unsqueeze(-1) - self.knots
        return torch.abs(diffs) ** 3

    def predict(self, x, chunk_days=300):
        outs = []
        for start in range(0, x.shape[0], chunk_days):
            chunk = x[start:start + chunk_days]
            days, stocks, feats = chunk.shape
            flat = chunk.reshape(days * stocks, feats)
            basis = self._basis(flat).reshape(flat.shape[0], -1)
            outs.append(self.coef(basis).reshape(days, stocks))
        return torch.cat(outs, dim=0)


class CNNAlpha(torch.nn.Module):
    """Causal 1D convolution over the day axis, independently per stock:
    each day's score depends on that day and the (window_size-1) days
    immediately before it -- a fixed local window. This is a genuinely
    different inductive bias from every other model here: LSTM carries
    unbounded state across the whole history, MLP/Linear/Spline see
    only the single current day (no history at all), CNNAlpha sees a
    short, fixed lookback and nothing beyond it.

    step()/predict() consistency and memory-safety follow the same
    pattern as SafeLSTM above: state carries the last (window_size-1)
    days of raw input across day-batches (not a learned hidden state,
    literal past features, since a plain conv has no memory of its
    own), and predict() chunks internally so a full un-batched call
    over the whole ~2700-day panel doesn't blow up memory. State is
    returned as a 1-tuple (not a bare tensor) to match the framework's
    own state-handling convention (run_epoch above does
    `tuple(s.detach() for s in state)`, which would silently misbehave
    on a bare tensor -- it would iterate over its first dimension
    instead of detaching it as a whole).

    Verified directly (not assumed) before use here: predict() gives
    bit-identical output regardless of internal chunk size, matches
    calling step() batch-by-batch the way the real training loop does,
    and -- the one that actually matters for a trading signal --
    scrambling the input from some day forward leaves every earlier
    day's score completely unchanged. A convolution with padding
    handled carelessly can leak future information into a "past"
    prediction; this one was checked not to.
    """

    def __init__(self, input_size, hidden=16, window_size=10):
        super().__init__()
        self.input_size = input_size
        self.window_size = window_size
        self.conv1 = torch.nn.Conv1d(input_size, hidden, kernel_size=window_size)
        self.relu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(hidden, hidden, kernel_size=1)
        self.out = torch.nn.Conv1d(hidden, 1, kernel_size=1)
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.out.weight)

    def init_state(self, num_stocks):
        pad = self.window_size - 1
        return (torch.zeros(pad, num_stocks, self.input_size),)

    def _forward_conv(self, x_with_context):
        # x_with_context: (T, stocks, feats), T >= window_size
        h = x_with_context.permute(1, 2, 0)   # (stocks, feats, T)
        h = self.relu(self.conv1(h))          # (stocks, hidden, T-window_size+1)
        h = self.relu(self.conv2(h))
        h = self.out(h)                       # (stocks, 1, T-window_size+1)
        return h.squeeze(1).transpose(0, 1)   # (T-window_size+1, stocks)

    def step(self, x, state):
        # x: (chunk_days, stocks, features) -- consecutive days, in order
        pad = self.window_size - 1
        batch_size = x.size(1)
        if state is None:
            state = self.init_state(batch_size)
        context = state[0].type_as(x)
        if context.size(1) < batch_size:               # pad new stocks
            extra = batch_size - context.size(1)
            zpad = torch.zeros(pad, extra, self.input_size).type_as(x)
            context = torch.cat([context, zpad], dim=1)
        x_with_context = torch.cat([context, x], dim=0)
        scores = self._forward_conv(x_with_context)     # (chunk_days, stocks)
        new_context = x_with_context[-pad:] if pad > 0 else x[:0]
        return scores, (new_context,)

    def predict(self, x, chunk_days=200):
        state = None
        outs = []
        for start in range(0, x.shape[0], chunk_days):
            chunk = x[start:start + chunk_days]
            scores, state = self.step(chunk, state)
            outs.append(scores)
        return torch.cat(outs, dim=0)


class TreeWrapper:
    """Wraps a scikit-learn/XGBoost regressor (fit outside torch, on
    the flattened (day*stock, features) matrix) to satisfy the MODEL
    CONTRACT -- predict(feat_t) -> scores -- so a tree model can sit
    in the exact same alpha pipeline as the torch ones above."""

    def __init__(self, m):
        self.m = m

    def eval(self):
        pass  # predict_scores calls model.eval() unconditionally

    def predict(self, x):
        days, stocks, feats = x.shape
        preds = self.m.predict(x.reshape(-1, feats).cpu().numpy())
        return torch.tensor(preds, dtype=torch.float32).reshape(days, stocks)


MODEL_CLASSES = {"mlp": SafeMLP, "lstm": SafeLSTM, "linear": LinearAlpha,
                  "spline": SplineAlpha, "cnn": CNNAlpha}


# =====================================================================
# 6. An OPTIONAL modular loss  (reward IR, penalise turnover)
# =====================================================================
# This is one loss you can use and shape; you are free to write your
# own instead.  A loss must return a dict with a "loss" key that the
# training loop can call .backward() on.

EPS = 1e-5


def _normalize(pred, returns, valid, epsilon=EPS):
    """Scores -> book: demean over VALID stocks, scale to L1 = 1.

    Longs and shorts balance (dollar neutral) and the gross
    exposure sums to 1 each day.  This runs inside the loss so
    gradients flow back to the model.
    """
    pred = pred * valid
    returns = returns * valid
    pred_mean = pred.sum(-1) / (valid.sum(-1) + epsilon)
    pred = (pred - pred_mean.unsqueeze(1)) * valid
    pred = pred / (pred.abs().sum(-1, keepdim=True) + epsilon)
    pred = pred * valid
    return pred, returns


def compute_ir(pred, returns, valid, epsilon=EPS):
    """Daily information ratio of the normalized book."""
    p, r = _normalize(pred, returns, valid, epsilon)
    pnl = (p * r).sum(-1)
    return pnl.mean() / (pnl.std() + epsilon)


def compute_tvr(pred, returns, valid, epsilon=EPS):
    """Average turnover: |book(t) - book(t-1)| summed per day."""
    p, _ = _normalize(pred, returns, valid, epsilon)
    diff = (p - p.roll(1, 0)).abs()
    return diff[1:, :].sum(axis=1).mean()


def piecewise_linear_map(perf, lines, shift=0.0):
    """Map a performance number through a piecewise-linear curve.

    `lines` is a list of (threshold, slope) segments, in order,
    with the last threshold = inf.  Below the first threshold the
    slope of the first segment applies, and so on.  This is how you
    shape HOW MUCH each performance number helps or hurts the loss.
    """
    if len(lines) == 0:
        return perf
    thr, slope = lines[0]
    if numpy.isinf(thr):
        return slope * perf + shift
    if perf < thr:
        return slope * perf + shift
    new_shift = slope * thr - thr * lines[1][1] + shift
    return piecewise_linear_map(perf, lines[1:], new_shift)


IR_STEPS = [(numpy.inf, -1)]
TVR_STEPS = [(0.1, 0), (0.3, 0.1), (numpy.inf, 1)]


def modular_loss(pred, returns, valid, ir_steps=None, tvr_steps=None):
    """One training objective, as a SUM of shaped pieces:

        loss = map(IR, ir_steps) + map(turnover, tvr_steps)

    A negative slope on IR means the loss falls as IR rises (so
    the model is pushed to raise IR).  A positive slope on turnover
    past a threshold penalises trading too much.  Change the
    slopes and thresholds -- or add more pieces here -- to change
    what the model optimises.
    """
    if ir_steps is None:
        ir_steps = IR_STEPS
    if tvr_steps is None:
        tvr_steps = TVR_STEPS
    ir = compute_ir(pred, returns, valid)
    tvr = compute_tvr(pred, returns, valid)
    loss = (piecewise_linear_map(ir, ir_steps)
            + piecewise_linear_map(tvr, tvr_steps))
    return {"loss": loss, "ir": ir, "tvr": tvr}


# =====================================================================
# 7. An OPTIONAL training loop
# =====================================================================
# Convenience only.  It works for any model that provides step();
# write your own loop if your model or loss need something else.


def run_epoch(model, loader, opt=None, loss_fn=modular_loss):
    """One pass over the days; train if an optimiser is given.

    The model state (if any) is carried from one day-batch to the
    next so a sequence model stays continuous across the split.
    """
    train_mode = opt is not None
    model.train() if train_mode else model.eval()
    totals, count = {}, 0
    state = None
    with torch.set_grad_enabled(train_mode):
        for x, y, v in loader:
            scores, state = model.step(x, state)
            if state is not None:
                state = tuple(s.detach() for s in state)
            result = loss_fn(scores, y, v)
            if train_mode:
                opt.zero_grad()
                result["loss"].backward()
                opt.step()
            for key, value in result.items():
                totals[key] = totals.get(key, 0.0) + float(
                    value.detach())
            count += 1
    return {k: v / max(count, 1) for k, v in totals.items()}


def train(model, train_ds, val_ds, epochs=8, batch_size=64,
          lr=1e-3, weight_decay=0.0,
          ir_steps=None, tvr_steps=None, loss_fn=None):
    """Train with Adam; print progress.  Returns (model, history).

    Days are fed in order (shuffle=False) so a sequence model sees
    a real time sequence.  `weight_decay` is L2 on the weights.
    Pass your own `loss_fn(pred, y, v)`, or shape the default via
    `ir_steps` / `tvr_steps` (see modular_loss).
    """
    if loss_fn is None:
        def loss_fn(pred, y, v):
            return modular_loss(pred, y, v, ir_steps, tvr_steps)

    opt = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay)
    loader = torch.utils.data.DataLoader
    train_loader = loader(
        train_ds, batch_size=batch_size, shuffle=False)
    val_loader = loader(
        val_ds, batch_size=batch_size, shuffle=False)
    history = {"train": [], "val": []}
    for ep in range(epochs):
        tr = run_epoch(model, train_loader, opt, loss_fn)
        va = run_epoch(model, val_loader, None, loss_fn)
        history["train"].append(tr)
        history["val"].append(va)
        print(
            "epoch %2d | train loss %.5f ir %.4f tvr %.4f"
            " | val loss %.5f ir %.4f tvr %.4f" % (
                ep, tr["loss"], tr.get("ir", 0), tr.get("tvr", 0),
                va["loss"], va.get("ir", 0), va.get("tvr", 0)))
    return model, history


# =====================================================================
# 8. Scores -> portfolio -> PnL -> performance  (the FIXED scorer)
# =====================================================================
# This is the grading boundary.  Your model produces scores; from
# here on everyone is scored the same way.


def predict_scores(model, feat_t, dates, stocks):
    """Run the model over every day -> raw score panel.

    Uses only the model contract: model.predict(feat_t).
    """
    model.eval()
    with torch.no_grad():
        pred = model.predict(feat_t)
    return pandas.DataFrame(
        pred.numpy(), index=dates, columns=stocks)


def scores_to_positions(scores, mask):
    """Turn scores into a DOLLAR-NEUTRAL, unit-gross book.

    Keep valid stocks, subtract the day's mean so longs and shorts
    balance to zero (dollar-neutral), then scale so the absolute
    weights sum to 1 (unit gross).  Same recipe the loss uses.
    """
    scores = scores.where(mask)
    weights = scores.sub(scores.mean(axis=1), axis=0).where(mask)
    gross = weights.abs().sum(axis=1)
    return weights.div(gross, axis=0).fillna(0.0)


def simulate(positions, forward_returns):
    """Daily PnL = sum over stocks of position * forward return."""
    aligned = forward_returns.reindex(
        index=positions.index, columns=positions.columns)
    return (positions * aligned.fillna(0.0)).sum(axis=1)


def performance(daily_pnl):
    """Sharpe-style metrics for a daily PnL (Sharpe = the grade)."""
    pnl = daily_pnl.dropna()
    mean, std = pnl.mean(), pnl.std()
    sharpe = mean / std * numpy.sqrt(252.0) if std > 0 else 0.0
    equity = pnl.cumsum()
    return {
        "sharpe": float(sharpe),
        "mean_daily_pnl": float(mean),
        "std_daily_pnl": float(std),
        "total_pnl": float(pnl.sum()),
        "max_drawdown": float((equity - equity.cummax()).min()),
        "num_days": int(len(pnl)),
    }


def turnover(positions):
    """Average fraction of the book traded per day."""
    return float(positions.diff().abs().sum(axis=1).mean())


def evaluate(positions, forward_returns):
    """One alpha's stats: performance metrics plus turnover."""
    stats = performance(simulate(positions, forward_returns))
    stats["turnover"] = turnover(positions)
    return stats


# =====================================================================
# 9. Correlation  (choose a set of un-alike alphas)
# =====================================================================
# An "alpha" here is the daily PnL of one trained MODEL, not a
# single feature.  Build 10 models, simulate each, then use these
# to keep only the ones that make genuinely different bets.


def pnl_correlation(pnls):
    """Correlation matrix of many alphas' daily PnL.

    `pnls` is {name: daily-PnL Series}.  Build it by simulating
    each model, e.g.:
        pnls = {name: simulate(pos, forward_returns) for ...}
    Two alphas with a high number here make the same bet -- keeping
    both adds risk, not edge.
    """
    frame = pandas.DataFrame(pnls)
    return frame.corr()


def select_uncorrelated(pnls, max_corr=0.3):
    """Greedy un-correlated subset, best Sharpe first.

    Order the alphas by descending Sharpe, then walk that order
    keeping an alpha only if its correlation to every one already
    kept stays at or below `max_corr`.  So the best alpha is always
    kept and only redundant, weaker ones are dropped.  Returns the
    list of kept names, highest Sharpe first.
    """
    corr = pnl_correlation(pnls)
    names = list(corr.index)
    matrix = corr.to_numpy().copy()
    numpy.fill_diagonal(matrix, 0.0)
    sharpe = numpy.array(
        [performance(pandas.Series(pnls[n]))["sharpe"] for n in names])
    order = numpy.argsort(-sharpe)
    selected = []
    for candidate in order:
        worst = matrix[candidate][selected].max() if selected else 0.0
        if worst <= max_corr:
            selected.append(candidate)
    return [names[i] for i in selected]


def combined_sharpe(pnls, names=None):
    """Sharpe of the equal-weight blend of chosen alphas.

    This is the number the grader cares about: several un-alike
    alphas averaged together are steadier than any one alone.
    """
    frame = pandas.DataFrame(pnls)
    if names is not None:
        frame = frame[names]
    return performance(frame.mean(axis=1))["sharpe"]


# =====================================================================
# 10. Plots and a small alpha search
# =====================================================================


def compute_kpis(alphas, forward_returns):
    """KPI table for many alphas.

    `alphas` is {name: positions panel}.  Returns a DataFrame with
    one row per alpha (sharpe, turnover, ...), sorted by sharpe.
    Use it to compare a whole search space at a glance.
    """
    rows = {}
    for name, positions in alphas.items():
        rows[name] = evaluate(positions, forward_returns)
    table = pandas.DataFrame(rows).T
    return table.sort_values("sharpe", ascending=False)


def plot_pnl(pnls, title="cumulative PnL"):
    """Plot one or more equity curves (needs matplotlib).

    `pnls` is a Series or a dict {name: Series}.  Plotting is for
    your own eyes only; it does not affect grading.
    """
    import matplotlib.pyplot as plt
    if isinstance(pnls, dict):
        items = pnls.items()
    else:
        items = [("pnl", pnls)]
    plt.figure(figsize=(9, 5))
    for name, series in items:
        plt.plot(series.dropna().cumsum(), label=name)
    plt.title(title)
    plt.xlabel("date")
    plt.ylabel("cumulative PnL")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_kpis(table, x="turnover", y="sharpe"):
    """Scatter the KPI space, one dot per alpha (matplotlib)."""
    import matplotlib.pyplot as plt
    plt.figure(figsize=(7, 6))
    plt.scatter(table[x], table[y])
    for name, row in table.iterrows():
        plt.annotate(str(name), (row[x], row[y]))
    plt.xlabel(x)
    plt.ylabel(y)
    plt.title("alpha search space")
    plt.tight_layout()
    plt.show()
