import os, sys, time, json
sys.path.insert(0, r"c:\Users\שקד כהן\Downloads\Applied-Machine-Learning\notebooks")
import numpy
import pandas
import torch
import finance_toolkit as toolkit

SEED = 42
torch.manual_seed(SEED)
numpy.random.seed(SEED)
torch.set_num_threads(2)  # low priority -- Part II is using the other 6 threads

DATA_DIR = r"c:\Users\שקד כהן\Downloads\Applied-Machine-Learning\data"
CKPT_DIR = r"C:\Users\A5EF~1\AppData\Local\Temp\claude\c--Users---------Downloads-Applied-Machine-Learning\eb784440-b41c-473f-8b03-a050955f2a7a\scratchpad\alpha_ckpt_round5"
os.makedirs(CKPT_DIR, exist_ok=True)

panels = toolkit.load_prices(DATA_DIR)
forward_returns = toolkit.forward_return(panels["returns"])
mask = toolkit.valid_mask(panels, min_history=60)
all_dates = panels["close"].index
tickers = panels["close"].columns

SPLIT_DATE = "2020-01-01"
is_train = all_dates < SPLIT_DATE
is_test = ~is_train

close, open_, high, low, volume, returns = (
    panels["close"], panels["open"], panels["high"], panels["low"],
    panels["volume"], panels["returns"])
sector, industry = toolkit.load_groups(os.path.join(DATA_DIR, "groups.csv"))

all_features = {
    "mom_5": toolkit.cs_zscore(toolkit.ts_mean(returns, 5), sector),
    "mom_20": toolkit.cs_zscore(toolkit.ts_mean(returns, 20), sector),
    "mom_60": toolkit.cs_zscore(toolkit.ts_mean(returns, 60), sector),
    "rev_1": toolkit.cs_zscore(-returns, sector),
    "intraday_rev_20": toolkit.cs_zscore(toolkit.ts_zscore(open_ - close, 20), sector),
    "industry_rev_5": toolkit.cs_zscore(-toolkit.ts_mean(returns, 5), industry),
    "range_pos": toolkit.cs_zscore((close - low) / (high - low), sector),
    "gap": toolkit.cs_zscore((open_ - close.shift(1)) / close.shift(1), sector),
    "price_to_ma20": toolkit.cs_zscore((close - toolkit.ts_mean(close, 20)) / toolkit.ts_mean(close, 20), sector),
    "vol_20": toolkit.cs_zscore(toolkit.ts_std(returns, 20), sector),
    "high_low_range_20": toolkit.cs_zscore(toolkit.ts_mean((high - low) / close, 20), sector),
    "volume_spike_20": toolkit.cs_zscore(toolkit.ts_zscore(volume, 20), sector),

    # NEW for round 5E -- 46 configs across 4 model classes on the original
    # 12 sector-neutralized features gave only 6 mutually-uncorrelated
    # survivors. Two genuinely new information sources:
    #
    # 1. New volume/volatility-family features (still sector-neutral). This
    #    family (volume_spike, range_pos) is what's actually been
    #    decorrelating from everything else so far -- doubling down with
    #    new constructions: a windowed range position, a volume trend, a
    #    volatility regime change (short vs long vol).
    "range_pos_5": toolkit.cs_zscore(toolkit.ts_mean((close - low) / (high - low), 5), sector),
    "volume_trend": toolkit.cs_zscore(toolkit.ts_mean(volume, 5) - toolkit.ts_mean(volume, 20), sector),
    "vol_change": toolkit.cs_zscore(toolkit.ts_std(returns, 20) - toolkit.ts_std(returns, 60), sector),

    # ROUND 5G -- staying fully sector/industry-neutralized (non-neutralized
    # features tried and dropped -- decided against, per the TIPS section's
    # explicit "neutralise by sector" recommendation). Already at 10 kept
    # (required minimum) but one pair is a tight 0.496 correlation, so this
    # is for real margin: industry-neutral variants of the two features
    # that have been decorrelating best (vol_20, volume_spike_20) -- a
    # genuinely different neutralization group, not just a new signal --
    # plus two low-raw-correlation combos from feature_corr.csv
    # (high_low_range_20 vs volume_spike_20 = 0.006 raw correlation).
    "vol_20_industry": toolkit.cs_zscore(toolkit.ts_std(returns, 20), industry),
    "volume_spike_20_industry": toolkit.cs_zscore(toolkit.ts_zscore(volume, 20), industry),
}
all_feature_panel = toolkit.build_features(all_features, panels)
all_feature_panel = {k: v.astype("float32") for k, v in all_feature_panel.items()}
all_feature_names = sorted(all_feature_panel)

momentum_set = ["mom_5", "mom_20", "mom_60"]
reversal_set = ["rev_1", "intraday_rev_20", "industry_rev_5"]
vol_volume_set = ["vol_20", "high_low_range_20", "volume_spike_20"]
technical_set = ["range_pos", "gap", "price_to_ma20"]
mixed_set = ["mom_20", "rev_1", "range_pos", "vol_20", "gap"]
all_set = all_feature_names


class SafeMLP(toolkit.MLP):
    def predict(self, x, chunk_days=300):
        outs = [toolkit.MLP.predict(self, x[start:start + chunk_days])
                for start in range(0, x.shape[0], chunk_days)]
        return torch.cat(outs, dim=0)


class SafeLSTM(toolkit.LSTM):
    def predict(self, x, chunk_days=150):
        outs, state = [], None
        with torch.no_grad():
            for start in range(0, x.shape[0], chunk_days):
                out, state = self.step(x[start:start + chunk_days], state)
                outs.append(out)
        return torch.cat(outs, dim=0)


class LinearAlpha(torch.nn.Module):
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


def build_alpha(features, model_kind="mlp", hidden=32, epochs=20,
                 lr=1e-3, weight_decay=0.0, seed=SEED):
    features = sorted(features)
    array, _ = toolkit.stack_features({n: all_feature_panel[n] for n in features})
    feat_t, label_t, valid_t = toolkit.make_tensors(array, forward_returns, mask)
    train_ds = toolkit.DayDataset(feat_t, label_t, valid_t, is_train)
    test_ds = toolkit.DayDataset(feat_t, label_t, valid_t, is_test)
    torch.manual_seed(seed)
    if model_kind == "mlp":
        model = SafeMLP(len(features), hidden=(hidden,))
    elif model_kind == "lstm":
        model = SafeLSTM(len(features), hidden_size=hidden)
    elif model_kind == "linear":
        model = LinearAlpha(len(features))
    elif model_kind == "spline":
        model = SplineAlpha(len(features))
    model, hist = toolkit.train(model, train_ds, test_ds, epochs=epochs, lr=lr,
                                 weight_decay=weight_decay)
    scores = toolkit.predict_scores(model, feat_t, all_dates, tickers)
    book = toolkit.scores_to_positions(scores, mask)
    positions = book.loc[is_test]
    pnl = toolkit.simulate(positions, forward_returns)
    return model, positions, pnl, features


configs = [
    # MLP -- broad across families, this class showed the most independence
    {"name": "mlp_mom",    "features": momentum_set,   "model_kind": "mlp", "hidden": 32, "epochs": 20, "lr": 1e-3, "weight_decay": 0.05},
    {"name": "mlp_rev",    "features": reversal_set,   "model_kind": "mlp", "hidden": 32, "epochs": 20, "lr": 1e-3, "weight_decay": 0.05},
    {"name": "mlp_volvol", "features": vol_volume_set, "model_kind": "mlp", "hidden": 32, "epochs": 20, "lr": 1e-3, "weight_decay": 0.05},
    {"name": "mlp_tech",   "features": technical_set,  "model_kind": "mlp", "hidden": 32, "epochs": 20, "lr": 1e-3, "weight_decay": 0.05},
    {"name": "mlp_mixed",  "features": mixed_set,      "model_kind": "mlp", "hidden": 32, "epochs": 20, "lr": 1e-3, "weight_decay": 0.05},
    {"name": "mlp_all",    "features": all_set,        "model_kind": "mlp", "hidden": 48, "epochs": 22, "lr": 1e-3, "weight_decay": 0.05},
    {"name": "mlp_solo_gap", "features": ["gap"],      "model_kind": "mlp", "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},

    # LSTM -- kept small in number, these cluster with each other regardless of features
    {"name": "lstm_all",           "features": all_set,       "model_kind": "lstm", "hidden": 32, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "lstm_mixed",         "features": mixed_set,     "model_kind": "lstm", "hidden": 24, "epochs": 16, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "lstm_solo_rangepos", "features": ["range_pos"], "model_kind": "lstm", "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},

    # Linear -- proven on all_set/tech in the pilot; extended to the other family sets
    {"name": "linear_mom",    "features": momentum_set,   "model_kind": "linear", "epochs": 20, "lr": 1e-3, "weight_decay": 0.1},
    {"name": "linear_rev",    "features": reversal_set,   "model_kind": "linear", "epochs": 20, "lr": 1e-3, "weight_decay": 0.1},
    {"name": "linear_volvol", "features": vol_volume_set, "model_kind": "linear", "epochs": 20, "lr": 1e-3, "weight_decay": 0.1},
    {"name": "linear_tech",   "features": technical_set,  "model_kind": "linear", "epochs": 20, "lr": 1e-3, "weight_decay": 0.4},  # raised from pilot's 0.1: turnover was 1.28, needs to drop below 0.8
    {"name": "linear_mixed",  "features": mixed_set,      "model_kind": "linear", "epochs": 20, "lr": 1e-3, "weight_decay": 0.1},
    {"name": "linear_all",    "features": all_set,        "model_kind": "linear", "epochs": 20, "lr": 1e-3, "weight_decay": 0.1},

    # Spline (GAM) -- proven on all_set/tech in the pilot; extended to the other family sets
    {"name": "spline_mom",    "features": momentum_set,   "model_kind": "spline", "epochs": 20, "lr": 1e-3, "weight_decay": 0.1},
    {"name": "spline_rev",    "features": reversal_set,   "model_kind": "spline", "epochs": 20, "lr": 1e-3, "weight_decay": 0.1},
    {"name": "spline_volvol", "features": vol_volume_set, "model_kind": "spline", "epochs": 20, "lr": 1e-3, "weight_decay": 0.1},
    {"name": "spline_tech",   "features": technical_set,  "model_kind": "spline", "epochs": 20, "lr": 1e-3, "weight_decay": 0.1},
    {"name": "spline_mixed",  "features": mixed_set,      "model_kind": "spline", "epochs": 20, "lr": 1e-3, "weight_decay": 0.1},
    {"name": "spline_all",    "features": all_set,        "model_kind": "spline", "epochs": 20, "lr": 1e-3, "weight_decay": 0.1},

    # ROUND 5B -- appended after seeing round 5's results: threshold pass rate
    # is strong (7/8 so far) but mutual correlation among same-class MLP
    # configs on overlapping feature sets (mixed_set/all_set vs their parts)
    # is high (up to 0.87), so the greedy corr filter will cut a chunk of
    # them. Padding the pool for real margin toward 15 kept, not just 10:
    # seed variants of the configs that showed LOW correlation with
    # everything else (mlp_rev, mlp_volvol, lstm_solo_rangepos), plus new
    # solo-feature configs on model classes not yet tried per feature
    # (range_pos/volume_spike/gap/intraday_rev alone, across linear/spline/mlp).
    {"name": "mlp_rev_s2",              "features": reversal_set,   "model_kind": "mlp",    "hidden": 32, "epochs": 20, "lr": 1e-3, "weight_decay": 0.05, "seed": 43},
    {"name": "mlp_volvol_s2",           "features": vol_volume_set, "model_kind": "mlp",    "hidden": 32, "epochs": 20, "lr": 1e-3, "weight_decay": 0.05, "seed": 43},
    {"name": "lstm_solo_rangepos_s2",   "features": ["range_pos"],  "model_kind": "lstm",   "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0,  "seed": 43},
    {"name": "linear_solo_rangepos",    "features": ["range_pos"],  "model_kind": "linear", "epochs": 20, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "spline_solo_rangepos",    "features": ["range_pos"],  "model_kind": "spline", "epochs": 20, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "mlp_solo_rev1",           "features": ["rev_1"],      "model_kind": "mlp",    "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "mlp_solo_volspike",       "features": ["volume_spike_20"], "model_kind": "mlp",    "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "linear_solo_gap",         "features": ["gap"],        "model_kind": "linear", "epochs": 20, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "spline_solo_gap",         "features": ["gap"],        "model_kind": "spline", "epochs": 20, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "mlp_solo_intraday_rev",   "features": ["intraday_rev_20"], "model_kind": "mlp", "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},

    # ROUND 5C -- select_uncorrelated actually run on the 16 threshold-passers
    # kept only 4 (mlp_all, lstm_mixed, spline_rev, lstm_solo_rangepos): every
    # "broad feature set" config (all_set/mixed_set) correlates >0.5 with
    # mlp_all regardless of model class, since feeding a model everything
    # converges to similar behavior no matter the architecture. Solo-feature
    # configs are what's actually surviving (lstm_solo_rangepos), so sweeping
    # the 7 raw features that have no solo config anywhere yet, across the
    # model classes that showed independence (mlp, lstm) plus a couple of
    # linear/spline spot checks.
    {"name": "mlp_solo_mom5",           "features": ["mom_5"],           "model_kind": "mlp",    "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "mlp_solo_mom20",          "features": ["mom_20"],          "model_kind": "mlp",    "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "mlp_solo_mom60",          "features": ["mom_60"],          "model_kind": "mlp",    "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "mlp_solo_industry_rev5",  "features": ["industry_rev_5"],  "model_kind": "mlp",    "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "mlp_solo_price_ma20",     "features": ["price_to_ma20"],   "model_kind": "mlp",    "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "mlp_solo_vol20",          "features": ["vol_20"],          "model_kind": "mlp",    "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "mlp_solo_hlrange20",      "features": ["high_low_range_20"], "model_kind": "mlp",  "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "lstm_solo_mom5",          "features": ["mom_5"],           "model_kind": "lstm",   "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "lstm_solo_industry_rev5", "features": ["industry_rev_5"],  "model_kind": "lstm",   "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "lstm_solo_volspike",      "features": ["volume_spike_20"], "model_kind": "lstm",   "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "lstm_solo_gap",           "features": ["gap"],             "model_kind": "lstm",   "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "linear_solo_industry_rev5", "features": ["industry_rev_5"], "model_kind": "linear", "epochs": 20, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "spline_solo_volspike",    "features": ["volume_spike_20"], "model_kind": "spline", "epochs": 20, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "spline_solo_industry_rev5", "features": ["industry_rev_5"], "model_kind": "spline", "epochs": 20, "lr": 1e-3, "weight_decay": 0.0},

    # ROUND 5D -- 31 qualifiers only gave 6 kept: most alphas, regardless of
    # feature or model class, converge onto a handful of shared factors
    # (mlp_solo_mom5 corr 0.75 with mlp_mom, 0.78 with lstm_solo_mom5, etc).
    # What's actually decorrelating is volume/volatility/range-positioning
    # signals (lstm_solo_volspike, mlp_solo_volspike, lstm_solo_rangepos are
    # 3 of the 6 survivors) -- structurally unlike the momentum/reversal
    # family. Doubling down on that direction: vol_20/high_low_range_20
    # solo on model classes not yet tried on them, plus volume+range
    # combinations (avoiding vol_20+high_low_range_20 together -- those two
    # raw features are 0.77 correlated with each other per feature_corr.csv).
    {"name": "lstm_solo_vol20",        "features": ["vol_20"],            "model_kind": "lstm",   "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "lstm_solo_hlrange20",    "features": ["high_low_range_20"], "model_kind": "lstm",   "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "spline_solo_vol20",      "features": ["vol_20"],            "model_kind": "spline", "epochs": 20, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "spline_solo_hlrange20",  "features": ["high_low_range_20"], "model_kind": "spline", "epochs": 20, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "linear_solo_vol20",      "features": ["vol_20"],            "model_kind": "linear", "epochs": 20, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "linear_solo_hlrange20",  "features": ["high_low_range_20"], "model_kind": "linear", "epochs": 20, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "mlp_volspike_rangepos",  "features": ["volume_spike_20", "range_pos"], "model_kind": "mlp",    "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "lstm_volspike_rangepos", "features": ["volume_spike_20", "range_pos"], "model_kind": "lstm",   "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "mlp_vol20_rangepos",     "features": ["vol_20", "range_pos"],          "model_kind": "mlp",    "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "lstm_solo_volspike_s2",  "features": ["volume_spike_20"],   "model_kind": "lstm",   "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0, "seed": 43},

    # ROUND 5E -- new volume/volatility-family constructions (windowed
    # range position, volume trend, volatility regime change), all still
    # sector-neutralized like every other feature in this project.
    {"name": "mlp_solo_rangepos5",    "features": ["range_pos_5"],  "model_kind": "mlp",    "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "lstm_solo_rangepos5",   "features": ["range_pos_5"],  "model_kind": "lstm",   "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "spline_solo_rangepos5", "features": ["range_pos_5"],  "model_kind": "spline", "epochs": 20, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "mlp_solo_volumetrend",  "features": ["volume_trend"], "model_kind": "mlp",    "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "lstm_solo_volumetrend", "features": ["volume_trend"], "model_kind": "lstm",   "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "spline_solo_volumetrend", "features": ["volume_trend"], "model_kind": "spline", "epochs": 20, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "linear_solo_volumetrend", "features": ["volume_trend"], "model_kind": "linear", "epochs": 20, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "mlp_solo_volchange",    "features": ["vol_change"],   "model_kind": "mlp",    "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "lstm_solo_volchange",   "features": ["vol_change"],   "model_kind": "lstm",   "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "spline_solo_volchange", "features": ["vol_change"],   "model_kind": "spline", "epochs": 20, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "linear_solo_volchange", "features": ["vol_change"],   "model_kind": "linear", "epochs": 20, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "lstm_rangepos_volumetrend", "features": ["range_pos", "volume_trend"], "model_kind": "lstm", "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},

    # ROUND 5G -- staying fully neutralized (non-neutralized direction
    # dropped). Already at 10 kept, but one pair sits at a tight 0.496
    # correlation, so this is for real margin: industry-neutral variants of
    # the two features that have decorrelated best (a genuinely different
    # neutralization group, not a new raw signal), plus low-raw-correlation
    # combos from feature_corr.csv.
    {"name": "mlp_solo_vol20industry",      "features": ["vol_20_industry"],          "model_kind": "mlp",    "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "lstm_solo_vol20industry",     "features": ["vol_20_industry"],          "model_kind": "lstm",   "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "spline_solo_vol20industry",   "features": ["vol_20_industry"],          "model_kind": "spline", "epochs": 20, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "mlp_solo_volspikeindustry",   "features": ["volume_spike_20_industry"], "model_kind": "mlp",    "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "lstm_solo_volspikeindustry",  "features": ["volume_spike_20_industry"], "model_kind": "lstm",   "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "spline_solo_volspikeindustry", "features": ["volume_spike_20_industry"], "model_kind": "spline", "epochs": 20, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "mlp_hlrange_volspike",   "features": ["high_low_range_20", "volume_spike_20"], "model_kind": "mlp",  "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "lstm_hlrange_volspike",  "features": ["high_low_range_20", "volume_spike_20"], "model_kind": "lstm", "hidden": 16, "epochs": 18, "lr": 1e-3, "weight_decay": 0.0},
    {"name": "mlp_all_s2",             "features": all_set,                                  "model_kind": "mlp",  "hidden": 24, "epochs": 22, "lr": 1e-3, "weight_decay": 0.05, "seed": 43},
    {"name": "spline_rev_s2",          "features": reversal_set,                             "model_kind": "spline", "epochs": 20, "lr": 1e-3, "weight_decay": 0.1, "seed": 43},
]

t_start = time.time()
results = []
for i, cfg in enumerate(configs):
    name = cfg["name"]
    kpi_path = os.path.join(CKPT_DIR, name + "_kpi.json")
    pnl_path = os.path.join(CKPT_DIR, name + "_pnl.csv")
    if os.path.exists(kpi_path) and os.path.exists(pnl_path):
        with open(kpi_path) as f:
            kpi = json.load(f)
        print("[%2d/%d] %-18s SKIP (checkpoint found) sharpe=%6.2f turnover=%.2f" % (
            i + 1, len(configs), name, kpi["sharpe"], kpi["turnover"]), flush=True)
        results.append((name, kpi))
        continue

    t0 = time.time()
    kwargs = {k: v for k, v in cfg.items() if k != "name"}
    model, book, pnl, used = build_alpha(**kwargs)
    kpi = toolkit.evaluate(book, forward_returns)
    kpi = {k: float(v) for k, v in kpi.items()}

    with open(kpi_path, "w") as f:
        json.dump(kpi, f)
    pnl.to_csv(pnl_path, header=True)

    dt = time.time() - t0
    total = time.time() - t_start
    print("[%2d/%d] %-18s DONE sharpe=%6.2f turnover=%.2f  (%.1fs, total %.1fmin)" % (
        i + 1, len(configs), name, kpi["sharpe"], kpi["turnover"], dt, total / 60), flush=True)
    results.append((name, kpi))

print("\nALL CONFIGS DONE. total time %.1f min" % ((time.time() - t_start) / 60), flush=True)

kpi_df = pandas.DataFrame({name: kpi for name, kpi in results}).T.sort_values("sharpe", ascending=False)
pandas.set_option("display.width", 200)
pandas.set_option("display.max_rows", 50)
print("\n", kpi_df.round(3), flush=True)

qualifying = [name for name, kpi in results if kpi["sharpe"] > 1.2 and kpi["turnover"] < 0.8]
print("\nqualifying (sharpe>1.2, turnover<0.8): %d / %d" % (len(qualifying), len(results)), flush=True)
print(qualifying, flush=True)

if len(qualifying) >= 2:
    pnls = {name: pandas.read_csv(os.path.join(CKPT_DIR, name + "_pnl.csv"), index_col=0).iloc[:, 0]
            for name in qualifying}
    corr = toolkit.pnl_correlation(pnls)
    corr.to_csv(os.path.join(CKPT_DIR, "qualifying_corr.csv"))
    print("\nfull correlation matrix among qualifying alphas:", flush=True)
    print(corr.round(2), flush=True)
