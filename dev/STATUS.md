# Part I search status (handoff snapshot)

`dev/alpha_search.py` is a checkpointed alpha-config search script (separate from
`notebooks/financial_template.ipynb` — it's a dev/scratch tool, not the submission).
Run it with `python dev/alpha_search.py` from the `notebooks/` directory (it needs
`finance_toolkit.py` importable, and `../data/` for the price panels). It skips any
config whose checkpoint already exists in `dev/alpha_ckpt/`, so re-running is cheap —
only new configs at the bottom of the file actually train. `dev/final_configs.py` is
one intermediate snapshot of the config list partway through the search, kept for
reference, not the final one (see the notebook's own FREE 3 cell for that).

## Final result

170+ configs tried across many rounds, spanning 4 model classes (MLP, LSTM, Linear,
Spline/GAM — a 5th, a causal 1D CNN over a fixed lookback window, was added late and
lives directly in the notebook's FREE 2 cell, not this dev script). The final
submitted 10 (`notebooks/financial_template.ipynb`, FREE 3):

`lstm_mixed, spline_tech, lstm_solo_volspikeindustry, lstm_solo_vol20,
lstm_solo_hlrangeindustry, linear_volvol, spline_solo_hlrange20, mlp_solo_volchange,
spline_solo_mom5, lstm_solo_rangepos5`

Verified by actually running the notebook end to end (not just a dev-script
estimate): all 10 qualify (sharpe 1.2-3.58, turnover < 0.8), max pairwise
correlation **0.43** (comfortably under the 0.5 limit, directives.txt line 56),
combined Sharpe 2.97.

## Key finding

Correlation is checked between **trained alphas' PnL**, not between input features.
Across every model class tried, most alphas converge onto a small number of shared
factors regardless of which features or architecture produced them — e.g.
`mlp_solo_mom5` correlates 0.75 with `mlp_mom`, 0.78 with `lstm_solo_mom5`. This is a
property of the data (limited independent signal directions), not a config mistake.
What reliably decorrelated:
- Solo (single-feature) configs generally, over broad "kitchen sink" feature sets —
  any config using many features together tends to converge toward the same
  dominant factor regardless of model class.
- Volume/volatility-family signals (`volume_spike_20`, `vol_20`, `high_low_range_20`,
  and derived variants: windowed range position, volume trend, volatility-regime
  change) — structurally unlike the momentum/reversal family.
- Model-class diversity itself: swapping a config to a different model class on the
  *same* feature (e.g. `spline_solo_mom5` instead of `mlp_solo_mom5`/`lstm_solo_mom5`)
  measurably lowered correlation with the rest of the kept set, even holding the
  feature fixed.

An explicit swap search (every 1-for-1 and 2-for-2 swap over ~50 sharpe-AND-turnover-
verified candidates) is what found the final 0.43 combination — plain greedy
selection over a large mixed-quality pool is not an optimal solver and can land on a
technically-passing but barely-under-the-limit result, or (confirmed directly) even
fewer than 10 kept if a high-Sharpe new candidate outranks and correlates too much
with an already-good one. Dumping the entire ~50-candidate verified pool into greedy
at once was tried and kept only 7.

## On neutralization

Checked `directives.txt` directly (it's the only source of requirements — no other
lecturer instructions exist beyond this file, confirmed with the student). The only
hard rule is that every feature ends with a `cs_*` op (`cs_zscore`/`cs_rank`) so
stocks share a scale each day. Passing a `group` to neutralize
(`cs_zscore(sig, sector)`) is given as *an example technique*, and the TIPS section
recommends it as good practice for avoiding sector bets. The final submission keeps
every feature neutralized (sector- or industry-) rather than using un-neutralized
variants: out-of-sample robustness on the instructor's hidden, later period matters
more than any in-sample edge un-neutralized features might show, and sector-rotation
bets are structurally less stable across time periods than the tested search already
showed was needed just to hit the correlation bar honestly.

## What's done

- `notebooks/financial_template.ipynb` FREE 1/FREE 2/FREE 3 reflect the final config
  list above. The notebook has been executed end to end for real; FIXED 4/FIXED 5
  wrote real `output/model_*.pt` + `output/params.json` matching the 10 alphas above
  (verified: no stale files from earlier rounds left in `output/`).
- Part II: `notebooks/colorization_perceptual_loss.ipynb` uses a base=64 residual
  U-Net (warm-started from a verified-compatible reference checkpoint), color-
  weighted L1 + VGG perceptual loss, and inverse-frequency hue-balanced sampling
  over Natural Images + Flowers102 so training isn't dominated by whichever color
  happens to be most common in the raw data. Training is in progress on GPU as of
  this snapshot (resumable from checkpoint at any point, see the training cell's own
  comments for the mechanism).
