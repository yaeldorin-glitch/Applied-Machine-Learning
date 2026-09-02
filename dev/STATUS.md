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
  (verified: no stale files from earlier rounds left in `output/`). **This solo-10
  version is a complete, valid, submittable state on its own** — everything below is
  an in-progress bonus improvement (a joint submission with a partner's alphas plus
  more backup candidates), not something the existing submission depends on.
- Part II: `notebooks/colorization_perceptual_loss.ipynb` — training finished (real
  GPU run, `dev/train_local_gpu.py`, COCO + Natural Images + Flowers102, patience-
  based early stopping). `weights.pth` is the confirmed final checkpoint. Both files
  are in `הגשה_חלק_2/`, ready to submit as-is.

## Joint submission with a partner's alphas — IN PROGRESS, NOT YET WIRED IN

Goal: submit up to 14 alphas instead of 10 — the same 9 above (all neutralized, kept
fixed per the student's explicit instruction: never swap these out) plus 5 of a
partner's already-trained, already-pickled alphas (`friend_166`, `friend_404`,
`friend_407`, `friend_341`, `friend_387` — verified via her exact feature formulas
that none of these 5 are purely neutralized, so per the student's rule all 5 are the
swappable ones, not the fixed 9). Verified together: max pairwise correlation 0.4541,
combined Sharpe 3.32.

The 9 fixed alphas, freshly reproduced today (retrained from scratch with the real
FREE 2/3 recipe, matches the originally-submitted numbers):

| alpha | sharpe | turnover |
|---|---|---|
| `lstm_mixed` | 3.577 | 0.277 |
| `spline_tech` | 3.286 | 0.678 |
| `lstm_solo_volspikeindustry` | 2.594 | 0.264 |
| `lstm_solo_hlrangeindustry` | 2.067 | 0.177 |
| `lstm_solo_vol20` | 2.540 | 0.170 |
| `linear_volvol` | 2.035 | 0.287 |
| `spline_solo_hlrange20` | 1.599 | 0.154 |
| `mlp_solo_volchange` | 1.472 | 0.216 |
| `spline_solo_mom5` | 1.434 | 0.534 |

Friend's 5 (`friend_166/404/407/341/387`): individual sharpe/turnover not
re-verified in today's run (it crashed right as it reached them, before printing
their numbers) — the 0.4541/3.32 combined figures above are from an earlier,
separate verification pass against just these 14, done before this search started.
Re-confirming their individual numbers is part of finishing the integration test.

On top of that 14, a further alpha search (rounds 6–10, `dev/alpha_search_round*.py`
in scratchpad, not yet copied into this repo) went looking for more backup candidates
in case the instructor's hidden data changes which alphas qualify. **Confirmed** (real
sharpe/turnover, real correlation against the fixed 14's actual PnL, not an estimate):

| candidate | sharpe | turnover | max corr to the 14 |
|---|---|---|---|
| `r8_rf_range_pos_5` | 4.13 | 0.58 | 0.486 |
| `r8_xgb_range_pos_5` | 4.04 | 0.70 | 0.497 |
| `r8_xgb_mom_120_range_pos_5` | 3.89 | 0.62 | 0.465 |
| `r6_lstm_price_to_ma20` | 3.44 | 0.27 | 0.478 |
| `r8_rf_range_pos_20` | 2.30 | 0.48 | 0.404 |
| `r6_linear_price_to_ma20` | 1.77 | 0.37 | 0.476 |
| `r8_xgb_mom_10_volume_spike_20` | 1.82 | 0.79 | 0.469 |
| `r8_rf_close_ma60` | 1.49 | 0.42 | 0.299 |
| `r6_lstm_range_pos` | 1.40 | 0.31 | 0.361 |
| `r6_spline_range_pos_5` | 1.38 | 0.74 | 0.256 |

Caveat: the top 3 all share the `range_pos_5` feature (different models) and are
almost certainly correlated *with each other*, even though each passes against the
14 individually — the real count of independent additions is more like 6-8, not 10.
This has not yet been cross-checked (see below).

**What's NOT done yet**: actually writing this into `financial_template.ipynb` —
adding the partner's feature formulas + the backup candidates' raw features to a new
cell, loading/retraining everything together, and re-running `select_uncorrelated`
over the full pool to get the real final kept list. A standalone test of this
(`joint_integration_test.py`, scratchpad) confirmed the 9 fixed alphas retrain
correctly in isolation, but repeatedly crashed this machine (8GB RAM, already tight
from VS Code + extensions) before completing a full run — the script now checkpoints
per-alpha so a re-run resumes instead of restarting, but that run has not yet
finished. Rounds 9-10 of the search itself are also still running in the background
(round 9: GBR/ExtraTrees/Ridge/ElasticNet, ~128 configs; round 10: corrected CNN, 12
configs) and may add a few more candidates to the table above.

**Next steps for whoever picks this up**: (1) let `joint_integration_test.py` finish
(or re-run it — it resumes from `joint_integration_ckpt/`) to get the real combined
`select_uncorrelated` result over all 14 + backups: (2) cross-check the backup
candidates above against each other, not just against the 14, since the `range_pos_5`
trio is likely redundant; (3) write the verified final list into the notebook's FREE
1/3 and a new cell before FIXED 4, matching the pattern already planned (separate
`friend_feature_panel` / `raw_feature_panel` dicts, no collision with the notebook's
own neutralized features); (4) re-run FIXED 4/5 for real and update `output/` and
`הגשה_חלק_1/`.
