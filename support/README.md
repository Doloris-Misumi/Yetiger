# YesTiger Support Recommender

Rule-based MVP for turning predicted coarse music-structure segments into
candidate support actions.

The current frozen structure model outputs seven coarse sections:

- `intro`
- `verse`
- `pre_chorus`
- `chorus`
- `instrumental`
- `bridge`
- `outro`

This layer reads the exported prediction JSON from `train/test_bar.py`, matches
each section against `knowledge/call_mix_library.json`, and emits ranked
recommendations with time windows, confidence, risk, and a short reason.

## Example

```powershell
python support/recommend.py `
  --prediction train/predictions_bar_coarse_paper_test/shunkansummerday.prediction.json `
  --library knowledge/call_mix_library.json `
  --out support/recommendations/shunkansummerday.support.json
```

## Current rule posture

- `instrumental`, `intro`, and long `bridge` sections are good candidates for
  MIX-like actions.
- `pre_chorus` is treated as a build / entry zone; recommendations are
  conservative.
- `chorus` allows low-risk rhythm calls and high-risk optional
  `underground_gei` actions.
- `verse` defaults to low-risk calls or `keepspace`.
- `outro` allows light calls or short closing actions, but recommendations are
  still conservative because outro recognition remains imperfect.

This is intentionally not a final taste model. It is a product scaffold that
should be tuned with human edits and live-style policy.
