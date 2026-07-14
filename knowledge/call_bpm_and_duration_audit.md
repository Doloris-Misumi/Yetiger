# Call BPM And Action Duration Audit

Updated: 2026-06-09

## Rule

From this point, action duration fields in `knowledge/call_mix_library.json` should be interpreted as practical support bars:

```text
call_bars, based on call_bpm / call_bar_multiplier
```

Music analysis can still use `bpm`, allin1 downbeats, and analysis bars. Action filling should convert between analysis bars and call bars before applying knowledge-library duration rules.

## Annotation BPM Normalization

| song_id | bpm | call_bpm | call_bar_multiplier | note |
|---|---:|---:|---:|---|
| dododo | 98 | 98 | 1.0 | same grid |
| godknows | 150 | 75 | 0.5 | half-time support grid |
| hitoshizuku | 102 | 102 | 1.0 | same grid |
| ishizue_no_hanakanmuri | 176 | 88 | 0.5 | half-time support grid |
| jibun_restart | 171 | 85.5 | 0.5 | half-time support grid |
| kizunamusic | 95 | 95 | 1.0 | same grid |
| louder | 97 | 97 | 1.0 | same grid |
| mayoiuta | 95 | 95 | 1.0 | same grid |
| more_jump_more | 102 | 102 | 1.0 | same grid |
| nijuu_no_niji | 94 | 94 | 1.0 | same grid |
| poppindream | 100 | 100 | 1.0 | same grid |
| starttruedreams | 90 | 90 | 1.0 | same grid |
| teardrops | 95 | 95 | 1.0 | same grid |
| xiuwaxiuwa | 88 | 88 | 1.0 | same grid; analysis grid still worth checking later |

## Duration Categories To Revise

| Priority | Category | Actions | Current pattern | Risk | Proposed rule |
|---:|---|---|---|---|---|
| 1 | Underground-gei fixed choreography | `long_zhi_mao`, `cun_zheng`, `lei_she`, `luo_man_si`, `tian_zhao` | `allowed_bars` often `[4, 8]`, `can_extend: true` | The 8-bar option can be an artifact of double analysis BPM, not a real longer choreography. | Use strict call-bar lengths. If both 4 and 8 call-bar versions are real, split IDs, e.g. `lei_she_4bar` / `lei_she_8bar`. |
| 1 | Underground-gei long choreography | `dian_bo_she` | `allowed_bars: [8, 16]`, `can_extend: true` | Same action can be stretched to fill long slots. | Split into explicit call-grid versions or keep one strict length after checking the teaching video. |
| 1 | Underground-gei irregular length | `gongfu_she` | `min_bars: 8`, `max_bars: 9`, no `allowed_bars`, `can_extend: true` | 9 bars may be grid drift or ending padding. | Confirm real support length, then set `allowed_bars` exactly, likely `[8]` or a renamed 9-bar variant. |
| 2 | High-risk MIX flexible | `highclap`, `sekai_konton_mix`, `sekai_konton_mix_first_half`, `ietora`, `tsunagaridai_mix` | high risk but non-strict, extendable, or compressible | High-risk calls can be inserted as stretch fillers. | Make high-risk MIX strict in call bars; split short/long variants instead of using `can_extend`. |
| 2 | Medium-risk MIX flexible | `mix_leadin_aaa_ikuzo`, `japanese_mix_second_half`, `myhontousuke`, `haiseno_activation`, `bismarck_mix`, `aiai_mix`, `imi_fumei_ai_mix`, `pan_mix`, `gozen_ichiban_mix` | non-strict, extendable, or compressible | Medium-risk calls can absorb leftover bars. | Prefer explicit `allowed_bars`; reserve `can_extend` only for true repeated-unit calls. |
| 2 | Strict MIX with broad min/max but missing `allowed_bars` | `standard_mix`, `japanese_mix`, `ainu_mix`, `ainu_kahen_mix`, `kaho_sanren_mix`, `shuumaku_kahen_mix`, `tiger_fire_activation`, `gachikoi_koujou` | `strict_bars: true`, but `min_bars`/`max_bars` are broad | Current code mostly uses preferred length, but the schema is ambiguous. | Add explicit `allowed_bars` matching the real call-grid lengths. |
| 3 | Fractional-bar actions | `jyajya_mix`, `fuwa_fuwa`, `fufu_call`, `hai_hai`, `o_call`, `vocal_chant` | 0.25/0.5/3.5 bar specs | `barfit` rounds to integer bars, so the declared length differs from planned length. | Represent these as beat/eight-count units or snap them to integer call bars with repeat metadata. |
| 3 | Repeatable rhythm calls | `ppph`, `fuwa_fuwa`, `hai_hai`, `oi_oi`, `clap`, `name_call`, `aizuchi`, `kecha`, `sing_along` | broad ranges such as 1-16 or 1-32 bars | They should repeat as units, not become one huge action. | Model as `unit_bars` plus `repeatable: true`; let the renderer repeat the unit. |

## Applied Length Updates

These updates were applied directly to `call_mix_library.json` using only the 11 original-speed support-grid annotations as duration evidence. Half-time songs were excluded from this pass.

| Group | Updated actions | Applied call-bar lengths |
|---|---|---|
| 4-bar underground-gei | `long_zhi_mao`, `cun_zheng`, `lei_she`, `luo_man_si`, `tian_zhao`, `ali_shizi`, `heibai_shuangyi` | `[4]`, strict |
| 8-bar underground-gei | `gongfu_she`, `dian_bo_she`, `zi_he_she`, `chan_she` | `[8]`, strict |
| long underground-gei | `tian_qu_short`, `tian_qu_long` | `[12]` / `[20]`, strict |
| fixed 1/2/4/5/7/8-bar MIX | `standard_mix`, `standard_mix_first_half`, `standard_mix_second_half`, `standard_mix_long`, `std_mix_kouhan_dt`, `japanese_mix`, `japanese_mix_long`, `japanese_mix_second_half`, `japanese_mix_second_half_double_time`, `jp_mix_seigyaku_dt`, `ainu_mix`, `ainu_mix_long`, `ainu_second_half_mix`, `ainu_kahen_mix`, `myhontousuke`, `myohon_activation`, `mix_leadin_aaa_ikuzo`, `haiseno_activation`, `mouikai`, `tiger_fire_activation`, `short_tiger_fire_activation`, `aiai_mix`, `imi_fumei_ai_mix`, `pan_mix`, `gozen_ichiban_mix`, `bismarck_mix_first_half`, `bismarck_mix`, `bismarck+kanzenseniki_mix`, `bariyado_mix`, `sae_mix`, `wakage_no_itari_mix`, `bandor_mix`, `hogwarts_mix`, `ietora_konzetsu_mix`, `lin_xiu_mix`, `kaho_sanren_mix`, `sekai_konton_mix`, `sekai_konton_mix_first_half`, `shuumaku_kahen_mix`, `popipa_mix`, `takamatsu_mix`, `tsunagaridai_mix`, `highclap`, `ietora` | explicit `allowed_bars`; high/medium-risk actions no longer compress or extend |
| conservative outlier handling | `chikipa_mix`, `gachikoi_koujou` | locked to `[7]` and `[8]`; shorter original-speed annotations are treated as annotation/grid outliers |
| observed rhythmcall units | `chokokoro_call`, `popipapipopa`, `fufu_call`, `vocal_chant`, `hai_hai`, `ppph`, `name_call`, `oi_oi`, `clap`, `sing_along` | explicit observed `allowed_bars`; still non-strict, but no broad auto-extension |

## Implementation Notes

- `collect_candidate_action_ids` currently adds every action in the matching category to the candidate pool. Once a span is predicted as `underground_gei`, all underground-gei actions can compete.
- `select_barfit_actions` currently sorts plans by fill ratio first. Flexible high-risk actions are therefore attractive when they fill more bars.
- `avoid`, `vocal_density`, and `beat_stability` are mostly descriptive today; duration is the main hard constraint.
- Remaining actions without explicit `allowed_bars` are low-risk or spacing/repeat units with no evidence in the 11 original-speed songs: `keepspace`, `fuwa_fuwa`, `o_call`, `aizuchi`, `kecha`.
- After switching action fitting to call bars, regenerate `experiments/signal_callability/*/*.barfit_action_*` and `webapp/static/examples/*.json`.
