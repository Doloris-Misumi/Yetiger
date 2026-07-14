# MixAndCall coverage audit for YesTiger

Source: https://dzyxdd.github.io/MixAndCall/mix/mix/

Accessed: 2026-06-29

This audit compares the public MixAndCall MIX page with YesTiger's executable action library at `knowledge/call_mix_library.json`.

The goal is not to copy every chant into the product. MixAndCall is closer to an encyclopedia: it contains hundreds of base, derived, meme, language, triple, variable, and speech-style entries. YesTiger's library should stay action-oriented: every executable entry needs duration rules, risk, context, and a live-style policy.

## Current local library

`call_mix_library.json` currently contains:

| Category | Count |
|---|---:|
| keepspace | 1 |
| rhythmcall | 14 |
| mix | 47 |
| underground_gei | 13 |
| total | 75 |

The local MIX library already covers the MVP core well:

- Standard / English MIX family
- Japanese MIX family
- Ainu MIX family
- second-half / lead-in / double-time fragments
- Myohon / Myohon activation
- variable and terminal variable candidates
- Ie Tiger / tiger-fire activations
- Bismarck, Pan, Hogwarts, Chikipa, Gozen, Bariyado, Tsunagaridai, Gachikoi
- several BanG Dream / Popipa related candidates

## MixAndCall page structure

The page exposes 457 section headings. They are better understood as families than as flat actions:

| Family | MixAndCall examples | Local coverage |
|---|---|---|
| Core language MIX | English/Standard, Japanese, Ainu, Chinese | strong for English/Japanese/Ainu; Chinese missing |
| Core timing variants | lead-in, first half, second half, +2 bars, reverse打ち, 12-prefix starts | partial; reverse and 12-prefix variants mostly missing |
| Speed variants | double-speed English/Japanese/Ainu/Chinese, reverse double-speed, cancel-to-double-speed | partial; only some second-half double-time entries exist |
| Triple MIX | basic triple, 2.5, variable triple, language triples | partial; variable triple exists, basic triple missing |
| Variable MIX | Myohon, Myohon driver, Ainu-Japanese variable, terminal variable, wiper variable | partial; Myohon and terminal variable exist, many variants missing |
| Koujou / speech | Gachikoi, Bismarck, Rocket-dan, short/long speech variants | partial; Gachikoi and Bismarck exist, many speech variants missing |
| Thematic / meme MIX | Encho, Yami, Kurofuku, Hogwarts, Lin Xiu, Pan, Chikipa, etc. | partial; some high-value examples exist |
| Foreign-language expansions | Spanish, German, Italian, Korean, Portuguese, Vietnamese, Arabic, Latin, Sanskrit, Nepalese | mostly missing |
| Localized Chinese/Japanese neta | many topic-specific and regional dialect entries | mostly missing; should be opt-in/reference only |

## Recommended knowledge-base layering

Use three layers instead of one giant action list:

1. `call_mix_library.json`
   - executable, product-safe actions;
   - consumed by the current generator;
   - should contain only validated entries with timing and risk.

2. `mix_reference_taxonomy.json`
   - non-executable reference taxonomy;
   - can include unvalidated families and candidate IDs;
   - used for annotation, search, RAG, and future curation.

3. profile policy
   - decides which high-risk / venue-sensitive entries are allowed;
   - examples: tame anisong, seiyuu live, underground idol, meme-heavy, BanG Dream-specific.

## High-priority gaps

These are worth adding or validating first because they are common enough and structurally meaningful:

| Priority | Candidate | Why |
|---:|---|---|
| 1 | `chinese_mix` | Completes the core language trio into English/Japanese/Ainu/Chinese. |
| 1 | `standard_mix_reverse` / `japanese_mix_reverse` | Reverse打ち is a basic timing variant. |
| 1 | `standard_mix_double_time` / `japanese_mix_double_time` / `ainu_mix_double_time` | Double-speed variants are common short-slot solutions. |
| 1 | `basic_sanren_mix` | Needed for long high-energy gaps and reference matching. |
| 2 | `encho_mix`, `yami_mix`, `kurofuku_mix` | Important encyclopedic families, but should be opt-in unless locally validated. |
| 2 | `spanish_mix`, `german_mix`, `italian_mix` | Foreign-language expansion families; useful for RAG/search but not default recommendations. |
| 2 | `rocket_dan_koujou`, `short_gachikoi_koujou` | Speech-style gaps need explicit duration handling. |
| 3 | meme/regional/neteta families | Keep as reference-only until a live-style profile explicitly allows them. |

## Product policy

For the online YesTiger system, the default profile should avoid encyclopedia-completeness. Recommended default behavior:

- allow low/medium-risk core MIX only when the slot has low vocal density and stable beat;
- allow high-risk actions only if the user selects an aggressive or underground profile;
- keep complete chant text optional and locally curated;
- use placeholders such as "phrase pending local validation" for entries sourced only from external references;
- never auto-insert venue-sensitive or fandom-sensitive calls without explicit profile permission.

## Next curation pass

The next safe implementation step is:

1. add missing core language/timing variants to `call_mix_library.json`;
2. keep meme/foreign/speech variants in `mix_reference_taxonomy.json`;
3. add `enabled_by_default` or `profile_policy` support to the generator before moving more high-risk entries into the executable library.

