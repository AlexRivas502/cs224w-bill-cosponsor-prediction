# Digest: Introduction and setup (section key intro)


---
## [intro-c0-md] markdown
# Choosing a train / validation / test split

**Goal:** decide how to split the Congress data (Congresses 108–118) for the two tasks in the proposal:

1. **Cosponsor prediction** (temporal link prediction): for a bill at time *t* (day 0 or day 7 after
   introduction), rank the eligible legislators who will cosponsor it in *(t, t+30]*.
2. **Bill advancement** (node classification): will a law-capable bill (HR, S, HJRES, SJRES) be
   reported out of committee within 180 days of introduction?

The default idea is a **by-Congress temporal split** (train on earlier Congresses, validate on the
next, test on the last). That split is only trustworthy if a set of conditions hold: similar label
rates, similar feature distributions, enough node overlap for the graph model, no leakage, fully
observed labels, and stable baseline scores across folds. Each section below tests a group of those
conditions and records a **check** (pass / fail / info). The last section collects every check and
compares the candidate splits.

**Run it:** JupyterLab, kernel with `pandas`, `pyarrow`, `numpy`, `scipy` (see
`notebooks/requirements.txt`). Charts use D3 v7, vendored at `notebooks/vendor/d3.v7.min.js` (no
internet needed). Run the cells top to bottom; charts appear as interactive SVGs (hover for values).

---
## [intro-c1-txt] text
```
repo: /Users/alex_rivas/Documents/Stanford/CS224W/Final Project/cs224w-bill-cosponsor-prediction
parsed tables: 21
```

---
## [intro-c2-txt] text
```
bills: 173,437 (145,738 law-capable); members rows: 6,638; congresses: 108-119
members: chamber filled from member_terms for 75/75 rows; party_code_asof differs from latest party_code in 30 member-Congress rows; policy areas folded into '(unstable/missing)': 4
snapshot (max update_date): 2026-10-17 (latest introduced_date 2026-10-15); policy area missing among 119th law-capable bills by age at snapshot: 0-7d 86% (n=56), 8-30d 14% (n=585), 31-90d 1% (n=1123), 91-180d 0% (n=2160), >180d 0% (n=13023) -> PA_MISS_T0 = 85.7% (n=56, small-sample proxy)
```

---
## [intro-c3-txt] text
```
cosponsorship events: 2,102,904; advancement labels: 145,738 (5.4% positive among uncensored; alternative label_alt = reported OR passed origin chamber within 180 d: 7.9%)
ROSTER_ASOF: 286 months, voting members in office per month (complete Congresses): House min 429 / median 435 / max 435; Senat min 99 / median 100 / max 100
ROSTER_ASOF_ALL (Task-1 pool, default of window_targets(point_in_time=True) / roster_asof()): adds delegates and resident commissioners, 5-6 per month (median 6); ROSTER_ASOF stays voting-only
Non-D/R members -> data-derived caucus (share of their cosponsorships on D-sponsored bills; near 0.5 = weak evidence): Amash I 116 (0.54)->D; Jeffords I 108-109 (0.51-0.60)->D; King I 113-119 (0.52-0.73)->D; Lieberman ID 110-112 (0.71-0.78)->D; Sanders I 108-119 (0.71-0.95)->D; Sinema I 118 (0.55)->D
Majority per chamber (point-in-time at mid-Congress, caucus-adjusted; ties -> tie_broken_by):
congress 108 109 110 111 112 113 114 115 116 117 118 119
chamber                                                 
House      R   R   D   D   R   R   R   R   D   D   R   R
Senat      R   R   D   D   D   D   R   R   R   D   D   R
ties: 117 Senate: 50-50 -> D by VP party (reported-bill sponsors D 65% (agrees))
REGIME (House/Senate majority): 108 RH/RS, 109 RH/RS, 110 DH/DS, 111 DH/DS, 112 RH/DS, 113 RH/DS, 114 RH/RS, 115 RH/RS, 116 DH/RS, 117 DH/DS, 118 RH/DS, 119 RH/RS | complete Congresses with 119's regime (RH/RS): [108, 109, 114, 115]
```

---
## [intro-c4-txt] text
```
split                                name                                                                                                                                        definition  eval units
  S1a    Expanding by Congress (test 117)                                                                                        train 108-115 | val 116 | test 117 (test refit on 108-116)           2
  S1b    Expanding by Congress (test 118)                                                                                        train 108-116 | val 117 | test 118 (test refit on 108-117)           2
   S2                  Sliding window k=3                                                                                 for test c: train c-4..c-2 | val c-1 | test c (refit on c-3..c-1)          14
   S3           Rolling origin, expanding                                                                   for test c: train 108..c-2 | val c-1 | test c (refit on 108..c-1); c in 112-118          14
   S4            Within-Congress temporal prior Congresses + bills of c introduced before its 60% date quantile | val 60-80% | test last 20% (only events dated before the cutoff are used)           6
   S5 Random bill split (leaky reference)                                                                             law-capable bills of 108-118 shuffled 70/15/15; ignores time entirely           2
variants (SPLIT_VARIANTS, not in iter_eval_units): S1a+stream: S1a + eval-Congress events dated <= introduced_date + t0; S1b+stream: S1b + eval-Congress events dated <= introduced_date + t0; S3+stream: S3 rolling folds + eval-Congress events dated <= introduced_date + t0; S1pool: train 108-115 | val 116 | test 117+118 pooled (refit on 108-116 once)
S5_ROLE: {'train': 90178, 'test': 19423, 'val': 19190} | S4_CUT_DATES: 116: 2019-12-12 / 2020-06-22; 117: 2021-11-04 / 2022-05-18; 118: 2023-11-17 / 2024-05-24
```

---
## [intro-c7-txt] text
```
setup complete: load(), cosp, adv (label, label_alt), adv_cutoff(), window_targets(point_in_time=), in_office_at(), ROSTER_ASOF / roster_asof(), majority_party(), REGIME / REGIME_HS / regime_match(), SNAPSHOT_DATE, PA_MISSING_AT_T0 / PA_MISS_T0, SPLIT_DEFS / SPLIT_VARIANTS / iter_eval_units(), S5_ROLE, S4_CUT_DATES, drift stats, rel_to_train() / rel_detail() / rel_by_split(mode=) / rel_split_table(), record_check(by_split=, kind=, na_reason=, t0_task2=) / checks_summary(kinds=) / checks_requirements() / checks_by_split() / checks_split_scores(), ROSTER_ASOF_ALL, show_d3() / line_chart() / bar_chart() / heatmap()
```