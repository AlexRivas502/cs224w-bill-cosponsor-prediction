# Digest: A. Label / target and covariate stationarity (section key A)


---
## [A-c8-md] markdown
## A. Label / target and covariate stationarity

Every forward split (train on earlier data, evaluate on later data: S1a/S1b/S2/S3 by Congress, and S4
within the last Congresses) evaluates Congresses that come **after** its training data. The model will
also be deployed on a later Congress (119). So any shift found here is **a difference between history and
the deployment target**, not a defect of a forward split: the forward split is the protocol that measures
it honestly, and the random split S5 is the one that hides it. A drift FAIL below therefore selects a
**remedy** (recency-weighted training, recalibration on the val Congress, density-normalised metrics,
per-Congress reporting), not a different split. All A checks are recorded as `kind='drift'`; every
forward split, including S4 (whose test units are the late parts of 116-118), gets the same verdict from
the candidate Congresses it evaluates, and S5 is left unjudged (None), which is not the same as passing.
Four kinds of shift are tested:

- **Label shift**: the positive rate P(y) changes (e.g. fewer bills get reported). Threshold-based
  metrics, calibration and AUPRC all move with the base rate even when the model is unchanged.
- **Label-definition drift**: the label stops measuring the same thing (e.g. bills advance without a
  committee report more often), so a base-rate change is partly an artifact.
- **Concept shift**: the relationship P(y | x) changes. This is the shift that makes a trained model
  wrong. It can be *regime-conditional* (it depends on which party controls the chamber), in which case
  a regime feature fixes it.
- **Covariate shift**: the inputs P(x) change, marginally or jointly.

**Train pool and candidates.** The reference pool is split S1a's train set (108-115, `REL_REF`). The
candidate val/test Congresses are 116, 117 and 118 (S1a: val 116, test 117; S1b: val 117, test 118).

**Verdict rule (shared `rel_to_train`).** A candidate **fails** a check part when it falls outside the
95% **t prediction interval** for one new Congress built from the 8 reference Congresses
(mean +/- t(7, 0.975) * sd * sqrt(1 + 1/8), about 2.5 SD; log scale for rates and counts; one-sided for
distances such as PSI or AUROC, where a reference Congress's value is its leave-one-out distance to the
other 7). Three extra readings guard against over-calling shift:

- **Multiplicity.** Each check has several parts (features, chambers, t0 values) times 3 candidates. A
  check is recorded as **FAIL only if at least one test survives a Holm correction** within that check.
  The family holds one test per (part x candidate) cell plus one **joint test per part** that asks
  whether the 3 candidates as a group sit off the reference mean (two-sample t of the candidate mean vs
  the 8 reference Congresses; it catches a drop that recurs in every candidate but is too small to
  survive in any single cell). The raw per-part t-PI failures are still printed and kept in `detail`,
  next to the family-wise nested-LOO false-alarm rate. `by_split`: a split fails if a candidate Congress
  it evaluates has a Holm-surviving cell, or a raw cell failure confirmed by a Holm-surviving joint test
  (same direction).
- **Trend vs break.** Each failing level quantity gets one of three labels: **break** (also outside the
  prediction interval of a linear trend fitted on 108-115), **consistent with trend** (inside the trend
  interval AND the 108-115 slope is itself significant), or **undetermined** (inside the trend interval
  but no significant slope: with n = 8 and 1-3 Congresses of extrapolation, the trend test simply has
  too little power to separate trend from level shift). Only the first two map to a remedy.
- **Deployment proxy (A15).** The in-progress 119th Congress is where the model would be used; A15 checks
  which Congresses look most like it, and A15b whether similarity follows **recency** or **party regime**
  (same House / Senate majorities), with and without the sponsor-party features that encode party
  control by construction.

Fixed bands (e.g. 0.8-1.25x) are kept as a secondary column. Congress 119 is excluded from every split.

---
## [A-c9-txt] text
```
train pool 108-115 | candidates [116, 117, 118] | uncensored advancement labels: 110,802 | bill features: 47 columns over 128,791 law-capable bills | verdict: outside the 95% t prediction interval of 108-115
majority party of 119 (for A15): House of Representatives: R; Senate: R | bills whose sponsor has no caucus/party (treated as minority; own level in the classifiers): {108: 1, 109: 1, 110: 2, 111: 1, 117: 11}
```

---
## [A-c10-md] markdown
### A1. Advancement base rate (Task 2) per Congress, and what drives its change
**Why it matters:** if the share of bills reported within 180 days differs a lot between train and the
held-out Congress, a model trained on 108-115 is mis-calibrated there, and AUPRC / F1 on the test
Congress cannot be compared with validation scores. **How to read it:** `rate` per Congress; `ratio` =
rate / pooled-train rate, `h` = Cohen's h and `p` = chi-square p (train rows: vs the other train
Congresses). With about 10k bills almost every gap is "significant", so the verdict uses the t prediction
interval of the train Congresses (`t` = standardised distance, `tr` = residual vs the 108-115 trend). The decomposition splits each candidate's log rate change into a **volume part**
(more bills introduced) and a **count part** (fewer bills reported), both relative to the mean train
Congress.

---
## [A-c11-txt] text
```
pooled train rate = 0.0650 (n=72,139)
              n  n_reported   rate  censored_share  ratio       h      p
congress                                                                
108        7533         606 0.0804          0.1262 1.2732  0.0670 0.0000
109        9494         524 0.0552          0.1125 0.8303 -0.0472 0.0000
110        9784         761 0.0778          0.1283 1.2350  0.0579 0.0000
111        9382         444 0.0473          0.1289 0.6998 -0.0876 0.0000
112        9592         414 0.0432          0.0961 0.6316 -0.1104 0.0000
113        7727         447 0.0578          0.1493 0.8786 -0.0332 0.0071
114        8948         653 0.0730          0.1247 1.1429  0.0362 0.0011
115        9679         839 0.0867          0.1520 1.4066  0.0960 0.0000
116       11866         491 0.0414          0.1725 0.6367 -0.1058 0.0000
117       12953         364 0.0281          0.1502 0.4324 -0.1787 0.0000
118       13844         611 0.0441          0.1643 0.6791 -0.0922 0.0000
  A1: ref 0.0432..0.0867, 95% PI [0.0329, 0.1219] (LOO false alarms 0/8) | 116: 0.0414 t=-1.5 tr=-1.3 | 117: 0.0281 t=-2.9 tr=-2.3 FAIL-undetermined | 118: 0.0441 t=-1.3 tr=-1.1 | joint t=-3.0 p=0.02
  A1: raw t-PI failures ['117 (undetermined)']; Holm survivors (decide the verdict): cells none, joint none
[PASS] A1: Advancement base rate of each candidate within the spread of train Congresses | metric=-2.933 (threshold outside 95% t prediction interval of ref 108-115 (two-sided, log scale; n=8); FAIL only if a test survives Holm over 3 cells + 1 joint tests)
decomposition vs the mean train Congress (log_change = volume_part + count_part):
          n_ratio  n_reported_ratio  rate_ratio  log_change  volume_part  count_part  volume_share
congress                                                                                          
116        1.3160            0.8380      0.6370     -0.4510      -0.2750     -0.1770        0.6080
117        1.4360            0.6210      0.4320     -0.8380      -0.3620     -0.4760        0.4320
118        1.5350            1.0430      0.6790     -0.3870      -0.4290      0.0420        1.1080
  A1d n_reported: ref 414..839, 95% PI [293, 1100] (LOO false alarms 0/8) | 116: 491 t=-0.5 tr=-0.7 | 117: 364 t=-1.6 tr=-1.5 | 118: 611 t=+0.3 tr=-0.1 | joint t=-1.0 p=0.36
  A1d: raw t-PI failures none; Holm survivors (decide the verdict): cells none, joint none
[PASS] A1d: Number of reported bills (numerator of the Task-2 rate) within the spread of train Congresses | metric=-1.594 (threshold outside 95% t prediction interval of ref 108-115 (two-sided, log scale; n=8); FAIL only if a test survives Holm over 3 cells + 1 joint tests)
```

---
## [A-c12-md] markdown
### A1b-A1c. Does the Task-2 label measure the same thing in every Congress?
**Why it matters:** "reported out of committee within 180 days" misses bills that advance without a
committee report: suspension of the rules in the House, unanimous consent in the Senate. If that route
grows in later Congresses, part of the base-rate drop is a **label-definition artifact**, and a model
would be penalised for correctly ranking bills that did advance. **How to read it:** among uncensored
bills that passed their chamber of origin within 180 days, the share *not* reported within 180 days
(`unrep_share`; `never_rep` = no Reported action at all). A1c recomputes the base rate with an
alternative label (reported **or** passed the origin chamber within 180 days) and checks whether the
candidates' drop shrinks under it.

---
## [A-c13-txt] text
```
n_passed  unrep_share  never_rep  unrep_share_House  unrep_share_Senate  alt_rate  label_rate
congress                                                                                               
108            645       0.4030     0.3710             0.4530              0.3150    0.1150      0.0800
109            584       0.5260     0.4850             0.5460              0.4840    0.0880      0.0550
110            787       0.4170     0.3710             0.4000              0.4930    0.1110      0.0780
111            593       0.5130     0.4640             0.4940              0.6040    0.0800      0.0470
112            353       0.4250     0.3800             0.4010              0.5060    0.0590      0.0430
113            411       0.4040     0.3940             0.4040              0.4050    0.0790      0.0580
114            608       0.3470     0.3190             0.3430              0.3640    0.0970      0.0730
115            786       0.3910     0.3520             0.3790              0.4610    0.1180      0.0870
116            580       0.6050     0.5740             0.5870              0.6540    0.0710      0.0410
117            563       0.5810     0.5470             0.5930              0.5420    0.0530      0.0280
118            414       0.4730     0.4280             0.4340              0.5900    0.0580      0.0440
  A1b: ref 0.347..0.526, 95% PI [0.306, 0.551] (LOO false alarms 1/8) | 116: 0.605 t=+2.7 tr=+3.3 FAIL-break | 117: 0.581 t=+2.4 tr=+3.0 FAIL-break | 118: 0.473 t=+0.7 tr=+1.6 | joint t=+3.0 p=0.0095
  A1b: raw t-PI failures ['116 (break)', '117 (break)']; Holm survivors (decide the verdict): cells ['116 (break)'], joint ['all joint (t=+3.0)']
[FAIL] A1b: Share of chamber-passed bills that the Task-2 label misses (no report within 180 d) not above train spread | metric=2.74 (threshold outside 95% t prediction interval of ref 108-115 (upper tail; n=8); FAIL only if a test survives Holm over 3 cells + 1 joint tests) [fails for: S1a, S1b, S2, S3, S1a+stream, S1b+stream, S3+stream, S1pool, S4]
candidate log ratio vs pooled train, Task-2 label vs alternative label:
          log_ratio_label  log_ratio_alt  drop_shrinks
congress                                              
116               -0.4510        -0.2720          True
117               -0.8380        -0.5580          True
118               -0.3870        -0.4690         False
  A1c: ref 0.0588..0.1184, 95% PI [0.0502, 0.1656] (LOO false alarms 1/8) | 116: 0.0710 t=-1.0 tr=-0.7 | 117: 0.0533 t=-2.1 tr=-1.4 | 118: 0.0583 t=-1.8 tr=-1.1 | joint t=-2.6 p=0.038
  A1c: raw t-PI failures none; Holm survivors (decide the verdict): cells none, joint none
[PASS] A1c: Alternative advancement label (reported OR passed origin chamber in 180 d): base rate within train spread | metric=-2.122 (threshold outside 95% t prediction interval of ref 108-115 (two-sided, log scale; n=8); FAIL only if a test survives Holm over 3 cells + 1 joint tests)
```

---
## [A-c14-md] markdown
### A2. Advancement base rate by chamber
**Why it matters:** House and Senate bills have different reporting rates, so a pooled rate can hide
opposite movements. A by-Congress split needs each chamber to be roughly stable, or the model needs
chamber as a feature (and per-chamber evaluation). **How to read it:** the chart shows the rate per
chamber, the pooled rate and the pooled rate under A1c's alternative label; the check applies the t
prediction-interval rule within each chamber.

---
## [A-c15-txt] text
```
congress    108    109    110    111    112    113    114    115    116    117    118
ch                                                                                   
House    0.0695 0.0499 0.0722 0.0538 0.0459 0.0576 0.0744 0.0905 0.0351 0.0257 0.0429
Senate   0.1005 0.0636 0.0888 0.0368 0.0381 0.0584 0.0705 0.0792 0.0525 0.0326 0.0465
  A2 House: ref 0.0459..0.0905, 95% PI [0.0352, 0.1120] (LOO false alarms 0/8) | 116: 0.0351 t=-2.4 tr=-2.5 FAIL-break | 117: 0.0257 t=-3.6 tr=-3.5 FAIL-break | 118: 0.0429 t=-1.6 tr=-1.8 | joint t=-4.0 p=0.0055
  A2 Senate: ref 0.0368..0.1005, 95% PI [0.0252, 0.1594] (LOO false alarms 0/8) | 116: 0.0525 t=-0.5 tr=-0.1 | 117: 0.0326 t=-1.7 tr=-1.0 | 118: 0.0465 t=-0.8 tr=-0.2 | joint t=-1.6 p=0.16
  A2: raw t-PI failures ['House 116 (break)', 'House 117 (break)']; Holm survivors (decide the verdict): cells none, joint ['House joint (t=-4.0)']; family-wise nested-LOO false-alarm rate 0% over 2 parts
[FAIL] A2: Per-chamber advancement rate of each candidate within the same chamber's train spread | metric=-3.645 (threshold outside 95% t prediction interval of ref 108-115 (two-sided, log scale; n=8); FAIL only if a test survives Holm over 6 cells + 2 joint tests) [fails for: S1a, S1b, S2, S3, S1a+stream, S1b+stream, S3+stream, S1pool, S4]
```

---
## [A-c15-o1] chart
```
title: Task-2 positive rate per Congress
subtitle: Uncensored law-capable bills; train 108-115, candidates 116-118; last line = alternative label
encoding: {'x': 'congress', 'y': 'rate', 'series': 'series', 'yfmt': '.1%', 'ylabel': 'positive share', 'xlabel': 'Congress'}
data (44 rows): congress | rate | series
  108 | 0.0695384615 | House
  108 | 0.1004514673 | Senate
  109 | 0.0499142367 | House
  109 | 0.0635917031 | Senate
  110 | 0.0722392638 | House
  110 | 0.0888480392 | Senate
  111 | 0.0538395168 | House
  111 | 0.0367995539 | Senate
  112 | 0.0459323953 | House
  112 | 0.0381343502 | Senate
  113 | 0.0575722092 | House
  113 | 0.0583941606 | Senate
  114 | 0.0743629745 | House
  114 | 0.0704624096 | Senate
  115 | 0.0904577192 | House
  115 | 0.0791589363 | Senate
  116 | 0.0350877193 | House
  116 | 0.0525087515 | Senate
  117 | 0.0256867642 | House
  117 | 0.0325704225 | Senate
  118 | 0.0428869146 | House
  118 | 0.0465411466 | Senate
  108 | 0.0804460374 | Both (Task-2 label)
  109 | 0.0551927533 | Both (Task-2 label)
  110 | 0.0777800491 | Both (Task-2 label)
  111 | 0.0473246643 | Both (Task-2 label)
  112 | 0.0431609675 | Both (Task-2 label)
  113 | 0.0578491006 | Both (Task-2 label)
  114 | 0.0729772016 | Both (Task-2 label)
  115 | 0.0866825085 | Both (Task-2 label)
  116 | 0.0413787291 | Both (Task-2 label)
  117 | 0.0281015981 | Both (Task-2 label)
  118 | 0.0441346432 | Both (Task-2 label)
  108 | 0.114960839 | Both (reported or passed)
  109 | 0.0875289657 | Both (reported or passed)
  110 | 0.1113041701 | Both (reported or passed)
  111 | 0.0797271371 | Both (reported or passed)
  112 | 0.0587989992 | Both (reported or passed)
  113 | 0.0793322117 | Both (reported or passed)
  114 | 0.09655789 | Both (reported or passed)
  115 | 0.1184006612 | Both (reported or passed)
  116 | 0.0709590426 | Both (reported or passed)
  117 | 0.053346715 | Both (reported or passed)
  118 | 0.058292401 | Both (reported or passed)
```

---
## [A-c16-md] markdown
### A3. Cosponsor-task labels (Task 1) at t0 = 0 and t0 = 7
**Why it matters:** ranking metrics (Hits@k, MRR, AUPRC) depend on how many positives each bill has
and on the positive density among eligible legislators. If later Congresses have fewer
post-introduction cosponsors, the test task is harder and scores drop for reasons unrelated to the
model. **How to read it:** per Congress and t0: share of bills with at least one positive, mean
positives per bill, positives / eligible (density) and censored share. The eligible pool is the
**point-in-time** roster (members in office, delegates included, in the month of introduction + t0), the
same pool section D ranks; `density_legacy` uses the full-Congress roster for comparison. A3 checks
density, A3b the share of bills with any positive.

---
## [A-c17-txt] text
```
any_pos        mean_pos        density        density_legacy        censored_share       
t0             0      7        0      7       0      7              0      7              0      7
congress                                                                                          
108       0.3417 0.3092   2.6315 2.2929  0.0084 0.0074         0.0084 0.0073         0.0017 0.0017
109       0.2966 0.2610   2.3410 1.9425  0.0077 0.0064         0.0076 0.0064         0.0143 0.0143
110       0.3498 0.3086   2.7635 2.3032  0.0086 0.0072         0.0084 0.0070         0.0052 0.0052
111       0.3426 0.2980   2.4237 1.9709  0.0079 0.0065         0.0077 0.0062         0.0123 0.0199
112       0.3091 0.2723   2.1881 1.7133  0.0069 0.0054         0.0068 0.0053         0.0150 0.0186
113       0.3868 0.3348   2.9735 2.3704  0.0093 0.0074         0.0091 0.0073         0.0142 0.0203
114       0.3789 0.3221   2.5330 1.8993  0.0080 0.0060         0.0079 0.0060         0.0157 0.0213
115       0.3661 0.3146   2.2656 1.7951  0.0071 0.0056         0.0068 0.0054         0.0255 0.0333
116       0.3843 0.3291   2.6298 2.0330  0.0084 0.0065         0.0083 0.0064         0.0218 0.0290
117       0.3454 0.2984   2.0403 1.5339  0.0065 0.0049         0.0063 0.0048         0.0311 0.0386
118       0.3200 0.2728   1.4744 1.0986  0.0047 0.0035         0.0045 0.0034         0.0300 0.0336
  A3 t0=0: ref 0.0069..0.0093, 95% PI [0.0062, 0.0102] (LOO false alarms 0/8) | 116: 0.0084 t=+0.6 tr=+0.8 | 117: 0.0065 t=-2.0 tr=-1.0 | 118: 0.0047 t=-5.2 tr=-3.2 FAIL-break | joint t=-3.5 p=0.011
  A3 t0=7: ref 0.0054..0.0074, 95% PI [0.0048, 0.0087] (LOO false alarms 0/8) | 116: 0.0065 t=+0.1 tr=+1.0 | 117: 0.0049 t=-2.2 tr=-0.9 | 118: 0.0035 t=-4.9 tr=-2.8 FAIL-break | joint t=-3.6 p=0.0086
  A3: raw t-PI failures ['t0=0 118 (break)', 't0=7 118 (break)']; Holm survivors (decide the verdict): cells ['t0=0 118 (break)', 't0=7 118 (break)'], joint none; family-wise nested-LOO false-alarm rate 0% over 2 parts
[FAIL] A3: Cosponsor positive density (t0=0 and t0=7) of each candidate within train spread | metric=-5.204 (threshold outside 95% t prediction interval of ref 108-115 (two-sided, log scale; n=8); FAIL only if a test survives Holm over 6 cells + 2 joint tests) [fails for: S1b, S2, S3, S1b+stream, S3+stream, S1pool, S4]
  A3b t0=0: ref 0.297..0.387, 95% PI [0.273, 0.436] (LOO false alarms 0/8) | 116: 0.384 t=+1.1 tr=+0.1 | 117: 0.345 t=+0.0 tr=-1.1 | 118: 0.320 t=-0.8 tr=-1.9 | joint t=+0.2 p=0.87
  A3b t0=7: ref 0.261..0.335, 95% PI [0.244, 0.373] (LOO false alarms 0/8) | 116: 0.329 t=+1.0 tr=+0.2 | 117: 0.298 t=-0.1 tr=-0.9 | 118: 0.273 t=-1.1 tr=-1.7 | joint t=-0.1 p=0.89
  A3b: raw t-PI failures none; Holm survivors (decide the verdict): cells none, joint none; family-wise nested-LOO false-alarm rate 0% over 2 parts
[PASS] A3b: Share of bills with at least one new cosponsor (t0=0 and t0=7) within train spread | metric=-1.124 (threshold outside 95% t prediction interval of ref 108-115 (two-sided, log scale; n=8); FAIL only if a test survives Holm over 6 cells + 2 joint tests)
```

---
## [A-c17-o1] chart
```
title: Cosponsor-task positive density (new cosponsors / eligible legislators)
subtitle: Positives = cosponsors joining in (t0, t0+30]; point-in-time pool (incl. delegates); uncensored law-capable bills
encoding: {'x': 'congress', 'y': 'density', 'series': 't0', 'yfmt': '.2%', 'ylabel': 'density', 'xlabel': 'Congress'}
data (22 rows): congress | density | t0
  108 | 0.0084283274 | t0=0
  109 | 0.0077492537 | t0=0
  110 | 0.0086360634 | t0=0
  111 | 0.0079206065 | t0=0
  112 | 0.0069376218 | t0=0
  113 | 0.0092946633 | t0=0
  114 | 0.008029144 | t0=0
  115 | 0.007101152 | t0=0
  116 | 0.0084470159 | t0=0
  117 | 0.0065140545 | t0=0
  118 | 0.004656196 | t0=0
  108 | 0.0073622026 | t0=7
  109 | 0.0064462686 | t0=7
  110 | 0.007217712 | t0=7
  111 | 0.0064589977 | t0=7
  112 | 0.005445417 | t0=7
  113 | 0.0074314448 | t0=7
  114 | 0.0060419426 | t0=7
  115 | 0.0056309383 | t0=7
  116 | 0.0065468296 | t0=7
  117 | 0.0049081148 | t0=7
  118 | 0.0034746781 | t0=7
```

---
## [A-c18-md] markdown
### A4. Original-cosponsor share (what is left to predict after day 0)
**Why it matters:** Task 1 only counts cosponsors who join *after* t0. Cosponsors listed at
introduction are inputs (t0 = 7) or simply absent from the target. If the share of cosponsorships that
are original rises, the target shrinks in later Congresses and the day-0 graph is denser: both the label
and the input distribution move. **How to read it:** share of active cosponsorships (law-capable
bills) that are original, and the shares joining in days 1-30 and after day 30.

---
## [A-c19-txt] text
```
congress           108    109    110    111    112    113    114    115    116    117    118
original_share  0.3890 0.3650 0.3300 0.3410 0.3310 0.3300 0.3590 0.3940 0.3830 0.4600 0.4660
post_intro_le30 0.1890 0.2030 0.2130 0.2070 0.2050 0.2200 0.1960 0.1820 0.2040 0.1720 0.1410
post_intro_gt30 0.4220 0.4320 0.4560 0.4520 0.4640 0.4500 0.4450 0.4240 0.4140 0.3680 0.3930
  A4: ref 0.330..0.394, 95% PI [0.289, 0.420] (LOO false alarms 0/8) | 116: 0.383 t=+1.0 tr=+0.8 | 117: 0.460 t=+3.8 tr=+2.8 FAIL-break | 118: 0.466 t=+4.0 tr=+2.7 FAIL-break | joint t=+4.6 p=0.0024
  A4: raw t-PI failures ['117 (break)', '118 (break)']; Holm survivors (decide the verdict): cells ['117 (break)', '118 (break)'], joint ['all joint (t=+4.6)']
[FAIL] A4: Original-cosponsor share of each candidate within train spread | metric=4.03 (threshold outside 95% t prediction interval of ref 108-115 (two-sided; n=8); FAIL only if a test survives Holm over 3 cells + 1 joint tests) [fails for: S1a, S1b, S2, S3, S1a+stream, S1b+stream, S3+stream, S1pool, S4]
```

---
## [A-c20-md] markdown
### A5. Concept stability: majority vs minority sponsor advancement, overall and by regime
**Why it matters:** majority-party sponsorship is one of the strongest Task-2 signals. If its effect
(odds ratio, OR) changes between train and the held-out Congress, the learned relationship is wrong
there. The effect may depend on **which party controls the chamber**. If candidates match train
Congresses with the same controlling party, the remedy is a regime feature or interaction, not a
different split. **How to read it:** A5 compares the pooled-chamber OR per Congress with train
Congresses. A5b compares each candidate chamber-Congress with the train chamber-Congresses that have
the same chamber and controlling party (random-effects z: sampling variance plus the between-Congress
variance seen in train). A5c tests in train whether D-control and R-control ORs differ. Majority from
`majority_party()`; sponsors are assigned to their caucus.

---
## [A-c21-txt] text
```
congress          108    109    110    111    112    113    114    115    116    117    118
sponsor_status                                                                             
rate_majority  0.1200 0.0810 0.0950 0.0550 0.0600 0.0840 0.1050 0.1260 0.0500 0.0320 0.0590
rate_minority  0.0380 0.0230 0.0470 0.0290 0.0180 0.0250 0.0360 0.0440 0.0290 0.0220 0.0250
OR             3.4690 3.7180 2.1130 1.9560 3.5110 3.6010 3.1180 3.1240 1.7240 1.4350 2.4750
  A5: ref 1.96..3.72, 95% PI [1.61, 5.60] (LOO false alarms 0/8) | 116: 1.72 t=-2.1 tr=-1.7 | 117: 1.44 t=-2.8 tr=-2.1 FAIL-undetermined | 118: 2.48 t=-0.7 tr=-0.6 | joint t=-2.9 p=0.022
  A5: raw t-PI failures ['117 (undetermined)']; Holm survivors (decide the verdict): cells none, joint none
[PASS] A5: Majority-sponsor odds ratio (both chambers pooled) of each candidate within train spread | metric=-2.791 (threshold outside 95% t prediction interval of ref 108-115 (two-sided, log scale; n=8); FAIL only if a test survives Holm over 3 cells + 1 joint tests)
between-Congress tau^2 of log OR: within regime 0.023, within chamber (regime ignored) 0.088; candidate cells below (train cells are in the chart)
 congress     ch   regime     OR  n_ref  OR_regime_ref  z_regime  OR_chamber_ref  z_chamber
      116  House  House-D 1.6400      2         2.4300   -1.7000          3.8100    -2.5200
      116 Senate Senate-R 2.3900      4         2.3600    0.0600          1.8000     0.8500
      117  House  House-D 1.6900      2         2.4300   -1.5400          3.8100    -2.4200
      117 Senate Senate-D 1.1600      4         1.4500   -0.9200          1.8000    -1.2700
      118  House  House-R 3.4500      6         4.4600   -1.2300          3.8100    -0.3000
      118 Senate Senate-D 1.5400      4         1.4500    0.2500          1.8000    -0.4700
[PASS] A5b: Each candidate chamber-Congress's majority-sponsor OR matches train Congresses of the same chamber x controlling party | metric=1.699 (threshold |z_regime| > 1.96 (random-effects, 95% two-sided) AND survives Holm over 6 candidate cells)
    ch  n_D  OR_Dcontrol  n_R  OR_Rcontrol  diff_logOR       z
 House    2       2.3800    6       4.7800     -0.7000 -4.1700
Senate    4       1.6100    4       2.3800     -0.3900 -2.4000
[INFO] A5c: Majority-sponsor effect differs by controlling party within train (regime interaction needed?) | metric=0.6992 (threshold |z| > 1.96 => model needs a regime interaction)
```

---
## [A-c21-o1] chart
```
title: Majority-sponsor odds ratio of being reported, per chamber and Congress
subtitle: Colour = party controlling the chamber (majority_party()); compare candidates with same-colour train points
data (22 rows): congress | ch | majority | OR | lo | hi
  108 | House | R | 3.9697916718 | 3.0084160444 | 5.2383864749
  108 | Senate | R | 3.1850177197 | 2.406766901 | 4.2149232942
  109 | House | R | 7.320352645 | 4.9848073883 | 10.7501772231
  109 | Senate | R | 2.1512301813 | 1.6105587549 | 2.8734073059
  110 | House | D | 2.8741057502 | 2.2611210524 | 3.6532691845
  110 | Senate | D | 1.32087177 | 1.0133916637 | 1.7216465214
  111 | House | D | 1.9659062026 | 1.4739165824 | 2.6221207113
  111 | Senate | D | 1.9841966281 | 1.2535083684 | 3.1408137019
  112 | House | R | 5.7513192491 | 4.1083365618 | 8.0513542665
  112 | Senate | D | 1.3029457306 | 0.8495146172 | 1.9983971348
  113 | House | R | 4.8053108986 | 3.5549437111 | 6.495465107
  113 | Senate | D | 1.9827548185 | 1.3281506066 | 2.9599931294
  114 | House | R | 4.0589459014 | 3.159526968 | 5.2144013953
  114 | Senate | R | 2.0682093237 | 1.5509920971 | 2.7579056107
  115 | House | R | 3.672070518 | 2.985883469 | 4.5159504813
  115 | Senate | R | 2.2827829721 | 1.7463224888 | 2.9840411098
  116 | House | D | 1.6353400109 | 1.2207385848 | 2.1907531921
  116 | Senate | R | 2.3890068471 | 1.8116928672 | 3.1502876778
  117 | House | D | 1.6911010174 | 1.256424936 | 2.2761587812
  117 | Senate | D | 1.156374905 | 0.8270081102 | 1.6169163331
  118 | House | R | 3.4506958902 | 2.6647802889 | 4.4683992058
  118 | Senate | D | 1.5356480551 | 1.1568167196 | 2.0385380926
```

---
## [A-c22-md] markdown
### A6. Concept stability: partisanship of new cosponsors (P(y | x) for Task 1)
**Why it matters:** a link-prediction model will learn "same party as the sponsor" as a strong prior.
If the share of new cosponsors from the sponsor's party drifts (polarisation), the learned prior is off
in the held-out Congress. **How to read it:** among cosponsors joining in days 1-30 of law-capable bills
with a D/R sponsor, the share from the same party and from the other major party.

---
## [A-c23-txt] text
```
congress           108         109         110         111         112         113         114         115         116         117         118
same            0.7120      0.7280      0.7880      0.8250      0.8300      0.8080      0.8160      0.7900      0.8320      0.8600      0.8320
cross           0.2820      0.2650      0.2070      0.1710      0.1670      0.1900      0.1810      0.2080      0.1650      0.1380      0.1660
n_new_cosp 22,400.0000 24,633.0000 30,804.0000 25,863.0000 22,928.0000 26,534.0000 25,492.0000 25,272.0000 36,895.0000 30,200.0000 23,736.0000
  A6: ref 0.712..0.830, 95% PI [0.676, 0.898] (LOO false alarms 0/8) | 116: 0.832 t=+1.0 tr=-0.2 | 117: 0.860 t=+1.6 tr=+0.1 | 118: 0.832 t=+0.9 tr=-0.7 | joint t=+1.8 p=0.11
  A6: raw t-PI failures none; Holm survivors (decide the verdict): cells none, joint none
[PASS] A6: Same-party share of new cosponsors of each candidate within train spread | metric=1.557 (threshold outside 95% t prediction interval of ref 108-115 (two-sided; n=8); FAIL only if a test survives Holm over 3 cells + 1 joint tests)
```

---
## [A-c24-md] markdown
### A7-A8. Marginal covariate drift (PSI) and bill volume
**Why it matters:** even with stable labels, a held-out Congress with a different mix of policy areas,
bill types, chambers, sponsor status, original-cosponsor counts or introduction timing tests the model
on a differently weighted population. **How to read it:** PSI of each candidate vs pooled 108-115
bills; a train Congress's value is its PSI vs the other 7 train Congresses. The t prediction-interval rule
decides, on the **log scale** (PSI is bounded at 0 and right-skewed, so a raw-scale interval has a
negative lower bound and over-calls large values; section B judges PSI the same way) (the
usual 0.1 / 0.25 PSI bands are shown in the chart subtitle only). The chart shows PSI divided by the
largest train LOO PSI, so bars above 1 mean "more drift than any train Congress". `policy_area_harm`
folds categories absent from some Congress (taxonomy changes) into one bucket (`PA_UNSTABLE`).

---
## [A-c25-txt] text
```
raw (unharmonised) policy_area PSI vs train, inflated by taxonomy changes: {116: 0.436, 117: 0.357, 118: 0.382}; folded: ['(missing)', 'Commemorations', 'Private Legislation', 'Social Sciences and History']
  A7 policy_area_harm: ref 0.035..0.272, 95% PI [0.030, 0.661] (LOO false alarms 0/8) | 116: 0.244 t=+0.7 tr=-0.1 | 117: 0.238 t=+0.6 tr=-0.3 | 118: 0.288 t=+0.9 tr=-0.2 | joint t=+1.1 p=0.14
  A7 bill_type: ref 0.000..0.007, 95% PI [0.000, 0.045] (LOO false alarms 0/8) | 116: 0.002 t=+0.2 tr=+0.4 | 117: 0.002 t=+0.3 tr=+0.5 | 118: 0.002 t=+0.4 tr=+0.5 | joint t=+0.5 p=0.33
  A7 ch: ref 0.000..0.006, 95% PI [0.000, 0.127] (LOO false alarms 0/8) | 116: 0.000 t=-0.5 tr=-0.6 | 117: 0.000 t=-4.5 tr=-3.5 | 118: 0.000 t=-0.1 tr=-0.4 | joint t=-2.7 p=0.98
  A7 sponsor_status: ref 0.003..0.096, 95% PI [0.001, 0.163] (LOO false alarms 1/8) | 116: 0.000 t=-2.3 tr=-1.8 | 117: 0.000 t=-2.3 tr=-1.7 | 118: 0.002 t=-1.3 tr=-0.9 | joint t=-3.1 p=0.99
  A7 n_original: ref 0.003..0.072, 95% PI [0.002, 0.163] (LOO false alarms 0/8) | 116: 0.079 t=+1.2 tr=+0.5 | 117: 0.082 t=+1.3 tr=+0.4 | 118: 0.114 t=+1.6 tr=+0.4 | joint t=+2.1 p=0.034
  A7 day_of_congress: ref 0.016..0.196, 95% PI [0.008, 0.186] (LOO false alarms 1/8) | 116: 0.038 t=-0.1 tr=+0.3 | 117: 0.035 t=-0.2 tr=+0.3 | 118: 0.050 t=+0.3 tr=+0.7 | joint t=+0.0 p=0.49
  A7: raw t-PI failures none; Holm survivors (decide the verdict): cells none, joint none; family-wise nested-LOO false-alarm rate 25% over 6 parts
[PASS] A7: Marginal covariate drift (PSI of 6 bill features) of each candidate within train LOO spread | metric=1.58 (threshold outside 95% t prediction interval of ref 108-115 (upper tail, log scale; n=8); FAIL only if a test survives Holm over 18 cells + 6 joint tests)
  A8: ref 8621..11414, 95% PI [8008, 13213] (LOO false alarms 1/8) | 116: 14340 t=+3.1 tr=+2.1 FAIL-undetermined | 117: 15242 t=+3.7 tr=+2.3 FAIL-undetermined | 118: 16565 t=+4.5 tr=+2.6 FAIL-break | joint t=+5.9 p=0.00058
  A8: raw t-PI failures ['116 (undetermined)', '117 (undetermined)', '118 (break)']; Holm survivors (decide the verdict): cells ['116 (undetermined)', '117 (undetermined)', '118 (break)'], joint ['all joint (t=+5.9)']
[FAIL] A8: Law-capable bill volume of each candidate within train spread | metric=4.5 (threshold outside 95% t prediction interval of ref 108-115 (two-sided, log scale; n=8); FAIL only if a test survives Holm over 3 cells + 1 joint tests) [fails for: S1a, S1b, S2, S3, S1a+stream, S1b+stream, S3+stream, S1pool, S4]
```

---
## [A-c25-o1] chart
```
title: Covariate drift of each candidate relative to train noise (PSI / max train LOO PSI)
subtitle: > 1 = the candidate drifts more than any train Congress does from the other train Congresses
encoding: {'x': 'feature', 'y': 'ratio', 'series': 'congress', 'yfmt': '.2f', 'ylabel': 'PSI ratio'}
data (18 rows): feature | ratio | congress
  policy_area_harm | 0.8966449415 | 116
  policy_area_harm | 0.8743059819 | 117
  policy_area_harm | 1.0592410769 | 118
  bill_type | 0.2455039547 | 116
  bill_type | 0.3115539784 | 117
  bill_type | 0.3254899337 | 118
  ch | 0.0132133038 | 116
  ch | 7.63e-08 | 117
  ch | 0.0413750655 | 118
  sponsor_status | 0.0049507385 | 116
  sponsor_status | 0.005206927 | 117
  sponsor_status | 0.0204570188 | 118
  n_original | 1.0898276453 | 116
  n_original | 1.1387445432 | 117
  n_original | 1.5803266365 | 118
  day_of_congress | 0.1921608105 | 116
  day_of_congress | 0.1783261168 | 117
  day_of_congress | 0.2551323494 | 118
```

---
## [A-c26-md] markdown
### A9-A10. Congress x Congress drift matrix (policy-area mix)
**Why it matters:** the heatmap shows which Congresses look alike. If distance grows with the time
gap, drift is gradual: the most recent Congresses are the best training data and an expanding window
dilutes them. **How to read it:** JSD (base 2, 0 = identical) of the harmonised policy-area
distribution. A9 compares each candidate's mean JSD to the train Congresses with each train Congress's
mean JSD to the other train Congresses (same rule as PSI, so A7 and A9 agree in method).

---
## [A-c27-txt] text
```
A9: ref 0.0311..0.0661, 95% PI [0.0244, 0.0697] (LOO false alarms 1/8) | 116: 0.0492 t=+0.2 tr=-0.1 | 117: 0.0506 t=+0.3 tr=-0.1 | 118: 0.0573 t=+0.9 tr=+0.3 | joint t=+0.7 p=0.25
  A9: raw t-PI failures none; Holm survivors (decide the verdict): cells none, joint none
[PASS] A9: Each candidate's mean policy-area JSD to train within the train LOO spread | metric=0.8612 (threshold outside 95% t prediction interval of ref 108-115 (upper tail; n=8); FAIL only if a test survives Holm over 3 cells + 1 joint tests)
[INFO] A10: Drift is gradual in time (Spearman of Congress gap vs policy-area JSD) | metric=0.6822 (threshold rho > 0.5 => gradual drift)
```

---
## [A-c27-o0] chart
```
title: Policy-area mix divergence between Congresses (JSD)
subtitle: Law-capable bills, harmonised policy areas; 0 = identical mix. Darker = more different
encoding: {'x': 'vs', 'y': 'Congress', 'v': 'jsd', 'fmt': '.3f'}
data (121 rows): vs | Congress | jsd
  108 | 108 | 0.0
  109 | 108 | 0.0385347503
  110 | 108 | 0.017482568
  111 | 108 | 0.0378534197
  112 | 108 | 0.0846402919
  113 | 108 | 0.0386399435
  114 | 108 | 0.0439902441
  115 | 108 | 0.0459263236
  116 | 108 | 0.0506122116
  117 | 108 | 0.054065444
  118 | 108 | 0.0641722986
  108 | 109 | 0.0385347503
  109 | 109 | 0.0
  110 | 109 | 0.0134927436
  111 | 109 | 0.0287131717
  112 | 109 | 0.0301849395
  113 | 109 | 0.0914358781
  114 | 109 | 0.0948929045
  115 | 109 | 0.1000403429
  116 | 109 | 0.1054501929
  117 | 109 | 0.1035752148
  118 | 109 | 0.1132817204
  108 | 110 | 0.017482568
  109 | 110 | 0.0134927436
  110 | 110 | 0.0
  111 | 110 | 0.01488461
  112 | 110 | 0.0427367824
  113 | 110 | 0.0483020635
  114 | 110 | 0.0519072202
  115 | 110 | 0.0554641788
  116 | 110 | 0.0599455024
  117 | 110 | 0.0594969806
  118 | 110 | 0.0668438982
  108 | 111 | 0.0378534197
  109 | 111 | 0.0287131717
  110 | 111 | 0.01488461
  111 | 111 | 0.0
  112 | 111 | 0.0222908238
  113 | 111 | 0.0345882518
  114 | 111 | 0.0362080023
  115 | 111 | 0.0432385426
  116 | 111 | 0.0465570036
  117 | 111 | 0.0481322839
  118 | 111 | 0.0553384715
  108 | 112 | 0.0846402919
  109 | 112 | 0.0301849395
  110 | 112 | 0.0427367824
  111 | 112 | 0.0222908238
  112 | 112 | 0.0
  113 | 112 | 0.0886810858
  114 | 112 | 0.0932715239
  115 | 112 | 0.1008696472
  116 | 112 | 0.1056797381
  117 | 112 | 0.1038976451
  118 | 112 | 0.1079292761
  108 | 113 | 0.0386399435
  109 | 113 | 0.0914358781
  110 | 113 | 0.0483020635
  111 | 113 | 0.0345882518
  112 | 113 | 0.0886810858
  ... (61 more rows)
```

---
## [A-c28-md] markdown
### A11. Joint covariate shift: can a classifier tell a candidate's bills from train bills?
**Why it matters:** each marginal PSI can look fine while the joint distribution shifts (e.g. the same
policy areas, but sponsored by a different party). A **domain classifier** (logistic regression on the
bill features: chamber, type, sponsor caucus and majority status, harmonised policy area,
introduction-day bin, log original cosponsors) is trained to separate train-pool bills from one
Congress's bills; its 2-fold cross-validated AUROC is 0.5 when the two are indistinguishable. Policy area
is **not** known at t0 for most fresh bills (`PA_MISS_T0`), so a second part drops it (the t0-safe
feature set). **How to read it:** candidates are compared with train Congresses classified against the
other 7 (LOO). The top coefficients say which features drive the separation.

---
## [A-c29-txt] text
```
domain AUROC (train: vs other train Congresses; candidates: vs 108-115):
        108    109    110    111    112    113    114    115    116    117    118
all  0.6380 0.6580 0.6150 0.6690 0.6050 0.5860 0.6330 0.6510 0.6290 0.6300 0.6460
noPA 0.6040 0.6260 0.6030 0.6680 0.5930 0.5420 0.5960 0.6050 0.5870 0.5840 0.5910
top separating features (log-odds of being the candidate): 116: policy_area_harm=Foreign Trade and International Finance -2.19, policy_area_harm=(unstable/missing) -1.64 | 117: sponsor_caucus=(missing) +3.10, policy_area_harm=Foreign Trade and International Finance -1.89 | 118: policy_area_harm=(unstable/missing) -2.71, policy_area_harm=Foreign Trade and International Finance -1.78
  A11 all: ref 0.586..0.669, 95% PI [0.575, 0.689] (LOO false alarms 0/8) | 116: 0.629 t=-0.1 tr=+0.2 | 117: 0.630 t=-0.1 tr=+0.3 | 118: 0.646 t=+0.5 tr=+0.6 | joint t=+0.2 p=0.44
  A11 noPA: ref 0.542..0.668, 95% PI [0.534, 0.676] (LOO false alarms 1/8) | 116: 0.587 t=-0.5 tr=+0.1 | 117: 0.584 t=-0.6 tr=+0.1 | 118: 0.591 t=-0.4 tr=+0.3 | joint t=-0.7 p=0.76
  A11: raw t-PI failures none; Holm survivors (decide the verdict): cells none, joint none; family-wise nested-LOO false-alarm rate 12% over 2 parts
[PASS] A11: Joint covariate shift (domain-classifier AUROC vs train pool) of each candidate within train LOO spread | metric=0.4691 (threshold outside 95% t prediction interval of ref 108-115 (upper tail; n=8); FAIL only if a test survives Holm over 6 cells + 2 joint tests)
```

---
## [A-c30-md] markdown
### A12-A13. Calibration and concept stability of a fitted Task-2 model
**Why it matters:** marginal ORs are a partial view. A logistic model using the same bill features as A11
is fitted on earlier Congresses and scored on a later one. The main comparison is **forward-chained**
(one step ahead, like split S3): Congress c is scored by a model fitted on 108..c-1, for c = 110-115
(train reference) and 116-118 (candidates). A leave-one-out variant (train Congress scored by a model
fitted on the other 7, which is interpolation and so easier) is shown for context. **How to read it:**
`OE` = observed / mean predicted rate (calibration-in-the-large; this mostly tracks A1's base rate).
`slope` = calibration slope: 1 means relative risks transfer, < 1 means the model is over-confident (the
feature effects weakened). `coef_shift` = majority-status log-odds effect refitted on Congress c alone,
minus the effect in the fitted model. A12 checks slope and coefficient shift; A13 checks AUROC with and
without policy area (`auroc_noPA` = t0-safe features only, since policy area is mostly missing at t0).

---
## [A-c31-txt] text
```
OE  slope  auroc  coef_shift  auroc_noPA  slope_loo  auroc_loo
110 1.4610 0.9450 0.7450     -0.7870      0.6200     1.0500     0.7520
111 0.7000 0.9100 0.7490     -0.7360      0.6310     1.0670     0.7560
112 0.7880 0.9660 0.7810      0.2360      0.6590     1.0510     0.7840
113 0.8840 0.9940 0.7560      0.0810      0.6810     1.0700     0.7610
114 1.0940 0.7230 0.7120     -0.4670      0.6630     0.7450     0.7180
115 1.2750 0.8290 0.7270     -0.4500      0.6520     0.8290     0.7270
116 0.6160 0.6750 0.6660     -0.2890      0.5620     0.6750     0.6660
117 0.4680 0.7910 0.6740     -0.6950      0.5580     0.7560     0.6780
118 0.7260 0.6210 0.6640     -0.1360      0.5990     0.5910     0.6610
  A12 slope: ref 0.723..0.994, 95% PI [0.613, 1.176] (LOO false alarms 1/6) | 116: 0.675 t=-2.0 tr=-0.9 | 117: 0.791 t=-0.9 tr=+0.3 | 118: 0.621 t=-2.5 tr=-0.6 | joint t=-2.8 p=0.039
  A12 coef_shift: ref -0.787..0.236, 95% PI [-1.527, 0.819] (LOO false alarms 0/6) | 116: -0.289 t=+0.1 tr=-0.3 | 117: -0.695 t=-0.7 tr=-0.9 | 118: -0.136 t=+0.5 tr=-0.2 | joint t=-0.1 p=0.95
  A12: raw t-PI failures none; Holm survivors (decide the verdict): cells none, joint none; family-wise nested-LOO false-alarm rate 17% over 2 parts
[PASS] A12: Fitted Task-2 model transfers forward: calibration slope and majority-coefficient shift within train spread | metric=-2.496 (threshold outside 95% t prediction interval of ref 110-115 (two-sided; n=6); FAIL only if a test survives Holm over 6 cells + 2 joint tests)
  A13 auroc: ref 0.712..0.781, 95% PI [0.693, 0.797] (LOO false alarms 0/6) | 116: 0.666 t=-3.1 tr=-1.8 FAIL-undetermined | 117: 0.674 t=-2.7 tr=-1.2 FAIL-undetermined | 118: 0.664 t=-3.2 tr=-1.2 FAIL-undetermined | joint t=-4.6 p=0.003
  A13 auroc_noPA: ref 0.620..0.681, 95% PI [0.603, 0.699] (LOO false alarms 0/6) | 116: 0.562 t=-3.7 tr=-4.6 FAIL-break | 117: 0.558 t=-3.9 tr=-4.6 FAIL-break | 118: 0.599 t=-2.2 tr=-3.0 FAIL-break | joint t=-5.0 p=0.002
  A13: raw t-PI failures ['auroc 116 (undetermined)', 'auroc 117 (undetermined)', 'auroc 118 (undetermined)', 'auroc_noPA 116 (break)', 'auroc_noPA 117 (break)', 'auroc_noPA 118 (break)']; Holm survivors (decide the verdict): cells ['auroc_noPA 116 (break)', 'auroc_noPA 117 (break)'], joint ['auroc joint (t=-4.6)', 'auroc_noPA joint (t=-5.0)']; family-wise nested-LOO false-alarm rate 0% over 2 parts
[FAIL] A13: Fitted Task-2 model AUROC (forward-chained) on each candidate not below the train spread | metric=-3.911 (threshold outside 95% t prediction interval of ref 110-115 (lower tail; n=6); FAIL only if a test survives Holm over 6 cells + 2 joint tests) [fails for: S1a, S1b, S2, S3, S1a+stream, S1b+stream, S3+stream, S1pool, S4]
```

---
## [A-c32-md] markdown
### A18. Concept stability for Task 1: do pair-feature effects on cosponsoring stay the same?
**Why it matters:** A6 looks at one marginal share; A12/A13 test P(y | x) only for Task 2. A link
predictor for Task 1 learns how much each (bill, legislator) pair feature raises the chance of
cosponsoring: past cosponsoring of this sponsor's bills, reciprocity (the sponsor cosponsored the
legislator's bills), the legislator's general activity, same caucus, same state. If those effects moved,
model selection on the val Congress can still work (ranking survives) while calibrated scores and the
learned weights drift. **How to read it:** for each Congress, a logistic model on sampled (bill,
eligible legislator) pairs at t0 = 7 (point-in-time pool incl. delegates; history counted in the 730
days before introduction + 7 d). A18 judges each coefficient of 116-118 against 109-115 (108 is dropped
from the reference because its look-back window starts before the data). A18b scores each Congress
with a model fitted on all earlier Congresses (forward-chained, like A13): AUROC over pairs and the
calibration slope.

---
## [A-c33-txt] text
```
pairs per Congress after negative sampling (positives all, negatives x0.1): 47,290-52,115; positives 30,703. Coefficients (log-odds per unit; per Congress fit):
                      108    109     110    111     112     113     114     115     116     117     118
log1p(hist s->l)   0.4580 0.5630  0.5670 0.5990  0.5070  0.6510  0.7230  0.5850  0.4160  0.5360  0.5590
log1p(recip l->s) -0.0580 0.0270 -0.1060 0.0440  0.1240 -0.0100 -0.1990 -0.0700 -0.0750 -0.0780 -0.0690
log1p(activity l) -0.0930 0.4440  0.2070 0.0710 -0.0150  0.0390  0.1190  0.1490  0.1550  0.1930  0.1180
same_caucus        0.8870 0.6720  1.1170 0.8170  1.1480  1.0650  1.0470  0.9280  1.3150  1.1860  1.3620
same_state         0.8400 0.6930  0.7420 0.5210  0.8640  0.7020  0.7510  0.8970  0.6320  0.3700  0.8230
  A18 log1p(hist s->l): ref 0.507..0.723, 95% PI [0.417, 0.781] (LOO false alarms 1/7) | 116: 0.416 t=-2.5 tr=-2.8 FAIL-break | 117: 0.536 t=-0.9 tr=-1.5 | 118: 0.559 t=-0.5 tr=-1.3 | joint t=-2.0 p=0.093
  A18 log1p(recip l->s): ref -0.199..0.124, 95% PI [-0.307, 0.252] (LOO false alarms 0/7) | 116: -0.075 t=-0.4 tr=+0.2 | 117: -0.078 t=-0.4 tr=+0.3 | 118: -0.069 t=-0.4 tr=+0.4 | joint t=-0.6 p=0.55
  A18 log1p(activity l): ref -0.015..0.444, 95% PI [-0.249, 0.539] (LOO false alarms 1/7) | 116: 0.155 t=+0.1 tr=+0.9 | 117: 0.193 t=+0.3 tr=+1.2 | 118: 0.118 t=-0.2 tr=+1.0 | joint t=+0.1 p=0.92
  A18 same_caucus: ref 0.672..1.148, 95% PI [0.514, 1.427] (LOO false alarms 1/7) | 116: 1.315 t=+1.8 tr=+0.9 | 117: 1.186 t=+1.2 tr=+0.2 | 118: 1.362 t=+2.1 tr=+0.7 | joint t=+2.6 p=0.039
  A18 same_state: ref 0.521..0.897, 95% PI [0.415, 1.062] (LOO false alarms 1/7) | 116: 0.632 t=-0.8 tr=-1.5 | 117: 0.370 t=-2.8 tr=-3.1 FAIL-break | 118: 0.823 t=+0.6 tr=-0.5 | joint t=-1.5 p=0.18
  A18: raw t-PI failures ['log1p(hist s->l) 116 (break)', 'same_state 117 (break)']; Holm survivors (decide the verdict): cells none, joint none; family-wise nested-LOO false-alarm rate 43% over 5 parts
[PASS] A18: Task-1 pair-feature effects (logistic coefficients, t0=7) of each candidate within the 109-115 spread | metric=-2.791 (threshold outside 95% t prediction interval of ref 109-115 (two-sided; n=7); FAIL only if a test survives Holm over 15 cells + 5 joint tests)
forward-chained (fit 108..c-1): 110: AUROC 0.714, slope 1.16 | 111: AUROC 0.703, slope 1.02 | 112: AUROC 0.716, slope 1.17 | 113: AUROC 0.719, slope 1.14 | 114: AUROC 0.724, slope 1.11 | 115: AUROC 0.706, slope 1.02 | 116: AUROC 0.718, slope 0.98 | 117: AUROC 0.730, slope 1.02 | 118: AUROC 0.751, slope 1.14
  A18b auroc: ref 0.703..0.724, 95% PI [0.696, 0.731] (LOO false alarms 0/6) | 116: 0.718 t=+0.5 tr=+0.2 | 117: 0.730 t=+1.8 tr=+1.0 | 118: 0.751 t=+4.3 tr=+2.2 | joint t=+3.3 p=0.99
  A18b slope: ref 1.016..1.165, 95% PI [0.916, 1.289] (LOO false alarms 0/6) | 116: 0.976 t=-1.7 tr=-0.8 | 117: 1.016 t=-1.2 tr=-0.2 | 118: 1.143 t=+0.6 tr=+1.0 | joint t=-1.2 p=0.28
  A18b: raw t-PI failures none; Holm survivors (decide the verdict): cells none, joint none; family-wise nested-LOO false-alarm rate 0% over 2 parts
[PASS] A18b: Task-1 pair model fitted on earlier Congresses: AUROC and calibration slope on each candidate within forward-chained spread | metric=-1.739 (threshold outside 95% t prediction interval of ref 110-115 (lower tail; n=6); FAIL only if a test survives Holm over 6 cells + 2 joint tests)
```

---
## [A-c33-o1] chart
```
title: Task-1 pair-feature effects per Congress (logistic coefficients, t0=7)
subtitle: Fit per Congress on sampled (bill, eligible legislator) pairs; reference 109-115, candidates 116-118
encoding: {'x': 'congress', 'y': 'coef', 'series': 'series', 'yfmt': '.2f', 'ylabel': 'log-odds coefficient', 'xlabel': 'Congress'}
data (55 rows): congress | coef | series
  108 | 0.4579677863 | log1p(hist s->l)
  109 | 0.5631455544 | log1p(hist s->l)
  110 | 0.56700428 | log1p(hist s->l)
  111 | 0.5985880885 | log1p(hist s->l)
  112 | 0.5065181368 | log1p(hist s->l)
  113 | 0.6506012518 | log1p(hist s->l)
  114 | 0.7229516206 | log1p(hist s->l)
  115 | 0.5845271359 | log1p(hist s->l)
  116 | 0.4155864711 | log1p(hist s->l)
  117 | 0.5356729945 | log1p(hist s->l)
  118 | 0.5586944844 | log1p(hist s->l)
  108 | -0.0578304082 | log1p(recip l->s)
  109 | 0.0272336014 | log1p(recip l->s)
  110 | -0.1058211967 | log1p(recip l->s)
  111 | 0.0438174908 | log1p(recip l->s)
  112 | 0.1237932662 | log1p(recip l->s)
  113 | -0.0100688353 | log1p(recip l->s)
  114 | -0.1988081956 | log1p(recip l->s)
  115 | -0.0697699465 | log1p(recip l->s)
  116 | -0.0745692116 | log1p(recip l->s)
  117 | -0.0775610016 | log1p(recip l->s)
  118 | -0.0690014225 | log1p(recip l->s)
  108 | -0.0925761048 | log1p(activity l)
  109 | 0.4438633841 | log1p(activity l)
  110 | 0.2067577971 | log1p(activity l)
  111 | 0.0710895179 | log1p(activity l)
  112 | -0.0147868027 | log1p(activity l)
  113 | 0.0391731265 | log1p(activity l)
  114 | 0.1185812447 | log1p(activity l)
  115 | 0.149291845 | log1p(activity l)
  116 | 0.1545119328 | log1p(activity l)
  117 | 0.1929464968 | log1p(activity l)
  118 | 0.1183006657 | log1p(activity l)
  108 | 0.8871872906 | same_caucus
  109 | 0.6716759566 | same_caucus
  110 | 1.1171389144 | same_caucus
  111 | 0.8165924642 | same_caucus
  112 | 1.1484124636 | same_caucus
  113 | 1.065432152 | same_caucus
  114 | 1.0469973539 | same_caucus
  115 | 0.9277951876 | same_caucus
  116 | 1.3149951611 | same_caucus
  117 | 1.1857316475 | same_caucus
  118 | 1.3619713076 | same_caucus
  108 | 0.8398090313 | same_state
  109 | 0.6927805396 | same_state
  110 | 0.7424357237 | same_state
  111 | 0.521408552 | same_state
  112 | 0.8639869914 | same_state
  113 | 0.7016221983 | same_state
  114 | 0.750706832 | same_state
  115 | 0.8969021685 | same_state
  116 | 0.6316674008 | same_state
  117 | 0.3699075165 | same_state
  118 | 0.8229589008 | same_state
```

---
## [A-c34-md] markdown
### A14. Which candidate split's training pool is closest to the Congress it evaluates?
**Why it matters:** this compares the shared candidate splits (`SPLIT_DEFS`: S1a/S1b expanding, S2
sliding k=`SLIDE_K`, S3 rolling-origin expanding) on the distribution distance between each evaluation
unit's training Congresses and its evaluated Congress. A smaller distance means the evaluated Congress
is better represented by that split's training data. **How to read it:** mean distance per split and
role over its evaluation units (`iter_eval_units`). The chart shows domain-classifier AUROC per test
fold for S2 and S3 (S1a/S1b units are a subset of S3's units). This is a proxy: closer distributions
do not guarantee better scores (section D's backtest decides), and sliding pools have fewer examples.

---
## [A-c35-txt] text
```
domain_auc  adv_logratio  dens_logratio  topic_jsd
split role                                                    
S1a   test      0.6107        0.7857         0.2781     0.0312
      val       0.6292        0.4514         0.0117     0.0391
S1b   test      0.6244        0.2588         0.5906     0.0316
      val       0.6107        0.7857         0.2781     0.0312
S2    test      0.6216        0.3846         0.1818     0.0237
      val       0.6346        0.4264         0.1229     0.0255
S3    test      0.6314        0.3538         0.2140     0.0401
      val       0.6371        0.3738         0.1411     0.0382
[INFO] A14: Sliding (S2, k=3) training pools are closer to their test Congress than expanding (S3) pools | metric=0.6429 (threshold > 0.5 favours sliding window)
```

---
## [A-c35-o1] chart
```
title: How distinguishable is each test Congress from its training pool? (domain AUROC)
subtitle: S2 = sliding k=3 (train c-3..c-1), S3 = expanding (108..c-1); 0.5 = indistinguishable
encoding: {'x': 'eval_congress', 'y': 'domain_auc', 'series': 'split', 'yfmt': '.3f', 'ylabel': 'domain AUROC', 'xlabel': 'test Congress'}
data (14 rows): eval_congress | domain_auc | split
  112 | 0.6180082122 | S2
  113 | 0.6467442581 | S2
  114 | 0.6917166702 | S2
  115 | 0.6123382626 | S2
  116 | 0.6473322141 | S2
  117 | 0.5611785064 | S2
  118 | 0.5735883747 | S2
  112 | 0.6108797279 | S3
  113 | 0.6277484204 | S3
  114 | 0.6661433956 | S3
  115 | 0.6506236182 | S3
  116 | 0.6291648236 | S3
  117 | 0.6107056666 | S3
  118 | 0.6244019266 | S3
```

---
## [A-c36-md] markdown
### A15. Does the test Congress look like the deployment target (the in-progress 119th)?
**Why it matters:** a test score is only a good forecast of deployed performance if the test Congress
resembles the Congress the model will be used on. 119 is in progress (excluded from every split), but its
bills introduced so far can be compared with earlier Congresses. **How to read it:** to make the
comparison fair, every Congress is restricted to bills introduced by the same day of its Congress as the
latest 119th bill. The domain classifier uses t0-safe features only (no policy area, which young 119th
bills mostly lack). The **primary** feature set also drops sponsor caucus and majority status: the share
of bills sponsored by each party moves with the majority (table below), so those features make
same-majority Congresses look alike by construction. The `with party` column keeps them for comparison.
`119 vs c` = AUROC separating 119's bills from Congress c's (lower = more alike); `c vs c-1` shows how far
consecutive Congresses usually are. PSIs per feature compare 119 with 116-118 and with 108-117. INFO only:
it tells you which candidate test Congress (117 for S1a, 118 for S1b) is the better deployment proxy.

---
## [A-c37-txt] text
```
window: bills introduced in days 0-650 of each Congress; 119 has 16,947 law-capable bills; policy area present for 99.1% of them (118 same window: 99.9%)
sponsor-party mix (same window): the D-sponsored share follows the majority, so caucus / majority-status features encode the regime
congress              108    109    110    111    112    113    114    115    116    117    118    119
D_sponsored        0.4817 0.4436 0.6434 0.7066 0.5683 0.5542 0.4675 0.4896 0.6431 0.5894 0.4837 0.4833
majority_sponsored 0.5181 0.5564 0.6434 0.7066 0.6030 0.5566 0.5325 0.5104 0.5906 0.5894 0.5622 0.5167
regime              RH/RS  RH/RS  DH/DS  DH/DS  RH/DS  RH/DS  RH/RS  RH/RS  DH/RS  DH/DS  RH/DS  RH/RS
                         108    109    110    111    112    113    114    115    116    117    118
119 vs c (no party)   0.5750 0.6080 0.5660 0.5880 0.6210 0.5490 0.5530 0.5390 0.5410 0.5590 0.5480
119 vs c (with party) 0.5750 0.6080 0.6490 0.7130 0.6860 0.6430 0.5520 0.5400 0.6980 0.6000 0.5610
c vs c-1 (no party)      NaN 0.5950 0.5940 0.5250 0.6210 0.6080 0.5280 0.5360 0.5280 0.5370 0.5320
119 vs pooled 108-117: no party 0.552, with party 0.603; consecutive-pair reference (no party): median 0.536, max 0.621
PSI of each feature, 119 vs ... (same day window; policy_area_harm inflated by young bills without a policy area):
                  consecutive_median  consecutive_max  119_vs_116  119_vs_117  119_vs_118  119_vs_108-117
feature                                                                                                  
policy_area_harm              0.0550           0.6430      0.0420      0.0300      0.0290          0.1350
bill_type                     0.0040           0.0130      0.0110      0.0140      0.0030          0.0100
ch                            0.0020           0.0120      0.0010      0.0000      0.0000          0.0000
sponsor_status                0.0070           0.0480      0.0220      0.0210      0.0080          0.0180
n_original                    0.0080           0.0830      0.0060      0.0120      0.0310          0.0410
day_of_congress               0.0490           0.2500      0.0390      0.0460      0.0490          0.0310
[INFO] A15: Candidate test Congresses resemble the deployment target (119, in progress) | metric=0.5477 (threshold lower domain AUROC vs 119 = better deployment proxy (INFO))
```

---
## [A-c37-o1] chart
```
title: How distinguishable is the in-progress 119th Congress from each earlier Congress?
subtitle: Domain AUROC on t0-safe bill features, bills from days 0-650; 0.5 = indistinguishable; 'no party' drops sponsor caucus / majority status
encoding: {'x': 'congress', 'y': 'auroc', 'series': 'series', 'yfmt': '.3f', 'ylabel': 'domain AUROC', 'xlabel': 'Congress'}
data (32 rows): congress | auroc | series
  108 | 0.5749152391 | 119 vs c (no party)
  109 | 0.6076623712 | 119 vs c (no party)
  110 | 0.5661090975 | 119 vs c (no party)
  111 | 0.5880267883 | 119 vs c (no party)
  112 | 0.620558753 | 119 vs c (no party)
  113 | 0.5487941729 | 119 vs c (no party)
  114 | 0.5531669488 | 119 vs c (no party)
  115 | 0.5393674455 | 119 vs c (no party)
  116 | 0.5410461529 | 119 vs c (no party)
  117 | 0.5593076823 | 119 vs c (no party)
  118 | 0.5476941106 | 119 vs c (no party)
  108 | 0.574554119 | 119 vs c (with party)
  109 | 0.607962039 | 119 vs c (with party)
  110 | 0.6493192723 | 119 vs c (with party)
  111 | 0.7134639753 | 119 vs c (with party)
  112 | 0.6855653213 | 119 vs c (with party)
  113 | 0.6431611612 | 119 vs c (with party)
  114 | 0.5516783257 | 119 vs c (with party)
  115 | 0.5398174877 | 119 vs c (with party)
  116 | 0.6977319667 | 119 vs c (with party)
  117 | 0.6003170598 | 119 vs c (with party)
  118 | 0.561145403 | 119 vs c (with party)
  109 | 0.5950610895 | c vs c-1 (no party)
  110 | 0.5938303803 | c vs c-1 (no party)
  111 | 0.5249221349 | c vs c-1 (no party)
  112 | 0.6206263814 | c vs c-1 (no party)
  113 | 0.6079723347 | c vs c-1 (no party)
  114 | 0.5282438647 | c vs c-1 (no party)
  115 | 0.5357009794 | c vs c-1 (no party)
  116 | 0.5276277986 | c vs c-1 (no party)
  117 | 0.536825597 | c vs c-1 (no party)
  118 | 0.5319013028 | c vs c-1 (no party)
```

---
## [A-c38-md] markdown
### A15b. Does similarity to 119 follow recency or party regime, once party-encoding features are removed?
**Why it matters:** if Congresses with the **same House / Senate majorities** look alike in substance,
regime is a useful feature; if similarity follows **recency**, recent data matter most. A classifier that
sees sponsor caucus and majority status will find same-regime Congresses alike **by construction** (the
sponsor-party mix moves with the majority), so the test is run three times: `no party` (chamber, bill
type, day bin, original cosponsors), `majority only` (adds the caucus-agnostic 'sponsor in majority'
flag) and `with party` (adds sponsor caucus; the earlier version). **How to read it:** every pair of
Congresses (108-119, same day window) gets a domain AUROC (heatmap = `no party`). A regression over pairs,
AUROC ~ gap + [gap = 1] + same_regime, separates time distance (the dummy absorbs the consecutive pairs,
several of which share a regime; the count is printed) from sharing the regime; its one-sided p-value comes from permuting the
regime labels across Congresses. Regimes = shared `REGIME` from `majority_party()`. Whatever the result,
section D's D26 is the outcome-based test of regime weighting; this is a bill-feature proxy.

---
## [A-c39-txt] text
```
pairs regression AUROC ~ 1 + gap + [gap=1] + same_regime (n=66 pairs; 2000 permutations): p_perm = one-sided p that sharing the regime LOWERS the AUROC (regime labels permuted); p_gap = one-sided p that distance in time RAISES it (Congress time positions permuted); consecutive pairs sharing a regime: 4/11
               b_same  p_perm   b_gap  p_gap  b_gap1  auroc_119_same  auroc_119_other  closest_to_119
no party      -0.0035  0.3013  0.0034 0.0555 -0.0026          0.5688           0.5674             115
majority only -0.0127  0.0980  0.0023 0.1644 -0.0090          0.5685           0.5838             115
with party    -0.0619  0.0020 -0.0004 0.5527 -0.0156          0.5685           0.6501             115
[INFO] A15b: Similarity between Congresses follows party regime after removing sponsor-party features (deployment target 119) | metric=-0.003493 (threshold no-party same-regime coefficient < 0 with permutation p < 0.05 => regime is a real similarity axis (INFO))
```

---
## [A-c39-o1] chart
```
title: How distinguishable are two Congresses' bills? (pairwise domain AUROC)
subtitle: t0-safe features without sponsor caucus / majority status, bills from days 0-650; labels = Congress and House/Senate majority; 0.5 = alike
encoding: {'x': 'x', 'y': 'y', 'v': 'no party', 'fmt': '.2f'}
data (132 rows): x | y | no party
  108 RH/RS | 109 RH/RS | 0.5950610895
  108 RH/RS | 110 DH/DS | 0.5652710259
  108 RH/RS | 111 DH/DS | 0.574503262
  108 RH/RS | 112 RH/DS | 0.6193911541
  108 RH/RS | 113 RH/DS | 0.5580055463
  108 RH/RS | 114 RH/RS | 0.5315958693
  108 RH/RS | 115 RH/RS | 0.5459998875
  108 RH/RS | 116 DH/RS | 0.5664675885
  108 RH/RS | 117 DH/DS | 0.5738781538
  108 RH/RS | 118 RH/DS | 0.5708756787
  108 RH/RS | 119 RH/RS | 0.5749152391
  109 RH/RS | 110 DH/DS | 0.5938303803
  109 RH/RS | 111 DH/DS | 0.6022433284
  109 RH/RS | 112 RH/DS | 0.5802263375
  109 RH/RS | 113 RH/DS | 0.5761536505
  109 RH/RS | 114 RH/RS | 0.5762035604
  109 RH/RS | 115 RH/RS | 0.5978305638
  109 RH/RS | 116 DH/RS | 0.6038645749
  109 RH/RS | 117 DH/DS | 0.6135866087
  109 RH/RS | 118 RH/DS | 0.6088213233
  109 RH/RS | 119 RH/RS | 0.6076623712
  110 DH/DS | 111 DH/DS | 0.5249221349
  110 DH/DS | 112 RH/DS | 0.6091501729
  110 DH/DS | 113 RH/DS | 0.5568000979
  110 DH/DS | 114 RH/RS | 0.5631709926
  110 DH/DS | 115 RH/RS | 0.572436145
  110 DH/DS | 116 DH/RS | 0.575925047
  110 DH/DS | 117 DH/DS | 0.5917839393
  110 DH/DS | 118 RH/DS | 0.5870002637
  110 DH/DS | 119 RH/RS | 0.5661090975
  111 DH/DS | 112 RH/DS | 0.6206263814
  111 DH/DS | 113 RH/DS | 0.5707049943
  111 DH/DS | 114 RH/RS | 0.5631637335
  111 DH/DS | 115 RH/RS | 0.5794538239
  111 DH/DS | 116 DH/RS | 0.5893436164
  111 DH/DS | 117 DH/DS | 0.6063799318
  111 DH/DS | 118 RH/DS | 0.6025259994
  111 DH/DS | 119 RH/RS | 0.5880267883
  112 RH/DS | 113 RH/DS | 0.6079723347
  112 RH/DS | 114 RH/RS | 0.6099094816
  112 RH/DS | 115 RH/RS | 0.6132706275
  112 RH/DS | 116 DH/RS | 0.6242850135
  112 RH/DS | 117 DH/DS | 0.6364394482
  112 RH/DS | 118 RH/DS | 0.6337869369
  112 RH/DS | 119 RH/RS | 0.620558753
  113 RH/DS | 114 RH/RS | 0.5282438647
  113 RH/DS | 115 RH/RS | 0.5386802512
  113 RH/DS | 116 DH/RS | 0.5390981496
  113 RH/DS | 117 DH/DS | 0.5489156921
  113 RH/DS | 118 RH/DS | 0.546729136
  113 RH/DS | 119 RH/RS | 0.5487941729
  114 RH/RS | 115 RH/RS | 0.5357009794
  114 RH/RS | 116 DH/RS | 0.5413593422
  114 RH/RS | 117 DH/DS | 0.5437881914
  114 RH/RS | 118 RH/DS | 0.5437932611
  114 RH/RS | 119 RH/RS | 0.5531669488
  115 RH/RS | 116 DH/RS | 0.5276277986
  115 RH/RS | 117 DH/DS | 0.552067775
  115 RH/RS | 118 RH/DS | 0.5427707163
  115 RH/RS | 119 RH/RS | 0.5393674455
  ... (72 more rows)
```

---
## [A-c40-md] markdown
### Section A summary: multiplicity, trend vs break, and what survives
**Why it matters:** section A runs dozens of check parts per candidate. At a 5% level, about one part in
twenty fails by chance even if nothing changed, and the parts are correlated. Every check above is
already decided by its **Holm survivors** (cells plus joint candidate tests). A17 then asks whether a
candidate fails more than chance predicts. Because several checks track the same underlying quantity
(A1 / A1c / A1d / A2 all see the Task-2 rate drop), A17 counts **clusters of checks** (one per underlying
quantity family), each with a false-alarm rate of 1 - 0.95^m for m member checks, and compares the count
with a Poisson-binomial null. A drift FAIL here means "the later Congresses, and so the deployment
target, differ from history; the forward split measures it; apply the remedy", **not** "this split is
invalid". The trend-vs-break table labels every failing level part and the remedy each implies. The
heatmap shows the standardised distance `t` of every check part.

---
## [A-c41-txt] text
```
Section A: 13 pass, 6 fail, 5 info (verdicts = Holm survivors within each check, cells + joint tests). Per candidate: raw failing parts, failing checks (binomial, correlated: optimistic) and failing clusters (Poisson-binomial; decides A17):
          parts  raw_failing  ref_share_max checks_failing  check_binom_p clusters_failing  expected  cluster_p                                                                           failing_clusters
congress                                                                                                                                                                                                  
116          36            6         0.1330           4/19         0.0130              4/6    0.8900     0.0050                Covariates, Task-2 base rate, Task-2 concept / model, Task-2 label validity
117          36            9         0.1330           5/19         0.0020              5/6    0.8900     0.0000  Covariates, Task-1 label, Task-2 base rate, Task-2 concept / model, Task-2 label validity
118          36            6         0.1330           4/19         0.0130              3/6    0.8900     0.0440                                           Covariates, Task-1 label, Task-2 concept / model
Holm survivors (cell or joint): 116: A1b, A2 House, A8, A13 auroc, A13 auroc_noPA | 117: A1b, A2 House, A4, A8, A13 auroc, A13 auroc_noPA | 118: A3 t0=0, A3 t0=7, A4, A8, A13 auroc, A13 auroc_noPA
[FAIL] A17: Each candidate's number of failing A check clusters (Holm survivors) is consistent with chance | metric=5 (threshold FAIL if Poisson-binomial p < 0.05 for the number of failing clusters (6 clusters, false-alarm rate 1 - 0.95^m each)) [fails for: S1a, S1b, S2, S3, S1a+stream, S1b+stream, S3+stream, S1pool, S4]

Trend vs break (failing level parts; t / tr = distance from the stationary / linear-trend PI; * = Holm survivor (cell or joint); 'trend' needs a significant 108-115 slope):
  A1: 117 undetermined (t=-2.9, tr=-2.3) -> (none: low power)
  A1b: 116 break* (t=+2.7, tr=+3.3); 117 break* (t=+2.4, tr=+3.0) -> recency-weighted training + recalibrate on val
  A2 House: 116 break* (t=-2.4, tr=-2.5); 117 break* (t=-3.6, tr=-3.5) -> recency-weighted training + recalibrate on val
  A3 t0=0: 118 break* (t=-5.2, tr=-3.2) -> recency-weighted training + recalibrate on val
  A3 t0=7: 118 break* (t=-4.9, tr=-2.8) -> recency-weighted training + recalibrate on val
  A4: 117 break* (t=+3.8, tr=+2.8); 118 break* (t=+4.0, tr=+2.7) -> recency-weighted training + recalibrate on val
  A5: 117 undetermined (t=-2.8, tr=-2.1) -> (none: low power)
  A8: 116 undetermined* (t=+3.1, tr=+2.1); 117 undetermined* (t=+3.7, tr=+2.3); 118 break* (t=+4.5, tr=+2.6) -> undetermined: (none: low power) / break: recency-weighted training + recalibrate on val
  A13 auroc: 116 undetermined* (t=-3.1, tr=-1.8); 117 undetermined* (t=-2.7, tr=-1.2); 118 undetermined* (t=-3.2, tr=-1.2) -> (none: low power)
  A13 auroc_noPA: 116 break* (t=-3.7, tr=-4.6); 117 break* (t=-3.9, tr=-4.6); 118 break* (t=-2.2, tr=-3.0) -> recency-weighted training + recalibrate on val
  A18 log1p(hist s->l): 116 break (t=-2.5, tr=-2.8) -> recency-weighted training + recalibrate on val
  A18 same_state: 117 break (t=-2.8, tr=-3.1) -> recency-weighted training + recalibrate on val
[INFO] A16: Candidate drift is a predictable trend rather than a structural break (failing level quantities) | metric=0.6667 (threshold share of failing level parts that are breaks (INFO))

Section A split-dependent checks per split (judged = True/False verdicts; A14/A15/A16 carry numeric by_split metrics instead):
split    S1a  S1b  S2  S3  S4  S5  S1a+stream  S1b+stream  S3+stream  S1pool
judged    20   20  20  20  20   0          20          20         20      20
failing    6    7   7   7   7   0           6           7          7       7
forward splits judged on the same number of A drift checks: True ({'S1a': 20, 'S1b': 20, 'S2': 20, 'S3': 20, 'S4': 20, 'S1a+stream': 20, 'S1b+stream': 20, 'S3+stream': 20, 'S1pool': 20}); S5 judged on 0 (random split hides the shift)
```

---
## [A-c41-o1] chart
```
title: Section A: standardised distance t of each candidate from the 8 train Congresses
subtitle: t = (x - mean) / (sd * sqrt(1 + 1/n)); fails beyond the t(7) 95% point (one-sided for distances); red = above train, blue = below
encoding: {'x': 'congress', 'y': 'part', 'v': 't', 'fmt': '+.1f'}
data (108 rows): congress | part | t
  116 | A1 | -1.54
  117 | A1 | -2.93
  118 | A1 | -1.3
  116 | A1d n_reported | -0.52
  117 | A1d n_reported | -1.59
  118 | A1d n_reported | 0.26
  116 | A1b | 2.74
  117 | A1b | 2.36
  118 | A1b | 0.7
  116 | A1c | -0.99
  117 | A1c | -2.12
  118 | A1c | -1.77
  116 | A2 House | -2.37
  117 | A2 House | -3.64
  118 | A2 House | -1.55
  116 | A2 Senate | -0.48
  117 | A2 Senate | -1.71
  118 | A2 Senate | -0.79
  116 | A3 t0=0 | 0.55
  117 | A3 t0=0 | -1.96
  118 | A3 t0=0 | -5.2
  116 | A3 t0=7 | 0.1
  117 | A3 t0=7 | -2.15
  118 | A3 t0=7 | -4.86
  116 | A3b t0=0 | 1.09
  117 | A3b t0=0 | 0.01
  118 | A3b t0=0 | -0.77
  116 | A3b t0=7 | 0.98
  117 | A3b t0=7 | -0.12
  118 | A3b t0=7 | -1.12
  116 | A4 | 1.01
  117 | A4 | 3.81
  118 | A4 | 4.03
  116 | A5 | -2.1
  117 | A5 | -2.79
  118 | A5 | -0.73
  116 | A5b House | -1.7
  116 | A5b Senate | 0.06
  117 | A5b House | -1.54
  117 | A5b Senate | -0.92
  118 | A5b House | -1.23
  118 | A5b Senate | 0.25
  116 | A6 | 0.97
  117 | A6 | 1.56
  118 | A6 | 0.95
  116 | A7 policy_area_harm | 0.67
  117 | A7 policy_area_harm | 0.64
  118 | A7 policy_area_harm | 0.88
  116 | A7 bill_type | 0.21
  117 | A7 bill_type | 0.33
  118 | A7 bill_type | 0.36
  116 | A7 ch | -0.51
  117 | A7 ch | -4.46
  118 | A7 ch | -0.14
  116 | A7 sponsor_status | -2.31
  117 | A7 sponsor_status | -2.27
  118 | A7 sponsor_status | -1.29
  116 | A7 n_original | 1.25
  117 | A7 n_original | 1.29
  118 | A7 n_original | 1.58
  ... (48 more rows)
```