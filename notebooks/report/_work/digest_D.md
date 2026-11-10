# Digest: D. Candidate splits and baseline backtest (section key D)


---
## [D-c109-md] markdown
## D. Candidate splits and baseline backtest

Sections A-C test the *preconditions* of a temporal split (similar distributions, node overlap,
no leakage). This section tests the *consequence*: if we actually train simple, transparent
baselines under each candidate split, does the split do its job? A split is useful for model
selection if (1) validation picks the model that is best on test *among close candidates* (regret),
(2) the test set is large enough to resolve **paired** model-vs-model differences, (3) the validation
score level predicts the test level (for early stopping and reporting), and (4) its protocol choices
(expanding vs sliding window, streaming same-Congress history, refit on train + val) do not cost
accuracy. We also quantify how optimistic the leaky random split is and whether a within-Congress
split behaves differently, and (D27) what it costs to keep Congress 118 as a locked final test that
no model choice ever sees. One minimum detectable effect per task (`D_MDE` = 0.01 Recall@20 / 0.01
AUROC) is used by every selection, optimism and sizing check.

**Protocol.** Every split comes from `SPLIT_DEFS` / `iter_eval_units()` and `SPLIT_VARIANTS` (setup).
Each validation unit is scored with baselines fit on its training Congresses; each test unit with
baselines refit on train + validation ("select on val, refit, report test"; D19 checks that refit).
The baselines are counts, smoothed rates, one graph-structural score (`rw2hop`, a 2-step walk on
the time-masked cosponsorship graph) and one **learned, transductive embedding model** (`svd`: a
truncated SVD of the sponsor x cosponsor training matrix, rank and recency chosen on validation), the
closest cheap stand-in for a GNN with per-legislator embeddings (it can go stale and is cold for new
members). The goal is to measure the split, not to win: conclusions that hold for count, graph and
learned models are the robust ones.

**Prediction times.** Task 1 is scored at t0 = 0 and 7 days after introduction (task definition).
Task 2 has no prediction time in the task definition; **this section assumes prediction at
introduction** (same t0 as Task 1). At introduction the CRS policy area is usually missing, so the
deployable Task 2 baseline is `eb_rate_t0safe`; `eb_rate` (policy area always present) is reported as
the "predict about a month after introduction" variant. Task 2 selection (D12b) and test-set sizing
(D9) are judged on the t0-safe variant; the policy-area variant is shown for reference.

---
## [D-c110-txt] text
```
Protocol variants (SPLIT_VARIANTS): S1a+stream: S1a + eval-Congress events dated <= introduced_date + t0 | S1b+stream: S1b + eval-Congress events dated <= introduced_date + t0 | S3+stream: S3 rolling folds + eval-Congress events dated <= introduced_date + t0 | S1pool: train 108-115 | val 116 | test 117+118 pooled (refit on 108-116 once)
```

---
## [D-c111-md] markdown
### Baselines, metrics and the backtest engine

**Task 1 (cosponsor ranking).** For each evaluable bill (law-capable, window not censored, at least
one new cosponsor in *(t0, t0+30]*) we rank the eligible same-chamber legislators: voting members
**in office in the month of introduced_date + t0** (`ROSTER_ASOF_ALL`, the shared point-in-time pool,
delegates and resident commissioners included as the task definition requires) minus
the sponsor and anyone who already cosponsored by *t0*.
`popularity` = legislator's number of cosponsorships in the training events;
`party+pop` = same caucus as the sponsor first (as-of-Congress caucus, `members.caucus`), then popularity;
`sponsor_hist` = times the legislator cosponsored this sponsor's bills in the training events (ties
broken by popularity); `rw2hop` = a **graph-structural** score closer to what a GNN computes: a two-step
random walk from the sponsor on the time-masked cosponsorship graph, `h1 + 0.3 * h1 @ A`, where `h1` is
the sponsor's row of the sponsor->cosponsor count matrix (row-normalised) and `A` the row-normalised
symmetric graph (H + H^T). It rewards "cosponsors of my cosponsors" (multi-hop structure), not just direct
history; it is a fixed-weight version of the propagation step of LightGCN / SGC-style GNNs. Streaming
variants use the eval Congress's graph as of the first day of the month of *introduced_date + t0*.
`svd_r{rank}[_hl1]` = learned embeddings: rank-r truncated SVD of log(1 + sponsor x cosponsor counts) in
the training events (optionally recency-weighted, half-life 1 Congress), score = reconstructed affinity
sponsor -> candidate. Legislators absent from training have a zero embedding (transductive cold start).
Metrics: Recall@10, Recall@20, NDCG@20 per bill, averaged.
**Task 2 (advancement).** `prior` = constant training rate (AUROC 0.5 by construction);
`chamber_maj` = smoothed rate by chamber x sponsor-in-majority (`majority_party()`, sponsor caucus);
`eb_rate` = empirical-Bayes rate by policy area x chamber x sponsor-in-majority, shrunk toward
`chamber_maj` (20 pseudo-bills); `eb_rate_t0safe` = `eb_rate`, but bills whose policy area would not
yet be assigned at introduction fall back to `chamber_maj` (rate measured on fresh 119th bills).
Metrics: AUROC, average precision (AP) and **AP lift** = AP / positive rate (AP alone is not
comparable across Congresses whose positive rate differs).
**Uncertainty.** 95% percentile intervals from a **cluster bootstrap by sponsor** (500 resamples;
bills of the same sponsor share coalitions, so they are resampled together). `ci_low_bill` /
`ci_high_bill` give the naive bill-level bootstrap for comparison. The same resamples are used for
every baseline of a unit, so **paired differences** (model A - model B on the same bills) get their own,
much tighter, CIs.
**Near-tie families** (for model selection, D11-D12, D21): Task 1 `sponsor_hist` with recency decay
(half-life 1, 2, 4 Congresses or none) and a same-caucus tie-break variant; Task 1 `rw2hop` with
second-hop weight in {0.1, 0.3, 1} and recency half-life 1 (weights 0.1, 0.3); Task 2 `eb_rate` with shrinkage m in
{5, 20, 80} and a variant without sponsor-majority, and the same four members in their t0-safe form
(`*_t0s`, policy area hidden for fresh bills); Task 1 `svd` with rank in {8, ..., 512} (powers of 2) x recency
half-life {none, 1} (D22-D25).

---
## [D-c112-txt] text
```
engine ready: 1,452 legislators coded, 128,791 law-capable bills in 108-118, 110,802 uncensored advancement labels
policy area missing among 119th bills <= 7 days old at the snapshot (setup PA_MISS_T0): 86% of 56 bills (8-30 days: 14%); eb_rate_t0safe hides policy area for that share of evaluation bills (Task 2 prediction time = introduction)
```

---
## [D-c113-txt] text
```
backtest: 60 split/fold/role units, 2,160 result rows (+1,512 near-tie family rows), 26s (streaming 4s); no subsampling (all evaluable bills used)
Task 1 (t0=0) positives missing from the Congress roster (dropped): 0 of 580,002 (0.00%); positives outside the point-in-time pool (dropped): 443 of 580,002 (0.08%); median eligible pool 434

Mean test score per split (Task 1 Recall@20 at t0=0; Task 2 AUROC and AP lift = AP / positive rate):
split                                    S1a  S1a+stream    S1b  S1b+stream  S1pool     S2     S3  S3+stream     S4     S5
task         metric    baseline                                                                                           
advancement  AP_lift   chamber_maj    1.0930      1.0280 1.3100      1.2050  1.2180 1.3300 1.2860     1.3740 1.5180 1.2530
                       eb_rate        2.2150      2.1690 2.1140      2.1170  2.1040 2.1290 2.1590     2.1740 3.8940 2.4280
                       eb_rate_t0safe 1.2750      1.2550 1.3480      1.2710  1.2930 1.4570 1.4330     1.5220 2.5520 1.4720
             AUROC     chamber_maj    0.5450      0.5280 0.6150      0.5950  0.5870 0.6140 0.6030     0.6140 0.6250 0.6010
                       eb_rate        0.6920      0.6900 0.6970      0.6980  0.6920 0.7220 0.7240     0.7250 0.7430 0.7500
                       eb_rate_t0safe 0.5600      0.5480 0.6190      0.6050  0.5950 0.6320 0.6240     0.6310 0.6570 0.6250
cosponsor_t0 Recall@20 party+pop      0.1910      0.1950 0.1770      0.1820  0.1780 0.2290 0.2010     0.2050 0.1920 0.2170
                       popularity     0.1370      0.1400 0.1130      0.1140  0.1230 0.1630 0.1430     0.1470 0.1380 0.1620
                       rw2hop         0.2630      0.3080 0.2450      0.2960  0.2230 0.2710 0.2590     0.2950 0.3110 0.3300
                       sponsor_hist   0.2620      0.3070 0.2440      0.2970  0.2220 0.2720 0.2590     0.2960 0.3100 0.3320
```

---
## [D-c114-md] markdown
### D1-D2, D20. Does the validation score predict the test score *level*?

The validation Congress stands in for the test Congress when we tune and early-stop. If val and test
scores differ a lot (Congress *c* is systematically easier or harder than *c-1*), loss-based early
stopping and score thresholds are tuned on a different target. We report the gap (test - val) per
split and fold; a gap is *significant* when it exceeds the 95% half-width of the difference under the
cluster bootstrap (sqrt(hw_val^2 + hw_test^2)). **Calibration:** instead of a fixed relative tolerance,
a split FAILS when its gaps are larger than sampling noise explains: chi-square test of
sum (gap / se_diff)^2 over its folds (df = number of folds; a single-fold split reduces to |z| > 1.96).
Val and test share many sponsors, so se_diff is conservative. We also print the noise-expected mean
|relative gap| (sqrt(2/pi) * se_diff / val) next to the observed one. Note that a pure **level shift** does not harm model
*selection* as long as the ranking of candidates holds; rankings and selection regret are measured
directly in D11-D12, D17 and D21, so the Task 2 level gap (D2) is informational. D20 repeats D1 for
the graph-structural `rw2hop` baseline. In the chart each row is a split/fold, the two dots are val and
test with 95% cluster-bootstrap CIs; long segments mean the validation Congress is a poor proxy.

---
## [D-c115-txt] text
```
Mean |test - val| per split (Task 1 Recall@20 at t0=0: 'sponsor_hist', 'rw2hop'; Task 2 AUROC: 'eb_rate'):
            folds  T1 sponsor_hist  T1 max|rel|  T1 rw2hop  T2 eb_rate  T2 hw_diff  T2 share_signif
split                                                                                              
S1a             1           0.0070       0.0290     0.0100      0.0050      0.0450           0.0000
S1a+stream      1           0.0190       0.0650     0.0210      0.0000      0.0430           0.0000
S1b             1           0.0180       0.0680     0.0180      0.0050      0.0520           0.0000
S1b+stream      1           0.0100       0.0330     0.0120      0.0080      0.0480           0.0000
S1pool          1           0.0320       0.1280     0.0300      0.0050      0.0380           0.0000
S2              7           0.0120       0.0720     0.0110      0.0370      0.0390           0.5710
S3              7           0.0110       0.0680     0.0100      0.0300      0.0390           0.4290
S3+stream       7           0.0090       0.0650     0.0090      0.0280      0.0380           0.4290
S4              3           0.0260       0.1560     0.0200      0.0580      0.1330           0.0000
S5              1           0.0090       0.0270     0.0080      0.0030      0.0220           0.0000

Per-fold gaps, S3 (test c, val c-1): Task 1 Recall@20 'sponsor_hist' and rw2hop (t0=0), Task 2 AUROC 'eb_rate':
       T1_val  T1_test  T1_gap  T1_hw_diff  T1rw_gap  T2_val  T2_test  T2_gap  T2_hw_diff
fold                                                                                     
c=112  0.2750   0.2600 -0.0150      0.0290   -0.0120  0.7280   0.7820  0.0550      0.0370
c=113  0.2600   0.2580 -0.0020      0.0260   -0.0000  0.7820   0.7540 -0.0280      0.0370
c=114  0.2580   0.2650  0.0070      0.0250    0.0050  0.7540   0.7150 -0.0390      0.0380
c=115  0.2650   0.2720  0.0070      0.0240    0.0070  0.7150   0.7420  0.0270      0.0340
c=116  0.2720   0.2540 -0.0180      0.0240   -0.0170  0.7420   0.6880 -0.0550      0.0330
c=117  0.2540   0.2620  0.0070      0.0240    0.0100  0.6880   0.6920  0.0050      0.0450
c=118  0.2620   0.2440 -0.0180      0.0240   -0.0180  0.6920   0.6970  0.0050      0.0520
[PASS] D1: Task 1: validation Congress predicts the test score level (val-test gaps within sampling noise, S3 folds) | metric=0.03988 (threshold FAIL if gaps exceed cluster-bootstrap noise: chi-square p < 0.05 (sum of z^2 over the split's folds)) [fails for: S1pool]
[INFO] D2: Task 2: validation AUROC level vs test AUROC level (mean |val-test gap| over one-step-ahead folds; informational) | metric=0.03042 (threshold informational: level shifts do not harm selection if rankings hold (see D12/D17))
[FAIL] D2b: Task 2 at introduction (eb_rate_t0safe): val Congress predicts the test AUROC level within noise [calibration and reporting only, not model selection] | metric=0.06209 (threshold FAIL if gaps exceed cluster-bootstrap noise: chi-square p < 0.05 (sum of z^2 over the split's folds)) [fails for: S1b, S2, S3, S4, S1b+stream, S3+stream]
[PASS] D20: Task 1, graph-structural rw2hop: validation Congress predicts the test score level (gaps within noise, S3) | metric=0.03709 (threshold FAIL if gaps exceed cluster-bootstrap noise: chi-square p < 0.05 (sum of z^2 over the split's folds)) [fails for: S1pool]
```

---
## [D-c115-o1] chart
```
title: Validation vs test Recall@20 (sponsor_hist, t0=0) per split
subtitle: dots = point estimate, whiskers = 95% cluster-bootstrap CI (by sponsor); S5 is the leaky random split
encoding: {'xlabel': 'Recall@20'}
data (18 rows): row | eval_role | value | ci_low | ci_high
  S1a  | val | 0.2541615916 | 0.2371233092 | 0.27082705
  S1a  | test | 0.2615268317 | 0.2457476415 | 0.2785115606
  S1b  | val | 0.2615268317 | 0.2457476415 | 0.2785115606
  S1b+stream  | val | 0.3070785531 | 0.2893495248 | 0.3252990127
  S1b  | test | 0.2438587738 | 0.2255724228 | 0.2618068793
  S1b+stream  | test | 0.2970692298 | 0.2808835227 | 0.3133455782
  S2 c=117 | val | 0.2692113675 | 0.2521040946 | 0.2870769741
  S2 c=117 | test | 0.274427011 | 0.2575593691 | 0.2922008945
  S2 c=118 | val | 0.274427011 | 0.2575593691 | 0.2922008945
  S2 c=118 | test | 0.2601671653 | 0.2421556011 | 0.2794949251
  S4 c=116 | val | 0.2796027134 | 0.2550444896 | 0.3050355103
  S4 c=116 | test | 0.2959655205 | 0.265577594 | 0.3246142077
  S4 c=117 | val | 0.2967729653 | 0.2712756528 | 0.3219419293
  S4 c=117 | test | 0.3162447773 | 0.2832504852 | 0.3513974175
  S4 c=118 | val | 0.275919643 | 0.2501814659 | 0.3032509513
  S4 c=118 | test | 0.3189591591 | 0.2858223937 | 0.3528002364
  S5  | val | 0.3231684985 | 0.3103997264 | 0.3357240157
  S5  | test | 0.3318461406 | 0.3164381074 | 0.3460784919
```

---
## [D-c115-o3] table
```
 | folds | chi2 | mean_abs_rel | noise_rel | p | ok
split |  |  |  |  |  | 
S1a | 1 | 0.6438 | 0.0378 | 0.0376 | 0.4223 | True
S1a+stream | 1 | 2.7723 | 0.0740 | 0.0355 | 0.0959 | True
S1b | 1 | 2.0831 | 0.0667 | 0.0369 | 0.1489 | True
S1b+stream | 1 | 0.8694 | 0.0377 | 0.0322 | 0.3511 | True
S1pool | 1 | 6.2703 | 0.1196 | 0.0381 | 0.0123 | False
S2 | 7 | 5.2314 | 0.0389 | 0.0386 | 0.6318 | True
S3 | 7 | 5.9953 | 0.0371 | 0.0381 | 0.5403 | True
S3+stream | 7 | 4.8087 | 0.0299 | 0.0343 | 0.6833 | True
S4 | 3 | 3.0453 | 0.0695 | 0.0584 | 0.3847 | True
S5 | 1 | 0.6388 | 0.0241 | 0.0241 | 0.4241 | True
```

---
## [D-c116-md] markdown
### D3-D4. How much does the score *level* move from one Congress to the next?

The rolling-origin folds (S3) repeat the one-step-ahead protocol for every test Congress from
112 to 118. If a fixed baseline's score swings across folds more than sampling noise allows
(between-fold std vs the cluster-bootstrap standard error, hw/1.96), then a single test Congress
(as in S1) gives a Congress-specific number: report scores per Congress and their mean over rolling
folds. This concerns the *reported level*; comparisons between models on the same test bills are
paired and much more stable (D9). Lines are per baseline; shaded bands are 95% cluster-bootstrap CIs.

---
## [D-c117-txt] text
```
Test score across S3 folds (T1 Recall@20 at t0=0, T2 AUROC; std_over_se = between-fold std / cluster-bootstrap SE):
                    mean    std     cv    min    max  mean_ci_hw  std_over_se
   baseline                                                                  
T1 party+pop      0.2010 0.0150 0.0730 0.1770 0.2190      0.0170       1.6750
   popularity     0.1430 0.0160 0.1110 0.1130 0.1620      0.0150       2.1280
   rw2hop         0.2590 0.0080 0.0310 0.2450 0.2700      0.0170       0.9260
   sponsor_hist   0.2590 0.0090 0.0340 0.2440 0.2720      0.0170       0.9960
T2 chamber_maj    0.6030 0.0480 0.0800 0.5350 0.6560      0.0330       2.9100
   eb_rate        0.7240 0.0360 0.0500 0.6880 0.7820      0.0280       2.4730
   eb_rate_t0safe 0.6240 0.0450 0.0710 0.5600 0.6680      0.0290       2.9660
[INFO] D3: Task 1: test score level across rolling folds (CV of best baseline across S3 folds; informational) | metric=0.03396 (threshold informational (CV <= 0.10 = level stable))
[INFO] D4: Task 2: test AUROC level across rolling folds (std of best baseline across S3 folds; informational) | metric=0.03593 (threshold informational (std > 0.03 and > 1.5x SE = level moves))
```

---
## [D-c117-o0] chart
```
title: Task 1 test Recall@20 across rolling-origin folds (S3, t0=0)
subtitle: each point: baselines refit on 108..c-1, evaluated on Congress c; bands = cluster-bootstrap 95% CI
encoding: {'ylabel': 'Recall@20', 'yfmt': '.2f'}
data (28 rows): x | y | lo | hi | n | series
  112 | 0.1508808007 | 0.1340452004 | 0.1664569352 | 3230 | popularity
  112 | 0.2190211603 | 0.2019431134 | 0.2383251281 | 3230 | party+pop
  112 | 0.2598401168 | 0.2410337698 | 0.277805792 | 3230 | sponsor_hist
  112 | 0.2583206384 | 0.2398331583 | 0.2758209252 | 3230 | rw2hop
  113 | 0.136398965 | 0.1216878087 | 0.1501081791 | 3459 | popularity
  113 | 0.1958892074 | 0.1805205778 | 0.2155181958 | 3459 | party+pop
  113 | 0.2577569148 | 0.2405029525 | 0.2768235497 | 3459 | sponsor_hist
  113 | 0.2582558955 | 0.2407685245 | 0.2775191856 | 3459 | rw2hop
  114 | 0.1523964915 | 0.1373094487 | 0.1671475603 | 3812 | popularity
  114 | 0.2161057478 | 0.1985147025 | 0.2335719688 | 3812 | party+pop
  114 | 0.2647517 | 0.2486796524 | 0.2821567958 | 3812 | sponsor_hist
  114 | 0.2637486756 | 0.2484195912 | 0.2805785474 | 3812 | rw2hop
  115 | 0.1493734478 | 0.1344241777 | 0.1629287556 | 4072 | popularity
  115 | 0.205081275 | 0.1868892175 | 0.223258393 | 4072 | party+pop
  115 | 0.2720543262 | 0.2563650407 | 0.2894766971 | 4072 | sponsor_hist
  115 | 0.2704369608 | 0.2554605256 | 0.2878541761 | 4072 | rw2hop
  116 | 0.1619559062 | 0.1477932549 | 0.1754383668 | 5391 | popularity
  116 | 0.2037287518 | 0.1863150799 | 0.2196296898 | 5391 | party+pop
  116 | 0.2541615916 | 0.2371233092 | 0.27082705 | 5391 | sponsor_hist
  116 | 0.2532736209 | 0.2364669338 | 0.2704734855 | 5391 | rw2hop
  117 | 0.1374843983 | 0.1243780441 | 0.151401207 | 5101 | popularity
  117 | 0.1905368825 | 0.1762681905 | 0.205355382 | 5101 | party+pop
  117 | 0.2615268317 | 0.2457476415 | 0.2785115606 | 5101 | sponsor_hist
  117 | 0.2628509525 | 0.2467100629 | 0.2788463607 | 5101 | rw2hop
  118 | 0.1131723241 | 0.0963485265 | 0.1273420876 | 5138 | popularity
  118 | 0.1770818428 | 0.1607311693 | 0.1952469775 | 5138 | party+pop
  118 | 0.2438587738 | 0.2255724228 | 0.2618068793 | 5138 | sponsor_hist
  118 | 0.2453250633 | 0.2277799926 | 0.2628949589 | 5138 | rw2hop
```

---
## [D-c117-o1] chart
```
title: Task 2 test AUROC across rolling-origin folds (S3)
subtitle: prior omitted (AUROC = 0.5 by construction); eb_rate_t0safe hides policy area for fresh bills
encoding: {'ylabel': 'AUROC', 'yfmt': '.2f'}
data (21 rows): x | y | lo | hi | n | series
  112 | 0.5903142939 | 0.5483133236 | 0.6304664854 | 9592 | chamber_maj
  112 | 0.7822469558 | 0.7534645506 | 0.805768254 | 9592 | eb_rate
  112 | 0.6275979737 | 0.5934725169 | 0.6586019208 | 9592 | eb_rate_t0safe
  113 | 0.6555381419 | 0.6234138898 | 0.6869091545 | 7727 | chamber_maj
  113 | 0.7538748863 | 0.7254260658 | 0.7784484719 | 7727 | eb_rate
  113 | 0.668362496 | 0.6409146373 | 0.6956802354 | 7727 | eb_rate_t0safe
  114 | 0.6418355677 | 0.6145703355 | 0.6685080344 | 8948 | chamber_maj
  114 | 0.7151327346 | 0.6880254772 | 0.7411339082 | 8948 | eb_rate
  114 | 0.6635708332 | 0.6386865433 | 0.6889947684 | 8948 | eb_rate_t0safe
  115 | 0.6419526181 | 0.6201544269 | 0.6634889399 | 9679 | chamber_maj
  115 | 0.7421323462 | 0.7214345487 | 0.7626388074 | 9679 | eb_rate
  115 | 0.6584580059 | 0.6366578485 | 0.6792941635 | 9679 | eb_rate_t0safe
  116 | 0.5350318928 | 0.5052383404 | 0.5622124442 | 11866 | chamber_maj
  116 | 0.6875326873 | 0.6612951145 | 0.7122607678 | 11866 | eb_rate
  116 | 0.5688178868 | 0.5403967315 | 0.5929890124 | 11866 | eb_rate_t0safe
  117 | 0.5448527364 | 0.5088623364 | 0.5788353154 | 12953 | chamber_maj
  117 | 0.6923597393 | 0.6543527679 | 0.7291091356 | 12953 | eb_rate
  117 | 0.5598481668 | 0.524912802 | 0.5941143575 | 12953 | eb_rate_t0safe
  118 | 0.6145104431 | 0.5679502472 | 0.6549245684 | 13844 | chamber_maj
  118 | 0.6970890484 | 0.658595633 | 0.7319234541 | 13844 | eb_rate
  118 | 0.6188484054 | 0.5758491057 | 0.6536208223 | 13844 | eb_rate_t0safe
```

---
## [D-c118-md] markdown
### D5-D6, D11-D12, D17, D21. Does validation pick the model that is best on test?

The real job of a validation set is choosing among **close** candidates (hyperparameters, decay
rates, regularization). D11, D12 and D21 use near-tie families and measure, per rolling fold, whether
the validation-best candidate is also test-best (hit rate), how much test score is lost by trusting
validation (**regret** = test score of the test-best minus test score of the val-chosen candidate) and
the Kendall tau between the val and test orderings. Family members are often statistically
indistinguishable on one Congress, so a regret only counts as **significant** when the paired
cluster-bootstrap CI of (test-best minus val pick) on that test Congress excludes 0 (this is liberal:
the test-best is itself chosen on test). A check FAILs only if mean regret > D_TOL = MDE / 2 (0.005) **and** at least
2 of 7 folds have significant regret (2+ false alarms out of 7 at the 5% level has probability ~4%).
D17 repeats the selection using the mean val score of the last three rolling folds. D5-D6 compare
the main baselines (informational: most pairs are far apart). D12b repeats D12 for the t0-safe Task 2
family (the deployable one if Task 2 predicts at introduction). **Single-fold splits** (S1a, S1b) cannot
meet a "2 of 7 folds" rule, so their `by_split` entry is the regret measured on their own fold (a
metric, not a verdict), and the detail says where it falls in the S3 distribution of per-fold regret.

---
## [D-c119-txt] text
```
[INFO] D5: Task 1: main baselines keep their order from validation to test (share of S3 folds; informational) | metric=0.5714 (threshold informational (see D11/D21 for close candidates))
[INFO] D6: Task 2: main baselines keep their order from validation to test (share of S3 folds; informational) | metric=1 (threshold informational (candidates far apart))
Near-tie family selection on rolling folds (regret = test-best minus test of val pick; regret_hw = mean paired CI
half-width of that difference; n_sig = folds (of 7) where it is significant; *_3val = pick on mean of last 3 val folds):
                  family    metric split  hit_rate  mean_regret  max_regret  regret_hw  n_sig  regret_3val  n_sig_3val  mean_tau  test_spread  test_hw
  T1 sponsor_hist family Recall@20    S3    0.7143       0.0028      0.0143     0.0025      1       0.0007           0    0.8571       0.0238   0.0177
  T1 sponsor_hist family Recall@20    S2    0.4286       0.0026      0.0080     0.0042      1       0.0028           0    0.7429       0.0131   0.0182
        T1 rw2hop family Recall@20    S3    0.7143       0.0002      0.0013     0.0006      0       0.0000           0    0.9429       0.0323   0.0175
        T1 rw2hop family Recall@20    S2    0.7143       0.0004      0.0024     0.0005      1       0.0000           0    0.9143       0.0201   0.0182
       T2 eb_rate family     AUROC    S3    0.1429       0.0078      0.0439     0.0080      1       0.0015           1    0.3333       0.0368   0.0290
       T2 eb_rate family     AUROC    S2    0.1429       0.0025      0.0091     0.0078      0       0.0020           1    0.2381       0.0354   0.0282
T2 eb_rate_t0safe family     AUROC    S3    0.2857       0.0045      0.0168     0.0141      0       0.0019           0    0.2381       0.0863   0.0304
T2 eb_rate_t0safe family     AUROC    S2    0.1429       0.0011      0.0018     0.0023      0       0.0008           2    0.4286       0.0984   0.0319
Val picks (S3): T1 sponsor_hist Recall@20: sh_hl1:6, sh_none+caucus:1 | T1 rw2hop Recall@20: rw_b0.1_hl1:6, rw_b0.3_hl1:1 | T2 eb_rate AUROC: eb_m5:3, eb_m20:2, eb_m80:1, eb_nomaj_m20:1 | T2 eb_rate AP_lift: eb_m5:2, eb_m20:2, eb_nomaj_m20:2, eb_m80:1 | T2 eb_rate_t0safe AUROC: eb_m5_t0s:3, eb_m80_t0s:3, eb_nomaj_m20_t0s:1
[PASS] D11: Task 1: validation picks the test-best among close count-history candidates (regret vs paired noise, S3 folds) | metric=0.002763 (threshold FAIL if mean regret > D_TOL = MDE/2 = 0.005 Recall@20 AND regret significant (paired cluster CI) in >= 2 of 7 folds)
[PASS] D12: Task 2: validation picks the test-best among close candidates (regret vs paired noise, S3 folds, AUROC) | metric=0.007766 (threshold FAIL if mean regret > D_TOL = MDE/2 = 0.005 AUROC AND regret significant (paired cluster CI) in >= 2 of 7 folds)
[PASS] D12b: Task 2 at introduction (t0-safe family, policy area hidden for fresh bills): validation picks the test-best (regret vs paired noise, S3) | metric=0.004518 (threshold FAIL if mean regret > D_TOL = MDE/2 = 0.005 AUROC AND regret significant (paired cluster CI) in >= 2 of 7 folds)
[PASS] D21: Task 1, graph-structural rw2hop family: validation picks the test-best (regret vs paired noise, S3 folds) | metric=0.0002027 (threshold FAIL if mean regret > D_TOL = MDE/2 = 0.005 Recall@20 AND regret significant (paired cluster CI) in >= 2 of 7 folds)
[PASS] D17: Selecting on the mean of the last 3 rolling val folds picks near-test-best candidates (all families, S3) | metric=0.001879 (threshold per family: FAIL if mean regret > D_TOL (MDE/2 = 0.005) AND significant in >= 2 of 7 folds)
```

---
## [D-c119-o1] chart
```
title: Model-selection regret per rolling fold (S3, near-tie families)
subtitle: test score lost by picking the validation-best candidate instead of the test-best (0 = val picked right)
encoding: {'x': 'fold', 'y': 'regret', 'series': 'family', 'yfmt': '.4f', 'ylabel': 'regret'}
data (28 rows): fold | regret | family
  c=112 | 0.0050623931 | T1 sponsor_hist family (Recall@20)
  c=113 | 0.0142786506 | T1 sponsor_hist family (Recall@20)
  c=114 | 0.0 | T1 sponsor_hist family (Recall@20)
  c=115 | 0.0 | T1 sponsor_hist family (Recall@20)
  c=116 | 0.0 | T1 sponsor_hist family (Recall@20)
  c=117 | 0.0 | T1 sponsor_hist family (Recall@20)
  c=118 | 0.0 | T1 sponsor_hist family (Recall@20)
  c=112 | 0.0 | T1 rw2hop family (Recall@20)
  c=113 | 0.0 | T1 rw2hop family (Recall@20)
  c=114 | 0.0 | T1 rw2hop family (Recall@20)
  c=115 | 0.0 | T1 rw2hop family (Recall@20)
  c=116 | 0.000104065 | T1 rw2hop family (Recall@20)
  c=117 | 0.0013150657 | T1 rw2hop family (Recall@20)
  c=118 | 0.0 | T1 rw2hop family (Recall@20)
  c=112 | 0.0005339906 | T2 eb_rate family (AUROC)
  c=113 | 0.0016080955 | T2 eb_rate family (AUROC)
  c=114 | 0.0028536167 | T2 eb_rate family (AUROC)
  c=115 | 0.0003163106 | T2 eb_rate family (AUROC)
  c=116 | 0.0 | T2 eb_rate family (AUROC)
  c=117 | 0.0051541813 | T2 eb_rate family (AUROC)
  c=118 | 0.0438948134 | T2 eb_rate family (AUROC)
  c=112 | 0.000198174 | T2 eb_rate_t0safe family (AUROC)
  c=113 | 0.0 | T2 eb_rate_t0safe family (AUROC)
  c=114 | 0.0019015496 | T2 eb_rate_t0safe family (AUROC)
  c=115 | 0.0 | T2 eb_rate_t0safe family (AUROC)
  c=116 | 0.0126099237 | T2 eb_rate_t0safe family (AUROC)
  c=117 | 0.016804964 | T2 eb_rate_t0safe family (AUROC)
  c=118 | 0.0001145279 | T2 eb_rate_t0safe family (AUROC)
```

---
## [D-c120-md] markdown
### D22-D25. A learned, transductive embedding model: selection, staleness and cold start

Count heuristics cannot overfit and have no per-legislator parameters, so they cannot show the two
failure modes that matter most for a GNN under a by-Congress split: (1) a **high-capacity model's
validation choice may not carry over to test** (it can fit Congress-specific structure), and (2)
**learned per-legislator embeddings go stale** across an election (a member's role changes: chamber or
party switch, majority status flip, new committee portfolio), which a count heuristic hides. We fit a
truncated SVD of log(1 + sponsor x cosponsor counts) on each fold's training Congresses (ranks 8-128,
typical GNN sizes, plus 256 / 512, close to memorised history; each with or without recency weighting:
14 candidates; rank = capacity), choose the candidate on the validation Congress and refit on train + val for test, exactly as the S3 protocol prescribes. D22 is
the val-test level check (as D1), D23 the selection regret (as D11), D24 reports Recall@20 separately
for positives from **new** legislators (zero embedding) and **returning** ones, and D25 tests
staleness: among returning legislators, is the learned model's recall loss on members whose role
changed larger than the count heuristic's loss on the same positives (difference in differences,
cluster-bootstrap CI)? The member-committee roster is not in the data, so "committee shift" is a proxy:
cosine similarity < 0.5 between the committee-referral profiles of the member's sponsored bills in their
last training Congress and in the eval Congress (both with >= 5 referred bills); it is used only to
stratify the evaluation, never as a feature.

---
## [D-c121-txt] text
```
SVD embeddings: 16 factorizations, 8s. Per S3 fold (val = Congress c-1 with fit on 108..c-2; test = c with refit on 108..c-1), Recall@20 at t0=0:
           val_pick     test_best  regret  regret_hw  regret_sig    tau    val   test  svd_minus_sh
fold                                                                                               
c=112  svd_r256_hl1  svd_r256_hl1  0.0000     0.0000       False 0.9121 0.2860 0.2722        0.0123
c=113  svd_r256_hl1  svd_r512_hl1  0.0032     0.0062       False 0.7363 0.2722 0.2782        0.0205
c=114  svd_r512_hl1  svd_r256_hl1  0.0004     0.0053       False 0.9341 0.2814 0.2902        0.0255
c=115  svd_r256_hl1  svd_r512_hl1  0.0011     0.0059       False 0.8901 0.2906 0.2938        0.0217
c=116  svd_r512_hl1  svd_r512_hl1  0.0000     0.0000       False 0.9560 0.2948 0.2757        0.0215
c=117  svd_r512_hl1  svd_r512_hl1  0.0000     0.0000       False 0.9560 0.2757 0.2832        0.0217
c=118  svd_r512_hl1  svd_r512_hl1  0.0000     0.0000       False 0.9560 0.2832 0.2675        0.0236
[PASS] D22: Task 1, learned SVD embeddings (val-selected): validation Congress predicts the test score level (S3) | metric=0.03721 (threshold FAIL if gaps exceed cluster-bootstrap noise: chi-square p < 0.05 (sum of z^2 over the split's folds))
[PASS] D23: Task 1, learned SVD embeddings: validation picks the test-best rank / recency (regret vs paired noise, S3) | metric=0.0006684 (threshold FAIL if mean regret > D_TOL (MDE/2 = 0.005) Recall@20 AND regret significant (paired cluster CI) in >= 2 of 7 folds (as D11))
```

---
## [D-c121-o1] chart
```
title: Learned embeddings: test Recall@20 by capacity (mean over S3 folds 112-118)
subtitle: sponsor_hist reference 0.259; the val-selected member per fold is in the table
encoding: {'x': 'rank', 'y': 'test', 'series': 'series', 'yfmt': '.3f', 'ylabel': 'mean test Recall@20', 'xlabel': 'SVD rank (capacity)'}
data (14 rows): rank | test | series
  8 | 0.2406410335 | test, recency hl1
  8 | 0.1943566261 | test, recency none
  16 | 0.2531436275 | test, recency hl1
  16 | 0.2246783097 | test, recency none
  32 | 0.2626649613 | test, recency hl1
  32 | 0.2344237674 | test, recency none
  64 | 0.2686936247 | test, recency hl1
  64 | 0.2426806629 | test, recency none
  128 | 0.2736594222 | test, recency hl1
  128 | 0.2476364249 | test, recency none
  256 | 0.2786402492 | test, recency hl1
  256 | 0.2539643483 | test, recency none
  512 | 0.2803507633 | test, recency hl1
  512 | 0.2603119408 | test, recency none
```

---
## [D-c122-txt] text
```
Recall@20 per positive by legislator status, S3 test folds 112-118 pooled (svd = val-picked member per fold; CI = cluster bootstrap by sponsor):
                                    n_pos  share    svd  svd_r32  sponsor_hist  svd_minus_sh   ci_lo  ci_hi
stratum                                                                                                    
new (no training history)           30032 0.1574 0.0000   0.0000        0.0000        0.0000  0.0000 0.0001
returning: chamber or party switch   1440 0.0075 0.0681   0.0688        0.0694       -0.0014 -0.0152 0.0112
returning: majority status flipped  64778 0.3396 0.2444   0.2398        0.2276        0.0167  0.0141 0.0194
returning: committee shift (proxy)   3870 0.0203 0.1765   0.1840        0.1532        0.0233  0.0124 0.0330
returning: unchanged                90650 0.4752 0.2584   0.2529        0.2387        0.0197  0.0172 0.0223
[INFO] D24: Task 1: Recall@20 for new vs returning legislators, learned embeddings vs count history (S3 test, informational) | metric=3.33e-05 (threshold informational (cold start for a transductive model; section B sizes it))
[PASS] D25: Task 1: learned embeddings do not go stale for returning legislators whose role changed (DiD vs count history, S3) | metric=0.002947 (threshold FAIL if, for the val-picked or the fixed rank-32 SVD, the pooled difference in differences (svd recall loss on changed vs unchanged returning members, minus the same loss for sponsor_hist) has CI > 0 AND is significant in >= 2 of 7 folds)
```

---
## [D-c122-o1] chart
```
title: Recall@20 by legislator status: learned embeddings vs count history (S3 test folds pooled)
subtitle: new = no cosponsorship in the training Congresses (zero embedding); changes measured vs the member's last training Congress
encoding: {'x': 'stratum', 'y': 'recall', 'series': 'model', 'yfmt': '.2f', 'ylabel': 'Recall@20 (per positive)'}
data (10 rows): stratum | recall | model
  new (no training history) | 3.32978e-05 | svd
  returning: chamber or party switch | 0.0680555556 | svd
  returning: majority status flipped | 0.2443576523 | svd
  returning: committee shift (proxy) | 0.1764857881 | svd
  returning: unchanged | 0.2583673469 | svd
  new (no training history) | 0.0 | sponsor_hist
  returning: chamber or party switch | 0.0694444444 | sponsor_hist
  returning: majority status flipped | 0.2276235759 | sponsor_hist
  returning: committee shift (proxy) | 0.1532299742 | sponsor_hist
  returning: unchanged | 0.2386982901 | sponsor_hist
```

---
## [D-c123-md] markdown
### D7. How optimistic is the random bill split?

A random split mixes bills of the same Congress (and of future Congresses) across train and test,
so the baselines see the very legislators, coalitions and sponsor relationships they are tested on.
We compare the random split's test bills of each Congress with the temporal (S3) test score of
that same Congress, **on the same bills** (the S3 model scores the S5 test bills of Congress c too), so
the optimism has a paired cluster-bootstrap CI. Calibration: optimism counts when its mean exceeds the
section-wide minimum detectable effect `D_MDE` (0.01 Recall@20 / 0.01 AUROC, the smallest difference worth
acting on; the cell first compares it with the data) and it is significant in at least 2 of 7 Congresses.

**Why one fixed MDE.** Earlier rounds used the median validation margin between the two best near-tie
candidates as the MDE. That margin measures the *spacing of our candidate grid* (e.g. shrinkage 5 vs
20 pseudo-bills), not what matters in practice, and made almost every Task-2 pair "relevant". The cell
prints that margin next to the MDE: when the grid margin is far below the MDE, validation is choosing
among ties, which is harmless and is exactly what the regret tolerance (MDE / 2) allows. A positive gap confirms
the random split overstates performance and must only be used as a leaky upper-bound reference.

---
## [D-c124-txt] text
```
MDE used by every check: Task 1 0.01 Recall@20 (4% of 'sponsor_hist' mean test level), Task 2 0.01 AUROC; regret tolerance MDE/2. Median val margin pick vs runner-up (grid spacing, not practical importance): Task 1 0.0021 (IQR 0.0008-0.0051), Task 2 0.0005 (IQR 0.0002-0.0019) -> most family choices are ties below the MDE
Random (S5) minus temporal (S3) Recall@20 on the same S5 test bills, mean over Congresses 112-118:
                           mean_random  mean_temporal  mean_optimism  n_sig
task         baseline                                                      
cosponsor_t0 party+pop          0.2240         0.2030         0.0210      5
             popularity         0.1680         0.1510         0.0160      4
             rw2hop             0.3320         0.2620         0.0700      7
             sponsor_hist       0.3340         0.2590         0.0750      7
Task 2 AUROC eb_rate: random split test 0.750 vs mean S3 test 0.724
[FAIL] D7: Random bill split does not inflate scores relative to the temporal split (paired optimism on the same bills, Task 1) | metric=0.07453 (threshold FAIL if mean optimism > D_MDE (0.0100 Recall@20) AND significant (paired cluster CI) in >= 2 of 7 Congresses) [fails for: S5]
```

---
## [D-c124-o1] chart
```
title: Random vs temporal test Recall@20 per Congress (sponsor_hist, t0=0)
subtitle: random split = same Congress's bills seen in training; temporal = only earlier Congresses
encoding: {'x': 'congress', 'y': 'score', 'series': 'split', 'yfmt': '.2f', 'ylabel': 'Recall@20', 'xlabel': 'test Congress'}
data (14 rows): congress | score | split
  112 | 0.278791803 | temporal (S3)
  113 | 0.2521253252 | temporal (S3)
  114 | 0.2693232013 | temporal (S3)
  115 | 0.2601518853 | temporal (S3)
  116 | 0.2677163983 | temporal (S3)
  117 | 0.2618915171 | temporal (S3)
  118 | 0.2264332848 | temporal (S3)
  112 | 0.3246488674 | random (S5)
  113 | 0.3146419338 | random (S5)
  114 | 0.3076115127 | random (S5)
  115 | 0.3195555418 | random (S5)
  116 | 0.3581954789 | random (S5)
  117 | 0.3641744681 | random (S5)
  118 | 0.3492989007 | random (S5)
```

---
## [D-c125-md] markdown
### D8, D13-D14. Same-Congress history: within-Congress split (S4) and the streaming protocol

A temporal GNN deployed during a Congress can see that Congress's cosponsorships up to the
prediction time. S4 tests this on the last 20% of each Congress's bills. The **streaming** variants
(`S1a+stream`, `S1b+stream`, `S3+stream`) keep the by-Congress pools but let the baselines also
count every eval-Congress cosponsorship dated on or before *introduced_date + t0* of the bill being
scored (Task 2: eval-Congress bills whose 180-day label had resolved by the scored bill's
introduction). This is leakage-free by construction and is how a temporal model would be used. If
streaming adds a lot, the by-Congress split should be evaluated in streaming mode (and the cold-start
problem from section B is largely an artefact of forbidding same-Congress history).

---
## [D-c126-txt] text
```
S4 (last 20% of bills, same-Congress history) minus S3 test score on the same Congress, best baselines:
             S3_test                               S4_test                                  diff                          
task     advancement cosponsor_t0 cosponsor_t7 advancement cosponsor_t0 cosponsor_t7 advancement cosponsor_t0 cosponsor_t7
congress                                                                                                                  
116           0.6880       0.2540       0.2360      0.8100       0.2960       0.2890      0.1220       0.0420       0.0530
117           0.6920       0.2620       0.2350      0.6860       0.3160       0.2790     -0.0070       0.0550       0.0440
118           0.6970       0.2440       0.2230      0.7320       0.3190       0.2870      0.0350       0.0750       0.0630
[INFO] D8: Within-Congress split (S4) gives scores comparable to by-Congress split (mean S4-S3 test diff, Task 1) | metric=0.05721 (threshold informational)

Streaming (S3+stream) vs plain S3, test folds:
                                       mean_plain  mean_stream  mean_gain  min_gain  max_gain
series                                                                                       
advancement AUROC (eb_rate)                0.7243       0.7253     0.0010   -0.0020    0.0041
cosponsor_t0 Recall@20 (popularity)        0.1431       0.1465     0.0035    0.0008    0.0053
cosponsor_t0 Recall@20 (rw2hop)            0.2589       0.2947     0.0358    0.0227    0.0506
cosponsor_t0 Recall@20 (sponsor_hist)      0.2591       0.2961     0.0370    0.0258    0.0532
cosponsor_t7 Recall@20 (sponsor_hist)      0.2412       0.2751     0.0339    0.0207    0.0448
Val-test gaps: T1 S3: mean |rel gap| 0.040, signif 0%; T1 S3+stream: mean |rel gap| 0.029, signif 0%; T2 S3: mean |rel gap| 0.041, signif 43%; T2 S3+stream: mean |rel gap| 0.037, signif 43%
[INFO] D13: Streaming eval-Congress history changes test scores (mean gain S3+stream - S3, Task 1 Recall@20) | metric=0.03697 (threshold informational)
[PASS] D14: Task 1: under the streaming protocol the val Congress still predicts the test score (gaps within noise, S3+stream) | metric=0.02937 (threshold FAIL if gaps exceed cluster-bootstrap noise: chi-square p < 0.05 (sum of z^2 over the split's folds))
```

---
## [D-c126-o1] chart
```
title: Streaming same-Congress history: test Recall@20 per Congress (sponsor_hist, t0=0)
subtitle: streaming counts eval-Congress cosponsorships dated on or before each bill's prediction time
encoding: {'x': 'congress', 'y': 'score', 'series': 'protocol', 'yfmt': '.2f', 'ylabel': 'Recall@20', 'xlabel': 'test Congress'}
data (14 rows): congress | score | protocol
  112 | 0.2598401168 | S3 (train Congresses only)
  113 | 0.2577569148 | S3 (train Congresses only)
  114 | 0.2647517 | S3 (train Congresses only)
  115 | 0.2720543262 | S3 (train Congresses only)
  116 | 0.2541615916 | S3 (train Congresses only)
  117 | 0.2615268317 | S3 (train Congresses only)
  118 | 0.2438587738 | S3 (train Congresses only)
  112 | 0.2962571946 | S3+stream (+ eval-Congress history to t0)
  113 | 0.2897382873 | S3+stream (+ eval-Congress history to t0)
  114 | 0.2963078539 | S3+stream (+ eval-Congress history to t0)
  115 | 0.2978197969 | S3+stream (+ eval-Congress history to t0)
  116 | 0.2884473311 | S3+stream (+ eval-Congress history to t0)
  117 | 0.3070785531 | S3+stream (+ eval-Congress history to t0)
  118 | 0.2970692298 | S3+stream (+ eval-Congress history to t0)
```

---
## [D-c126-o3] table
```
 | folds | chi2 | mean_abs_rel | noise_rel | p | ok
split |  |  |  |  |  | 
S1a+stream | 1 | 2.1646 | 0.0646 | 0.0350 | 0.1412 | True
S1b+stream | 1 | 0.6562 | 0.0326 | 0.0321 | 0.4179 | True
S3+stream | 7 | 4.3074 | 0.0294 | 0.0339 | 0.7438 | True
```

---
## [D-c127-md] markdown
### D9, D15. Are the test sets large enough for what the split is used for?

A split is used to **compare** models and protocols on the same test bills, so the relevant noise is
that of the *paired difference*: Congress-level difficulty shifts move both models together and
cancel. For each decision-relevant pair we compute, on every rolling test Congress, the paired
difference and its cluster-bootstrap CI. Two different questions follow:
(a) **single-Congress risk**: how often would ONE test Congress (S1a / S1b) fail to confirm the
direction of a real difference? `p_miss_1C` = share of the 7 test Congresses whose paired CI does not
exclude 0 in the direction of the mean (wrong sign or not significant); `p_flip_1C` = share with the
opposite sign. More test Congresses do not reduce the Congress-to-Congress heterogeneity of an effect,
so the deployment Congress can still differ: `pi_next` = 95% prediction half-width of the effect in a
new Congress (t, 6 df). (b) **average effect**: the CI half-width of the mean effect over k = 1, 2, 7
test Congresses, 1.96 * max(sd_between, se_1C) / sqrt(k).
A pair is *relevant* if |mean difference| >= `D_MDE` (0.01, the same MDE as every other check here);
smaller differences are not worth acting on. The `by_split` verdict: a split holds if, for every relevant
model / near-tie pair, its miss risk is <= 20% (80% power): empirical `p_miss_1C` for one test Congress,
the normal approximation Phi(|mean| / (sd / sqrt(k)) - 1.96) for k = 2 (S1pool) and k = 7 (S2 / S3).
**Caveat: these are different quantities.** The one-Congress risk includes Congress-to-Congress
heterogeneity of the effect; the k-Congress risk is about the *average* effect, so multi-fold splits
pass more easily by construction. The deployment-relevant risk, `pi_next` relative to the MDE (how far
the effect can move in the next Congress), does not shrink with k; D9b records it for every by-Congress
split, also from the recent folds 115-118 only (the early folds 112-115 are least like 119). Task 2
pairs are judged on the **t0-safe** family; policy-area pairs (kind `ref@d30`) are shown for reference.

---
## [D-c128-txt] text
```
Paired differences on S3 test Congresses 112-118 (MDE: T1 0.0100, T2 0.0100; p_miss_1C = share of single test Congresses not confirming the direction):
    kind task                                 pair  mean_diff  sd_between  hw_1C  relevant  p_miss_1C  p_flip_1C  pi_next  pi_rel  hw_mean_k2  hw_mean_k7  miss_k2  miss_k7
   model   T1                rw2hop - sponsor_hist    -0.0002      0.0013 0.0035     False     1.0000     0.4286   0.0035  0.3451      0.0025      0.0013   0.9610   0.9440
   model   T1             sponsor_hist - party+pop     0.0581      0.0114 0.0127      True     0.0000     0.0000   0.0298  2.9848      0.0158      0.0085   0.0000   0.0000
   model   T1         svd(val pick) - sponsor_hist     0.0210      0.0041 0.0070      True     0.0000     0.0000   0.0109  1.0850      0.0057      0.0031   0.0000   0.0000
near-tie   T1                      sh_hl1 - sh_hl2     0.0065      0.0018 0.0039     False     0.1429     0.0000   0.0048  0.4818      0.0028      0.0015   0.0037   0.0000
near-tie   T1              sh_hl1 - sh_none+caucus     0.0142      0.0092 0.0075      True     0.1429     0.1429   0.0240  2.4019      0.0127      0.0068   0.4120   0.0169
near-tie   T1                    rw_b0.1 - rw_b0.3     0.0017      0.0007 0.0022     False     0.5714     0.0000   0.0018  0.1758      0.0016      0.0008   0.4541   0.0273
near-tie   T1                   svd_r32 - svd_r128    -0.0132      0.0040 0.0069      True     0.0000     0.0000   0.0105  1.0520      0.0056      0.0030   0.0036   0.0000
protocol   T1   expanding - sliding (sponsor_hist)    -0.0124      0.0056 0.0055      True     0.1429     0.0000   0.0147  1.4694      0.0078      0.0042   0.1243   0.0001
protocol   T1        stream - plain (sponsor_hist)     0.0370      0.0094 0.0085      True     0.0000     0.0000   0.0245  2.4461      0.0130      0.0069   0.0001   0.0000
   model   T2         eb_rate_t0safe - chamber_maj     0.0202      0.0117 0.0131      True     0.4286     0.0000   0.0307  3.0684      0.0163      0.0087   0.3168   0.0047
near-tie   T2               eb_m5_t0s - eb_m20_t0s    -0.0001      0.0004 0.0006     False     0.8571     0.4286   0.0011  0.1141      0.0006      0.0003   0.9383   0.8801
near-tie   T2              eb_m20_t0s - eb_m80_t0s     0.0003      0.0007 0.0013     False     1.0000     0.4286   0.0017  0.1714      0.0009      0.0005   0.9014   0.7597
near-tie   T2        eb_m20_t0s - eb_nomaj_m20_t0s     0.0827      0.0576 0.0423      True     0.2857     0.1429   0.1507 15.0721      0.0799      0.0427   0.4722   0.0331
 ref@d30   T2                eb_rate - chamber_maj     0.1209      0.0436 0.0338      True     0.0000     0.0000   0.1141 11.4148      0.0605      0.0323   0.0251   0.0000
 ref@d30   T2                eb_m20 - eb_nomaj_m20     0.0337      0.0251 0.0203      True     0.2857     0.1429   0.0656  6.5585      0.0347      0.0186   0.5245   0.0557
protocol   T2 expanding - sliding (eb_rate_t0safe)    -0.0087      0.0222 0.0201     False     0.8571     0.5714   0.0580  5.8047      0.0308      0.0164   0.9196   0.8205
[FAIL] D9: Test-set size: a single test Congress confirms the direction of every relevant model / near-tie difference (paired, S3) | metric=0.429 (threshold per split: max over relevant pairs (|mean diff| >= D_MDE) of the miss risk <= 0.20 (empirical share of single test Congresses for 1-Congress splits; normal approximation for k = 2 / 7)) [fails for: S1a, S1b, S1a+stream, S1b+stream, S1pool]
[INFO] D9b: Deployment risk: how far a relevant model difference can move in the next Congress (95% PI half-width / MDE; informational) | metric=15.07 (threshold informational: > 1 means the next Congress can reverse or erase an MDE-sized advantage seen on test)

Absolute-score CI half-widths ('sponsor_hist' Recall@20, 'eb_rate' AUROC; bill vs sponsor-cluster bootstrap; S3/S4 = mean over folds):
                         n_eval  n_clusters  value  hw_bill  hw_cluster  cluster/bill
task         split                                                                   
advancement  S1b    13,844.0000    543.0000 0.6971   0.0180      0.0367        2.0413
             S1pool 26,797.0000    647.0000 0.6920   0.0161      0.0283        1.7598
             S3     10,658.4286    544.8571 0.7243   0.0198      0.0285        1.4402
             S4        596.0000    289.3333 0.7425   0.0943      0.1149        1.2186
cosponsor_t0 S1b     5,138.0000    538.0000 0.2439   0.0091      0.0181        1.9942
             S1pool 10,239.0000    631.0000 0.2217   0.0062      0.0163        2.6072
             S3      4,314.7143    529.0000 0.2591   0.0102      0.0173        1.6928
             S4        777.0000    347.3333 0.3104   0.0264      0.0324        1.2256
[INFO] D15: Absolute score level: test Congresses needed for a precise reported score (informational) | T1 1, T2 13 Congresses (threshold informational: report test scores per Congress plus the mean over rolling folds)
```

---
## [D-c128-o1] chart
```
title: Risk that the test set fails to confirm a real model difference (relevant pairs, S3 test Congresses)
subtitle: share of evaluations whose paired CI misses the true direction; dashed line = 20% (80% power)
data (18 rows): pair | measure | value | mean_diff
  T1 sponsor_hist - party+pop | one test Congress (empirical, S1a / S1b) | 0.0 | 0.0580721982
  T1 svd(val pick) - sponsor_hist | one test Congress (empirical, S1a / S1b) | 0.0 | 0.020966248
  T1 sh_hl1 - sh_none+caucus | one test Congress (empirical, S1a / S1b) | 0.1428571429 | 0.0141691311
  T1 svd_r32 - svd_r128 | one test Congress (empirical, S1a / S1b) | 0.0 | -0.0132126575
  T2 eb_rate_t0safe - chamber_maj | one test Congress (empirical, S1a / S1b) | 0.4285714286 | 0.0202097248
  T2 eb_m20_t0s - eb_nomaj_m20_t0s | one test Congress (empirical, S1a / S1b) | 0.2857142857 | 0.0826970392
  T1 sponsor_hist - party+pop | mean of 2 test Congresses (normal approx.) | 8.13e-08 | 0.0580721982
  T1 svd(val pick) - sponsor_hist | mean of 2 test Congresses (normal approx.) | 1.06e-07 | 0.020966248
  T1 sh_hl1 - sh_none+caucus | mean of 2 test Congresses (normal approx.) | 0.4120210161 | 0.0141691311
  T1 svd_r32 - svd_r128 | mean of 2 test Congresses (normal approx.) | 0.0036149772 | -0.0132126575
  T2 eb_rate_t0safe - chamber_maj | mean of 2 test Congresses (normal approx.) | 0.3168226939 | 0.0202097248
  T2 eb_m20_t0s - eb_nomaj_m20_t0s | mean of 2 test Congresses (normal approx.) | 0.472189755 | 0.0826970392
  T1 sponsor_hist - party+pop | mean of 7 rolling test Congresses (normal approx.) | 0.0 | 0.0580721982
  T1 svd(val pick) - sponsor_hist | mean of 7 rolling test Congresses (normal approx.) | 0.0 | 0.020966248
  T1 sh_hl1 - sh_none+caucus | mean of 7 rolling test Congresses (normal approx.) | 0.0168852422 | 0.0141691311
  T1 svd_r32 - svd_r128 | mean of 7 rolling test Congresses (normal approx.) | 0.0 | -0.0132126575
  T2 eb_rate_t0safe - chamber_maj | mean of 7 rolling test Congresses (normal approx.) | 0.0046820743 | 0.0202097248
  T2 eb_m20_t0s - eb_nomaj_m20_t0s | mean of 7 rolling test Congresses (normal approx.) | 0.0330794599 | 0.0826970392
```

---
## [D-c129-md] markdown
### D10. Do older Congresses help (expanding) or hurt (sliding window)?

If the process drifts, old Congresses can teach stale patterns and a sliding window (only the last 3
Congresses) can match or beat the expanding window. Both windows are scored on the same test bills,
so the per-fold difference has a **paired** cluster-bootstrap CI. We run it for plain count
baselines, for their recency-weighted versions (half-life 1 Congress), and for the graph-structural
`rw2hop` (with and without recency weighting), and for the learned SVD embeddings (val-picked member,
same hyperparameters in both windows). If only the unweighted models prefer the sliding
window, the right conclusion is "weight recent edges", not "drop old Congresses". Folds share most
training Congresses, so the sign test is descriptive. Positive values = expanding window is better.

---
## [D-c130-txt] text
```
Expanding minus sliding (k=3) on the same test bills, S3/S2 folds:
                          mean_diff  fold_std  folds_expanding_better  folds_sliding_signif  folds_expanding_signif
series                                                                                                             
T1 sponsor_hist             -0.0124    0.0056                  0.0000                0.8571                  0.0000
T1 sh_hl1 (recency)         -0.0008    0.0011                  0.2857                0.1429                  0.0000
T1 rw2hop                   -0.0124    0.0053                  0.0000                0.8571                  0.0000
T1 rw_b0.3_hl1 (recency)    -0.0007    0.0015                  0.4286                0.1429                  0.0000
T1 popularity               -0.0202    0.0104                  0.0000                0.8571                  0.0000
T1 sponsor_hist t0=7        -0.0117    0.0055                  0.0000                0.8571                  0.0000
T2 eb_rate AUROC             0.0027    0.0057                  0.7143                0.0000                  0.1429
T2 eb_rate AP_lift           0.0299    0.1051                  0.5714                0.1429                  0.0000
T1 svd (val pick)           -0.0003    0.0023                  0.4286                0.0000                  0.0000
[PASS] D10: Older Congresses do not hurt a recency-weighted model: expanding >= sliding window (k=3) on test | T1 sh_hl1 diff=-0.0008; T1 svd -0.0003; T2 +0.0027 (threshold FAIL if, for a recency-weighted or val-tuned model (sh_hl1, rw_b0.3_hl1, val-picked svd, T2), mean diff < 0 and sliding is significantly better (paired cluster CI) in >= 50% of folds)
```

---
## [D-c130-o1] chart
```
title: Expanding window minus sliding window (k=3) test score per fold
subtitle: above 0 = using all earlier Congresses beats using only the last 3 (paired on the same test bills)
encoding: {'x': 'congress', 'y': 'diff', 'series': 'series', 'yfmt': '+.3f', 'ylabel': 'expanding − sliding', 'xlabel': 'test Congress'}
data (28 rows): congress | diff | series
  112 | -0.0008378085 | T1 sponsor_hist
  112 | 0.0005409354 | T1 sh_hl1 (recency)
  112 | 0.0012272047 | T2 eb_rate AUROC
  112 | 0.0028148568 | T1 svd (val pick)
  113 | -0.0112480429 | T1 sponsor_hist
  113 | -0.0022097449 | T1 sh_hl1 (recency)
  113 | 0.0046930698 | T2 eb_rate AUROC
  113 | -0.0003021952 | T1 svd (val pick)
  114 | -0.0121966057 | T1 sponsor_hist
  114 | -0.0007891472 | T1 sh_hl1 (recency)
  114 | 0.0110053566 | T2 eb_rate AUROC
  114 | 0.0015776472 | T1 svd (val pick)
  115 | -0.0180323238 | T1 sponsor_hist
  115 | 0.0003225392 | T1 sh_hl1 (recency)
  115 | -0.0034496195 | T2 eb_rate AUROC
  115 | 0.0015795876 | T1 svd (val pick)
  116 | -0.0150497759 | T1 sponsor_hist
  116 | -0.0006525856 | T1 sh_hl1 (recency)
  116 | 0.0084712518 | T2 eb_rate AUROC
  116 | -0.0026602895 | T1 svd (val pick)
  117 | -0.0129001793 | T1 sponsor_hist
  117 | -0.0023818275 | T1 sh_hl1 (recency)
  117 | 0.0015199472 | T2 eb_rate AUROC
  117 | -0.0031153118 | T1 svd (val pick)
  118 | -0.0163083915 | T1 sponsor_hist
  118 | -0.0003236713 | T1 sh_hl1 (recency)
  118 | -0.0043829819 | T2 eb_rate AUROC
  118 | -0.0019915308 | T1 svd (val pick)
```

---
## [D-c131-md] markdown
### D26. Regime-matched vs recency training weights

Section A finds the deployment Congress (119, unified Republican control) closest to the last
Congresses with the same House / Senate majorities, and the majority-sponsor effect depends on which
party controls the chamber. If that matters for prediction, "train on the most recent Congresses"
(recency weighting, sliding windows) is the wrong remedy and training should upweight Congresses with
the same regime as the target. The regime of a Congress is known when it starts (election results), so
this is leakage-free. On every S3 test fold we refit on 108..c-1 with four weightings and score the same
test bills: unweighted, recency (half-life 1 Congress), regime-matched (x2 for training Congresses with
Congress c's House and Senate majorities, from `majority_party()`), and regime x recency. Task 1 uses
`sponsor_hist`; Task 2 uses `eb_rate_t0safe` (prediction at introduction) and `eb_rate`. Positive
paired differences = regime weighting helps. Folds with no regime-matched training Congress give 0.

---
## [D-c132-txt] text
```
Regime per Congress (House/Senate majority): 108: RH/RS, 109: RH/RS, 110: DH/DS, 111: DH/DS, 112: RH/DS, 113: RH/DS, 114: RH/RS, 115: RH/RS, 116: DH/RS, 117: DH/DS, 118: RH/DS, 119: RH/RS
S3 folds with >= 1 regime-matched training Congress: 5/7. Paired test differences (folds better / worse = paired cluster CI excludes 0):
                                                 mean_diff  folds_better  folds_worse
model             comparison                                                         
T1 sponsor_hist   regime x2 - unweighted            0.0010             2            3
                  regime x2 - recency              -0.0156             0            5
                  regime x2 + recency - recency     0.0011             1            0
                  recency - unweighted              0.0209             7            0
T2 eb_rate_t0safe regime x2 - unweighted           -0.0001             0            1
                  regime x2 - recency              -0.0020             0            0
                  regime x2 + recency - recency    -0.0001             2            1
                  recency - unweighted              0.0096             2            0
[PASS] D26: Recency-only training is not beaten by regime-matched weighting (regime x2 + recency vs recency, S3 test folds) | T1 sponsor_hist +0.0011; T2 eb_rate_t0safe -0.0001; T2 eb_rate -0.0000 (threshold FAIL if, for T1 sponsor_hist or T2 eb_rate_t0safe, regime + recency beats recency alone on average AND is significantly better (paired cluster CI) in >= 2 folds and worse in none)
```

---
## [D-c132-o1] chart
```
title: Does upweighting same-regime Congresses beat recency weighting alone? (S3 test folds)
subtitle: paired difference on the same test bills; Task 1 Recall@20, Task 2 AUROC; 0 = no regime-matched training Congress
encoding: {'x': 'fold', 'y': 'diff', 'series': 'model', 'yfmt': '+.4f', 'ylabel': 'regime+recency − recency'}
data (21 rows): fold | diff | model
  c=112 RH/DS | 0.0 | T1 sponsor_hist
  c=112 RH/DS | 0.0 | T2 eb_rate_t0safe
  c=112 RH/DS | 0.0 | T2 eb_rate
  c=113 RH/DS | 0.0074601926 | T1 sponsor_hist
  c=113 RH/DS | -0.0004910637 | T2 eb_rate_t0safe
  c=113 RH/DS | -5.90014e-05 | T2 eb_rate
  c=114 RH/RS | 0.0005988678 | T1 sponsor_hist
  c=114 RH/RS | -0.0007580352 | T2 eb_rate_t0safe
  c=114 RH/RS | 0.0017926259 | T2 eb_rate
  c=115 RH/RS | 0.0009806777 | T1 sponsor_hist
  c=115 RH/RS | 0.0014502289 | T2 eb_rate_t0safe
  c=115 RH/RS | -0.0026402364 | T2 eb_rate
  c=116 DH/RS | 0.0 | T1 sponsor_hist
  c=116 DH/RS | 0.0 | T2 eb_rate_t0safe
  c=116 DH/RS | 0.0 | T2 eb_rate
  c=117 DH/DS | -0.0005861902 | T1 sponsor_hist
  c=117 DH/DS | 0.000155159 | T2 eb_rate_t0safe
  c=117 DH/DS | 0.000370985 | T2 eb_rate
  c=118 RH/DS | -0.0005707535 | T1 sponsor_hist
  c=118 RH/DS | -0.0013714907 | T2 eb_rate_t0safe
  c=118 RH/DS | 0.000466151 | T2 eb_rate
```

---
## [D-c133-md] markdown
### D16. Task 2 sensitivity to information not available at prediction time

CRS assigns the policy area some days after introduction, so a Task 2 model scored at introduction
often does not have it. `eb_rate_t0safe` hides the policy area for the share of bills that lack it
when fresh (measured on the in-progress 119th Congress). If the score drops noticeably, the Task 2
feature set must be built "as of" the prediction time, otherwise every split overstates performance.

---
## [D-c134-txt] text
```
eval_congress     112    113    114    115    116    117    118
baseline                                                       
eb_rate        0.7822 0.7539 0.7151 0.7421 0.6875 0.6924 0.6971
eb_rate_t0safe 0.6276 0.6684 0.6636 0.6585 0.5688 0.5598 0.6188
drop           0.1546 0.0855 0.0516 0.0837 0.1187 0.1325 0.0782
[INFO] D16: Task 2: AUROC drop when policy area is unavailable at introduction (eb_rate - eb_rate_t0safe, S3 test) | metric=0.1007 (threshold informational)
```

---
## [D-c135-md] markdown
### D18. Does the candidate pool definition matter?

Task 1 ranks the point-in-time pool (members in office in the month of introduction + t0). The
legacy pool (everyone who served in the chamber at any time that Congress) also contains departed and
not-yet-seated members, which inflates the negatives. We rerun the S3 test folds with the legacy pool
to see whether scores or the ordering of baselines depend on that choice.

---
## [D-c136-txt] text
```
Point-in-time minus legacy pool, mean Recall@20 over S3 test folds: popularity +0.0021, rw2hop +0.0021, sponsor_hist +0.0024
[INFO] D18: Task 1: scores and baseline ordering are insensitive to the candidate pool (point-in-time vs legacy roster, S3) | metric=0.002352 (threshold informational)
```

---
## [D-c137-md] markdown
### D19. Refit on train + validation, or keep the validation-selected model?

Test units "select on val, refit on train + val". A GNN chooses its epoch count by early stopping
on val, and the refit has no val left to stop on, so the refit must reuse a fixed budget (the epoch
or hyperparameter value chosen on val). For each S3 fold we compare the test score of the
val-selected family member **without refit** (fit on 108..c-2) and **with refit** (same member, fit
on 108..c-1). If refitting does not help, keeping the val-selected checkpoint (and its early-stopping
signal) loses nothing; if it helps, refit with the val-chosen budget.

---
## [D-c138-txt] text
```
Refit (fit on 108..c-1) minus no refit (fit on 108..c-2), same val-selected member, S3 test Congress c:
                           mean_gain  min_gain  max_gain  mean_hw  folds_refit_better  folds_refit_worse
family                                                                                                  
T1 rw2hop (+stream)           0.0257    0.0220    0.0308   0.0067                   7                  0
T1 rw2hop family              0.0648    0.0491    0.0762   0.0109                   7                  0
T1 sponsor_hist (+stream)     0.0244    0.0184    0.0281   0.0064                   7                  0
T1 sponsor_hist family        0.0589    0.0447    0.0731   0.0107                   7                  0
T1 svd (val pick)             0.0644    0.0503    0.0763   0.0115                   7                  0
T2 eb_rate family             0.0050   -0.0004    0.0122   0.0045                   3                  0
[PASS] D19: Refitting on train+val with hyperparameters fixed from val does not hurt the test score (S3 folds) | metric=0.005004 (threshold FAIL if refit is significantly worse (paired cluster CI) in >= 2 of 7 folds for any family)
```

---
## [D-c138-o1] chart
```
title: Test-score gain from refitting on train + validation (S3 folds)
subtitle: same hyperparameters (chosen on val); above 0 = refitting with the val Congress helps
encoding: {'x': 'fold', 'y': 'gain', 'series': 'family', 'yfmt': '+.4f', 'ylabel': 'refit − no refit'}
data (28 rows): fold | gain | family
  c=112 | 0.0554116634 | T1 sponsor_hist family
  c=112 | 0.0563611812 | T1 rw2hop family
  c=112 | 0.0030162971 | T2 eb_rate family
  c=112 | 0.0562320213 | T1 svd (val pick)
  c=113 | 0.0469844339 | T1 sponsor_hist family
  c=113 | 0.0761922971 | T1 rw2hop family
  c=113 | 0.0122197433 | T2 eb_rate family
  c=113 | 0.0762896307 | T1 svd (val pick)
  c=114 | 0.0628123246 | T1 sponsor_hist family
  c=114 | 0.0697070216 | T1 rw2hop family
  c=114 | 0.0003609252 | T2 eb_rate family
  c=114 | 0.0671492177 | T1 svd (val pick)
  c=115 | 0.0632821606 | T1 sponsor_hist family
  c=115 | 0.0582315907 | T1 rw2hop family
  c=115 | 0.0112740604 | T2 eb_rate family
  c=115 | 0.0606643622 | T1 svd (val pick)
  c=116 | 0.0446715641 | T1 sponsor_hist family
  c=116 | 0.0491305973 | T1 rw2hop family
  c=116 | 0.0039594458 | T2 eb_rate family
  c=116 | 0.0502809792 | T1 svd (val pick)
  c=117 | 0.0661379354 | T1 sponsor_hist family
  c=117 | 0.0696094974 | T1 rw2hop family
  c=117 | 0.0045511126 | T2 eb_rate family
  c=117 | 0.0665320437 | T1 svd (val pick)
  c=118 | 0.0730614209 | T1 sponsor_hist family
  c=118 | 0.0745730136 | T1 rw2hop family
  c=118 | -0.0003529835 | T2 eb_rate family
  c=118 | 0.0738251526 | T1 svd (val pick)
```

---
## [D-c139-md] markdown
### D27. A locked final test: develop on folds 112-117, report once on 118

In S3 / S2 every test Congress 112-117 is also the validation Congress of the next fold, and D9
recommends comparing models on the mean over rolling test folds. If the team compares GNN
architectures or feature sets on that mean and then reports it, the reported number was chosen on
test data. The fix is a **locked test**: develop, tune and compare on the rolling folds 112-117 only
(hyperparameters on each fold's val Congress, architectures on the mean over the 6 development folds
or on the mean of their val Congresses, D17), then freeze the pipeline (features, model, val-chosen
epochs), refit on 108-117 and score Congress 118 **once**. Congress 119 is the prospective test once
its windows close. This cell measures what reserving 118 costs: whether the selection checks still
pass on 6 folds, how much the miss risk (D9) and the level CI (D15) grow from k = 7 to k = 6, and
whether the development folds predict the locked fold (118 inside the development prediction interval).
In the chart each row is a relevant model pair: grey dots = development folds, band = 95% prediction
interval for a new Congress from those folds, orange dot = the locked fold 118 with its paired CI.

---
## [D-c140-txt] text
```
Selection on development folds 112-117 (k=6) vs all folds (k=7): mean regret, significant folds, verdict
                   regret_k7  sig_k7  ok_k7  regret_k6  sig_k6  ok_k6
item                                                                 
T1 sponsor_hist       0.0028       1   True     0.0032       1   True
T1 rw2hop             0.0002       0   True     0.0002       0   True
T2 eb_rate            0.0078       1   True     0.0017       0   True
T2 eb_rate_t0safe     0.0045       0   True     0.0053       0   True
T1 svd (learned)      0.0007       0   True     0.0008       0   True
Reported level: mean over test folds and its 95% CI half-width (k=7 vs k=6); 118 vs the dev prediction interval:
                          mean_k7  hw_mean_k7  mean_k6  hw_mean_k6  pi_dev  locked_118  in_pi
model                                                                                        
T1 sponsor_hist            0.2591      0.0065   0.2617      0.0050  0.0172      0.2439  False
T1 svd (val pick)          0.2801      0.0071   0.2822      0.0068  0.0235      0.2675   True
T2 eb_rate_t0safe          0.6236      0.0330   0.6244      0.0390  0.1354      0.6188   True
T2 eb_rate (policy area)   0.7243      0.0266   0.7289      0.0297  0.1030      0.6971   True
Paired sizing, relevant pairs: max miss risk k=7 0.033 -> k=6 0.115; median CI half-width of the mean difference 0.0076 -> 0.0080; 118 paired difference inside the dev prediction interval for 8/9 relevant model / protocol pairs
[PASS] D27: Final-test isolation: developing on rolling folds 112-117 and reserving 118 as a locked test keeps selection and paired sizing adequate | metric=0.115 (threshold PASS if every selection check (D11/D12/D12b/D21/D23 rule, regret tol MDE/2) still passes on the 6 dev folds AND the max miss risk over relevant pairs at k=6 is <= 0.20)
D_FINAL_PROTOCOL: develop: S3(+stream) rolling folds 112-117: train 108..c-2 -> val c-1 -> refit 108..c-1 -> test c | locked_test: 118 | prospective_test: 119
```

---
## [D-c140-o1] chart
```
title: Do the development folds 112-117 predict the locked test Congress 118?
subtitle: paired test difference per relevant model / protocol pair (Task 1 Recall@20, Task 2 AUROC at introduction)
encoding: {'xlabel': 'paired difference (MDE = 0.01)'}
data (72 rows): pair | kind | value | lo | hi | congress
  T1 sponsor_hist - party+pop | band | 0.0566214094 | 0.0239392396 | 0.0893035792 | 0
  T1 sponsor_hist - party+pop | dev | 0.0408189565 | 0.0408189565 | 0.0408189565 | 112
  T1 sponsor_hist - party+pop | dev | 0.0618677074 | 0.0618677074 | 0.0618677074 | 113
  T1 sponsor_hist - party+pop | dev | 0.0486459523 | 0.0486459523 | 0.0486459523 | 114
  T1 sponsor_hist - party+pop | dev | 0.0669730512 | 0.0669730512 | 0.0669730512 | 115
  T1 sponsor_hist - party+pop | dev | 0.0504328398 | 0.0504328398 | 0.0504328398 | 116
  T1 sponsor_hist - party+pop | dev | 0.0709899492 | 0.0709899492 | 0.0709899492 | 117
  T1 sponsor_hist - party+pop | lock | 0.066776931 | 0.0530858294 | 0.0792867094 | 118
  T1 svd(val pick) - sponsor_hist | band | 0.0205254067 | 0.0084187131 | 0.0326321003 | 0
  T1 svd(val pick) - sponsor_hist | dev | 0.0123430507 | 0.0123430507 | 0.0123430507 | 112
  T1 svd(val pick) - sponsor_hist | dev | 0.0204585846 | 0.0204585846 | 0.0204585846 | 113
  T1 svd(val pick) - sponsor_hist | dev | 0.025468692 | 0.025468692 | 0.025468692 | 114
  T1 svd(val pick) - sponsor_hist | dev | 0.0217069114 | 0.0217069114 | 0.0217069114 | 115
  T1 svd(val pick) - sponsor_hist | dev | 0.0214900379 | 0.0214900379 | 0.0214900379 | 116
  T1 svd(val pick) - sponsor_hist | dev | 0.0216851634 | 0.0216851634 | 0.0216851634 | 117
  T1 svd(val pick) - sponsor_hist | lock | 0.0236112957 | 0.0182147193 | 0.0294181118 | 118
  T1 sh_hl1 - sh_none+caucus | band | 0.013095657 | -0.0134620702 | 0.0396533842 | 0
  T1 sh_hl1 - sh_none+caucus | dev | -0.0050623931 | -0.0050623931 | -0.0050623931 | 112
  T1 sh_hl1 - sh_none+caucus | dev | 0.0142786506 | 0.0142786506 | 0.0142786506 | 113
  T1 sh_hl1 - sh_none+caucus | dev | 0.0161011827 | 0.0161011827 | 0.0161011827 | 114
  T1 sh_hl1 - sh_none+caucus | dev | 0.0232176899 | 0.0232176899 | 0.0232176899 | 115
  T1 sh_hl1 - sh_none+caucus | dev | 0.0169486328 | 0.0169486328 | 0.0169486328 | 116
  T1 sh_hl1 - sh_none+caucus | dev | 0.0130901789 | 0.0130901789 | 0.0130901789 | 117
  T1 sh_hl1 - sh_none+caucus | lock | 0.0206099761 | 0.0141831788 | 0.0269463278 | 118
  T1 svd_r32 - svd_r128 | band | -0.0130689793 | -0.0252467247 | -0.0008912339 | 0
  T1 svd_r32 - svd_r128 | dev | -0.0086362401 | -0.0086362401 | -0.0086362401 | 112
  T1 svd_r32 - svd_r128 | dev | -0.0099137176 | -0.0099137176 | -0.0099137176 | 113
  T1 svd_r32 - svd_r128 | dev | -0.0154613361 | -0.0154613361 | -0.0154613361 | 114
  T1 svd_r32 - svd_r128 | dev | -0.0200317694 | -0.0200317694 | -0.0200317694 | 115
  T1 svd_r32 - svd_r128 | dev | -0.0145104869 | -0.0145104869 | -0.0145104869 | 116
  T1 svd_r32 - svd_r128 | dev | -0.009860326 | -0.009860326 | -0.009860326 | 117
  T1 svd_r32 - svd_r128 | lock | -0.0140747264 | -0.0211006267 | -0.0077260574 | 118
  T1 expanding - sliding (sponsor_hist) | band | -0.0117107894 | -0.0279575986 | 0.0045360199 | 0
  T1 expanding - sliding (sponsor_hist) | dev | -0.0008378085 | -0.0008378085 | -0.0008378085 | 112
  T1 expanding - sliding (sponsor_hist) | dev | -0.0112480429 | -0.0112480429 | -0.0112480429 | 113
  T1 expanding - sliding (sponsor_hist) | dev | -0.0121966057 | -0.0121966057 | -0.0121966057 | 114
  T1 expanding - sliding (sponsor_hist) | dev | -0.0180323238 | -0.0180323238 | -0.0180323238 | 115
  T1 expanding - sliding (sponsor_hist) | dev | -0.0150497759 | -0.0150497759 | -0.0150497759 | 116
  T1 expanding - sliding (sponsor_hist) | dev | -0.0129001793 | -0.0129001793 | -0.0129001793 | 117
  T1 expanding - sliding (sponsor_hist) | lock | -0.0163083915 | -0.02165982 | -0.0108145518 | 118
  T1 stream - plain (sponsor_hist) | band | 0.0342595893 | 0.015975806 | 0.0525433725 | 0
  T1 stream - plain (sponsor_hist) | dev | 0.0364170777 | 0.0364170777 | 0.0364170777 | 112
  T1 stream - plain (sponsor_hist) | dev | 0.0319813726 | 0.0319813726 | 0.0319813726 | 113
  T1 stream - plain (sponsor_hist) | dev | 0.0315561539 | 0.0315561539 | 0.0315561539 | 114
  T1 stream - plain (sponsor_hist) | dev | 0.0257654707 | 0.0257654707 | 0.0257654707 | 115
  T1 stream - plain (sponsor_hist) | dev | 0.0342857395 | 0.0342857395 | 0.0342857395 | 116
  T1 stream - plain (sponsor_hist) | dev | 0.0455517213 | 0.0455517213 | 0.0455517213 | 117
  T1 stream - plain (sponsor_hist) | lock | 0.053210456 | 0.0432019422 | 0.0655312741 | 118
  T2 eb_rate_t0safe - chamber_maj | band | 0.0228550186 | -0.0057752342 | 0.0514852714 | 0
  T2 eb_rate_t0safe - chamber_maj | dev | 0.0372836798 | 0.0372836798 | 0.0372836798 | 112
  T2 eb_rate_t0safe - chamber_maj | dev | 0.0128243541 | 0.0128243541 | 0.0128243541 | 113
  T2 eb_rate_t0safe - chamber_maj | dev | 0.0217352655 | 0.0217352655 | 0.0217352655 | 114
  T2 eb_rate_t0safe - chamber_maj | dev | 0.0165053878 | 0.0165053878 | 0.0165053878 | 115
  T2 eb_rate_t0safe - chamber_maj | dev | 0.033785994 | 0.033785994 | 0.033785994 | 116
  T2 eb_rate_t0safe - chamber_maj | dev | 0.0149954303 | 0.0149954303 | 0.0149954303 | 117
  T2 eb_rate_t0safe - chamber_maj | lock | 0.0043379623 | -0.0085275859 | 0.0153079717 | 118
  T2 eb_m20_t0s - eb_nomaj_m20_t0s | band | 0.081290023 | -0.0935923373 | 0.2561723833 | 0
  T2 eb_m20_t0s - eb_nomaj_m20_t0s | dev | 0.097275516 | 0.097275516 | 0.097275516 | 112
  T2 eb_m20_t0s - eb_nomaj_m20_t0s | dev | 0.1303459264 | 0.1303459264 | 0.1303459264 | 113
  T2 eb_m20_t0s - eb_nomaj_m20_t0s | dev | 0.120848368 | 0.120848368 | 0.120848368 | 114
  ... (12 more rows)
```

---
## [D-c141-md] markdown
### Section D summary

One line per check. FAIL does not automatically rule out a split: read the detail for the
implication (e.g. report the mean over rolling folds instead of a single test Congress).

---
## [D-c142-txt] text
```
FAIL (3): D2b [fails for S1b,S2,S3,S4,S1b+stream,S3+stream], D7 [fails for S5], D9 [fails for S1a,S1b,S1a+stream,S1b+stream,S1pool]
PASS (15): D1 [fails for S1pool], D20 [fails for S1pool], D11, D12, D12b, D21, D17, D22, D23, D25, D14, D10, D26, D19, D27
INFO (12): D2, D3, D4, D5, D6, D24, D8, D13, D9b, D15, D16, D18

SPLIT_DEFS + SPLIT_VARIANTS (setup): ['S1a', 'S1b', 'S2', 'S3', 'S4', 'S5', 'S1a+stream', 'S1b+stream', 'S3+stream', 'S1pool'] | SPLIT_RESULTS: 2,160 rows (3 tasks x 8 baselines; CIs = cluster bootstrap by sponsor, ci_*_bill = bill bootstrap) | D_FAMILY_RESULTS: 1,512 rows | D_PAIRED: paired-difference sizing | D_LEARNED: SVD selection per fold | D_STALENESS: recall by legislator status | D_REGIME: weighting comparison | D_LOCKED: selection / level on dev folds vs all folds | D_FINAL_PROTOCOL: locked-test protocol
```