# Digest: Decision: which split to use (section key decision)


---
## [decision-c143-md] markdown
## Decision: which split to use

Sections A-D each tested one family of conditions a split relies on and recorded every result in the
shared check registry (`CHECKS`) and, for D, in the backtest table `SPLIT_RESULTS`. This section reads
only those two objects: nothing below is hardcoded, so if you change a threshold or rerun on new data,
the tables, scores and the recommendation update with it.

**How checks are used.** Every check has a *kind*. **Validity** checks ask whether a split gives an
honest, usable estimate (leakage, val predicts test, enough test Congresses, censoring): a FAIL is a
reason to prefer a different split, and only these rank the splits. **Drift** checks ask whether later
Congresses differ from earlier ones: every forward split fails them by design (that is the split doing
its job), so a drift FAIL selects a *remedy* (recalibration, recency weighting, per-Congress reporting),
not a different split. **Requirements** are construction rules every split must follow (time-filter
edges, rebuild features as of t0); they never count as split failures. A few split-dependent checks
have no kind ("unclassified"); they are not used for the main ranking, but the recommendation is
re-run with them included as a robustness test.

---
## [decision-c144-txt] text
```
102 checks recorded; 100 remain after de-duplicating shared quantities ('dup' = verdict owned by another section's check of the same quantity; not counted twice)

--- A. targets & covariates (26 checks) ---
  id status  kind dup    metric                          threshold                                                                                              condition / detail
  A1   PASS drift        -2.933 outside 95% t prediction interval~ Advancement base rate of each candidate within the spread~ | VERDICT RULE: FAIL iff >= 1 test survives a Holm ~
 A1d   PASS drift        -1.594 outside 95% t prediction interval~ Number of reported bills (numerator of the Task-2 rate) w~ | VERDICT RULE: FAIL iff >= 1 test survives a Holm ~
 A1b   FAIL drift          2.74 outside 95% t prediction interval~ Share of chamber-passed bills that the Task-2 label misse~ | VERDICT RULE: FAIL iff >= 1 test survives a Holm ~
 A1c   PASS drift        -2.122 outside 95% t prediction interval~ Alternative advancement label (reported OR passed origin ~ | VERDICT RULE: FAIL iff >= 1 test survives a Holm ~
  A2   FAIL drift        -3.645 outside 95% t prediction interval~ Per-chamber advancement rate of each candidate within the~ | VERDICT RULE: FAIL iff >= 1 test survives a Holm ~
  A3   FAIL drift        -5.204 outside 95% t prediction interval~ Cosponsor positive density (t0=0 and t0=7) of each candid~ | VERDICT RULE: FAIL iff >= 1 test survives a Holm ~
 A3b   PASS drift        -1.124 outside 95% t prediction interval~ Share of bills with at least one new cosponsor (t0=0 and ~ | VERDICT RULE: FAIL iff >= 1 test survives a Holm ~
  A4   FAIL drift          4.03 outside 95% t prediction interval~ Original-cosponsor share of each candidate within train s~ | VERDICT RULE: FAIL iff >= 1 test survives a Holm ~
  A5   PASS drift        -2.791 outside 95% t prediction interval~ Majority-sponsor odds ratio (both chambers pooled) of eac~ | VERDICT RULE: FAIL iff >= 1 test survives a Holm ~
 A5b   PASS drift         1.699 |z_regime| > 1.96 (random-effects~ Each candidate chamber-Congress's majority-sponsor OR mat~ | VERDICT RULE: FAIL iff a candidate cell survives ~
 A5c   INFO drift        0.6992 |z| > 1.96 => model needs a regim~ Majority-sponsor effect differs by controlling party with~ | log OR difference D-control minus R-control per c~
  A6   PASS drift         1.557 outside 95% t prediction interval~ Same-party share of new cosponsors of each candidate with~ | VERDICT RULE: FAIL iff >= 1 test survives a Holm ~
  A7   PASS drift          1.58 outside 95% t prediction interval~ Marginal covariate drift (PSI of 6 bill features) of each~ | VERDICT RULE: FAIL iff >= 1 test survives a Holm ~
  A8   FAIL drift           4.5 outside 95% t prediction interval~ Law-capable bill volume of each candidate within train sp~ | VERDICT RULE: FAIL iff >= 1 test survives a Holm ~
  A9   PASS drift        0.8612 outside 95% t prediction interval~ Each candidate's mean policy-area JSD to train within the~ | VERDICT RULE: FAIL iff >= 1 test survives a Holm ~
 A10   INFO drift        0.6822         rho > 0.5 => gradual drift Drift is gradual in time (Spearman of Congress gap vs pol~ | rho=0.68. Strongly positive rho means distant Con~
 A11   PASS drift        0.4691 outside 95% t prediction interval~ Joint covariate shift (domain-classifier AUROC vs train p~ | VERDICT RULE: FAIL iff >= 1 test survives a Holm ~
 A12   PASS drift        -2.496 outside 95% t prediction interval~ Fitted Task-2 model transfers forward: calibration slope ~ | VERDICT RULE: FAIL iff >= 1 test survives a Holm ~
 A13   FAIL drift        -3.911 outside 95% t prediction interval~ Fitted Task-2 model AUROC (forward-chained) on each candi~ | VERDICT RULE: FAIL iff >= 1 test survives a Holm ~
 A18   PASS drift        -2.791 outside 95% t prediction interval~ Task-1 pair-feature effects (logistic coefficients, t0=7)~ | VERDICT RULE: FAIL iff >= 1 test survives a Holm ~
A18b   PASS drift        -1.739 outside 95% t prediction interval~ Task-1 pair model fitted on earlier Congresses: AUROC and~ | VERDICT RULE: FAIL iff >= 1 test survives a Holm ~
 A14   INFO drift        0.6429       > 0.5 favours sliding window Sliding (S2, k=3) training pools are closer to their test~ | S2 closer than S3 in {'domain_auc': 3, 'adv_logra~
 A15   INFO drift        0.5477 lower domain AUROC vs 119 = bette~ Candidate test Congresses resemble the deployment target ~ | Domain AUROC of 119 vs each complete Congress (t0~
A15b   INFO drift     -0.003493 no-party same-regime coefficient ~ Similarity between Congresses follows party regime after ~ | Pairs regression AUROC ~ 1 + gap + [gap=1] + same~
 A17   FAIL drift             5 FAIL if Poisson-binomial p < 0.05~ Each candidate's number of failing A check clusters (Holm~ | 116: 4/6 clusters failing (Covariates, Task-2 bas~
 A16   INFO drift        0.6667 share of failing level parts that~ Candidate drift is a predictable trend rather than a stru~ | 14 break, 0 consistent with trend, 7 undetermined~

--- B. graph & nodes (16 checks) ---
 id   status        kind dup  metric                          threshold                                                                                              condition / detail
 B1     INFO    unclass.      0.3145   info: model/protocol requirement Share of val/test Task-1 positive edges from legislators ~ | Max over by-Congress units of positive edges whos~
B1b     PASS    validity      0.0071 < 0.02 of positives (t0=0, every ~ With eval-Congress history up to intro + t0 (+stream), fe~ | +stream protocol: max share of positives whose co~
B1c     PASS    unclass.      0.0145 cold in train AND < 5 in-Congress~ Positives from train-cold legislators have a usable in-Co~ | +stream protocol, t0=0: positives cold in train w~
 B2     FAIL       drift      0.2172 <= 0.144 (outside 95% t predictio~ Few val/test bills have a sponsor with no sponsorship his~ | Max share of law-capable bills whose sponsor neve~
 B3     PASS       drift           2 per split: failures (PSI > fixed ~ Val/test bill and legislator degree distributions are no ~ | Fixed reference (pool-size independent): bounds b~
 B4     PASS    unclass.      0.0128                             < 0.05 Val/test committee-referral edges point to committees see~ | Max share of referral edges with a committee code~
B5a     PASS    unclass.      0.0161                             < 0.05 Val/test bill-subject links use subject terms seen in tra~ | Max unseen-subject link share: S1a 0.02%, S1b 0.0~
B5b INFO/REQ requirement dup  0.0096 info: feature-exclusion requireme~ Requirement: do not use legislative subjects as node / ed~ | At t0 only 1.0% of bills aged 0-30 d (in-progress~
B6a     INFO    unclass.     0.00014 info: uninformative (source store~ Related-bill links from complete Congresses to later Cong~ | 20 links from 108-118 point to a later Congress a~
B6b FAIL/REQ requirement      0.4266 < 0.05 (requirement: filter if it~ Within-Congress related links are time-safe without filte~ | On average 42.7% of same-Congress related links p~
 B7 INFO/REQ requirement           1                  info: must filter Graph builder must keep only referral committee edges kno~ | 100.0% of advancement positives have a non-referr~
 B8     PASS       drift           2 per split: |t| > 2.36 failures (f~ Val/test graph structure (density, mean degree, homophily~ | Failures / tests (p excess): S1a 1/12 (p=0.46), S~
 B9     PASS    unclass.      0.1439 per split: gaps above the 110-115~ Each split's val unit has a new-pair share similar to its~ | |val - test| new-pair share, max per split (fails~
B10 INFO/REQ requirement      0.5197         info: must time-mask edges Share of the eval Congress's final cosponsor graph that i~ | For eval bills of 116-118 at t0=0, on average 52.~
B12 FAIL/REQ requirement     -0.0198 FAIL (= the rule is binding) if t~ Training inputs match eval inputs: train examples from as~ | Val R@20 (snapshot inputs at eval): snapshot-trai~
B11     PASS       drift       0.014 per unit evaluating 116-118: swit~ Chamber / party switchers (one node id, train neighbourho~ | Max switcher share of Task-1 positives (t0=0) per~

--- C. temporal & leakage (30 checks) ---
  id   status        kind dup                       metric                          threshold                                                                                              condition / detail
  C1     PASS    unclass.                          0.06177 outside 95% t prediction interval~ Introduction-day distribution (KS vs 108-115) of every ev~ | ref 108-115 PI [0.004, 0.119] (outside 95% t pred~
  C2     PASS    unclass.                            1.959 outside 95% t prediction interval~ Session-1 share and end-of-Congress (last 90 d) share of ~ | session-1 share: ref 108-115 PI [0.543, 0.751] (o~
  C3     FAIL       drift                          0.09342 outside 95% t prediction interval~ Timing of non-original cosponsorships (lag > 0) in eval C~ | ref 108-115 PI [0.012, 0.046] (outside 95% t pred~
  C4     INFO       drift dup                       0.4746         INFO (A3 owns the verdict) Task-1 positive rate per eligible legislator, eval vs tra~ | max relative diff vs 108-115 mean; t0=0: {116: 0.~
 C23     PASS    unclass.                                0 metric = number of breaks failing~ No significant structural break (beyond a linear trend) i~ | breaks: orig_share30/House: raw 2021-01 (p=0.000)~
C23b     FAIL    unclass.                                1 by_split False if binomial P(>= o~ Share of test units scored on a post-break regime unseen ~ | significant detrended breaks (p < 0.0125): [('ori~
  C5     PASS    validity                          0.03864          max censored share < 0.05 Task-1 censoring (t0=7) drops < 5% of law-capable bills i~ | censored share t0=7 ranges 0.002-0.039; eval Cong~
  C6     PASS    unclass.                           0.1725 outside 95% t prediction interval~ Task-2 censored share (180 d) of eval Congresses within t~ | ref 108-115 PI [0.082, 0.173] (outside 95% t pred~
 C6b     FAIL    unclass.                           0.8078 task-2 censored share of S4 test ~ Within-Congress split S4: its test slice (last 20% of int~ | S4 test slices lose 0.808 of their bills to task-~
  C7     INFO    unclass.               4 regimes, 7 flips                                nan Party-control regimes and flips across 108-118 (informati~ | regimes: {108: 'RH/RS', 109: 'RH/RS', 110: 'DH/DS~
  C8     PASS       drift                                1 per-chamber coverage = 1.0 for ev~ Each eval Congress's House majority and Senate majority b~ | per-chamber coverage per split {'S1a': 1.0, 'S1b'~
 C8b     INFO    unclass.                           0.2455 FAIL only if any exact permutatio~ Joint House x Senate regime (divided vs unified) changes ~ | divided-unified differences (108-118): House rate~
  C9     PASS    unclass.                                1         all ties confirmed by data Regime labels are robust: every tie-broken chamber is con~ | ties: [(117, 'Senat', 50, 50, 'D')]; tie-break ev~
 C10     PASS    unclass.                           0.9987 >= 0.995 agreement and >= latest-~ Per-Congress party labels (members.party_code_asof) match~ | party_code_asof agrees 99.87% vs latest party_cod~
 C11 INFO/REQ requirement                           0.4846 INFO (requirement for every split) Requirement: rebuild bill-level count/outcome fields as o~ | headline: the snapshot n_cosponsors differs from ~
 C12     PASS    unclass.                           0.9762 min share referred by day 0 >= 0.~ Committee referral is known at t0=0 for >= 95% of law-cap~ | min share by day 0 0.976, by day 7 0.979; bills w~
 C13 INFO/REQ requirement                          0.03503 requirement (a t0 feature must ex~ Requirement: do not use CRS summary text as a t0 feature ~ | share with a summary last updated within 7 days: ~
 C24     PASS    unclass.                            1.211 consistency/stale shares = 0 in e~ Labels of the most recent Congresses (cosponsor events, c~ | max consistency/stale share 0.0000 (n_cosponsors ~
 C14     INFO    unclass.                            0.292 INFO unless C24 fails (then: 118 ~ Recent-edit share of 118 records (INFO; becomes a split v~ | share of bills updated within 365 days of the sna~
C15a INFO/REQ requirement                           0.3722 requirement (INFO; subjects are n~ Requirement: legislative subjects are not a t0 feature (c~ | subjects coverage: ref 108-115 PI [0.956, 1.020] ~
C15b     PASS       drift                           0.9904 outside 95% t prediction interval~ Coverage of t0-usable annotations (referral; policy area ~ | referral: ref 108-115 PI [0.978, 0.993] (outside ~
C15c     INFO    unclass.                           0.5776                               INFO CRS-summary coverage of eval Congresses (INFO: summaries ~ | summary: ref 108-115 PI [1.000, 1.000] (outside 9~
 C16     INFO    unclass.     max coverage gap 0.426 (sum~                                nan Annotations exist within 30 days of introduction (119 bil~ | coverage for bills aged 0-30 d: {'policy_area': 0~
 C18 INFO/REQ requirement                           0.1429 requirement (INFO; the rule binds~ Requirement: mask policy area at t0 <= 7 d (present for <~ | shared proxy PA_MISSING_AT_T0: present for 14.3% ~
 C25 INFO/REQ requirement                               30 INFO (design decision; metric = s~ Requirement: fix task 2's prediction time t0 and use only~ | availability by t0 [0, 7, 30, 60]: cosponsors / a~
 C17 INFO/REQ requirement                          0.09655 INFO (construction requirement fo~ Requirement: eligible pool = members in office at introdu~ | legacy pool share out of office: max 0.097, House~
 C22     INFO    unclass.                          0.00971 INFO (requirement for every split~ Point-in-time pool vs full-Congress roster: effect on tas~ | Recall@20 gain vs full roster, pit+delegates / pi~
 C19     PASS    unclass.                                0 < 1% of eval bills with a later t~ Temporal splits (S1-S4) put no twin introduced after the ~ | share of eval bills with a later twin in train (s~
 C20     FAIL    unclass.                           0.4519                             < 0.01 Random split S5: eval bills with a later twin (same or la~ | S5: same-Congress twin introduced later in train ~
 C21     INFO    unclass.                           0.3585                               INFO Same-Congress twin signal a by-Congress split forgoes unl~ | 35.9% of by-Congress eval bills have a same-Congr~

--- D. backtest (30 checks) ---
  id status     kind dup                       metric                          threshold                                                                                              condition / detail
  D1   PASS validity                          0.03988 FAIL if gaps exceed cluster-boots~ Task 1: validation Congress predicts the test score level~ | 'sponsor_hist', S3: mean |rel gap| 0.040 vs 0.038~
  D2   INFO    drift                          0.03042 informational: level shifts do no~ Task 2: validation AUROC level vs test AUROC level (mean ~ | 'eb_rate' over S3 folds: 43% of folds have |gap| ~
 D2b   FAIL    drift                          0.06209 FAIL if gaps exceed cluster-boots~ Task 2 at introduction (eb_rate_t0safe): val Congress pre~ | 'eb_rate_t0safe', S3: mean |rel gap| 0.062 vs 0.0~
 D20   PASS validity                          0.03709 FAIL if gaps exceed cluster-boots~ Task 1, graph-structural rw2hop: validation Congress pred~ | 'rw2hop', S3: mean |rel gap| 0.037 vs 0.038 expec~
  D3   INFO validity                          0.03396 informational (CV <= 0.10 = level~ Task 1: test score level across rolling folds (CV of best~ | 'sponsor_hist' Recall@20 ranges 0.244-0.272 over ~
  D4   INFO validity                          0.03593 informational (std > 0.03 and > 1~ Task 2: test AUROC level across rolling folds (std of bes~ | 'eb_rate' AUROC ranges 0.688-0.782; between-fold ~
  D5   INFO unclass.                           0.5714 informational (see D11/D21 for cl~ Task 1: main baselines keep their order from validation t~ | Kendall tau(val, test) of the 4 main baselines av~
  D6   INFO unclass.                                1 informational (candidates far apa~ Task 2: main baselines keep their order from validation t~ | prior < chamber_maj < eb_rate by wide margins; se~
 D11   PASS validity                         0.002763 FAIL if mean regret > D_TOL = MDE~ Task 1: validation picks the test-best among close count-~ | Family ['sh_none', 'sh_hl4', 'sh_hl2', 'sh_hl1', ~
 D12   PASS validity                         0.007766 FAIL if mean regret > D_TOL = MDE~ Task 2: validation picks the test-best among close candid~ | Family ['eb_m5', 'eb_m20', 'eb_m80', 'eb_nomaj_m2~
D12b   PASS validity                         0.004518 FAIL if mean regret > D_TOL = MDE~ Task 2 at introduction (t0-safe family, policy area hidde~ | Family ['eb_m5_t0s', 'eb_m20_t0s', 'eb_m80_t0s', ~
 D21   PASS validity                        0.0002027 FAIL if mean regret > D_TOL = MDE~ Task 1, graph-structural rw2hop family: validation picks ~ | Family ['rw_b0.1', 'rw_b0.3', 'rw_b1', 'rw_b0.1_h~
 D17   PASS validity                         0.001879 per family: FAIL if mean regret >~ Selecting on the mean of the last 3 rolling val folds pic~ | Regret with 3-fold val selection (single val in b~
 D22   PASS validity                          0.03721 FAIL if gaps exceed cluster-boots~ Task 1, learned SVD embeddings (val-selected): validation~ | 'svd_pick', S3: mean |rel gap| 0.037 vs 0.038 exp~
 D23   PASS validity                        0.0006684 FAIL if mean regret > D_TOL (MDE/~ Task 1, learned SVD embeddings: validation picks the test~ | Hit rate 57%, max regret 0.0032, significant in 0~
 D24   INFO unclass.                         3.33e-05 informational (cold start for a t~ Task 1: Recall@20 for new vs returning legislators, learn~ | New legislators: 15.7% of test positives; Recall@~
 D25   PASS    drift                         0.002947 FAIL if, for the val-picked or th~ Task 1: learned embeddings do not go stale for returning ~ | DiD pooled +0.0029 [-0.0002, +0.0065]; by change ~
  D7   FAIL validity                          0.07453 FAIL if mean optimism > D_MDE (0.~ Random bill split does not inflate scores relative to the~ | Mean S5-minus-S3 Recall@20 for 'sponsor_hist' ove~
  D8   INFO unclass.                          0.05721                      informational Within-Congress split (S4) gives scores comparable to by-~ | Task 1 'sponsor_hist' S4-S3 Recall@20 diff by Con~
 D13   INFO unclass.                          0.03697                      informational Streaming eval-Congress history changes test scores (mean~ | Task 1 'sponsor_hist' gain t0=0 +0.037 (range +0.~
 D14   PASS validity                          0.02937 FAIL if gaps exceed cluster-boots~ Task 1: under the streaming protocol the val Congress sti~ | 'sponsor_hist', S3+stream: mean |rel gap| 0.029 v~
  D9   FAIL validity                            0.429 per split: max over relevant pair~ Test-set size: a single test Congress confirms the direct~ | Relevant decision pairs: 6 of 11 (MDE T1 0.0100, ~
 D9b   INFO    drift                            15.07 informational: > 1 means the next~ Deployment risk: how far a relevant model difference can ~ | pi_next / MDE per relevant pair (all 7 folds; rec~
 D15   INFO unclass.           T1 1, T2 13 Congresses informational: report test scores~ Absolute score level: test Congresses needed for a precis~ | Between-fold std of the absolute score (S3): T1 0~
 D10   PASS validity     T1 sh_hl1 diff=-0.0008; T1 ~ FAIL if, for a recency-weighted o~ Older Congresses do not hurt a recency-weighted model: ex~ | Mean diff (folds sliding signif. better / expandi~
 D26   PASS    drift     T1 sponsor_hist +0.0011; T2~ FAIL if, for T1 sponsor_hist or T~ Recency-only training is not beaten by regime-matched wei~ | Mean paired diff (folds better/worse): T1 sponsor~
 D16   INFO unclass.                           0.1007                      informational Task 2: AUROC drop when policy area is unavailable at int~ | Policy area hidden for 86% of evaluation bills (s~
 D18   INFO unclass.                         0.002352                      informational Task 1: scores and baseline ordering are insensitive to t~ | Mean Recall@20 point-in-time minus legacy: popula~
 D19   PASS validity                         0.005004 FAIL if refit is significantly wo~ Refitting on train+val with hyperparameters fixed from va~ | Mean gain (folds refit signif. better / worse): T~
 D27   PASS validity                            0.115 PASS if every selection check (D1~ Final-test isolation: developing on rolling folds 112-117~ | Selection with k=6 (k=7): T1 sponsor_hist regret ~

De-duplicated, non-requirement checks by section x kind x status (these are the counts the decision uses):
kind    drift           unclassified           validity          
status   FAIL INFO PASS         FAIL INFO PASS     FAIL INFO PASS
section                                                          
A           7    6   13            0    0    0        0    0    0
B           1    0    3            0    2    4        0    0    1
C           1    0    2            3    7    9        0    0    1
D           1    2    2            0    8    0        2    2   13
Construction requirements (apply to every split, not counted as failures): B5b, B6b, B7, B10, B12, C11, C13, C15a, C18, C25, C17
```

---
## [decision-c145-o0] chart
```
title: 
subtitle: 
data (25 rows): sec | grp | st | n
  A. targets & covariates | drift | FAIL | 7
  A. targets & covariates | drift | INFO | 6
  A. targets & covariates | drift | PASS | 13
  B. graph & nodes | drift | FAIL | 1
  B. graph & nodes | drift | PASS | 3
  B. graph & nodes | requirement | FAIL | 2
  B. graph & nodes | requirement | INFO | 3
  B. graph & nodes | unclassified | INFO | 2
  B. graph & nodes | unclassified | PASS | 4
  B. graph & nodes | validity | PASS | 1
  C. temporal & leakage | drift | FAIL | 1
  C. temporal & leakage | drift | INFO | 1
  C. temporal & leakage | drift | PASS | 2
  C. temporal & leakage | requirement | INFO | 6
  C. temporal & leakage | unclassified | FAIL | 3
  C. temporal & leakage | unclassified | INFO | 7
  C. temporal & leakage | unclassified | PASS | 9
  C. temporal & leakage | validity | PASS | 1
  D. backtest | drift | FAIL | 1
  D. backtest | drift | INFO | 2
  D. backtest | drift | PASS | 2
  D. backtest | unclassified | INFO | 8
  D. backtest | validity | FAIL | 2
  D. backtest | validity | INFO | 2
  D. backtest | validity | PASS | 13
```

---
## [decision-c146-md] markdown
### Split scorecard: what supports and what undermines each candidate

For every candidate split (the five base splits plus D's protocol variants) the scorecard combines
(1) the **validity** checks that judged it (passes support the split, fails undermine it), (2) the
unclassified split-dependent checks (shown separately), (3) how many drift checks it exposes, and (4) the
backtest in `SPLIT_RESULTS`: for each task the baseline that **validation** would pick (best mean val
score, as a modeler would choose), its mean val and test score, the mean |val - test| gap per fold
relative to the test score's own 95% CI half-width (ratio < 1 = the gap is within sampling noise),
the fold-to-fold standard deviation of the test score, and the number of test bills. Task 2 uses only
baselines available at introduction (no policy area, per requirement C18 and the D16 drop).
The `SPLIT_RESULTS` baselines are all **unweighted** (every training Congress counts the same). Section D
also scored recency-weighted family members (e.g. `sh_hl1`, half-life 1 Congress) in `D_FAMILY_RESULTS`,
but only for S2 and S3. The `*_rw` columns repeat the val pick with those members added, so S2 and S3
can be compared on a model that is allowed to down-weight old Congresses; `-` means "not evaluated".
**How to read it:** a usable split has 0 validity fails with many checks judged; a split judged by few
checks has not "passed", it was simply not tested. Splits with 1 test fold have no fold variability.

---
## [decision-c147-txt] text
```
WARNING checks_split_scores: 15 split-dependent check(s) have kind=None and are NOT scored: B1, B1c, B4, B5a, B9, C1, C2, C23b, C6, C6b, C24, C19, C20, D8, D13. Classify them (record_check(kind=...) or QUANTITY_KIND) or pass kinds including None to score them.
Unclassified split-dependent checks (not ranked; used in the robustness re-run): B1, B1c, B4, B5a, B9, C1, C19, C2, C20, C23b, C24, C6, C6b, D13, D8

Scorecard (T1 = cosponsor t0=0 Recall@20, T2 = advancement AUROC; val-selected baseline; gap/hw = mean |val-test| / test 95% CI half-width; *_rw = val pick incl. recency-weighted members, evaluated for S2, S3 only):
                 kind val_judged val_fail unc_judged unc_fail drift_fail drift_judged folds       T1_base T1_test T1_gap/hw T1_fold_sd T1_n_eval T1_base_rw T1_test_rw         T2_base T2_test T2_gap/hw T2_fold_sd  T2_base_rw T2_test_rw
split                                                                                                                                                                                                                                     
S1a         expanding          6        1         10        0          6           28     1  sponsor_hist  0.2620    0.4500          -      5101          -          -  eb_rate_t0safe  0.5600    0.2590          -           -          -
S1b         expanding          6        1         10        0          9           28     1        rw2hop  0.2450    0.9980          -      5138          -          -  eb_rate_t0safe  0.6190    1.5170          -           -          -
S2            sliding         11        0         10        0         10           29     7  sponsor_hist  0.2720    0.6620     0.0100     30203     sh_hl1     0.2830  eb_rate_t0safe  0.6320    1.2830     0.0490  eb_m20_t0s     0.6320
S3            rolling         13        0         10        0          9           29     7  sponsor_hist  0.2590    0.6120     0.0090     30203     sh_hl1     0.2820  eb_rate_t0safe  0.6240    1.2800     0.0450   eb_m5_t0s     0.6240
S4             within          4        1          7        1          9           25     3        rw2hop  0.3110    0.6260     0.0170      2331          -          -  eb_rate_t0safe  0.6570    0.8650     0.0800           -          -
S5             random          4        1          3        2          0            2     1  sponsor_hist  0.3320    0.5860          -      6676          -          -  eb_rate_t0safe  0.6250    0.6150          -           -          -
S1a+stream  expanding          7        1         11        0          6           28     1  sponsor_hist  0.3070    1.0370          -      5101          -          -  eb_rate_t0safe  0.5480    1.2140          -           -          -
S1b+stream  expanding          7        1         11        0          9           28     1        rw2hop  0.2960    0.7020          -      5138          -          -  eb_rate_t0safe  0.6050    1.4540          -           -          -
S3+stream     rolling          8        0         11        0          9           28     7  sponsor_hist  0.2960    0.5080     0.0060     30203          -          -  eb_rate_t0safe  0.6310    1.6370     0.0520           -          -
S1pool      expanding          5        3         10        1          9           28     1  sponsor_hist  0.2220    1.9940          -     10239          -          -  eb_rate_t0safe  0.5950    1.1610          -           -          -
Splits with no +stream variant evaluated (their streaming behaviour is unknown, not a protocol mismatch): S2

What undermines (-) / supports (+) each split (validity checks; unclassified fails in [ ]):
  S1a        - D9           + C5, D1, D20, D22, D10
  S1b        - D9           + C5, D1, D20, D22, D10
  S2         - none         + C5, D1, D20, D11, D12, D12b, D21, D17, D9, D10, D27
  S3         - none         + C5, D1, D20, D11, D12, D12b, D21, D17, D22, D23, D9, D10, D27
  S4         - C5           + B1b, D1, D20  [unclassified fails: C6b]
  S5         - D7           + C5, D1, D20  [unclassified fails: C19, C20]
  S1a+stream - D9           + B1b, C5, D1, D20, D14, D10
  S1b+stream - D9           + B1b, C5, D1, D20, D14, D10
  S3+stream  - none         + B1b, C5, D1, D20, D14, D9, D10, D27
  S1pool     - D1, D20, D9  + C5, D10  [unclassified fails: C23b]
```

---
## [decision-c148-o0] chart
```
title: 
subtitle: 
data (310 rows): id | split | v | grp | why
  B1b | S1a | not judged | validity | base split = protocol without eval-Congress history; its cold share is B1 (INFO); the pro~
  B1b | S1b | not judged | validity | base split = protocol without eval-Congress history; its cold share is B1 (INFO); the pro~
  B1b | S2 | not judged | validity | protocol without eval-Congress history; cold share is B1 (INFO)
  B1b | S3 | not judged | validity | base split = protocol without eval-Congress history; its cold share is B1 (INFO); the pro~
  B1b | S4 | PASS | validity | 
  B1b | S5 | not judged | validity | random bill split: ignores time and Congress boundaries (leaky reference); cross-Congress~
  B1b | S1a+stream | PASS | validity | 
  B1b | S1b+stream | PASS | validity | 
  B1b | S3+stream | PASS | validity | 
  B1b | S1pool | not judged | validity | protocol without eval-Congress history; cold share is B1 (INFO)
  C5 | S1a | PASS | validity | 
  C5 | S1b | PASS | validity | 
  C5 | S2 | PASS | validity | 
  C5 | S3 | PASS | validity | 
  C5 | S4 | FAIL | validity | 
  C5 | S5 | PASS | validity | 
  C5 | S1a+stream | PASS | validity | 
  C5 | S1b+stream | PASS | validity | 
  C5 | S3+stream | PASS | validity | 
  C5 | S1pool | PASS | validity | 
  D1 | S1a | PASS | validity | 
  D1 | S1b | PASS | validity | 
  D1 | S2 | PASS | validity | 
  D1 | S3 | PASS | validity | 
  D1 | S4 | PASS | validity | 
  D1 | S5 | PASS | validity | 
  D1 | S1a+stream | PASS | validity | 
  D1 | S1b+stream | PASS | validity | 
  D1 | S3+stream | PASS | validity | 
  D1 | S1pool | FAIL | validity | 
  D10 | S1a | PASS | validity | 
  D10 | S1b | PASS | validity | 
  D10 | S2 | PASS | validity | 
  D10 | S3 | PASS | validity | 
  D10 | S4 | not judged | validity | S4 / S5 train on all earlier Congresses plus part of the eval Congress: the window choice~
  D10 | S5 | not judged | validity | S4 / S5 train on all earlier Congresses plus part of the eval Congress: the window choice~
  D10 | S1a+stream | PASS | validity | 
  D10 | S1b+stream | PASS | validity | 
  D10 | S3+stream | PASS | validity | 
  D10 | S1pool | PASS | validity | 
  D11 | S1a | not judged | validity | near-tie families / learned model are fit only on the S2 / S3 rolling folds (S1a / S1b = ~
  D11 | S1b | not judged | validity | near-tie families / learned model are fit only on the S2 / S3 rolling folds (S1a / S1b = ~
  D11 | S2 | PASS | validity | 
  D11 | S3 | PASS | validity | 
  D11 | S4 | not judged | validity | near-tie families / learned model are fit only on the S2 / S3 rolling folds (S1a / S1b = ~
  D11 | S5 | not judged | validity | near-tie families / learned model are fit only on the S2 / S3 rolling folds (S1a / S1b = ~
  D11 | S1a+stream | not judged | validity | near-tie families / learned model are fit only on the S2 / S3 rolling folds (S1a / S1b = ~
  D11 | S1b+stream | not judged | validity | near-tie families / learned model are fit only on the S2 / S3 rolling folds (S1a / S1b = ~
  D11 | S3+stream | not judged | validity | near-tie families / learned model are fit only on the S2 / S3 rolling folds (S1a / S1b = ~
  D11 | S1pool | not judged | validity | near-tie families / learned model are fit only on the S2 / S3 rolling folds (S1a / S1b = ~
  D12 | S1a | not judged | validity | near-tie families / learned model are fit only on the S2 / S3 rolling folds (S1a / S1b = ~
  D12 | S1b | not judged | validity | near-tie families / learned model are fit only on the S2 / S3 rolling folds (S1a / S1b = ~
  D12 | S2 | PASS | validity | 
  D12 | S3 | PASS | validity | 
  D12 | S4 | not judged | validity | near-tie families / learned model are fit only on the S2 / S3 rolling folds (S1a / S1b = ~
  D12 | S5 | not judged | validity | near-tie families / learned model are fit only on the S2 / S3 rolling folds (S1a / S1b = ~
  D12 | S1a+stream | not judged | validity | near-tie families / learned model are fit only on the S2 / S3 rolling folds (S1a / S1b = ~
  D12 | S1b+stream | not judged | validity | near-tie families / learned model are fit only on the S2 / S3 rolling folds (S1a / S1b = ~
  D12 | S3+stream | not judged | validity | near-tie families / learned model are fit only on the S2 / S3 rolling folds (S1a / S1b = ~
  D12 | S1pool | not judged | validity | near-tie families / learned model are fit only on the S2 / S3 rolling folds (S1a / S1b = ~
  ... (250 more rows)
```

---
## [decision-c149-o0] chart
```
title: 
subtitle: 
data (120 rows): task | split | role | fold | value | base | ev
  T1 t0=0 Recall@20 | S1a | val | S1a | 0.2541615916 | sponsor_hist | 116
  T1 t0=0 Recall@20 | S1a | test | S1a | 0.2615268317 | sponsor_hist | 117
  T2 AUROC | S1a | val | S1a | 0.5688178868 | eb_rate_t0safe | 116
  T2 AUROC | S1a | test | S1a | 0.5598481668 | eb_rate_t0safe | 117
  T1 t0=0 Recall@20 | S1b | val | S1b | 0.2628509525 | rw2hop | 117
  T1 t0=0 Recall@20 | S1b | test | S1b | 0.2453250633 | rw2hop | 118
  T2 AUROC | S1b | val | S1b | 0.5598481668 | eb_rate_t0safe | 117
  T2 AUROC | S1b | test | S1b | 0.6188484054 | eb_rate_t0safe | 118
  T1 t0=0 Recall@20 | S2 | val | c=112 | 0.2747155471 | sponsor_hist | 111
  T1 t0=0 Recall@20 | S2 | test | c=112 | 0.2606779254 | sponsor_hist | 112
  T1 t0=0 Recall@20 | S2 | val | c=113 | 0.2606779254 | sponsor_hist | 112
  T1 t0=0 Recall@20 | S2 | test | c=113 | 0.2690049576 | sponsor_hist | 113
  T1 t0=0 Recall@20 | S2 | val | c=114 | 0.2690049576 | sponsor_hist | 113
  T1 t0=0 Recall@20 | S2 | test | c=114 | 0.2769483057 | sponsor_hist | 114
  T1 t0=0 Recall@20 | S2 | val | c=115 | 0.2769483057 | sponsor_hist | 114
  T1 t0=0 Recall@20 | S2 | test | c=115 | 0.29008665 | sponsor_hist | 115
  T1 t0=0 Recall@20 | S2 | val | c=116 | 0.29008665 | sponsor_hist | 115
  T1 t0=0 Recall@20 | S2 | test | c=116 | 0.2692113675 | sponsor_hist | 116
  T1 t0=0 Recall@20 | S2 | val | c=117 | 0.2692113675 | sponsor_hist | 116
  T1 t0=0 Recall@20 | S2 | test | c=117 | 0.274427011 | sponsor_hist | 117
  T1 t0=0 Recall@20 | S2 | val | c=118 | 0.274427011 | sponsor_hist | 117
  T1 t0=0 Recall@20 | S2 | test | c=118 | 0.2601671653 | sponsor_hist | 118
  T2 AUROC | S2 | val | c=112 | 0.5718474264 | eb_rate_t0safe | 111
  T2 AUROC | S2 | test | c=112 | 0.6859848114 | eb_rate_t0safe | 112
  T2 AUROC | S2 | val | c=113 | 0.6859848114 | eb_rate_t0safe | 112
  T2 AUROC | S2 | test | c=113 | 0.6664689198 | eb_rate_t0safe | 113
  T2 AUROC | S2 | val | c=114 | 0.6664689198 | eb_rate_t0safe | 113
  T2 AUROC | S2 | test | c=114 | 0.6642441294 | eb_rate_t0safe | 114
  T2 AUROC | S2 | val | c=115 | 0.6642441294 | eb_rate_t0safe | 114
  T2 AUROC | S2 | test | c=115 | 0.6582798958 | eb_rate_t0safe | 115
  T2 AUROC | S2 | val | c=116 | 0.6582798958 | eb_rate_t0safe | 115
  T2 AUROC | S2 | test | c=116 | 0.5670965287 | eb_rate_t0safe | 116
  T2 AUROC | S2 | val | c=117 | 0.5670965287 | eb_rate_t0safe | 116
  T2 AUROC | S2 | test | c=117 | 0.5681197129 | eb_rate_t0safe | 117
  T2 AUROC | S2 | val | c=118 | 0.5681197129 | eb_rate_t0safe | 117
  T2 AUROC | S2 | test | c=118 | 0.6165265555 | eb_rate_t0safe | 118
  T1 t0=0 Recall@20 | S3 | val | c=112 | 0.2747155471 | sponsor_hist | 111
  T1 t0=0 Recall@20 | S3 | test | c=112 | 0.2598401168 | sponsor_hist | 112
  T1 t0=0 Recall@20 | S3 | val | c=113 | 0.2598401168 | sponsor_hist | 112
  T1 t0=0 Recall@20 | S3 | test | c=113 | 0.2577569148 | sponsor_hist | 113
  T1 t0=0 Recall@20 | S3 | val | c=114 | 0.2577569148 | sponsor_hist | 113
  T1 t0=0 Recall@20 | S3 | test | c=114 | 0.2647517 | sponsor_hist | 114
  T1 t0=0 Recall@20 | S3 | val | c=115 | 0.2647517 | sponsor_hist | 114
  T1 t0=0 Recall@20 | S3 | test | c=115 | 0.2720543262 | sponsor_hist | 115
  T1 t0=0 Recall@20 | S3 | val | c=116 | 0.2720543262 | sponsor_hist | 115
  T1 t0=0 Recall@20 | S3 | test | c=116 | 0.2541615916 | sponsor_hist | 116
  T1 t0=0 Recall@20 | S3 | val | c=117 | 0.2541615916 | sponsor_hist | 116
  T1 t0=0 Recall@20 | S3 | test | c=117 | 0.2615268317 | sponsor_hist | 117
  T1 t0=0 Recall@20 | S3 | val | c=118 | 0.2615268317 | sponsor_hist | 117
  T1 t0=0 Recall@20 | S3 | test | c=118 | 0.2438587738 | sponsor_hist | 118
  T2 AUROC | S3 | val | c=112 | 0.5718474264 | eb_rate_t0safe | 111
  T2 AUROC | S3 | test | c=112 | 0.6275979737 | eb_rate_t0safe | 112
  T2 AUROC | S3 | val | c=113 | 0.6275979737 | eb_rate_t0safe | 112
  T2 AUROC | S3 | test | c=113 | 0.668362496 | eb_rate_t0safe | 113
  T2 AUROC | S3 | val | c=114 | 0.668362496 | eb_rate_t0safe | 113
  T2 AUROC | S3 | test | c=114 | 0.6635708332 | eb_rate_t0safe | 114
  T2 AUROC | S3 | val | c=115 | 0.6635708332 | eb_rate_t0safe | 114
  T2 AUROC | S3 | test | c=115 | 0.6584580059 | eb_rate_t0safe | 115
  T2 AUROC | S3 | val | c=116 | 0.6584580059 | eb_rate_t0safe | 115
  T2 AUROC | S3 | test | c=116 | 0.5688178868 | eb_rate_t0safe | 116
  ... (60 more rows)
```

---
## [decision-c150-md] markdown
### Recommendation: explicit rules applied to the recorded checks

The recommendation is a short chain of rules. Each rule names the checks it reads, prints the values
it used and its verdict, and says what it implies. A conclusion is labelled **robust** when at least two
independent checks support it and none contradicts it (and no section disagrees on that quantity);
otherwise it is **borderline** and the printout says what to watch. The split ranking is computed per
task (checks are assigned to Task 1, Task 2 or both from their wording) and is re-run with the
unclassified checks included. A pick is labelled ROBUST only if (a) both runs pick the same split, (b) the
criterion that separates it from the runner-up comes from rules that are all robust, and (c) its
model-selection checks were run on that split itself rather than inherited from its base split.

| Rule | Question | If it fails |
|---|---|---|
| R1 | Any validity FAIL for the split (leakage, val does not predict test, too few test Congresses, censoring)? | drop the split for model selection |
| R2 | Cold start: share of eval positives from legislators unseen in train (B1) vs with in-Congress history (B1b/B1c) | inductive features and/or the `+stream` protocol |
| R3 | Does streaming eval-Congress history up to t0 change scores (D13, SPLIT_RESULTS) and keep val honest (D14)? | evaluate a temporal GNN with `+stream` |
| R4 | Party regimes: do they change the effects (A5b, D26, C23b)? | evaluate per regime |
| R4b | Does the split drop the eval Congress's regime from training (C8), and does that matter given R4? | prefer a window that keeps every regime |
| R5 | Expanding vs sliding window, with and without recency weighting (D10, SPLIT_RESULTS, D_FAMILY_RESULTS, A14) | sliding window, or expanding + recency weights |
| R6 | Val predicts test level and model choice (D1, D20, D14, D2b, D11/D12/D12b/D21/D23/D17), per split actually tested | report per-Congress scores; flag inferred evidence |
| R7 | Fold stability and test size (D3, D4, D9, D27) | use several rolling test folds; lock the last Congress |
| R8 | Drift FAILs exposed by the chosen split | remedies (below), not a different split |

---
## [decision-c151-txt] text
```
[R1-T1] T1: which splits have a validity FAIL? -> drop S1a, S1b, S4, S5, S1a+stream, S1b+stream, S1pool  <ROBUST; support: C5, D1, D20, D7, D9>
    evidence: S1a: D9; S1b: D9; S4: C5; S5: D7; S1a+stream: D9; S1b+stream: D9; S1pool: D1, D20, D9
    implies : not usable for model selection / honest reporting of this task
[R1-T2] T2: which splits have a validity FAIL? -> drop S1a, S1b, S1a+stream, S1b+stream, S1pool  <BORDERLINE; support: D9>
    evidence: S1a: D9; S1b: D9; S1a+stream: D9; S1b+stream: D9; S1pool: D9
    implies : not usable for model selection / honest reporting of this task
[R2] Cold start: are eval positives from legislators unseen in training? -> cold start is LARGE for a transductive model without in-Congress history; resolved once eval-Congress history up to t0 is in the graph  <ROBUST; support: B1, B1b, B1c>
    evidence: B1 train-cold share of positives by split S1a 18.0%, S1b 18.1%, S2 20.5%, S3 20.4%, S1pool 31.4% (bound 2%); B1b +stream = 0.0071, B1c = 0.0145
    implies : use inductive node features (party, chamber, state, committee, tenure) AND time-filtered in-Congress history (+stream); never rely on per-legislator embeddings learned only from training Congresses
[R3-T1] T1: does eval-Congress history up to t0 (+stream) change the test score? -> yes: evaluate with +stream  <ROBUST; support: D13, D14, B1b>
    evidence: S3+stream - S3 test Recall@20 (val-selected baseline) = +0.0370 vs MDE 0.01; D13 = 0.03697; D14 (val honest under stream) = PASS
    implies : the GNN must see the eval Congress's own edges dated <= intro + t0 (time-filtered neighbor sampling)
[R3-T2] T2: does eval-Congress history up to t0 (+stream) change the test score? -> no material gain: plain or +stream are equivalent  <BORDERLINE; support: ->
    evidence: S3+stream - S3 test AUROC (val-selected baseline) = +0.0077 vs MDE 0.01; D14 (val honest under stream) = PASS
    implies : streaming is optional for this task; keep it for consistency with Task 1
[R4] Party control: do regime changes alter the effects a model learns? -> regime effects are stable given chamber-majority features  <ROBUST; support: A5b, D26>
    evidence: A5b (majority effect matches same regime) = PASS; D26 (regime weighting not needed) = PASS; C23b (post-break test units) fails for S1pool
    implies : include chamber-majority / sponsor-in-majority features; report test folds per regime
[R4b] Does the split drop the eval Congress's party regime from training, and does that matter? -> mild penalty for S2  <BORDERLINE; support: C8; against: A5b, D26>
    evidence: C8 regime coverage fails for S2; but R4 finds regime effects stable (A5b PASS, D26 PASS)
    implies : a missing regime is a coverage gap, not a demonstrated score loss; use it only as a tie-break
[R5] Do older Congresses hurt (expanding S3 vs sliding S2), with and without recency weighting? -> window length is a wash ONLY with recency weighting; without it the sliding window is better; Task 2 is equivalent within the MDE  <BORDERLINE; support: D10; against: A14>
    evidence: test S3 - S2: T1 unweighted sponsor_hist -0.0124, T1 unweighted rw2hop -0.0124, T1 recency sh_hl1 -0.0008, T1 recency rw_b0.3_hl1 -0.0007, T2 eb_rate_t0safe -0.0087, T1 val-picked incl. recency -0.0008, T2 val-picked incl. near-tie family -0.0089 (tolerance MDE/2 = 0.005 T1, MDE = 0.01 T2); D10 (recency-weighted not worse) = PASS; A14 distribution closeness favouring sliding = 0.6429 (> 0.5 favours sliding, proxy only)
    implies : CONDITIONAL: use an expanding window only together with recency weighting (decay chosen on val); with unweighted count / graph features S2 is as good or better. The window choice gives no score reason to prefer S3 over S2; only R4b (regime coverage) and fold evidence do
[R6-T1] T1: on the splits where selection was tested, does val predict the test level and pick the test-best model? -> yes for S2, S3 (by-Congress val is a faithful proxy there)  <ROBUST; support: D1, D20, D14, D22, D11, D21, D23, D17>
    evidence: S2: level D1=PASS,D20=PASS; selection D11=PASS,D21=PASS,D17=PASS; S3: level D1=PASS,D20=PASS,D22=PASS; selection D11=PASS,D21=PASS,D23=PASS,D17=PASS
    implies : tune / early-stop on val Congress c-1, refit on 108..c-1 (D19), test on c
[R6-T1s] T1: is val a faithful proxy under S3+stream (streaming)? -> level carries over (tested); model selection NOT tested, inferred from the base split  <BORDERLINE; support: D14; against: D11, D17, D21, D23>
    evidence: S3+stream: level D1=PASS,D20=PASS,D14=PASS; selection NOT RUN (inferred from S3: D11,D21,D23,D17)
    implies : before trusting model choices made under S3+stream, rerun D11 / D17 (sponsor_hist and rw2hop families) on its folds; until then report that selection reliability is inferred from plain S3
[R6-T2] T2: does the val Congress predict the test AUROC level and the model choice? -> model ranking transfers but the AUROC LEVEL does not transfer between Congresses  <ROBUST; support: D12, D12b, D2b, D4>
    evidence: level: D2b = FAIL (fails for S1b, S2, S3, S4, S1b+stream, S3+stream); D4 std = 0.03593; selection D12/D12b = PASS/PASS
    implies : select Task-2 models on val (rankings transfer), but report per-Congress AUROC and the fold mean and never quote a val AUROC as the expected test level (D2b). AUROC is invariant to monotone recalibration, so no recalibration fixes a D2b failure; if calibrated probabilities are needed, recalibrate on the eval Congress's already-resolved bills (D2b), not on Congress c-1
[R7] Is one test Congress enough, and can the last Congress be locked away? -> use rolling one-step-ahead test folds and lock 118 as the final test  <ROBUST; support: D9, D27, D17>
    evidence: D9 single-test-Congress miss risk fails for S1a, S1b, S1a+stream, S1b+stream, S1pool; D3 T1 fold CV = 0.03396; D27 locked test = PASS (locked 118)
    implies : compare architectures on the mean over development folds (paired, same bills); touch the locked Congress once
[R8] Which drift checks fail (remedies, not a different split)? -> 10 drift FAILs  <ROBUST; support: A1b, A2, A3, A4, A8, A13, A17, B2, C3, D2b>
    evidence: A1b (label_validity_t2), A2 (label_rate_t2_chamber), A3 (cosp_density), A4 (orig_share), A8 (bill_volume), A13 (model_auroc_transfer_t2), A17 (cand_fail_count), B2 (sponsor_overlap), C3 (lag_dist), D2b
    implies : see the remedy list below
```

---
## [decision-c152-txt] text
```
=== T1 (cosponsor_t0) ranking (validity checks for this task + shared ones) ===
     split  fail  judged  low_evidence  protocol_mismatch                      protocol  rule_penalty pen_rules  sel_checks  sel_inferred  folds   test  drift_fail
 S3+stream     0       8             0                  0                         match             0         -           0             1      7 0.2960           9
        S3     0      11             0                  1 plain (stream variant exists)             0         -           4             0      7 0.2590           9
        S2     0       9             0                  1  no +stream variant evaluated             1       R4b           3             0      7 0.2720          10
S1a+stream     1       7             0                  0                         match             0         -           0             0      1 0.3070           6
S1b+stream     1       7             0                  0                         match             0         -           0             0      1 0.2960           9
With unclassified checks included the top split is S3+stream (same); usable (0 validity fails, enough evidence): S3+stream, S3, S2; too little evidence: -

=== T2 (advancement) ranking (validity checks for this task + shared ones) ===
    split  fail  judged  low_evidence  protocol_mismatch            protocol  rule_penalty pen_rules  sel_checks  sel_inferred  folds   test  drift_fail
       S3     0       6             0                  0               match             0         -           3             0      7 0.6240           9
       S2     0       6             0                  0               match             1       R4b           3             0      7 0.6320          10
S3+stream     0       4             0                  1 stream (not needed)             0         -           0             1      7 0.6310           9
       S4     0       1             1                  0               match             0         -           0             0      3 0.6570           9
       S5     0       0             1                  0               match             0         -           0             0      1 0.6250           0
With unclassified checks included the top split is S3 (same); usable (0 validity fails, enough evidence): S3, S2, S3+stream; too little evidence: S4, S5

====================================================================================================
RECOMMENDATION
====================================================================================================

T1 (cosponsor_t0): use S3+stream = Rolling origin, expanding + streaming eval-Congress history  [BORDERLINE]
  definition : S3 rolling folds + eval-Congress events dated <= introduced_date + t0
  folds      : development test Congresses 112-117 (val = c-1, train = 108..c-2); LOCKED final test 118 (train 108-117, touched once); prospective 119
  backtest   : val-picked 'sponsor_hist' (unweighted SPLIT_RESULTS baselines) test Recall@20 0.296 over 7 fold(s), fold sd 0.006, mean |val-test| = 0.51x the test CI half-width, 30,203 test bills
  relies on  : validity checks passed for S3+stream (T1 + shared): B1b, C5, D1, D20, D14, D9, D10, D27
  selection  : NOT tested on S3+stream; selection reliability under S3+stream is INFERRED from plain S3 (D11=PASS, D21=PASS, D23=PASS, D17=PASS). Rerun D11 / D17 (sponsor_hist and rw2hop families) on S3+stream folds to confirm
  risks      : drift FAILs on this split for T1 -> A3, A4, A8, A17, C3
  runner-up  : S3 (loses on 'protocol_mismatch': 1 vs 0 [plain (stream variant exists)]; decided by R3-T1 (robust); runner-up test 0.259 vs pick 0.296)
  borderline : selection reliability (does val pick the test-best model?) was not tested on this split; inferred from plain S3 (D11, D21, D23, D17 pass there)
  condition  : expanding only WITH recency weighting; otherwise S2 is as good or better (R5, S3 minus S2: T1 unweighted sponsor_hist -0.0124, T1 unweighted rw2hop -0.0124, T1 recency sh_hl1 -0.0008, T1 recency rw_b0.3_hl1 -0.0007, T1 val-picked incl. recency -0.0008). S2 is a legitimate alternative (no +stream variant of S2 evaluated yet)
  not for selection: S1a (D9), S1b (D9), S4 (C5), S5 (D7), S1a+stream (D9), S1b+stream (D9), S1pool (D1, D20, D9)

T2 (advancement): use S3 = Rolling origin, expanding  [BORDERLINE]
  definition : for test c: train 108..c-2 | val c-1 | test c (refit on 108..c-1); c in 112-118
  folds      : development test Congresses 112-117 (val = c-1, train = 108..c-2); LOCKED final test 118 (train 108-117, touched once); prospective 119
  backtest   : val-picked 'eb_rate_t0safe' (unweighted SPLIT_RESULTS baselines) test AUROC 0.624 over 7 fold(s), fold sd 0.045, mean |val-test| = 1.28x the test CI half-width, 74,609 test bills
  relies on  : validity checks passed for S3 (T2 + shared): D12, D12b, D17, D9, D10, D27
  selection  : tested on S3: D12=PASS, D12b=PASS, D17=PASS
  risks      : drift FAILs on this split for T2 -> A1b, A2, A8, A13, A17, D2b
  runner-up  : S2 (loses on 'rule_penalty': 1 vs 0; decided by R4b (borderline); runner-up test 0.632 vs pick 0.624)
  borderline : separated from S2 only by borderline rule(s) R4b
  condition  : S3 vs S2 test (S3 minus S2): T2 eb_rate_t0safe -0.0087, T2 val-picked incl. near-tie family -0.0089; |diff| <= MDE 0.01, so the windows are equivalent on score and S2 is a legitimate alternative; no recency-weighted Task-2 baseline was evaluated
  reporting  : AUROC level does not transfer between Congresses (D2b): report per-Congress AUROC and the fold mean; never quote a val AUROC as the expected test level
  not for selection: S1a (D9), S1b (D9), S4 (too little evidence, unclassified fail C6b), S5 (too little evidence, unclassified fail C19, C20), S1a+stream (D9), S1b+stream (D9), S1pool (D9)

One pipeline for both tasks: S3+stream for Task 2 is usable and changes the Task-2 test AUROC by +0.0077 (MDE 0.01) -> run both tasks on S3+stream (same folds; Task 2 simply gains little from streaming). Note: Task-2 selection checks were not run on S3+stream either (inferred from plain S3)

Remedies for the drift the chosen splits expose (R8):
  A1b   label_validity_t2        -> Task-2 label misses more chamber-passed bills lately: also report label_alt (reported OR passed)
  A2    label_rate_t2_chamber    -> per-chamber advancement rate moves: chamber-specific intercepts; if probabilities are needed, recalibrate on the eval Congress's already-resolved bills (as D2b), not on c-1
  A3    cosp_density             -> positive density moves: compare ranking metrics per Congress; do not compare raw recall across Congresses
  A4    orig_share               -> original-cosponsor share moves: t0=0 vs t0=7 positives differ by Congress; report both t0
  A8    bill_volume              -> bill volume moves: normalise counts per Congress (degree / activity features as rates)
  A13   model_auroc_transfer_t2  -> Task-2 model degrades forward: retrain every Congress, recency-weight, report per Congress
  A17   cand_fail_count          -> several drifts at once: rely on rolling folds rather than one test Congress
  B2    sponsor_overlap          -> new sponsors: inductive sponsor features plus in-Congress history (+stream)
  C3    lag_dist                 -> cosponsor timing moves: window labels at fixed t0 (already) and monitor lag per Congress
  D2b   (no quantity)            -> Task-2 AUROC level does not transfer between Congresses: report per-Congress AUROC and the fold mean, never a val AUROC as the expected test level (AUROC ignores recalibration, so recalibrating cannot fix this); for calibrated probabilities recalibrate on the eval Congress's already-resolved bills, not on c-1

Robust vs borderline conclusions:
  picks     : T1 S3+stream [BORDERLINE]; T2 S3 [BORDERLINE]
  robust    : R1-T1 (drop S1a, S1b, S4, S5, S1a+stream, S1b+stream, S1pool); R2 (cold start is LARGE for a transductive model without in-Congress history; resolved once eval-Congress history up to t0 is in the graph); R3-T1 (yes: evaluate with +stream); R4 (regime effects are stable given chamber-majority features); R6-T1 (yes for S2, S3 (by-Congress val is a faithful proxy there)); R6-T2 (model ranking transfers but the AUROC LEVEL does not transfer between Congresses); R7 (use rolling one-step-ahead test folds and lock 118 as the final test); R8 (10 drift FAILs)
  borderline: R1-T2 (drop S1a, S1b, S1a+stream, S1b+stream, S1pool); R3-T2 (no material gain: plain or +stream are equivalent); R4b (mild penalty for S2); R5 (window length is a wash ONLY with recency weighting; without it the sliding window is better; Task 2 is equivalent within the MDE); R6-T1s (level carries over (tested); model selection NOT tested, inferred from the base split)

Evidence gaps (close these before treating the borderline picks as settled):
  - recency-weighted baselines (e.g. sh_hl1) exist only for S2, S3; not evaluated for S1a, S1b, S1a+stream, S1b+stream, S3+stream, S1pool (add them to SPLIT_RESULTS so a recency-weighted val pick is possible there)
  - no +stream variant for S2: Task-1 standing under streaming is unknown
  - T1 pick S3+stream: no model-selection check (D11 / D17 / D21 / D23) run on its folds

Monitor on every new fold / Congress: bill_volume, cand_fail_count, cosp_density, label_rate_t2_chamber, label_validity_t2, lag_dist, model_auroc_transfer_t2, orig_share, sponsor_overlap; plus the validity checks the choice relies on (rerun this notebook when a Congress completes).
```

---
## [decision-c153-md] markdown
### Practical guidance for the GNN

Read this together with the RECOMMENDATION printout above; the guidance assumes its rules came out the
way they do in the current run (re-check if a rerun flips a verdict).

- **Folds.** Develop on rolling-origin folds: for each development test Congress *c*, train on 108..c-2,
  tune and early-stop on *c-1*, refit on 108..c-1 with the tuned hyperparameters, and score *c*. Compare
  architectures on the mean over the development folds (paired, same bills), never on one Congress. Keep
  the last complete Congress (`D_FINAL_PROTOCOL['locked_test']`) as a locked test you score once, and treat
  the in-progress 119th as a prospective check only. A single by-Congress split (train 108-115 / val 116 /
  test 117 or 118) is fine for quick debugging, but R7/D9 show one test Congress cannot settle close model
  comparisons.
- **Inductive, not purely transductive.** Many eval positives come from legislators absent from the
  training Congresses (R2). Give legislator nodes inductive features (party as of the Congress, chamber,
  state, committee seats, tenure, majority status) and do not depend on per-node ID embeddings learned only
  from the training Congresses. GraphSAGE-style aggregation or a heterogeneous GNN with feature encoders
  fits; a lookup-table embedding model does not.
- **Time-filtered neighbor sampling (the `+stream` protocol).** For a bill scored at t0, the graph may
  contain every edge dated <= introduced_date + t0: earlier Congresses plus the eval Congress's own
  sponsorships and cosponsorships so far. Never use the eval Congress's final graph (B10), committee edges
  recorded after t0 (B7), related-bill links not yet known (B6b), or snapshot count fields (C11). Build
  training examples the same way, from as-of-t0 snapshots (B12), so train and eval inputs match.
- **Window and weighting (conditional).** The expanding window is only as good as the sliding one *if*
  the model down-weights old Congresses (R5/D10). With unweighted count or graph features, the last-3-
  Congress sliding window (S2) scores as well or better. So: if you use an expanding window, add a recency
  weight or edge-time decay and choose its half-life on validation. If you do not, S2 is a legitimate
  choice. The remaining arguments for the expanding window are weak (R4b: S2 leaves the eval Congress's
  party regime out of training, but regime effects look stable, A5b/D26) or about evidence (more checks
  were run on S3). Add chamber-majority / sponsor-in-majority features rather than regime-specific weighting (R4/D26).
- **Model selection under streaming.** The selection checks (D11/D17/D21/D23) were run on plain S3
  folds, not on S3+stream. Until they are rerun under streaming, treat "val picks the test-best model" for
  the streaming protocol as inferred, not tested. D14 only shows that the score *level* carries over.
- **Task 2.** Pick models on validation (rankings transfer, D12/D12b). The AUROC *level* does not carry
  over from one Congress to the next (D2b), so report per-Congress AUROC plus the fold mean, and never quote
  a val AUROC as the expected test level. AUROC does not change under monotone recalibration, so
  recalibrating cannot fix this. If you need calibrated probabilities, recalibrate on the eval Congress's
  already-resolved bills (labels are known 180 days after introduction), not on Congress c-1. Hide the
  policy area for bills younger than its availability lag (C18/C25). Task 2 can share Task 1's folds.
- **Monitor.** Every quantity in the "Monitor" line drifts between Congresses. Rerun this notebook when a
  Congress completes. If a validity check the choice relies on flips, the ranking changes automatically.