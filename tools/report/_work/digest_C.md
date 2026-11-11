# Digest: C. Temporal structure, regime changes, censoring, leakage (section key C)


---
## [C-c72-md] markdown
## C. Temporal structure, regime changes, censoring, leakage

A by-Congress split assumes that each Congress is a comparable "draw" of the same process, and that
whatever we feed the model at prediction time (day 0 or day 7 after introduction) was really known at
that time. This section tests the time-related conditions behind that assumption:

1. **Within-Congress dynamics**: are bills introduced on the same calendar rhythm in every Congress?
2. **Cosponsorship timing**: is the 30-day prediction window equally hard in every Congress?
3. **Procedural breakpoints**: does the original-cosponsor share jump on a date inside a Congress, and does
   each split's fit set contain the post-break regime it is evaluated on?
4. **Censoring**: how many bills lose their label because the window runs past the end of the Congress?
5. **Regime changes**: is each chamber's controlling party in every validation/test Congress represented in
   training, and does the joint House x Senate regime matter at all?
6. **Snapshot leakage**: which bill fields describe the bill *after* introduction and must never be features?
7. **Data latency and label completeness**: is the most recent Congress (118) less complete than older ones, and
   are its labels (cosponsor events, committee reports) fully loaded, or only its annotations?
8. **Member turnover / point-in-time pool**: how much does the full-Congress roster distort task-1 densities
   and Recall@20 compared with members in office at t0?
9. **Feature availability and companion bills**: is the policy area known at t0, and do companion /
   reintroduced bills connect evaluation bills to the training portion of each candidate split?
10. **Task-2 prediction time**: which t0 makes which features legitimate, and how much label is spent by waiting?

"Differs from train" verdicts use the shared `rel_to_train()` rule (95% t prediction interval of the 108-115
reference, with nested leave-one-out false-alarm rates). Pool-dependent checks iterate over the shared
evaluation units `iter_eval_units()` (plus the S1pool variant) and record a `by_split` verdict per candidate
split, so the decision section can score splits without parsing text. Quantities owned by another section
(label rates -> A) are shown here only as cross-references (INFO).
Congress 119 is still in session and is used only as context.

---
## [C-c73-txt] text
```
law-capable bills 108-119: 145,738; train pool 108-115, evaluation candidates [116, 117, 118]
```

---
## [C-c74-md] markdown
### C1-C2. When are bills introduced inside a Congress?

**Why it matters.** Bills introduced in the first weeks of a Congress (reintroductions, leadership
priorities) and bills dropped in the final months (messaging bills, the end-of-Congress rush) behave
differently: early bills have two years to gather cosponsors, late bills are mostly censored. If the
introduction-time profile differs between train and test Congresses, a model learns a mix that the test
set does not have. It also tells us whether a test set must be a **full Congress**: a partial window
(for example session 1 only, as in S4) has a very different bill mix from a full Congress.

**How to read it.** The heatmap shows the share of each Congress's bills introduced in each quarter of
the Congress (Q1 = first three months). The table gives the session-1 share, the share introduced in the
last 90 days, and the KS distance of `day_of_congress` versus the 108-115 pool (KS is an effect size here:
with ~10k bills every p-value is tiny, so we threshold the statistic, not the p-value).

---
## [C-c75-txt] text
```
Introduction timing per Congress (law-capable bills; KS vs 108-115 pool excluding the row's own Congress)
congress        108     109     110     111     112     113     114     115     116     117     118
n_bills       8,621  10,697  11,224  10,770  10,612   9,083  10,223  11,414  14,340  15,242  16,565
session1      0.674   0.647   0.673   0.688   0.558   0.641   0.669   0.627   0.615   0.648   0.641
last90        0.044   0.020   0.012   0.029   0.024   0.036   0.033   0.056   0.060   0.058   0.061
median_day  199.000 246.000 251.000 262.500 306.500 211.000 201.000 253.000 266.000 214.000 252.000
ks_vs_train   0.075   0.046   0.057   0.060   0.123   0.030   0.058   0.038   0.062   0.051   0.060
[PASS] C1: Introduction-day distribution (KS vs 108-115) of every eval Congress within train-to-train noise | metric=0.06177 (threshold outside 95% t prediction interval of ref 108-115 (upper tail; n=8))
Train pool 108-115, task 1 at t0=7: session-1 vs session-2 bills
           bills  pos_rate  any_pos
session1                           
session 2  27794    0.0051   0.2431
session 1  53431    0.0074   0.3324
[PASS] C2: Session-1 share and end-of-Congress (last 90 d) share of eval Congresses within train-to-train noise | metric=1.959 (threshold outside 95% t prediction interval of ref 108-115 (two-sided; n=8), for either share (metric = max |t| over candidates and both shares))
```

---
## [C-c75-o1] chart
```
title: Share of each Congress's law-capable bills introduced per quarter of the Congress
subtitle: Q1 = first 3 months, Q8 = last 3 months; rows are Congresses
encoding: {'x': 'quarter', 'y': 'congress', 'v': 'share', 'fmt': '.0%'}
data (88 rows): quarter | congress | share
  Q1 | 108 | 0.2875536481
  Q2 | 108 | 0.1879132351
  Q3 | 108 | 0.1104280246
  Q4 | 108 | 0.0879248347
  Q5 | 108 | 0.0878088389
  Q6 | 108 | 0.1078761165
  Q7 | 108 | 0.0851409349
  Q8 | 108 | 0.0453543672
  Q1 | 109 | 0.203982425
  Q2 | 109 | 0.2290361784
  Q3 | 109 | 0.1150789941
  Q4 | 109 | 0.0987192671
  Q5 | 109 | 0.0729176405
  Q6 | 109 | 0.1678040572
  Q7 | 109 | 0.0929232495
  Q8 | 109 | 0.0195381883
  Q1 | 110 | 0.2638096935
  Q2 | 110 | 0.1593905916
  Q3 | 110 | 0.1062900927
  Q4 | 110 | 0.1436208125
  Q5 | 110 | 0.0959550962
  Q6 | 110 | 0.1022808268
  Q7 | 110 | 0.1165359943
  Q8 | 110 | 0.0121168924
  Q1 | 111 | 0.2640668524
  Q2 | 111 | 0.1595171773
  Q3 | 111 | 0.0897864438
  Q4 | 111 | 0.1751160631
  Q5 | 111 | 0.0797585887
  Q6 | 111 | 0.1028783658
  Q7 | 111 | 0.100185701
  Q8 | 111 | 0.0286908078
  Q1 | 112 | 0.2011873351
  Q2 | 112 | 0.157651715
  Q3 | 112 | 0.0954579721
  Q4 | 112 | 0.1032793064
  Q5 | 112 | 0.0800980023
  Q6 | 112 | 0.2662080663
  Q7 | 112 | 0.0719939691
  Q8 | 112 | 0.0241236336
  Q1 | 113 | 0.2314213366
  Q2 | 113 | 0.2015853793
  Q3 | 113 | 0.1070131014
  Q4 | 113 | 0.1010679291
  Q5 | 113 | 0.103159749
  Q6 | 113 | 0.1061323351
  Q7 | 113 | 0.1135087526
  Q8 | 113 | 0.0361114169
  Q1 | 114 | 0.2598063191
  Q2 | 114 | 0.2003325834
  Q3 | 114 | 0.114936907
  Q4 | 114 | 0.0936124425
  Q5 | 114 | 0.0869607747
  Q6 | 114 | 0.1106328866
  Q7 | 114 | 0.1007532036
  Q8 | 114 | 0.0329648831
  Q1 | 115 | 0.24820396
  Q2 | 115 | 0.1730331172
  Q3 | 115 | 0.1036446469
  Q4 | 115 | 0.1028561416
  ... (28 more rows)
```

---
## [C-c76-md] markdown
### C3-C4. Cosponsorship timing and task difficulty

**Why it matters.** Task 1 predicts cosponsors who join in *(t0, t0+30]*. Beyond *how many* cosponsors a
bill gets (label rate, owned by section A: A3/A4), what matters for the window is *when* the later
cosponsors arrive: if in some Congresses non-original cosponsors trickle in over months while in others
they join in the first weeks, the same 30-day window captures a different slice of the cosponsor list.

**How to read it.** The line chart shows, per Congress, the cumulative share of cosponsorships that are
original (lag 0) and that arrive within 7, 30 and 37 days (37 = end of the t0=7 window). C3 tests the
**timing of non-original cosponsorships** (lag > 0) with a KS distance against the 108-115 pool, and judges
it against the leave-one-out spread among the train Congresses themselves (a FAIL means "more different than
train-to-train noise", not just "different"). C4 (INFO) is a cross-reference to section A's label rate.

---
## [C-c77-txt] text
```
Cosponsorship lag per Congress (median_lag and KS over non-original cosponsorships, lag > 0; neg = date before introduction; train rows use leave-one-out pools)
          original  within7  within30  within37  median_lag  late_in_t7_window  ks_late_vs_train
congress                                                                                        
108         0.3890   0.4450    0.5780    0.6090     63.0000             0.2690            0.0360
109         0.3640   0.4300    0.5680    0.5980     63.0000             0.2650            0.0270
110         0.3300   0.4000    0.5440    0.5780     62.0000             0.2650            0.0260
111         0.3410   0.4120    0.5480    0.5780     65.0000             0.2520            0.0140
112         0.3310   0.4030    0.5360    0.5630     70.0000             0.2390            0.0300
113         0.3300   0.4050    0.5500    0.5790     63.0000             0.2600            0.0200
114         0.3590   0.4330    0.5550    0.5790     74.0000             0.2280            0.0370
115         0.3940   0.4580    0.5760    0.6010     75.0000             0.2350            0.0390
116         0.3830   0.4560    0.5860    0.6120     65.0000             0.2530            0.0240
117         0.4600   0.5280    0.6320    0.6560     71.0000             0.2370            0.0410
118         0.4660   0.5210    0.6070    0.6250     92.0000             0.1960            0.0930
[FAIL] C3: Timing of non-original cosponsorships (lag > 0) in eval Congresses within train-to-train variation | metric=0.09342 (threshold outside 95% t prediction interval of ref 108-115 (upper tail; n=8)) [fails for: S1b, S2, S3, S4, S1b+stream, S3+stream, S1pool]
```

---
## [C-c77-o1] chart
```
title: Cumulative share of cosponsorships by lag after introduction, per Congress
subtitle: Law-capable bills, active (non-withdrawn) cosponsorships
encoding: {'x': 'congress', 'y': 'share', 'series': 'share_of', 'yfmt': '.0%', 'ylabel': '', 'xlabel': 'Congress'}
data (44 rows): congress | share | share_of
  108 | 0.3886962321 | original (lag 0)
  109 | 0.3636490456 | original (lag 0)
  110 | 0.3304800431 | original (lag 0)
  111 | 0.3414548614 | original (lag 0)
  112 | 0.3308152232 | original (lag 0)
  113 | 0.3295322337 | original (lag 0)
  114 | 0.358725026 | original (lag 0)
  115 | 0.3938805874 | original (lag 0)
  116 | 0.3827682682 | original (lag 0)
  117 | 0.4604211065 | original (lag 0)
  118 | 0.4664646596 | original (lag 0)
  108 | 0.4447482494 | <= 7 d
  109 | 0.4302571762 | <= 7 d
  110 | 0.3999944775 | <= 7 d
  111 | 0.4121336745 | <= 7 d
  112 | 0.4026977194 | <= 7 d
  113 | 0.4049408794 | <= 7 d
  114 | 0.432759749 | <= 7 d
  115 | 0.458330638 | <= 7 d
  116 | 0.4556209107 | <= 7 d
  117 | 0.5275367261 | <= 7 d
  118 | 0.5209059543 | <= 7 d
  108 | 0.5775008336 | <= 30 d
  109 | 0.5675764389 | <= 30 d
  110 | 0.543717469 | <= 30 d
  111 | 0.5483187082 | <= 30 d
  112 | 0.5362947979 | <= 30 d
  113 | 0.5495614433 | <= 30 d
  114 | 0.5551372368 | <= 30 d
  115 | 0.575759536 | <= 30 d
  116 | 0.5864560615 | <= 30 d
  117 | 0.6322336828 | <= 30 d
  118 | 0.6071870321 | <= 30 d
  108 | 0.6092447482 | <= 37 d
  109 | 0.5983325401 | <= 37 d
  110 | 0.577549668 | <= 37 d
  111 | 0.5781428082 | <= 37 d
  112 | 0.5627260034 | <= 37 d
  113 | 0.579068098 | <= 37 d
  114 | 0.5790891943 | <= 37 d
  115 | 0.6009228713 | <= 37 d
  116 | 0.611933558 | <= 37 d
  117 | 0.6555962834 | <= 37 d
  118 | 0.6254386847 | <= 37 d
```

---
## [C-c78-txt] text
```
Task-1 difficulty per Congress (pos_rate = mean positives / eligible; any_pos = share of bills with >= 1 positive)
congress       108    109    110    111    112    113    114    115    116    117    118
pos_rate_t0 0.0094 0.0079 0.0091 0.0078 0.0074 0.0098 0.0085 0.0074 0.0084 0.0066 0.0047
pos_rate_t7 0.0078 0.0064 0.0074 0.0062 0.0056 0.0075 0.0065 0.0059 0.0066 0.0049 0.0035
any_pos_t7  0.3092 0.2610 0.3086 0.2980 0.2723 0.3348 0.3221 0.3146 0.3291 0.2984 0.2728
[INFO] C4: Task-1 positive rate per eligible legislator, eval vs train (cross-reference; verdict owned by A3) | metric=0.4746 (threshold INFO (A3 owns the verdict))
```

---
## [C-c79-md] markdown
### C23. Procedural breakpoints inside a Congress (original-cosponsor share and timing)

**Why it matters.** The jump in the original-cosponsor share in 117-118 (table above) may come from a
procedural change (for example electronic cosponsor submission and remote operations in the House in 2020-2021)
rather than from gradual drift. A procedural change is a *break*: it happens on a date, not at a Congress
boundary. If the break falls inside a Congress, that Congress is a mix of two regimes, a validation Congress
can straddle it, and a fit set that ends before it has never seen the new regime. That decides whether a
by-Congress cut is enough or whether a within-Congress cut (or dropping pre-break data) is needed.

**How to read it.** Monthly series by introduction month (law-capable bills, uncensored at t0 = 0): share of
30-day cosponsors that are original, and the median lag of non-original cosponsors joining within 180 days (bills
with a full 180-day window). Each series is de-seasonalised by its month-of-Congress mean (to remove the
start/end-of-Congress cycle). A constant-mean CUSUM ("raw") would call any steady trend a break, because block
permutation destroys trends; the verdict therefore uses the largest level shift BEYOND a linear trend (trend and
step fitted jointly, segments >= 6 months; null = linear trend + 2,000 permutations of 6-month residual blocks,
approximate). C23b then asks whether a split has more test folds on a never-seen post-break regime than
deployment itself would. The chart shows the raw monthly series with Congress boundaries and the trend-adjusted break.

---
## [C-c80-txt] text
```
Largest level shift per monthly series: raw = constant-mean CUSUM on the de-seasonalised series; trend-adjusted (break_month .. p) = step beyond a jointly fitted linear trend (verdict); before/after = raw means; trend_per_yr = slope without a step
       series chamber raw_break  raw_p  trend_per_yr break_month  congress  month_of_congress  before   after  shift_sd      p  inside
 orig_share30   House   2021-01 0.0000        0.0030     2021-01       117                  1  0.6200  0.7130    1.2050 0.0000   False
 orig_share30   Senat   2017-04 0.0000        0.0080     2008-01       110                 13  0.7360  0.7820   -1.5450 0.0000    True
median_lag180   House   2024-01 0.0320        0.0430     2024-01       118                 13 42.0450 49.1670    1.6330 0.0380    True
median_lag180   Senat   2017-11 0.2040        0.1030     2017-11       115                 11 40.7950 46.1110    1.0240 0.0230    True
[PASS] C23: No significant structural break (beyond a linear trend) in original-cosponsor share / cosponsor timing inside a Congress | metric=0 (threshold metric = number of breaks failing the rule; FAIL if any: trend-adjusted break with block-permutation p < 0.0125 (0.05 Bonferroni over 4 series), location identified (raw and trend-adjusted fits within 6 months), >= 3 months inside a Congress)
Test units scored on a post-break regime unseen in their fit set vs the deployment rate 0.10 (first post-break Congresses [117] out of 10 transitions)
        test_units  post_break_unseen  share  expected  binom_p
S1a              1                  1 1.0000    0.1000   0.1000
S1b              1                  0 0.0000    0.1000   1.0000
S1pool           2                  2 1.0000    0.1000   0.0100
S2               7                  1 0.1430    0.1000   0.5220
S3               7                  1 0.1430    0.1000   0.5220
S4               3                  0 0.0000    0.1000   1.0000
C23b sensitivity: A16-style break list [116, 117, 118] (section A (A_ZTAB: Holm-surviving parts labelled 'break', as in A16)); expected rate 0.30 (reference Congresses unbroken) or 1.00 (candidate transitions only)
        test_units  post_break  share  p_rate_lo  p_rate_cand  c23b_narrow  a16_lo  a16_cand
S1a              1           1 1.0000     0.3000       1.0000         True    True      True
S1b              1           1 1.0000     0.3000       1.0000         True    True      True
S1pool           2           2 1.0000     0.0900       1.0000        False    True      True
S2               7           3 0.4290     0.3530       1.0000         True    True      True
S3               7           3 0.4290     0.3530       1.0000         True    True      True
S4               3           3 1.0000     0.0270       1.0000         True   False      True
[FAIL] C23b: Share of test units scored on a post-break regime unseen in training is not above the deployment rate | metric=1 (threshold by_split False if binomial P(>= observed post-break test units | n test units, rate 0.10) < 0.05) [fails for: S1pool]
```

---
## [C-c80-o1] chart
```
title: 
subtitle: 
data (454 rows): t | v | chamber | congress
  2003.042 | 0.5631776941801283 | House | 108
  2003.125 | 0.6947693098573449 | House | 108
  2003.208 | 0.6465721040189125 | House | 108
  2003.292 | 0.7380851063829788 | House | 108
  2003.375 | 0.6949080017115961 | House | 108
  2003.458 | 0.6962199312714776 | House | 108
  2003.542 | 0.7893043827905106 | House | 108
  2003.708 | 0.5195852534562212 | House | 108
  2003.792 | 0.6252999657182037 | House | 108
  2003.875 | 0.7001553599171414 | House | 108
  2003.958 | 1.0 | House | 108
  2004.042 | 0.3770949720670391 | House | 108
  2004.125 | 0.6776807980049875 | House | 108
  2004.208 | 0.6371933827723902 | House | 108
  2004.292 | 0.6295715778474399 | House | 108
  2004.375 | 0.6361646234676007 | House | 108
  2004.458 | 0.5346820809248555 | House | 108
  2004.542 | 0.8626653102746694 | House | 108
  2004.708 | 0.4496359223300971 | House | 108
  2004.792 | 0.9169472502805837 | House | 108
  2004.875 | 0.897196261682243 | House | 108
  2005.042 | 0.4416 | House | 109
  2005.125 | 0.6630026436621678 | House | 109
  2005.208 | 0.7491668128398526 | House | 109
  2005.292 | 0.6380202940567405 | House | 109
  2005.375 | 0.7028504490433425 | House | 109
  2005.458 | 0.6465281574630946 | House | 109
  2005.542 | 0.7404549147034931 | House | 109
  2005.708 | 0.5706674144699944 | House | 109
  2005.792 | 0.5813854653633659 | House | 109
  2005.875 | 0.5304259634888439 | House | 109
  2005.958 | 0.6783132530120481 | House | 109
  2006.125 | 0.6403995560488346 | House | 109
  2006.208 | 0.4876847290640394 | House | 109
  2006.292 | 0.5696757230499562 | House | 109
  2006.375 | 0.49659591509811774 | House | 109
  2006.458 | 0.5970453387671931 | House | 109
  2006.542 | 0.670793472144063 | House | 109
  2006.708 | 0.5903045903045903 | House | 109
  2007.042 | 0.5221346839242168 | House | 110
  2007.125 | 0.589397431675996 | House | 110
  2007.208 | 0.6430963900260513 | House | 110
  2007.292 | 0.4772049424797614 | House | 110
  2007.375 | 0.6482471776589424 | House | 110
  2007.458 | 0.4986321810494902 | House | 110
  2007.542 | 0.5955710955710956 | House | 110
  2007.625 | 0.9627906976744186 | House | 110
  2007.708 | 0.4667328699106256 | House | 110
  2007.792 | 0.5164156044804944 | House | 110
  2007.875 | 0.7341650671785028 | House | 110
  2007.958 | 0.817258883248731 | House | 110
  2008.042 | 0.442719881744272 | House | 110
  2008.125 | 0.5123152709359606 | House | 110
  2008.208 | 0.5142314990512334 | House | 110
  2008.292 | 0.5844042056074766 | House | 110
  2008.375 | 0.5340990667623834 | House | 110
  2008.458 | 0.5436524673133699 | House | 110
  2008.542 | 0.6379310344827587 | House | 110
  2008.625 | 1.0 | House | 110
  2008.708 | 0.6317204301075269 | House | 110
  ... (394 more rows)
```

---
## [C-c81-md] markdown
### C5-C6. Label censoring at the end of each Congress

**Why it matters.** Unpassed bills die when a Congress adjourns, so a bill introduced 20 days before the
end has no full 30-day window (task 1) and one introduced 100 days before the end has no full 180-day
window (task 2). These bills are excluded. If the excluded share differs by Congress, or if the excluded
bills are systematically different (the end-of-Congress rush), the evaluation population differs from
the training population. Note that 119 is censored *much* more because it is in progress.

**How to read it.** The chart shows the censored share per Congress for task 1 (t0=0, t0=7) and task 2.
The table adds the share of evaluation bills each candidate split loses (from `iter_eval_units()`); S4's
test slice is the end of the Congress, so it is hit hardest.

---
## [C-c82-txt] text
```
Censored share per Congress (119 in progress, context only)
congress        108   109   110   111   112   113   114   115   116   117   118   119
t1_t0=0       0.002 0.014 0.005 0.012 0.015 0.014 0.016 0.025 0.022 0.031 0.030 0.000
t1_t0=7       0.002 0.014 0.005 0.020 0.019 0.020 0.021 0.033 0.029 0.039 0.034 0.000
t2_180d       0.126 0.112 0.128 0.129 0.096 0.149 0.125 0.152 0.173 0.150 0.164 0.119
t2_bills_lost 1,088 1,203 1,440 1,388 1,020 1,356 1,275 1,735 2,474 2,289 2,721 2,025

Share of eval bills lost to censoring per split and role (mean over folds; t2_lost = worst fold)
       units      eval_bills             t1_lost        t2_lost       
role    test val        test         val    test    val    test    val
split                                                                 
S1a        1   1 15,242.0000 14,340.0000  0.0390 0.0290  0.1500 0.1730
S1b        1   1 16,565.0000 15,242.0000  0.0340 0.0390  0.1640 0.1500
S1pool     2   1 15,903.5000 14,340.0000  0.0360 0.0290  0.1640 0.1730
S2         7   7 12,497.0000 11,669.1430  0.0280 0.0260  0.1730 0.1730
S3         7   7 12,497.0000 11,669.1430  0.0280 0.0260  0.1730 0.1730
S4         3   3  3,090.6670  3,093.6670  0.1680 0.0000  0.8580 0.0000
S5         1   1 19,423.0000 19,190.0000  0.0230 0.0250  0.1400 0.1390
[PASS] C5: Task-1 censoring (t0=7) drops < 5% of law-capable bills in every Congress 108-118 | metric=0.03864 (threshold max censored share < 0.05) [fails for: S4]
[PASS] C6: Task-2 censored share (180 d) of eval Congresses within train-to-train noise | metric=0.1725 (threshold outside 95% t prediction interval of ref 108-115 (two-sided; n=8))
[FAIL] C6b: Within-Congress split S4: its test slice (last 20% of introductions) keeps >= 50% of bills for task 2 | metric=0.8078 (threshold task-2 censored share of S4 test < 0.50) [fails for: S4]
```

---
## [C-c82-o1] chart
```
title: Share of law-capable bills whose label window runs past the end of the Congress
subtitle: Task 1: 30-day window after t0; task 2: 180-day reporting window
encoding: {'x': 'congress', 'y': 'share', 'series': 'task', 'yfmt': '.0%', 'ylabel': '', 'xlabel': 'Congress'}
data (33 rows): congress | share | task
  108 | 0.0017399374 | t1_t0=0
  109 | 0.0143030756 | t1_t0=0
  110 | 0.0051674982 | t1_t0=0
  111 | 0.0122562674 | t1_t0=0
  112 | 0.0149830381 | t1_t0=0
  113 | 0.014202356 | t1_t0=0
  114 | 0.0157488017 | t1_t0=0
  115 | 0.0254950061 | t1_t0=0
  116 | 0.0218270572 | t1_t0=0
  117 | 0.0310982811 | t1_t0=0
  118 | 0.0300030184 | t1_t0=0
  108 | 0.0017399374 | t1_t0=7
  109 | 0.0143030756 | t1_t0=7
  110 | 0.0051674982 | t1_t0=7
  111 | 0.0198700093 | t1_t0=7
  112 | 0.0185638899 | t1_t0=7
  113 | 0.0202576241 | t1_t0=7
  114 | 0.0213244644 | t1_t0=7
  115 | 0.0332924479 | t1_t0=7
  116 | 0.0290097629 | t1_t0=7
  117 | 0.0386432227 | t1_t0=7
  118 | 0.0335647449 | t1_t0=7
  108 | 0.1262034567 | t2_180d
  109 | 0.1124614378 | t2_180d
  110 | 0.1282965075 | t2_180d
  111 | 0.1288765088 | t2_180d
  112 | 0.0961176027 | t2_180d
  113 | 0.1492898822 | t2_180d
  114 | 0.1247187714 | t2_180d
  115 | 0.152006308 | t2_180d
  116 | 0.1725244073 | t2_180d
  117 | 0.1501771421 | t2_180d
  118 | 0.1642619982 | t2_180d
```

---
## [C-c83-md] markdown
### C7-C10. Party-control regimes and party labels

**Why it matters.** Cosponsorship and committee action depend on who holds the gavel. A model trained only
on Republican-House Congresses may not transfer to a Democratic House. A by-Congress split works best if
each chamber's controlling party in the validation and test Congresses also appears in training (and,
only if it measurably matters, the joint House x Senate regime). If not, the test measures extrapolation
to a new regime, and we should at least report per-regime results.

**How to read it.** The strip shows the majority party per chamber per Congress from the shared
`majority_party()` (point-in-time at mid-Congress, party as of the Congress, independents counted with their
caucus, Senate ties broken by the Vice President's party and cross-checked against the data). Diamonds mark
a control flip versus the previous Congress. C8 checks, for every evaluation unit of the candidate splits, that
each chamber's controlling party appears in training (task 2 depends on the bill's own chamber). C8b asks whether
the *joint* regime (divided vs unified government) changes anything measurable beyond the chamber majority; only
then does joint-regime coverage become a requirement. C9 shows how narrow the margins are and how C8 would change
if a tie went the other way;
C10 checks that per-Congress party labels are used (a time leak otherwise).

---
## [C-c84-txt] text
```
Majority party per chamber (from majority_party(); point-in-time, caucus-adjusted)
chamber  house senate regime  unified_congress  house_flip  senate_flip
congress                                                               
108          R      R  RH/RS              True       False        False
109          R      R  RH/RS              True       False        False
110          D      D  DH/DS              True        True         True
111          D      D  DH/DS              True       False        False
112          R      D  RH/DS             False        True        False
113          R      D  RH/DS             False       False        False
114          R      R  RH/RS              True       False         True
115          R      R  RH/RS              True       False        False
116          D      R  DH/RS             False        True        False
117          D      D  DH/DS              True       False         True
118          R      D  RH/DS             False        True        False
119          R      R  RH/RS              True       False         True
[INFO] C7: Party-control regimes and flips across 108-118 (informational) | 4 regimes, 7 flips
```

---
## [C-c84-o1] chart
```
title: 
subtitle: 
data (24 rows): congress | chamber | majority | seats_D | seats_R | seats_other | n_voting | tie_broken_by | flip
  108 | House | R | 206 | 228 | 0 | 434 | none (seat count) | False
  108 | Senate | R | 49 | 51 | 0 | 100 | none (seat count) | False
  109 | House | R | 203 | 232 | 0 | 435 | none (seat count) | False
  109 | Senate | R | 45 | 55 | 0 | 100 | none (seat count) | False
  110 | House | D | 232 | 200 | 0 | 432 | none (seat count) | True
  110 | Senate | D | 51 | 49 | 0 | 100 | none (seat count) | True
  111 | House | D | 257 | 178 | 0 | 435 | none (seat count) | False
  111 | Senate | D | 60 | 40 | 0 | 100 | none (seat count) | False
  112 | House | R | 192 | 242 | 0 | 434 | none (seat count) | True
  112 | Senate | D | 53 | 47 | 0 | 100 | none (seat count) | False
  113 | House | R | 201 | 233 | 0 | 434 | none (seat count) | False
  113 | Senate | D | 55 | 45 | 0 | 100 | none (seat count) | False
  114 | House | R | 188 | 246 | 0 | 434 | none (seat count) | False
  114 | Senate | R | 46 | 54 | 0 | 100 | none (seat count) | True
  115 | House | R | 193 | 239 | 0 | 432 | none (seat count) | False
  115 | Senate | R | 49 | 51 | 0 | 100 | none (seat count) | False
  116 | House | D | 234 | 197 | 0 | 431 | none (seat count) | True
  116 | Senate | R | 47 | 53 | 0 | 100 | none (seat count) | False
  117 | House | D | 222 | 213 | 0 | 435 | none (seat count) | False
  117 | Senate | D | 50 | 50 | 0 | 100 | VP party (reported-bill sponsors D 65% (agrees)) | True
  118 | House | R | 212 | 220 | 0 | 432 | none (seat count) | True
  118 | Senate | D | 51 | 49 | 0 | 100 | none (seat count) | False
  119 | House | R | 213 | 220 | 0 | 433 | none (seat count) | False
  119 | Senate | R | 47 | 53 | 0 | 100 | none (seat count) | True
```

---
## [C-c85-txt] text
```
Share of evaluation units whose chamber majorities (primary) / joint House x Senate regime (INFO) appear in training
        units  chambers_covered  house_covered  senate_covered  joint_regime_covered
split                                                                               
S1a         2            1.0000         1.0000          1.0000                0.5000
S1b         2            1.0000         1.0000          1.0000                1.0000
S2         14            0.5710         0.8570          0.7140                0.3570
S3         14            1.0000         1.0000          1.0000                0.7140
S1pool      3            1.0000         1.0000          1.0000                0.6670
[PASS] C8: Each eval Congress's House majority and Senate majority both appear in its training Congresses | metric=1 (threshold per-chamber coverage = 1.0 for every eval unit (judged on S1a, S1b, S3; by_split for all)) [fails for: S2]
Divided minus unified government, per chamber (exact permutation over Congress labels; 108-115 has only 2 divided)
chamber quantity congresses  n_divided    diff  p_exact  n_perm
  House rate_maj    108-118          4 -0.0210   0.3000     330
  House rate_maj    108-115          2 -0.0140   0.4640      28
  House   log_or    108-118          4  0.0730   0.8210     330
  House   log_or    108-115          2  0.3590   0.3210      28
  Senat rate_maj    108-118          4 -0.0270   0.2450     330
  Senat rate_maj    108-115          2 -0.0410   0.2140      28
  Senat   log_or    108-118          4 -0.0930   0.6390     330
  Senat   log_or    108-115          2 -0.2650   0.2500      28
[INFO] C8b: Joint House x Senate regime (divided vs unified) changes task-2 rate or majority OR beyond the chamber majority | metric=0.2455 (threshold FAIL only if any exact permutation p < 0.0125 (0.05 Bonferroni over 4 tests, 108-118); else INFO)
val regime = test regime share of folds: {'S1a': 0.0, 'S1b': 0.0, 'S1pool': 0.0, 'S2': 0.29, 'S3': 0.29}; regimes occurring once in 108-118: [(116, 'DH/RS')]
```

---
## [C-c86-txt] text
```
Seat margin (|D - R| caucus-adjusted, voting members at mid-Congress) per chamber
congress  108  109  110  111  112  113  114  115  116  117  118
chamber                                                        
House      22   29   32   79   50   32   58   46   37    9    8
Senat       2   10    2   20    6   10    8    2    6    0    2
ties: 117 Senate: 50-50, VP party (reported-bill sponsors D 65% (agrees))
C8 sensitivity: 117 senate: label D -> regime DH/DS (2x in 108-115); if R: DH/RS (0x in 108-115)
[PASS] C9: Regime labels are robust: every tie-broken chamber is confirmed by the data (committee-reported sponsors) | metric=1 (threshold all ties confirmed by data)
```

---
## [C-c87-txt] text
```
members.party_code (latest) differs from party_code_asof in 29 member-Congress rows (10 members), e.g.:
Alexander 108-108 latest R / as-of D; Amash 112-116 latest L / as-of R; Hall 108-108 latest R / as-of D; Kiley 118-118 latest I / as-of R; Lieberman 108-109 latest ID / as-of D; Manchin 111-118 latest I / as-of D; Mitchell 115-116 latest I / as-of R; Sinema 113-117 latest I / as-of D; Specter 108-110 latest D / as-of R; Van Drew 116-116 latest R / as-of D
bill_cosponsors.party agrees with party_code_asof for 99.87% of member-Congress pairs, with the latest party_code for 99.47%
[PASS] C10: Per-Congress party labels (members.party_code_asof) match the party recorded on that Congress's bills | metric=0.9987 (threshold >= 0.995 agreement and >= latest-party agreement)
```

---
## [C-c88-md] markdown
### C11-C14. Snapshot leakage: which fields are known at t0?

**Why it matters.** The parquet tables are a **snapshot taken years after each Congress ended**. Many
bill-level fields summarize the bill's whole life (latest action, became law, number of actions,
cosponsors, text versions, summaries). Using any of them as a feature at t0 leaks the future, and a
random split (S5) or even a temporal split would look far better than reality. Fields with a timestamp
can be rebuilt "as of t0"; fields without one (policy area, subjects) are only safe if we can show they
are normally assigned before t0.

**How to read it.** C11 is a requirement, not a pass/fail test. The leakage table compares each snapshot field with its value as of t0 = 7 days
(where a timestamped table lets us rebuild it): `share_changed` is the share of law-capable bills whose
snapshot value differs from the t0 value. Then: when are committee referrals and CRS summaries recorded
relative to introduction, and when was each bill last updated relative to the end of its Congress?

---
## [C-c89-txt] text
```
Snapshot fields vs value as of t0=7 days (law-capable bills 108-118)
              field                           encodes  share_changed_after_t0
       n_cosponsors      count incl. later cosponsors                  0.4850
          n_actions               all actions to date                  0.3580
    n_text_versions reported/engrossed/enrolled texts                  0.1190
    latest_action_*    last action of the bill's life                  0.3580
law_type/law_number              became law (outcome)                  0.0320
        n_summaries   CRS summaries (no publish date)                  0.1130
        update_date              record last modified                  0.9320
For law_type: share of bills that became law. For n_summaries: share with > 1 summary (later versions). For update_date: share modified after the Congress ended.
[INFO/REQ] C11: Requirement: rebuild bill-level count/outcome fields as of t0 (snapshot values are post-t0) | metric=0.4846 (threshold INFO (requirement for every split))
```

---
## [C-c90-txt] text
```
Committee referral timing (first 'Referred to' activity, days after introduction)
congress        108    109    110    111    112    113    114    115    116    117    118
has_referral 0.9800 0.9860 0.9810 0.9860 0.9850 0.9830 0.9870 0.9920 0.9930 0.9900 0.9930
by_day0      0.9760 0.9800 0.9780 0.9840 0.9810 0.9820 0.9850 0.9920 0.9910 0.9900 0.9830
by_day7      0.9790 0.9840 0.9800 0.9850 0.9830 0.9820 0.9860 0.9920 0.9920 0.9900 0.9870
median_lag   0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
p99_lag      0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.5500
[PASS] C12: Committee referral is known at t0=0 for >= 95% of law-capable bills in every Congress | metric=0.9762 (threshold min share referred by day 0 >= 0.95)
```

---
## [C-c91-txt] text
```
Introduced-version CRS summary: coverage and last-update lag after introduction (days)
congress             108     109     110     111     112     113     114      115      116      117      118
has_summary       1.0000  1.0000  1.0000  1.0000  1.0000  1.0000  1.0000   0.9980   1.0000   0.9990   0.5730
within7           0.0280  0.1040  0.1450  0.1610  0.2410  0.0950  0.0870   0.0470   0.0260   0.0350   0.0120
within30          0.1500  0.4920  0.4100  0.4970  0.4920  0.4770  0.4000   0.2250   0.1650   0.1480   0.1020
median_lag_days 126.0000 31.0000 40.0000 31.0000 32.0000 33.0000 46.0000 155.0000 183.0000 112.0000 136.0000
[INFO/REQ] C13: Requirement: do not use CRS summary text as a t0 feature (available by t0=7 for too few bills) | metric=0.03503 (threshold requirement (a t0 feature must exist for >= 90% of eval bills at t0; observed max share shown))
```

---
## [C-c92-md] markdown
### C24, C14. Are the labels of the most recent Congresses fully loaded?

**Why it matters.** 118 is the natural test Congress for task 1, and it looks different: lower positive density (A3),
later non-original cosponsors (C3), and many records still being edited (C14). If cosponsor events or committee-report
actions of 117/118 were still missing from the snapshot, those differences would be data latency, not behaviour, and
118 would understate task-1 and task-2 performance. Three tests: (1) internal consistency: the snapshot counts
(bills.n_cosponsors, n_actions, latest_action_date) against the event tables; (2) staleness: events dated after the
record's own update_date; (3) the end-of-Congress tail: if late events were missing, the share of a Congress's
cosponsorships dated in its last 90 days (and of committee reports in its last 180 days) would be too low.

**How to read it.** Mismatch and stale shares should be 0; the tail shares of 117/118 are judged with `rel_to_train`
(lower tail: only a deficit signals missing events). The last line compares, inside 118, how often bills with and
without cosponsor rows were modified after the Congress ended (post-end edits driven by cosponsor loading would differ).

---
## [C-c93-txt] text
```
Label completeness per Congress (law-capable bills; mismatch / stale shares should be 0)
congress                  108    109    110    111    112    113    114    115    116    117    118
cosp_count_mismatch    0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
action_count_mismatch  0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
latest_action_mismatch 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
event_after_update     0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
withdrawn_row_share    0.0012 0.0012 0.0009 0.0008 0.0012 0.0007 0.0007 0.0005 0.0004 0.0004 0.0034
cosp_last90            0.0255 0.0133 0.0045 0.0197 0.0209 0.0293 0.0252 0.0385 0.0442 0.0416 0.0437
report_last180         0.2590 0.2672 0.2447 0.3467 0.2769 0.3423 0.2695 0.3024 0.3377 0.4237 0.4452
118: modified after Congress end / in last 365 d of snapshot, bills with cosponsor rows 0.758 / 0.317 vs without 0.720 / 0.186
[PASS] C24: Labels of the most recent Congresses (cosponsor events, committee reports) are fully loaded | metric=1.211 (threshold consistency/stale shares = 0 in every Congress AND end-of-Congress tail shares not below the PI (outside 95% t prediction interval of ref 108-115 (lower tail; n=8)); metric = min t of the two tails over candidates)
```

---
## [C-c94-txt] text
```
Bill record update_date vs Congress end (snapshot date = max update_date = 2026-10-17)
congress                     108        109        110        111        112        113        114        115      116    117      118       119
after_end                 1.0000     1.0000     1.0000     1.0000     1.0000     1.0000     1.0000     1.0000   1.0000 0.6950   0.7510    0.0000
median_days_after_end 6,586.0000 5,854.0000 5,121.0000 4,391.0000 3,273.0000 2,930.0000 2,199.0000 1,469.0000 667.0000 8.0000 144.0000 -164.0000
updated_last_365d         0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000     0.0000   0.0000 0.0000   0.2920    0.8880
[INFO] C14: Recent-edit share of 118 records (INFO; becomes a split verdict only if C24 finds incomplete labels) | metric=0.292 (threshold INFO unless C24 fails (then: 118 records unsettled = labels may still change))
```

---
## [C-c95-md] markdown
### C15-C16. Data completeness and annotation latency

**Why it matters.** Policy areas, subjects, summaries and committee referrals are added by CRS and
clerks after introduction. If the newest Congress (118, the natural test set) is less annotated than
older ones, a model that uses those features is tested on degraded inputs, and the test score
understates (or misstates) real performance. Congress 119, which is still in progress, lets us measure
how fast annotations arrive: compare coverage for young versus old bills.

**How to read it.** The line chart shows per-Congress coverage of each annotation for law-capable bills.
The second table shows, for Congress 119 only, coverage by bill age at the snapshot date.

---
## [C-c96-txt] text
```
Annotation coverage per Congress (law-capable bills; 119 in progress)
congress       108    109    110    111    112    113    114    115    116    117    118    119
policy_area 1.0000 1.0000 1.0000 0.9990 0.9980 1.0000 0.9960 0.9940 0.9950 0.9960 0.9990 0.9910
subjects    1.0000 1.0000 1.0000 0.9900 0.9940 0.9930 0.9620 0.9640 0.9840 0.6180 0.3720 0.2370
summary     1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 1.0000 0.9990 0.5780 0.3080
referral    0.9800 0.9860 0.9810 0.9860 0.9850 0.9830 0.9870 0.9920 0.9930 0.9900 0.9930 0.9930
[INFO/REQ] C15a: Requirement: legislative subjects are not a t0 feature (coverage of recent Congresses still filling in) | metric=0.3722 (threshold requirement (INFO; subjects are not available at any t0 <= 90 d, C16/C25))
[PASS] C15b: Coverage of t0-usable annotations (referral; policy area for task 2 at t0 >= 30 d) in eval Congresses within train noise | metric=0.9904 (threshold outside 95% t prediction interval of ref 108-115 (lower tail; n=8), per field)
[INFO] C15c: CRS-summary coverage of eval Congresses (INFO: summaries are not a t0 feature) | metric=0.5776 (threshold INFO)
```

---
## [C-c96-o1] chart
```
title: Share of law-capable bills with each annotation, per Congress
subtitle: Coverage in the snapshot; 119 excluded (in progress)
encoding: {'x': 'congress', 'y': 'coverage', 'series': 'annotation', 'yfmt': '.0%', 'ylabel': '', 'xlabel': 'Congress'}
data (44 rows): congress | coverage | annotation
  108 | 1.0 | policy_area
  109 | 1.0 | policy_area
  110 | 1.0 | policy_area
  111 | 0.9993500464 | policy_area
  112 | 0.9980211082 | policy_area
  113 | 0.9995596169 | policy_area
  114 | 0.9958916169 | policy_area
  115 | 0.9939547924 | policy_area
  116 | 0.9950488145 | policy_area
  117 | 0.9963259415 | policy_area
  118 | 0.9990944763 | policy_area
  108 | 0.9998840042 | subjects
  109 | 0.9999065158 | subjects
  110 | 1.0 | subjects
  111 | 0.9896007428 | subjects
  112 | 0.9941575575 | subjects
  113 | 0.9931740614 | subjects
  114 | 0.9616550915 | subjects
  115 | 0.9637287542 | subjects
  116 | 0.9842398884 | subjects
  117 | 0.6181603464 | subjects
  118 | 0.3722306067 | subjects
  108 | 1.0 | summary
  109 | 1.0 | summary
  110 | 1.0 | summary
  111 | 1.0 | summary
  112 | 1.0 | summary
  113 | 0.9998899042 | summary
  114 | 1.0 | summary
  115 | 0.9999123883 | summary
  116 | 1.0 | summary
  117 | 0.9988190526 | summary
  118 | 0.5776033806 | summary
  108 | 0.9803967057 | referral
  109 | 0.9862578293 | referral
  110 | 0.9812009979 | referral
  111 | 0.9859795729 | referral
  112 | 0.9852996608 | referral
  113 | 0.9830452494 | referral
  114 | 0.986794483 | referral
  115 | 0.9922025583 | referral
  116 | 0.9925383543 | referral
  117 | 0.9903555964 | referral
  118 | 0.9928161787 | referral
```

---
## [C-c97-txt] text
```
Congress 119: coverage by bill age at snapshot (2026-10-17)
         bills  policy_area  subjects  summary  referral
age                                                     
0-7         56       0.1430    0.0180   0.0000    0.8210
8-30       585       0.8600    0.0100   0.0460    0.9900
31-90     1123       0.9920    0.0440   0.0850    0.9970
91-180    2160       1.0000    0.1250   0.1100    0.9880
181-365   4046       0.9990    0.1540   0.1630    0.9930
>365      8977       0.9980    0.3420   0.4680    0.9950
[INFO] C16: Annotations exist within 30 days of introduction (119 bills aged 0-30 d vs > 365 d) | max coverage gap 0.426 (summary)
```

---
## [C-c98-md] markdown
### C18. Is the policy area known at prediction time?

**Why it matters.** Sections B and D use the CRS policy area as a t0 feature (D's empirical-Bayes rate per
policy area, B's bill attributes). In the snapshot almost every bill of 108-118 has one, but CRS assigns it
*after* introduction, and the historical assignment date is not in the data. The in-progress Congress 119 is
the only window onto that lag: if bills a few days old often lack a policy area, then a model trained on
complete snapshots sees a feature at t0 that it would not have in deployment.

**How to read it.** Policy-area missingness of 119 bills by age at the snapshot, from the shared
`PA_MISSING_AT_T0` (so sections B, C and D quote the same numbers; 0-7 and 8-30 days matter for t0 = 0 and 7),
and whether the youngest bills were introduced during a recess (floor actions per day from `bill_actions`).
Proxy: CRS's current speed, not the speed in earlier Congresses.

---
## [C-c99-txt] text
```
policy area missing by age at snapshot (shared PA_MISSING_AT_T0, 119 law-capable bills): 0-7d 85.7% (n=56), 8-30d 14.0% (n=585), 31-90d 0.8% (n=1123), 91-180d 0.0% (n=2160), >180d 0.2% (n=13023)
youngest bucket introduced 2026-10-10..2026-10-17: floor actions/day 5.6 (119 median 8-day window 15.6), law-capable introductions/day 7.0 (median 26.4)
[INFO/REQ] C18: Requirement: mask policy area at t0 <= 7 d (present for < 95% of 119 bills aged 0-7 and 8-30 days) | metric=0.1429 (threshold requirement (INFO; the rule binds while availability < 0.95 in either age bucket: binds))
```

---
## [C-c100-md] markdown
### C25. Which prediction time t0 should task 2 use?

**Why it matters.** Task 2 (reported within 180 days) has no fixed prediction time. Whether the policy area, referral
details and early cosponsors are legitimate features depends on it, and that decides whether the latency checks (C15b's
policy-area part, C18, A13 with vs without policy area) constrain the split at all. A later t0 makes more features
available but "spends" part of the label: bills already reported by t0 are no longer a prediction.

**How to read it.** For candidate t0 values: availability at that bill age of each feature. Timestamped features
(referral, cosponsors, actions) use the worst Congress in 108-118; untimestamped ones (policy area, subjects) use 119
bills of about that age at the snapshot (small-sample proxy, n shown); summaries use the update_date lower bound. The
last rows give the share of task-2 positives already reported by t0 (worst Congress), i.e. label lost by waiting.

---
## [C-c101-txt] text
```
Feature availability at bill age t0 (share of law-capable bills) and task-2 label spent by t0
                                                 t0=0 (n119=41)  t0=7 (n119=27)  t0=30 (n119=261)  t0=60 (n119=117)
cosponsors / actions so far (timestamped)                1.0000          1.0000            1.0000            1.0000
referral committee (min 108-118)                         0.9760          0.9790            0.9790            0.9800
CRS summary (lower bound, min 116-118)                   0.0000          0.0120            0.1020            0.1680
policy_area (119 proxy)                                  0.1220          0.2590            0.9690            0.9910
subjects (119 proxy)                                     0.0000          0.1480            0.0270            0.0770
task-2 positives already reported (max 108-118)          0.0020          0.0460            0.2470            0.4410
Share of task-2 positives reported by t0, per Congress
congress    108    109    110    111    112    113    114    115    116    117    118
t0=0     0.0000 0.0000 0.0000 0.0020 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000 0.0000
t0=7     0.0280 0.0460 0.0430 0.0290 0.0270 0.0220 0.0400 0.0150 0.0240 0.0080 0.0250
t0=30    0.2080 0.2270 0.2010 0.2090 0.2030 0.1450 0.2470 0.1820 0.1710 0.1180 0.1490
t0=60    0.4040 0.4410 0.4240 0.3780 0.3940 0.3020 0.4260 0.3950 0.3580 0.3190 0.3550
[INFO/REQ] C25: Requirement: fix task 2's prediction time t0 and use only features available at that age | metric=30 (threshold INFO (design decision; metric = smallest candidate t0 at which policy area is present for >= 95% of 119 bills))
```

---
## [C-c102-md] markdown
### C17, C22. Member turnover and the point-in-time candidate pool

**Why it matters.** The legacy pool per bill is "everyone who served in the chamber that Congress". Members
who died, resigned or were seated mid-Congress are then candidates for bills introduced while they were out of
office: guaranteed negatives that change the positive density and can take top-k slots from real candidates
(a member who resigned in 118 still has a strong 117 history). This is a construction requirement for every
split, not something that separates splits, but its size tells us whether D's backtest and the labels must
switch to the shared point-in-time pool (`window_targets(point_in_time=True)` / `ROSTER_ASOF`).

**How to read it.** C17: share of the legacy pool not in office in the month of `introduced_date + 7` (per
Congress and chamber), and the share of task-1 positives who are outside the point-in-time pool (delegates and
roster approximation errors: unrankable positives). C22: for S3 test folds, Recall@20 of a simple history
ranker (sponsor-cosponsor affinity in the fit Congresses + recent activity) and the positive density, with the
full-Congress roster vs the point-in-time pool.

---
## [C-c103-txt] text
```
Share of the legacy pool not in office at introduced_date + 7 (mean over bills), and task-1 positives outside the pit pool
congress               108    109    110    111    112    113    114    115    116    117    118
out_of_office_House 0.0090 0.0130 0.0270 0.0220 0.0200 0.0180 0.0140 0.0350 0.0190 0.0320 0.0250
out_of_office_Senat 0.0000 0.0140 0.0200 0.0970 0.0220 0.0480 0.0000 0.0480 0.0200 0.0200 0.0480
pos_outside_pit     0.0160 0.0100 0.0110 0.0170 0.0140 0.0130 0.0120 0.0140 0.0200 0.0190 0.0230
of_which_delegates  0.9970 0.9430 1.0000 0.9330 0.9720 0.9570 1.0000 0.9930 0.9980 0.9950 0.8150
[INFO/REQ] C17: Requirement: eligible pool = members in office at introduced_date + t0 (not the full-Congress roster) | metric=0.09655 (threshold INFO (construction requirement for every split; effect quantified in C22))
```

---
## [C-c104-txt] text
```
S3 test folds, t0=7: Recall@20 and positive density, full-Congress roster vs point-in-time pool (oop_in_top20 = out-of-office members in the full-roster top 20, per bill)
 fold chamber  bills  recall_full  oop_in_top20  recall_pit  recall_pit_deleg  dens_full  dens_pit  gain_pit  gain_pit_deleg
c=115   House   2649       0.1974        0.3681      0.1930            0.1999     0.0055    0.0057   -0.0043          0.0026
c=115   Senat    822       0.4620        0.4988      0.4717            0.4717     0.0066    0.0070    0.0097          0.0097
c=118   House   3568       0.1857        0.5751      0.1718            0.1852     0.0035    0.0036   -0.0138         -0.0004
c=118   Senat    800       0.4361        0.3262      0.4418            0.4418     0.0034    0.0036    0.0057          0.0057
[INFO] C22: Point-in-time pool vs full-Congress roster: effect on task-1 Recall@20 and positive density (S3 test folds) | metric=0.00971 (threshold INFO (requirement for every split; metric = max |Recall@20 gain| of pit+delegates vs full roster))
```

---
## [C-c105-md] markdown
### C19-C21. Companion and reintroduced bills across the split boundary

**Why it matters.** Many bills have a twin: an identical House/Senate companion in the same Congress, or
a reintroduction of the same bill from an earlier Congress. Twins share sponsors, cosponsors and fate.
A reintroduction in a *training* Congress is legitimate signal (it was known when the new bill was
introduced). A same-Congress companion in the training portion is legitimate only if it was introduced
earlier; if it was introduced *after* the evaluation bill (possible only under the random split S5), the
model sees the future. Same-Congress companions also show what a by-Congress split forgoes: they are known
at t0 only if the graph includes the evaluation Congress's own history up to t0.

**How to read it.** Twins = same-Congress `bill_related` links (any relationship) plus identical official
titles as introduced (normalized; titles shared by > 30 bills are dropped as generic). For every evaluation
unit: the share of eval bills with a same-Congress twin in the training portion (split into introduced
before vs after the eval bill) and with an identical-title bill in a training Congress (earlier = reintroduction,
later = future leak). Twins are themselves a CRS/clerk annotation assigned after introduction (proxy).

---
## [C-c106-txt] text
```
pairs: 250,731 directed twin links (114,953 same-Congress, 135,778 cross-Congress)
Share of eval bills linked to the training portion, mean over each split's evaluation units
       twin_in_train_before  twin_in_train_after  reintro_in_train  future_congress_twin  twin_known_t0_in_eval_congress
split                                                                                                                   
S1a                  0.0000               0.0000            0.3370                0.0000                          0.3890
S1b                  0.0000               0.0000            0.3640                0.0000                          0.3620
S2                   0.0000               0.0000            0.3250                0.0000                          0.3560
S3                   0.0000               0.0000            0.3270                0.0000                          0.3560
S4                   0.1720               0.0000            0.1500                0.0000                          0.3730
S5                   0.2690               0.2030            0.2500                0.2490                          0.3330
[PASS] C19: Temporal splits (S1-S4) put no twin introduced after the eval bill into training (no future twin leak) | metric=0 (threshold < 1% of eval bills with a later twin in train) [fails for: S5]
[FAIL] C20: Random split S5: eval bills with a later twin (same or later Congress) in training < 1% | metric=0.4519 (threshold < 0.01) [fails for: S5]
[INFO] C21: Same-Congress twin signal a by-Congress split forgoes unless the eval Congress's history up to t0 is in the graph | metric=0.3585 (threshold INFO)
```

---
## [C-c106-o1] chart
```
title: Share of evaluation bills with a twin bill in the training portion, by candidate split
subtitle: Twins = same-Congress related / identical-title bills; mean over each split's val and test units
encoding: {'x': 'split', 'y': 'share', 'series': 'link', 'yfmt': '.0%', 'ylabel': 'share of eval bills'}
data (24 rows): split | share | link
  S1a | 0.0 | same-Congress twin, earlier
  S1b | 0.0 | same-Congress twin, earlier
  S2 | 0.0 | same-Congress twin, earlier
  S3 | 0.0 | same-Congress twin, earlier
  S4 | 0.1723897345 | same-Congress twin, earlier
  S5 | 0.2690581041 | same-Congress twin, earlier
  S1a | 0.0 | same-Congress twin, LATER (leak)
  S1b | 0.0 | same-Congress twin, LATER (leak)
  S2 | 0.0 | same-Congress twin, LATER (leak)
  S3 | 0.0 | same-Congress twin, LATER (leak)
  S4 | 0.0 | same-Congress twin, LATER (leak)
  S5 | 0.2025591601 | same-Congress twin, LATER (leak)
  S1a | 0.3366476723 | reintroduction (earlier Congress)
  S1b | 0.3643770096 | reintroduction (earlier Congress)
  S2 | 0.3245496534 | reintroduction (earlier Congress)
  S3 | 0.3271274664 | reintroduction (earlier Congress)
  S4 | 0.1500538443 | reintroduction (earlier Congress)
  S5 | 0.2498349142 | reintroduction (earlier Congress)
  S1a | 0.0 | twin in LATER Congress (leak)
  S1b | 0.0 | twin in LATER Congress (leak)
  S2 | 0.0 | twin in LATER Congress (leak)
  S3 | 0.0 | twin in LATER Congress (leak)
  S4 | 0.0 | twin in LATER Congress (leak)
  S5 | 0.2492927552 | twin in LATER Congress (leak)
```

---
## [C-c107-md] markdown
### Section C summary

The table below lists every check recorded in this section. FAIL does not automatically rule out a
by-Congress split; read the `detail` column for what each failure implies. Three kinds of result appear:
**split-discriminating** checks with a `by_split` verdict (C1, C2, C3, C5, C6, C6b, C8, C15b, C19, C20, C23b, C24;
C14 only if C24 finds incomplete labels; every split in ALL_SPLITS gets a verdict or a stated reason, and S4 is judged on
the candidate Congresses its slices test wherever the verdict is about that Congress's bills: C3, C15b, C24), **drift / timing** checks (C1-C3, C23: how different the recent Congresses
are and whether changes happen inside a Congress), and **requirements** that hold for every split (C11, C13, C15a,
C17, C18, C25 recorded with `requirement=True`; C10, C12, C22 state the same kind of rule: build features as of t0,
use as-of party, mask policy area at t0 <= 7, fix task 2's t0, use the point-in-time pool incl. delegates).
C24 decides whether 118's differences are data latency (it finds no missing cosponsor events or committee reports),
C8b whether the joint House x Senate regime needs to be covered. C4 is a cross-reference to section A.

---
## [C-c108-txt] text
```
FAIL (4): C3=0.09342, C23b=1, C6b=0.8078, C20=0.4519
PASS (12): C1=0.06177, C2=1.959, C23=0, C5=0.03864, C6=0.1725, C8=1, C9=1, C10=0.9987, C12=0.9762, C24=1.211, C15b=0.9904, C19=0
INFO (8): C4=0.4746, C7=4 regimes, 7 flips, C8b=0.2455, C14=0.292, C15c=0.5776, C16=max coverage gap 0.426 (summary), C22=0.00971, C21=0.3585
REQUIREMENT (6): C11=0.4846, C13=0.03503, C15a=0.3722, C18=0.1429, C25=30, C17=0.09655
Section C: 12 pass, 4 fail, 8 info, 6 requirements (one line per check with its condition is printed where it is recorded above)
```