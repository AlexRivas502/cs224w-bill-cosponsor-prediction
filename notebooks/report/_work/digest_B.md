# Digest: B. Graph and node conditions (section key B)


---
## [B-c42-md] markdown
## B. Graph and node conditions

A GNN does not only see a table of bills: it sees **nodes** (legislators, bills, committees,
subjects, policy areas) and **edges** (cosponsorship, sponsorship, committee referral, subject tags,
related-bill links). A by-Congress split is only trustworthy for a graph model if
(1) the nodes the model must score at test time are known, either from training or from history
inside the test Congress that is available before the prediction time t0 (otherwise learned node
embeddings are useless and the model must be *inductive*), (2) the graph's degree structure and
density look the same across Congresses (message passing aggregates over neighbours, so a shift in
degree changes every representation), (3) the relation schema and vocabularies (committee codes,
subject terms, policy areas) are stable, so a "committee" or "subject" node means the same thing in
train and test, and (4) no edge carries information from after t0 (related-bill links to later
bills, committee edges that record the label, cosponsor edges dated after t0).

**Evaluation pools.** Every pool here comes from the shared `SPLIT_DEFS` / `iter_eval_units()` (the
same units section D backtests): S1a (train 108-115, val 116, test 117), S1b (train 108-116, val 117,
test 118), S2 (sliding, k = `SLIDE_K`) and S3 (rolling, expanding) folds, plus the `S1pool` variant
(val 116, test 117 + 118). Many units share the same (train Congresses, evaluated Congress) pair, so
metrics are computed once per distinct **pool** (ids P01, P02, ...; the mapping to units is printed
once below, later tables are keyed by pool id). Tables show the pools that evaluate 116-118; the
per-split lines summarise all folds. The `+stream` variants (eval-Congress events dated <= intro + t0
allowed) and S4 (within-Congress: prior Congresses + events before the cut date) are evaluated where
the protocol changes the answer (cold start, sponsor history). S5 (random) ignores time and is not
evaluated. Congress 108 is the first in the data, so "never seen before" is **left-censored** there.

**Noise rule (no pool-size bias).** "Differs from train" verdicts use ONE fixed reference: the
leave-one-out spread of the S1a train Congresses 108-115 (`REL_REF`, shared `rel_to_train`), whatever
pool is evaluated. A short S2 pool is not judged against its own 3 Congresses (the max of 3 values is
exceeded by chance about 1/(3+1) = 25% of the time). Because a split has many evaluation units, a split
is judged on the **excess** of failures over the number expected by chance (binomial test: observed
failures vs alpha x tests), not on "any unit fails".

---
## [B-c43-txt] text
```
members (complete Congresses): 6,083 rows, 1,369 legislators; active cosponsorships: 1,920,006; Task-1 positive edges t0=0: 294,865, t0=7: 232,856
16 distinct evaluation pools (35 units of S1a, S1b, S2, S3, S1pool); pool ids used in every table below:
pool   train  eval                                         used_by
 P01 108-110   111                                  S2 val, S3 val
 P02 108-111   112                                 S3 test, S3 val
 P03 109-111   112                                 S2 test, S2 val
 P04 108-112   113                                 S3 test, S3 val
 P05 110-112   113                                 S2 test, S2 val
 P06 108-113   114                                 S3 test, S3 val
 P07 111-113   114                                 S2 test, S2 val
 P08 108-114   115                                 S3 test, S3 val
 P09 112-114   115                                 S2 test, S2 val
 P10 108-115   116            S1a val, S3 test, S3 val, S1pool val
 P11 113-115   116                                 S2 test, S2 val
 P12 108-116   117 S1a test, S1b val, S3 test, S3 val, S1pool test
 P13 114-116   117                                 S2 test, S2 val
 P14 108-117   118                               S1b test, S3 test
 P15 108-116   118                                     S1pool test
 P16 115-117   118                                         S2 test
```

---
## [B-c44-md] markdown
### B1. Legislator node overlap with train (transductive view)

**Why it matters.** Task 1 ranks legislators for a bill. A transductive model (a learned embedding
per legislator, e.g. an embedding table or a GNN with ID features) can only score legislators it saw
in training. A brand-new member falls back to an untrained vector. **This is a model and protocol
requirement, not a split-ranking criterion**: every election replaces a sizeable share of members
(first table), so every by-Congress split has a similar cold-start share (see the per-split line). Whether it hurts
depends on whether the model may use history inside the evaluated Congress, which B1b measures.
**How to read it.** The first table is per Congress against *all* earlier Congresses. The second
table evaluates the candidate pools. "New to chamber" counts legislators who served before, but only
in the other chamber.

---
## [B-c45-txt] text
```
One-step-ahead cold start (legislator never a member in any earlier Congress in the data):
congress                109   110   111   112   113   114   115   116   117   118
members                 546   554   560   552   554   547   561   550   557   554
new_members           0.095 0.130 0.150 0.196 0.173 0.128 0.132 0.182 0.140 0.164
new_to_chamber        0.009 0.007 0.007 0.011 0.014 0.013 0.005 0.009 0.005 0.009
new_active_cosponsors 0.095 0.130 0.150 0.193 0.173 0.128 0.133 0.182 0.139 0.165
new_pos_edges         0.083 0.119 0.121 0.204 0.172 0.115 0.116 0.180 0.127 0.181

Share UNSEEN as a member in the train Congresses, by pool:
pool   train  eval  members  active_cosponsors  pos_edges
 P10 108-115   116   0.1820             0.1820     0.1800
 P11 113-115   116   0.1840             0.1840     0.1840
 P12 108-116   117   0.1400             0.1390     0.1270
 P13 114-116   117   0.1400             0.1390     0.1270
 P14 108-117   118   0.1640             0.1650     0.1810
 P15 108-116   118   0.2870             0.2880     0.3150
 P16 115-117   118   0.1640             0.1650     0.1810
max share of positive edges from unseen legislators over all units: S1a 18.0%, S1b 18.1%, S2 20.5%, S3 20.4%, S1pool 31.5%
[INFO] B1: Share of val/test Task-1 positive edges from legislators unseen in train (transductive-only model) | metric=0.3145 (threshold info: model/protocol requirement)
```

---
## [B-c45-o1] chart
```
title: Legislator cold start per Congress (never seen in any earlier Congress)
subtitle: Share of members / active cosponsors / Task-1 positive edges involving a new legislator; 108 omitted (left-censored)
encoding: {'x': 'congress', 'y': 'share', 'series': 'measure', 'yfmt': '.0%', 'ylabel': 'share new', 'xlabel': ''}
data (40 rows): congress | share | measure
  109 | 0.0950639854 | new_members
  110 | 0.1299638989 | new_members
  111 | 0.1497326203 | new_members
  112 | 0.1956521739 | new_members
  113 | 0.1732851986 | new_members
  114 | 0.1279707495 | new_members
  115 | 0.1319073084 | new_members
  116 | 0.1818181818 | new_members
  117 | 0.1400359066 | new_members
  118 | 0.1636690647 | new_members
  109 | 0.095412844 | new_active_cosponsors
  110 | 0.1299638989 | new_active_cosponsors
  111 | 0.1502683363 | new_active_cosponsors
  112 | 0.1930783242 | new_active_cosponsors
  113 | 0.1727272727 | new_active_cosponsors
  114 | 0.1279707495 | new_active_cosponsors
  115 | 0.1326164875 | new_active_cosponsors
  116 | 0.1818181818 | new_active_cosponsors
  117 | 0.1387387387 | new_active_cosponsors
  118 | 0.164556962 | new_active_cosponsors
  109 | 0.0825230919 | new_pos_edges
  110 | 0.1194866643 | new_pos_edges
  111 | 0.1210099678 | new_pos_edges
  112 | 0.2039174537 | new_pos_edges
  113 | 0.172056338 | new_pos_edges
  114 | 0.1154706321 | new_pos_edges
  115 | 0.1157936508 | new_pos_edges
  116 | 0.1801127738 | new_pos_edges
  117 | 0.1266469749 | new_pos_edges
  118 | 0.1808703727 | new_pos_edges
  109 | 0.0091407678 | new_to_chamber
  110 | 0.0072202166 | new_to_chamber
  111 | 0.0071301248 | new_to_chamber
  112 | 0.0108695652 | new_to_chamber
  113 | 0.0144404332 | new_to_chamber
  114 | 0.012797075 | new_to_chamber
  115 | 0.0053475936 | new_to_chamber
  116 | 0.0090909091 | new_to_chamber
  117 | 0.0053859964 | new_to_chamber
  118 | 0.0089928058 | new_to_chamber
```

---
## [B-c45-o2] chart
```
title: Member overlap: share of column Congress's members who also served in row Congress
subtitle: Overlap decays with distance: a train pool far in the past covers fewer test legislators
encoding: {'x': 'eval_congress', 'y': 'seen_in', 'v': 'share', 'fmt': '.2f'}
data (121 rows): eval_congress | seen_in | share
  108 | 108 | 1.0
  109 | 108 | 0.9047619048
  110 | 108 | 0.785198556
  111 | 108 | 0.6571428571
  112 | 108 | 0.5489130435
  113 | 108 | 0.4314079422
  114 | 108 | 0.3491773309
  115 | 108 | 0.2959001783
  116 | 108 | 0.2363636364
  117 | 108 | 0.197486535
  118 | 108 | 0.1534296029
  108 | 109 | None
  109 | 109 | 1.0
  110 | 109 | 0.8646209386
  111 | 109 | 0.7285714286
  112 | 109 | 0.6141304348
  113 | 109 | 0.4837545126
  114 | 109 | 0.39488117
  115 | 109 | 0.3351158645
  116 | 109 | 0.2690909091
  117 | 109 | 0.2244165171
  118 | 109 | 0.1768953069
  108 | 110 | None
  109 | 110 | None
  110 | 110 | 1.0
  111 | 110 | 0.85
  112 | 110 | 0.6992753623
  113 | 110 | 0.5631768953
  114 | 110 | 0.4680073126
  115 | 110 | 0.4081996435
  116 | 110 | 0.3254545455
  117 | 110 | 0.2782764811
  118 | 110 | 0.2238267148
  108 | 111 | None
  109 | 111 | None
  110 | 111 | None
  111 | 111 | 1.0
  112 | 111 | 0.7916666667
  113 | 111 | 0.6534296029
  114 | 111 | 0.550274223
  115 | 111 | 0.4759358289
  116 | 111 | 0.3818181818
  117 | 111 | 0.328545781
  118 | 111 | 0.2689530686
  108 | 112 | None
  109 | 112 | None
  110 | 112 | None
  111 | 112 | None
  112 | 112 | 1.0
  113 | 112 | 0.8158844765
  114 | 112 | 0.7129798903
  115 | 112 | 0.6024955437
  116 | 112 | 0.48
  117 | 112 | 0.4111310592
  118 | 112 | 0.3285198556
  108 | 113 | None
  109 | 113 | None
  110 | 113 | None
  111 | 113 | None
  112 | 113 | None
  ... (61 more rows)
```

---
## [B-c46-md] markdown
### B1b / B1c. Within-Congress warm-up before t0 (streaming view of cold start) and neighbourhood depth

**Why it matters.** In the realistic streaming setup the model may use every edge of the evaluated
Congress dated **on or before** the prediction time (bill introduction + t0): earlier cosponsorships
and sponsorships by the same legislator. A new member who already cosponsored other bills before t0
is no longer an empty node. But one or two edges are not a neighbourhood: message passing over a node
with 1-2 neighbours gives a representation dominated by those few bills. B1b counts a positive as
warm after >= 1 in-Congress edge; **B1c asks for >= 5** (the low end of typical GraphSAGE-style
neighbour-sampling fan-outs) and also shows >= 10. The verdicts are per protocol: base splits forbid
eval-Congress history (their cold share is B1's train-cold share), `+stream` variants allow it up to
intro + t0, S4 allows events of the Congress dated before its cut date.
**How to read it.** One row per (pool, t0). `cold_train` = positives whose cosponsor has no edge in
the train Congresses; `cold_lt1/5/10` = of all positives, cold in train AND fewer than 1 / 5 / 10
in-Congress edges by intro + t0 (`cold_lt1` is B1b's streaming-cold share); `new_ge1/5/10` = of the
train-cold positives, share with >= 1 / 5 / 10 in-Congress edges by t0; `all_ge5` = of ALL positives,
share whose cosponsor has >= 5 in-Congress edges by t0; `spon_cold` = evaluated bills whose sponsor
has no train edge and no other in-Congress edge by t0.

---
## [B-c47-txt] text
```
pool   train  eval  cold_train  cold_lt1  cold_lt5  cold_lt10  new_ge1  new_ge5  new_ge10  all_ge5  spon_cold
 P10 108-115   116      0.1800    0.0070    0.0140     0.0190   0.9610   0.9210    0.8970   0.9530     0.0000
 P11 113-115   116      0.1840    0.0070    0.0150     0.0190   0.9610   0.9210    0.8980   0.9530     0.0000
 P12 108-116   117      0.1270    0.0020    0.0060     0.0090   0.9800   0.9490    0.9260   0.9670     0.0010
 P13 114-116   117      0.1270    0.0020    0.0060     0.0090   0.9800   0.9490    0.9260   0.9670     0.0010
 P14 108-117   118      0.1840    0.0040    0.0110     0.0180   0.9800   0.9420    0.9050   0.9680     0.0000
 P15 108-116   118      0.3150    0.0040    0.0140     0.0260   0.9870   0.9550    0.9160   0.9680     0.0000
 P16 115-117   118      0.1840    0.0040    0.0110     0.0180   0.9800   0.9420    0.9050   0.9680     0.0000
t0 = 7, same pools: cold_lt1 <= 0.14%, cold_lt5 <= 0.78%, train-cold with >= 5 edges >= 95.0%
per split, max over units at t0=0 (base split | +stream protocol): cold_train S1a 18.0%, S1b 18.4%, S2 20.5%, S3 20.4%, S1pool 31.5% | cold_lt1 S1a 0.70%, S1b 0.36%, S2 0.71%, S3 0.70%, S1pool 0.70% | cold_lt5 S1a 1.43%, S1b 1.07%, S2 1.45%, S3 1.43%, S1pool 1.43%
S4 units (history before the cut date): max cold_lt1 0.66%, cold_lt5 0.66%; min share of train-cold positives with >= 5 in-Congress edges by t0=0: S1a 92.1%, S1b 94.2%, S2 91.2%, S3 91.2%, S1pool 92.1%
[PASS] B1b: With eval-Congress history up to intro + t0 (+stream), few positives involve a legislator with no edge at all | metric=0.0071 (threshold < 0.02 of positives (t0=0, every unit))
[PASS] B1c: Positives from train-cold legislators have a usable in-Congress neighbourhood (>= 5 edges) by intro + t0 | metric=0.0145 (threshold cold in train AND < 5 in-Congress edges at intro + t0: < 0.02 of positives (t0=0, every unit))
```

---
## [B-c47-o1] chart
```
title: Share of Task-1 positives with a cold cosponsor, by how much history is allowed (t0 = 0)
subtitle: Pools evaluating 116-118; streaming history up to intro + t0 removes most cold start, but a 5-edge neighbourhood is a stricter bar
encoding: {'x': 'pool', 'y': 'share', 'series': 'measure', 'yfmt': '.0%', 'ylabel': 'share of positives'}
data (21 rows): pool | share | measure
  P10 108-115->116 | 0.1801127738 | no train edge (base split)
  P11 113-115->116 | 0.183609846 | no train edge (base split)
  P12 108-116->117 | 0.1266469749 | no train edge (base split)
  P13 114-116->117 | 0.1266469749 | no train edge (base split)
  P14 108-117->118 | 0.1839939217 | no train edge (base split)
  P15 108-116->118 | 0.3145076189 | no train edge (base split)
  P16 115-117->118 | 0.1839939217 | no train edge (base split)
  P10 108-115->116 | 0.0142864888 | no train edge, < 5 in-Congress edges by t0
  P11 113-115->116 | 0.0145304706 | no train edge, < 5 in-Congress edges by t0
  P12 108-116->117 | 0.0064053632 | no train edge, < 5 in-Congress edges by t0
  P13 114-116->117 | 0.0064053632 | no train edge, < 5 in-Congress edges by t0
  P14 108-117->118 | 0.0106791609 | no train edge, < 5 in-Congress edges by t0
  P15 108-116->118 | 0.0140559706 | no train edge, < 5 in-Congress edges by t0
  P16 115-117->118 | 0.0106791609 | no train edge, < 5 in-Congress edges by t0
  P10 108-115->116 | 0.0069941444 | no train edge, no in-Congress edge by t0
  P11 113-115->116 | 0.0071296899 | no train edge, no in-Congress edge by t0
  P12 108-116->117 | 0.0024891308 | no train edge, no in-Congress edge by t0
  P13 114-116->117 | 0.0024891308 | no train edge, no in-Congress edge by t0
  P14 108-117->118 | 0.0036300705 | no train edge, no in-Congress edge by t0
  P15 108-116->118 | 0.0040099616 | no train edge, no in-Congress edge by t0
  P16 115-117->118 | 0.0036300705 | no train edge, no in-Congress edge by t0
```

---
## [B-c48-md] markdown
### B2. Sponsor overlap

**Why it matters.** The sponsor is the strongest single signal for who cosponsors (sponsor-cosponsor
ties repeat across bills). Bills whose sponsor is new to the train period have no sponsor history,
so the model sees a cold-start *bill neighbourhood*. New sponsors are mostly freshmen, who
sponsor fewer bills, so the bill-level share is usually smaller than the member-level share. Unlike
B1, this depends on the train window: a short sliding window (S2) forgets sponsors who were absent
in its few Congresses. **How to read it.** Share of law-capable bills (and of Task-1 positives) in the
evaluated Congress whose sponsor was not a member during the train Congresses, and the stricter
version, "never sponsored a law-capable bill in the train Congresses" (`fail` = above the noise bound).
The bound is not a fixed band: it is the 95% upper t prediction interval of the **one-step-ahead**
share (train 108..c-1, evaluate c) over the reference Congresses 109-115, i.e. how many new sponsors an
ordinary election brings. As elsewhere, only units evaluating 116-118 decide a split. The streaming
version is in B1b.

---
## [B-c49-txt] text
```
pool   train  eval  bills  sponsor_not_member  sponsor_never_sponsored  pos_edges_new_sponsor  fail
 P10 108-115   116  14340              0.1300                   0.1320                 0.1480 False
 P11 113-115   116  14340              0.1320                   0.1330                 0.1480 False
 P12 108-116   117  15242              0.0880                   0.0910                 0.1150 False
 P13 114-116   117  15242              0.0880                   0.0920                 0.1150 False
 P14 108-117   118  16565              0.1040                   0.1050                 0.1210 False
 P15 108-116   118  16565              0.2130                   0.2170                 0.2500  True
 P16 115-117   118  16565              0.1040                   0.1060                 0.1210 False
one-step-ahead new-sponsor share (expanding pools): 109=0.051, 110=0.079, 111=0.097, 112=0.126, 113=0.118, 114=0.076, 115=0.089, 116=0.132, 117=0.091, 118=0.105; 95% upper t-PI of ref 109-115 = 0.144 (nested-LOO false alarms 0/7)
max share over units evaluating 116-118 (bills whose sponsor never sponsored in train): S1a 13.2%, S1b 10.5%, S2 13.3%, S3 13.2%, S1pool 21.7%
[FAIL] B2: Few val/test bills have a sponsor with no sponsorship history in train, relative to train-to-train noise | metric=0.2172 (threshold <= 0.144 (outside 95% t prediction interval of ref 109-115 (upper tail; n=7); one-step-ahead share)) [fails for: S1pool]
```

---
## [B-c50-md] markdown
### B3. Degree distribution stationarity

**Why it matters.** GNN layers aggregate over neighbours. If bills get many more cosponsors, or a
few legislators dominate cosponsoring in later Congresses, the scale of aggregated messages and
the base rate of positives per bill both move. A model trained on old degree regimes is then
miscalibrated at test time. **How to read it.** First table: per Congress, per-bill cosponsor counts
(law-capable bills) and per-legislator cosponsorship counts, KS / PSI against the previous Congress
(KS p-values are tiny at this sample size, so read the *statistic*), Gini and top-10% share (tail
concentration). Second table: PSI of each pool's evaluated Congress against that pool's train
Congresses. The noise reference is **fixed** for every pool: the one-vs-rest PSI of each Congress in
108-115 against the other seven, with a 95% upper t prediction interval on the log scale
(`rel_to_train`). `*` marks a PSI above that bound. The split table compares observed failures with
the number expected by chance (5% per test). Degrees count all active cosponsorships (final
snapshot), a proxy for what is seen at prediction time.

---
## [B-c51-txt] text
```
congress  bill_mean  bill_zero  bill_gini  leg_mean  leg_gini  leg_top10  bill_ks  bill_psi  leg_ks  leg_psi
      108    13.9150     0.2810     0.7730  267.8930    0.2890     0.2240      NaN       NaN     NaN      NaN
      109    11.3920     0.3630     0.8050  289.3280    0.2860     0.2180   0.0910    0.0360  0.1010   0.0600
      110    12.9060     0.3020     0.7840  356.5880    0.2750     0.2100   0.0660    0.0180  0.1930   0.1910
      111    11.6500     0.2960     0.7850  321.1500    0.2620     0.2020   0.0240    0.0050  0.1090   0.0720
      112    10.5280     0.3800     0.8020  240.2540    0.2360     0.1970   0.0840    0.0350  0.2940   0.4750
      113    13.3430     0.2310     0.7600  256.0670    0.2520     0.2050   0.1490    0.1060  0.0760   0.0440
      114    12.7050     0.2220     0.7570  286.9870    0.2510     0.2010   0.0140    0.0030  0.1250   0.0910
      115    12.1900     0.2000     0.7550  301.1660    0.3070     0.2220   0.0220    0.0060  0.0860   0.1080
      116    12.6620     0.1850     0.7530  396.4070    0.3270     0.2260   0.0160    0.0020  0.2060   0.2170
      117    11.5450     0.1960     0.7500  385.1490    0.2950     0.2150   0.0180    0.0030  0.0720   0.0660
      118    10.2000     0.1900     0.7400  371.7260    0.2880     0.2130   0.0270    0.0090  0.0420   0.0160

Fixed noise reference (one-vs-rest PSI inside 108-115; 95% upper t-PI, log scale): bill 0.003-0.060 -> bound 0.191 (nested-LOO false alarms 0/8); leg 0.033-0.251 -> bound 0.390 (nested-LOO false alarms 0/8); leg_rel 0.012-0.076 -> bound 0.127 (nested-LOO false alarms 0/8)

PSI of the evaluated Congress vs its pool's train Congresses (* = above the fixed noise bound):
pool   train  eval   bill    leg leg_rel
 P10 108-115   116 0.061  0.353   0.135*
 P11 113-115   116 0.008  0.423*  0.123 
 P12 108-116   117 0.040  0.226   0.017 
 P13 114-116   117 0.003  0.107   0.011 
 P14 108-117   118 0.054  0.151   0.010 
 P15 108-116   118 0.066  0.195   0.012 
 P16 115-117   118 0.012  0.037   0.031 

Per split, all pools: failures vs chance (raw = bill + raw legislator degree; scale-free = bill + legislator degree / Congress mean):
        pools  tests  fails  expected  p_excess  fails_scalefree  p_excess_scalefree
split                                                                               
S1a         2      4      0    0.2000    1.0000                1              0.1850
S1b         2      4      0    0.2000    1.0000                0              1.0000
S2          8     16      2    0.8000    0.1890                1              0.5600
S3          8     16      0    0.8000    1.0000                1              0.5600
S1pool      3      6      0    0.3000    1.0000                1              0.2650
[PASS] B3: Val/test bill and legislator degree distributions are no further from the train pool than train Congresses are from each other | metric=2 (threshold per split: failures (PSI > fixed 108-115 95% upper t-PI) not in excess of chance (binomial p >= 0.05))
```

---
## [B-c51-o1] chart
```
title: Degree concentration per Congress
subtitle: Gini of bill / legislator degree, top-10% legislators' share, share of bills with no cosponsor
encoding: {'x': 'congress', 'y': 'value', 'series': 'measure', 'yfmt': '.2f', 'ylabel': 'value', 'xlabel': ''}
data (44 rows): congress | value | measure
  108 | 0.7726989962 | bill_gini
  109 | 0.8048163746 | bill_gini
  110 | 0.7837316055 | bill_gini
  111 | 0.7846628138 | bill_gini
  112 | 0.8023883228 | bill_gini
  113 | 0.7599334841 | bill_gini
  114 | 0.7567253285 | bill_gini
  115 | 0.7551691299 | bill_gini
  116 | 0.753221473 | bill_gini
  117 | 0.75010331 | bill_gini
  118 | 0.7396284282 | bill_gini
  108 | 0.2893472717 | leg_gini
  109 | 0.2862515988 | leg_gini
  110 | 0.2754002231 | leg_gini
  111 | 0.2624258087 | leg_gini
  112 | 0.2362074053 | leg_gini
  113 | 0.2521039879 | leg_gini
  114 | 0.2506497269 | leg_gini
  115 | 0.3069724154 | leg_gini
  116 | 0.3274872991 | leg_gini
  117 | 0.2952851543 | leg_gini
  118 | 0.288256146 | leg_gini
  108 | 0.2242373091 | leg_top10
  109 | 0.2176701082 | leg_top10
  110 | 0.210007593 | leg_top10
  111 | 0.201841596 | leg_top10
  112 | 0.1969235409 | leg_top10
  113 | 0.205355947 | leg_top10
  114 | 0.2014434776 | leg_top10
  115 | 0.2223090309 | leg_top10
  116 | 0.2261540014 | leg_top10
  117 | 0.2145547434 | leg_top10
  118 | 0.2132070158 | leg_top10
  108 | 0.2814058694 | bill_zero
  109 | 0.3629054875 | bill_zero
  110 | 0.302209551 | bill_zero
  111 | 0.2961002786 | bill_zero
  112 | 0.3797587637 | bill_zero
  113 | 0.231201145 | bill_zero
  114 | 0.2223417783 | bill_zero
  115 | 0.2003679692 | bill_zero
  116 | 0.1852859135 | bill_zero
  117 | 0.1959060491 | bill_zero
  118 | 0.1897977664 | bill_zero
```

---
## [B-c52-md] markdown
### B4. Committee schema stability

**Why it matters.** Bill-committee referral edges are a natural relation for both tasks (the committee
is the gatekeeper for advancement). Committees are renamed, merged, and subcommittees are reshuffled
across Congresses. A committee node that exists only in test has no learned embedding, and a
renamed committee splits one concept into two nodes. **How to read it.** Per Congress: share of
committee codes and of *referral* edges (activity "Referred to") whose code never appeared in an
earlier Congress, split into full committees vs subcommittees. Codes in this data appear to be
normalised to current codes (e.g. 108 rows use today's subcommittee names). That helps stability
but means historical renames are hidden, not absent.

---
## [B-c53-txt] text
```
Referral edges whose committee code is new vs all earlier Congresses:
congress         109   110   111   112   113   114   115   116   117   118
codes            108   115   110   114    99    99    99    97    98    88
new_codes      0.120 0.087 0.073 0.053 0.030 0.000 0.000 0.031 0.020 0.068
new_edges_full 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000 0.000
new_edges_sub  0.082 0.022 0.013 0.038 0.035 0.000 0.000 0.009 0.002 0.006
sub_edge_share 0.296 0.288 0.255 0.337 0.299 0.297 0.259 0.217 0.255 0.191
By pool (116-118): P10 unseen 0.19% (full 0.00%), Jaccard 0.61; P11 unseen 0.19% (full 0.00%), Jaccard 0.91; P12 unseen 0.05% (full 0.00%), Jaccard 0.62; P13 unseen 0.09% (full 0.00%), Jaccard 0.89; P14 unseen 0.11% (full 0.00%), Jaccard 0.51; P15 unseen 0.12% (full 0.00%), Jaccard 0.50; P16 unseen 0.13% (full 0.00%), Jaccard 0.72
[PASS] B4: Val/test committee-referral edges point to committees seen in train (every by-Congress unit) | metric=0.0128 (threshold < 0.05)
```

---
## [B-c54-md] markdown
### B5. Subject and policy-area vocabulary stability and coverage

**Why it matters.** Legislative subjects (CRS index terms) and policy areas are candidate bill
features or bill-subject edges. Two failure modes break a temporal split: (a) the **vocabulary
changes** (a thesaurus revision makes old and new subject nodes disjoint), and (b) **coverage lag**:
CRS assigns subjects after introduction, sometimes much later, so recent Congresses have fewer
tagged bills. Coverage lag means test bills look different from train bills purely because of data
latency. It is also a leakage risk (tags are not timestamped, so they may encode later events).
**How to read it.** Per Congress: vocabulary size, share of subject terms / bill-subject links not
seen in any earlier Congress, Jaccard vs previous Congress, and share of law-capable bills with at
least one subject and with a policy area (final snapshot). The last line estimates policy-area
availability **at t0** from the youngest bills of the in-progress Congress 119. Because subjects are
essentially absent at t0, B5b is recorded as a feature-exclusion **requirement** (every split), not as
a split condition.

---
## [B-c55-txt] text
```
Subject vocabulary and coverage per Congress (119 in progress, context only):
congress                108    109    110   111   112   113   114   115   116    117   118   119
vocab                 4,070  4,050  4,247   957   989   996   980 1,012 1,031  1,053   961   947
subj_coverage         1.000  1.000  1.000 0.990 0.994 0.993 0.962 0.964 0.984  0.618 0.372 0.237
policy_coverage       1.000  1.000  1.000 0.999 0.998 1.000 0.996 0.994 0.995  0.996 0.999 0.991
subj_per_tagged_bill 21.691 18.311 15.434 6.771 6.742 8.385 8.223 8.317 9.664 10.108 8.299 7.552
new_terms               NaN  0.050  0.095 0.099 0.013 0.010 0.000 0.003 0.005  0.001 0.003 0.003
new_links               NaN  0.004  0.012 0.016 0.002 0.002 0.000 0.000 0.000  0.000 0.000 0.007
jaccard_prev            NaN  0.901  0.830 0.194 0.923 0.940 0.933 0.936 0.942  0.937 0.893 0.945
By pool (116-118): P10 unseen links 0.02%, Jaccard 0.21, unseen PA bills 0.00%; P11 unseen links 0.04%, Jaccard 0.95, unseen PA bills 0.00%; P12 unseen links 0.00%, Jaccard 0.22, unseen PA bills 0.00%; P13 unseen links 0.04%, Jaccard 0.93, unseen PA bills 0.11%; P14 unseen links 0.02%, Jaccard 0.20, unseen PA bills 0.00%; P15 unseen links 0.03%, Jaccard 0.20, unseen PA bills 0.00%; P16 unseen links 0.04%, Jaccard 0.87, unseen PA bills 0.00%

At-t0 proxy (in-progress 119): policy area present for 14.3% of bills aged 0-7 d (n=56, shared PA_MISSING_AT_T0); >=1 subject for 1.0% of bills aged 0-30 d (n=732)
[PASS] B5a: Val/test bill-subject links use subject terms seen in train (every by-Congress unit) | metric=0.0161 (threshold < 0.05)
[INFO/REQ] B5b: Requirement: do not use legislative subjects as node / edge features at t0 (not available at prediction time; coverage lags in 117-118) | metric=0.0096 (threshold info: feature-exclusion requirement (all splits))
```

---
## [B-c55-o1] chart
```
title: Subject coverage and vocabulary continuity per Congress
subtitle: Share of law-capable bills with >=1 subject / a policy area; Jaccard of subject vocab vs previous Congress
encoding: {'x': 'congress', 'y': 'share', 'series': 'measure', 'yfmt': '.0%', 'ylabel': 'share', 'xlabel': ''}
data (35 rows): congress | share | measure
  108 | 0.9998840042 | subj_coverage
  109 | 0.9999065158 | subj_coverage
  110 | 1.0 | subj_coverage
  111 | 0.9896007428 | subj_coverage
  112 | 0.9941575575 | subj_coverage
  113 | 0.9931740614 | subj_coverage
  114 | 0.9616550915 | subj_coverage
  115 | 0.9637287542 | subj_coverage
  116 | 0.9842398884 | subj_coverage
  117 | 0.6181603464 | subj_coverage
  118 | 0.3722306067 | subj_coverage
  119 | 0.2369740957 | subj_coverage
  108 | 1.0 | policy_coverage
  109 | 1.0 | policy_coverage
  110 | 1.0 | policy_coverage
  111 | 0.9993500464 | policy_coverage
  112 | 0.9980211082 | policy_coverage
  113 | 0.9995596169 | policy_coverage
  114 | 0.9958916169 | policy_coverage
  115 | 0.9939547924 | policy_coverage
  116 | 0.9950488145 | policy_coverage
  117 | 0.9963259415 | policy_coverage
  118 | 0.9990944763 | policy_coverage
  119 | 0.990558801 | policy_coverage
  109 | 0.9007490637 | jaccard_prev
  110 | 0.8295479603 | jaccard_prev
  111 | 0.1943998164 | jaccard_prev
  112 | 0.9229249012 | jaccard_prev
  113 | 0.9403714565 | jaccard_prev
  114 | 0.9334637965 | jaccard_prev
  115 | 0.9358600583 | jaccard_prev
  116 | 0.9420152091 | jaccard_prev
  117 | 0.936802974 | jaccard_prev
  118 | 0.8928571429 | jaccard_prev
  119 | 0.9449541284 | jaccard_prev
```

---
## [B-c56-md] markdown
### B6. Cross-split edges: related-bill links and within-Congress future links

**Why it matters.** `bill_related` connects bills. In a temporal split, an edge from a train bill to
a bill in a *later* Congress (or a later-introduced bill in the same Congress) lets the model see
the future when building train representations. Edges that cross the val/test boundary leak
test structure into training. Links to *earlier* Congresses are legitimate context. **How to read
it.** For each source Congress: number of links to earlier / same / later Congresses. Second
table: among same-Congress links, the share pointing to a bill introduced *after* the source bill
(these must be dropped or time-filtered when building a snapshot at t0). Caveat: the source stores
almost only same-Congress links, so the cross-Congress count says little about cross-Congress leakage.

---
## [B-c57-txt] text
```
bill_related links by direction of the related bill's Congress, and same-Congress links whose target bill was introduced after the source (119 = in progress):
congress           108   109    110    111    112    113    114    115    116    117    118    119
earlier          0.000 0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000     36    784
same             5,321 9,000 10,590 10,365 10,000 11,654 13,710 14,844 19,294 17,646 15,593 13,173
later            0.000 0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000     20  0.000
later_share      0.000 0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.000  0.001  0.000
to_later_bill    0.390 0.420  0.432  0.439  0.442  0.428  0.435  0.437  0.433  0.429  0.408  0.401
later_beyond_7d  0.307 0.337  0.354  0.358  0.376  0.369  0.379  0.380  0.362  0.354  0.333  0.330
later_beyond_30d 0.214 0.254  0.278  0.284  0.307  0.310  0.320  0.316  0.290  0.289  0.267  0.268
[INFO] B6a: Related-bill links from complete Congresses to later Congresses (cross-split leak) | metric=0.00014 (threshold info: uninformative (source stores same-Congress links))
[FAIL/REQ] B6b: Within-Congress related links are time-safe without filtering | metric=0.4266 (threshold < 0.05 (requirement: filter if it fails))
```

---
## [B-c58-md] markdown
### B7. Committee edges that encode the advancement label

**Why it matters.** `bill_committees` mixes the referral edge (known on day 0-7) with later activity
edges such as "Reported by", "Markup by", "Discharged from", "Hearings by". "Reported by" *is* the
Task 2 label, and markup / hearings are strong post-introduction signals. If the graph builder keeps
all rows as bill-committee edges, every split (temporal or random) leaks the label. **How to read
it.** Activity mix per Congress, the share of referral edges that occur more than 7 days after
introduction (late referrals are not in a t0=7 snapshot), and how strongly having a non-referral edge
predicts the advancement label.

---
## [B-c59-txt] text
```
Committee-edge activity mix per Congress (row shares) and share of referral edges after day 7:
congress                   108   109   110   111   112   113   114   115   116   117   118
markup/hearing/discharge 0.139 0.109 0.132 0.110 0.101 0.122 0.122 0.129 0.104 0.101 0.114
other                    0.004 0.000 0.001 0.004 0.002 0.002 0.001 0.004 0.000 0.000 0.010
referral                 0.782 0.835 0.795 0.830 0.850 0.819 0.811 0.801 0.852 0.861 0.822
reported                 0.075 0.056 0.072 0.056 0.047 0.057 0.065 0.066 0.044 0.038 0.054
referral_after_day7      0.271 0.245 0.192 0.169 0.200 0.206 0.211 0.179 0.133 0.126 0.128
Advancement label rate by whether the bill has any non-referral committee edge: False: n=92,423, rate 0.000; True: n=18,379, rate 0.335
[INFO/REQ] B7: Graph builder must keep only referral committee edges known by t0 (label leakage) | metric=1 (threshold info: must filter)
```

---
## [B-c60-md] markdown
### B8. Network structure stability per Congress

**Why it matters.** Structural summaries of the bill-legislator cosponsor graph (density, mean
degree) and of homophily (share of cosponsorships within the sponsor's party) describe what a GNN
learns to exploit. Rising polarisation changes how much "same party as sponsor" explains, so a model
trained in a bipartisan era over-predicts cross-party cosponsors in a polarised one. **How to read
it.** Per Congress and chamber: density = edges / (bills x members), mean cosponsors per bill,
same-party share (cosponsor party == sponsor party, D/R only). Pool test: the evaluated value is
compared with the **mean of that pool's train Congresses**, scaled by ONE fixed noise SD (the spread of
the same metric over 108-115): t = (value - pool mean) / (SD_ref x sqrt(1 + 1/k)), fail if |t| exceeds
the 95% t quantile (df = 7). The pool sets the centre (a recent sliding pool is centred near the
present), the fixed reference sets the width, so S2 is not penalised for having 3 Congresses. Last
table: for 116-118, the fixed-reference verdict and the trend residual (`trend_z`), which separates a
steady trend (large t, small trend_z) from a break (large trend_z).

---
## [B-c61-txt] text
```
congress                       108     109     110     111     112     113     114     115     116     117     118
                   chamber                                                                                        
density            House    0.0420  0.0358  0.0368  0.0361  0.0308  0.0382  0.0373  0.0340  0.0367  0.0329  0.0286
                   Senate   0.0536  0.0417  0.0545  0.0377  0.0439  0.0555  0.0538  0.0549  0.0584  0.0523  0.0495
mean_cosp_per_bill House   18.6641 15.9702 16.6469 16.2633 13.8567 17.1737 16.6663 15.4905 16.4399 14.9778 12.8776
                   Senate   5.3578  4.2068  5.5609  4.1475  4.4797  5.8285  5.3834  5.7648  5.9568  5.3359  5.1932
same_party         House    0.7192  0.7214  0.7792  0.8068  0.7885  0.7695  0.7743  0.7621  0.8114  0.8486  0.8024
                   Senate   0.6350  0.6497  0.6751  0.7428  0.7754  0.7569  0.7364  0.7218  0.7439  0.7652  0.7747

Pool-centred t (fixed 108-115 SD; * = |t| > 2.36); columns dens / mean / same x House (H) / Senate (S):
pool   train  eval  fails dens_H dens_S mean_H mean_S same_H same_S
 P10 108-115   116      0  +0.1   +1.2   +0.1   +1.2   +1.4   +0.6 
 P11 113-115   116      0  +0.1   +0.4   -0.0   +0.4   +1.2   +0.1 
 P12 108-116   117      1  -1.0   +0.2   -0.9   +0.2   +2.4*  +0.9 
 P13 114-116   117      0  -0.8   -0.4   -0.8   -0.5   +1.9   +0.5 
 P14 108-117   118      0  -2.2   -0.2   -2.3   -0.0   +0.8   +1.0 
 P15 108-116   118      1  -2.3   -0.1   -2.4*  +0.0   +1.0   +1.1 
 P16 115-117   118      0  -1.6   -0.7   -1.7   -0.6   -0.1   +0.5 
Per split, all pools, failures vs chance (5% per test, 6 tests per pool) fails/tests (p_excess): S1a 1/12 (0.46); S1b 1/12 (0.46); S2 0/48 (1.00); S3 1/48 (0.91); S1pool 2/18 (0.23)

Fixed reference 108-115 per Congress: t vs mean / trend_z (T = outside PI but on trend; B = break from trend):
congress                          116          117          118
metric             chamber                                     
density            House    +0.1/+0.8    -1.0/-0.0    -2.3/-0.9
                   Senate   +1.2/+0.5    +0.4/-0.2    +0.0/-0.6
mean_cosp_per_bill House    +0.1/+0.7    -0.9/-0.0  -2.4/-1.0 T
                   Senate   +1.2/+0.4    +0.3/-0.4    +0.1/-0.7
same_party         House    +1.4/+0.5  +2.6/+1.3 T    +1.1/-0.1
                   Senate   +0.6/-0.8    +1.0/-0.6    +1.1/-0.7
[PASS] B8: Val/test graph structure (density, mean degree, homophily per chamber) close to the pool's train mean, relative to fixed train-to-train noise | metric=2 (threshold per split: |t| > 2.36 failures (fixed 108-115 SD, pool-centred) not in excess of chance (binomial p >= 0.05))
```

---
## [B-c61-o1] chart
```
title: Party homophily of cosponsorship per Congress
subtitle: Share of cosponsorships (law-capable bills, D/R only) where the cosponsor is in the sponsor's party
encoding: {'x': 'congress', 'y': 'same_party', 'series': 'chamber', 'yfmt': '.0%', 'ylabel': 'same-party share', 'xlabel': ''}
data (22 rows): congress | same_party | chamber
  108 | 0.7191595386 | House
  108 | 0.6349953546 | Senate
  109 | 0.7213621391 | House
  109 | 0.6496885733 | Senate
  110 | 0.7792460603 | House
  110 | 0.6750725508 | Senate
  111 | 0.8067674719 | House
  111 | 0.7427950311 | Senate
  112 | 0.7885270272 | House
  112 | 0.7754218953 | Senate
  113 | 0.7695345393 | House
  113 | 0.7568590901 | Senate
  114 | 0.7743388445 | House
  114 | 0.7363578343 | Senate
  115 | 0.7620592818 | House
  115 | 0.7217912389 | Senate
  116 | 0.8113620681 | House
  116 | 0.743920512 | Senate
  117 | 0.8485582319 | House
  117 | 0.7651924799 | Senate
  118 | 0.8024258818 | House
  118 | 0.7746538421 | Senate
```

---
## [B-c61-o2] chart
```
title: Density of the bill-legislator cosponsor graph per Congress
subtitle: Active cosponsorships / (law-capable bills x members), by chamber of origin
encoding: {'x': 'congress', 'y': 'density', 'series': 'chamber', 'yfmt': '.4f', 'ylabel': 'density', 'xlabel': ''}
data (22 rows): congress | density | chamber
  108 | 0.0420363545 | House
  108 | 0.0535781605 | Senate
  109 | 0.0358075249 | House
  109 | 0.041651703 | Senate
  110 | 0.0368294262 | House
  110 | 0.0545182953 | Senate
  111 | 0.0360605495 | House
  111 | 0.0377047727 | Senate
  112 | 0.0307926305 | House
  112 | 0.0439185496 | Senate
  113 | 0.0382487828 | House
  113 | 0.0555094943 | Senate
  114 | 0.0372847085 | House
  114 | 0.053833937 | Senate
  115 | 0.0339702871 | House
  115 | 0.0549032623 | Senate
  116 | 0.0366962631 | House
  116 | 0.0584004887 | Senate
  117 | 0.0329182178 | House
  117 | 0.0523128728 | Senate
  118 | 0.0285534749 | House
  118 | 0.0494591183 | Senate
```

---
## [B-c62-md] markdown
### B9. Dyadic (sponsor-cosponsor pair) novelty: val vs its own test

**Why it matters.** Link-prediction models often succeed by memorising recurring pairs: "X always
cosponsors Y's bills". If most test positives are pairs already seen in train, a pair-memorising
baseline will look strong; if most are new pairs, the model must generalise. Each split's **val**
unit should have a share similar to its **own test** unit, or model selection on val may not
transfer. Consecutive Congresses differ through elections, so some val-test gap is unavoidable. The
gap is therefore judged against the **year-to-year swing among train Congresses**: |new_pair(c) -
new_pair(c-1)| for c in 110-115 with expanding pools (fixed reference, 95% upper t-PI). **How to read
it.** First table: per pool, share of positives whose (cosponsor, sponsor) pair never occurred in the
train Congresses, and the share where both legislators were seen (warm nodes, new pair). Second: per
split, val-test gaps vs the noise bound and failures vs chance. Per-fold gaps are kept in `B_PAIR_GAPS`
so the decision section can relate them to D's Task-1 val-test gap.

---
## [B-c63-txt] text
```
pool   train  eval  pos_edges  new_pair  new_pair_warm_nodes
 P10 108-115   116      36888    0.4130               0.1240
 P11 113-115   116      36888    0.4230               0.1310
 P12 108-116   117      30131    0.3300               0.1110
 P13 114-116   117      30131    0.3360               0.1170
 P14 108-117   118      23691    0.3880               0.1190
 P15 108-116   118      23691    0.5830               0.1200
 P16 115-117   118      23691    0.3940               0.1260

Year-to-year swing |new_pair(c) - new_pair(c-1)| (expanding pools), ref 110-115: 110=0.013, 111=0.090, 112=0.135, 113=0.012, 114=0.079, 115=0.025, 116=0.098, 117=0.083, 118=0.057; 95% upper t-PI 0.168 (nested-LOO false alarms 1/6)
|val - test| new-pair share per split vs that bound, max / mean, fails/folds (p_excess): S1a 0.083 / 0.083, 0/1 (1.00); S1b 0.057 / 0.057, 0/1 (1.00); S2 0.144 / 0.072, 0/7 (1.00); S3 0.135 / 0.070, 0/7 (1.00); S1pool 0.028 / 0.028, 0/1 (1.00)
[PASS] B9: Each split's val unit has a new-pair share similar to its own test unit (relative to train year-to-year swing) | metric=0.1439 (threshold per split: gaps above the 110-115 swing bound (0.168, 95% upper t-PI) not in excess of chance)
```

---
## [B-c63-o1] chart
```
title: Share of Task-1 positives that are a new (cosponsor, sponsor) pair, by evaluated Congress
subtitle: Pair unseen in the train Congresses; swings track elections, and every by-Congress split inherits them
encoding: {'x': 'congress', 'y': 'share', 'series': 'pool', 'yfmt': '.0%', 'ylabel': 'new-pair share', 'xlabel': ''}
data (18 rows): congress | share | pool
  109 | 0.3732377248 | expanding 108..c-1
  110 | 0.3862332696 | expanding 108..c-1
  111 | 0.2965519916 | expanding 108..c-1
  112 | 0.4316631689 | expanding 108..c-1
  113 | 0.4198685446 | expanding 108..c-1
  114 | 0.3404480716 | expanding 108..c-1
  115 | 0.3153571429 | expanding 108..c-1
  116 | 0.4129798308 | expanding 108..c-1
  117 | 0.3303574392 | expanding 108..c-1
  118 | 0.3876999705 | expanding 108..c-1
  111 | 0.2965519916 | sliding 3 Congresses
  112 | 0.4404512067 | sliding 3 Congresses
  113 | 0.4283943662 | sliding 3 Congresses
  114 | 0.350727822 | sliding 3 Congresses
  115 | 0.3256746032 | sliding 3 Congresses
  116 | 0.4231728475 | sliding 3 Congresses
  117 | 0.3356012081 | sliding 3 Congresses
  118 | 0.3939048584 | sliding 3 Congresses
```

---
## [B-c64-md] markdown
### B10. As-of-t0 graph completeness

**Why it matters.** A full-graph GNN trained and evaluated on the final snapshot of a Congress sees
every cosponsor edge of that Congress, including the evaluated bill's own future cosponsors (the
Task 1 labels) and the later activity of its neighbours. At prediction time (introduction + t0), only
part of that graph exists. This is the graph-level version of the leakage in B6b / B7: it applies to
every split, so it is a construction requirement (snapshot or temporal-edge masking), not a split
preference. Its size also shows how different the as-of-t0 graph is from the final one. **How to
read it.** For each eval bill (law-capable, uncensored): `visible` = share of its Congress's final
cosponsor edges dated <= introduction + t0. `nbr_deg_after` = share of the final degree of the bill's
legislator neighbours known at t0 (sponsor + cosponsors by t0) that accrues after t0. `hop1_vis` =
(sponsor + cosponsors by t0) / (sponsor + all cosponsors): share of the bill's final 1-hop neighbourhood
that exists at t0; `hop2_vis` = edges of the as-of 1-hop legislators dated <= t0 / final edges of the final
1-hop legislators (2-hop edge mass). Means over bills. The train Congresses (108-115) matter as much as the
eval ones: training examples must be built the same way (B12).

---
## [B-c65-txt] text
```
congress           108   109   110   111   112   113   114   115   116   117   118
              t0                                                                  
visible       0  0.473 0.471 0.474 0.458 0.507 0.464 0.457 0.475 0.497 0.479 0.465
              7  0.485 0.484 0.486 0.466 0.517 0.473 0.467 0.482 0.505 0.487 0.475
nbr_deg_after 0  0.519 0.525 0.517 0.535 0.488 0.528 0.533 0.524 0.501 0.519 0.510
              7  0.506 0.512 0.504 0.525 0.477 0.518 0.523 0.516 0.493 0.510 0.499
hop1_vis      0  0.703 0.741 0.698 0.711 0.735 0.676 0.692 0.706 0.700 0.736 0.745
              7  0.737 0.773 0.739 0.749 0.769 0.722 0.734 0.745 0.742 0.772 0.778
hop2_vis      0  0.354 0.364 0.357 0.347 0.403 0.331 0.337 0.349 0.362 0.364 0.381
              7  0.381 0.392 0.387 0.373 0.427 0.361 0.364 0.373 0.390 0.388 0.405
90th percentile of visible share over bills (t0=0): 108 0.93, 109 0.88, 110 0.92, 111 0.90, 112 0.85, 113 0.91, 114 0.90, 115 0.89, 116 0.89, 117 0.90, 118 0.89
[INFO/REQ] B10: Share of the eval Congress's final cosponsor graph that is NOT available at t0 (full-graph GNN leakage) | metric=0.5197 (threshold info: must time-mask edges)
```

---
## [B-c65-o1] chart
```
title: How much of a Congress's final cosponsor graph exists at prediction time
subtitle: Mean over eval bills: share of the Congress's final edges dated <= intro + t0; share of neighbours' degree accrued after t0
encoding: {'x': 'congress', 'y': 'share', 'series': 'measure', 'yfmt': '.0%', 'ylabel': 'share', 'xlabel': ''}
data (44 rows): congress | share | measure
  108 | 0.4728410227 | final edges visible (t0=0)
  109 | 0.4712713423 | final edges visible (t0=0)
  110 | 0.4738566779 | final edges visible (t0=0)
  111 | 0.458487024 | final edges visible (t0=0)
  112 | 0.5074299588 | final edges visible (t0=0)
  113 | 0.4644792977 | final edges visible (t0=0)
  114 | 0.4571386209 | final edges visible (t0=0)
  115 | 0.4752258794 | final edges visible (t0=0)
  116 | 0.4966173764 | final edges visible (t0=0)
  117 | 0.4791245732 | final edges visible (t0=0)
  118 | 0.4653031078 | final edges visible (t0=0)
  108 | 0.4854480146 | final edges visible (t0=7)
  109 | 0.4837447485 | final edges visible (t0=7)
  110 | 0.4860739043 | final edges visible (t0=7)
  111 | 0.4663212349 | final edges visible (t0=7)
  112 | 0.5167110337 | final edges visible (t0=7)
  113 | 0.4728611442 | final edges visible (t0=7)
  114 | 0.4666407474 | final edges visible (t0=7)
  115 | 0.4824954843 | final edges visible (t0=7)
  116 | 0.5045344036 | final edges visible (t0=7)
  117 | 0.4874180697 | final edges visible (t0=7)
  118 | 0.4748469082 | final edges visible (t0=7)
  108 | 0.5187199518 | neighbour degree after t0 (t0=0)
  109 | 0.5251384472 | neighbour degree after t0 (t0=0)
  110 | 0.5172280495 | neighbour degree after t0 (t0=0)
  111 | 0.5349978195 | neighbour degree after t0 (t0=0)
  112 | 0.487854262 | neighbour degree after t0 (t0=0)
  113 | 0.5279800907 | neighbour degree after t0 (t0=0)
  114 | 0.5332918807 | neighbour degree after t0 (t0=0)
  115 | 0.5243228329 | neighbour degree after t0 (t0=0)
  116 | 0.501179909 | neighbour degree after t0 (t0=0)
  117 | 0.5193130125 | neighbour degree after t0 (t0=0)
  118 | 0.5101076131 | neighbour degree after t0 (t0=0)
  108 | 0.5058583958 | neighbour degree after t0 (t0=7)
  109 | 0.5116485861 | neighbour degree after t0 (t0=7)
  110 | 0.5044387905 | neighbour degree after t0 (t0=7)
  111 | 0.5252532761 | neighbour degree after t0 (t0=7)
  112 | 0.4774964196 | neighbour degree after t0 (t0=7)
  113 | 0.5184824203 | neighbour degree after t0 (t0=7)
  114 | 0.5228711608 | neighbour degree after t0 (t0=7)
  115 | 0.5163045897 | neighbour degree after t0 (t0=7)
  116 | 0.4925612008 | neighbour degree after t0 (t0=7)
  117 | 0.5101565394 | neighbour degree after t0 (t0=7)
  118 | 0.4994487398 | neighbour degree after t0 (t0=7)
```

---
## [B-c66-md] markdown
### B12. Train / eval input consistency for a temporal GNN

**Why it matters.** B10 shows the eval graph must be the as-of-(intro + t0) snapshot. If the model is
nevertheless *trained* on final graphs of the train Congresses (with each training bill's own target
edges still in the message-passing graph), its inputs in training are complete and partly label-bearing,
while at val/test time they are partial and label-free. The model learns weights for features that look
different at test time (a covariate shift created by the pipeline, not by the data), and the val score
understates what a consistent pipeline would reach. This affects every split equally, so it is a
construction **requirement**; its size tells how strictly the training pipeline must replay time.
**How it is measured.** A count-feature logistic ranker (log1p of: the candidate's pair count with the
sponsor in the 3 previous Congresses, its in-Congress pair count with the sponsor, its in-Congress
degree) is trained on sampled bills of the train Congresses of the S2 pools evaluating 116 and 117, with
in-Congress features taken (a) from the final graph including the bill's own edges, (b) from the final
graph minus the bill's own edges, (c) from the as-of-intro snapshot (t0 = 0). All are evaluated on val
bills with snapshot features (the only inputs available at prediction time). `R20_*` = mean per-bill
share of positives in the top 20 of the point-in-time same-chamber pool (sponsor and cosponsors at t0
excluded); `d_*` = R@20 minus the snapshot-trained model's, with a paired bootstrap 95% CI over val
bills. `hist_only` drops the in-Congress features (reference: what in-Congress history adds at all).
Proxy: a 3-feature linear model, not a GNN; a GNN aggregates the same in-Congress structure.

---
## [B-c67-txt] text
```
pool   train  eval  train_bills  val_bills  R20_snapshot  R20_final_minus_target  R20_final_with_target  R20_hist_only
 P11 113-115   116         3000       1000        0.2990                  0.3140                 0.2830         0.2610
 P13 114-116   117         3000       1000        0.3100                  0.3080                 0.2900         0.2690
d_final_minus_target (vs snapshot-trained, 95% CI): P11 +0.015 [-0.001, +0.030]; P13 -0.001 [-0.017, +0.014]; pooled +0.007 [-0.005, +0.017]
d_final_with_target (vs snapshot-trained, 95% CI): P11 -0.016 [-0.038, +0.004]; P13 -0.020 [-0.041, +0.000]; pooled -0.018 [-0.031, -0.003]
d_hist_only (vs snapshot-trained, 95% CI): P11 -0.037 [-0.053, -0.022]; P13 -0.041 [-0.055, -0.027]; pooled -0.039 [-0.050, -0.028]
weights hist / in-Congress pair / degree (mean over folds): final_with_target -0.11/+1.54/+0.11; final_minus_target +0.30/+0.61/+0.45; snapshot +0.53/+0.43/-0.06; hist_only +0.67/+nan/+nan. Train positives' in-Congress pair count: final 3.92 vs snapshot 1.24
[FAIL/REQ] B12: Training inputs match eval inputs: train examples from as-of-(intro + t0) snapshots, target edges excluded from message passing | metric=-0.0198 (threshold FAIL (= the rule is binding) if training on final-graph features lowers val Recall@20 vs snapshot training (paired 95% CI < 0) in any fold or pooled)
```

---
## [B-c67-o1] chart
```
title: Val Recall@20 change vs a ranker trained on as-of-t0 snapshots (eval inputs are always snapshots)
subtitle: Count-feature logistic ranker, S2 pools evaluating 116 / 117, t0 = 0; below 0 = inconsistent training inputs cost recall
encoding: {'x': 'label', 'y': 'delta', 'series': 'variant', 'yfmt': '+.3f', 'ylabel': 'Recall@20 minus snapshot-trained'}
data (6 rows): label | delta | variant
  P11 val 116 | 0.0149742885 | trained on final graph, own edges removed
  P13 val 117 | -0.001410531 | trained on final graph, own edges removed
  P11 val 116 | -0.0161320479 | trained on final graph incl. own edges
  P13 val 117 | -0.0198160261 | trained on final graph incl. own edges
  P11 val 116 | -0.0374925645 | no in-Congress features
  P13 val 117 | -0.0413320602 | no in-Congress features
```

---
## [B-c68-md] markdown
### B11. Node identity across Congresses: chamber and party switchers

**Why it matters.** A legislator keeps one node id across Congresses. When a House member moves to
the Senate, or a member changes party, that node's training neighbourhood comes from the *other*
chamber (whose bills they can no longer cosponsor in the same way) or from the other party's
coalition, while Task 1 ranks only same-chamber legislators and party homophily is strong (B8). A
learned embedding (or sponsor-history feature) then describes the wrong role. If such nodes carry a
visible share of test positives and are ranked much worse, the model needs chamber-specific node
features (or a chamber-switch reset) and the split's test score depends on how many switchers that
Congress happens to have. **How to read it.** Per pool, Task-1 positives (t0 = 0) are classed by
their cosponsor's status in their **most recent train Congress**: `cold` (never a member in train),
`chamber_switch` (served only in the other chamber then), `party_switch` (same chamber, different
`party_code_asof`), `stable`. `other_ch_edges` = for chamber switchers, share of their train
cosponsorships on bills of the other chamber. `R@20` = share of positives of each class that a
`sponsor_hist` ranker (train counts of cosponsoring this sponsor, popularity tie-break, point-in-time
same-chamber pool incl. delegates, sponsor and t0 cosponsors excluded; section D's baseline,
re-implemented per positive) places in the top 20. Computed on the S3 folds and all pools evaluating
116-118. Impact = switcher share x (R@20 stable - R@20 switcher) = the Recall@20 lost overall.

---
## [B-c69-txt] text
```
pool   train  eval  share_chamber_switch  share_party_switch  share_cold  other_ch_edges  R20_stable  R20_switch  R20_cold  n_switch_pos  impact
 P10 108-115   116                0.0080              0.0000      0.1800          1.0000      0.2350      0.0780    0.0000           293  0.0010
 P11 113-115   116                0.0080              0.0000      0.1840          1.0000      0.2510      0.0780    0.0000           293  0.0010
 P12 108-116   117                0.0030              0.0050      0.1270          1.0000      0.2260      0.0410    0.0000           243  0.0010
 P13 114-116   117                0.0030              0.0050      0.1270          1.0000      0.2370      0.0330    0.0000           243  0.0020
 P14 108-117   118                0.0060              0.0010      0.1810          1.0000      0.2450      0.0710    0.0000           168  0.0010
 P15 108-116   118                0.0100              0.0040      0.3150          1.0000      0.2250      0.0760    0.0000           330  0.0020
 P16 115-117   118                0.0060              0.0010      0.1810          1.0000      0.2610      0.0650    0.0000           168  0.0010
S3 folds, all eval Congresses: switcher share of positives 111 0.67%, 112 0.87%, 113 0.80%, 114 0.99%, 115 0.37%, 116 0.79%, 117 0.81%, 118 0.71%; positives in the point-in-time pool >= 99.6%
Pooled S3 R@20 by class: chamber_switch eval 111-115 0.080 (n=810); chamber_switch eval 116-118 0.059 (n=538); cold eval 111-115 0.000 (n=18,212); cold eval 116-118 0.000 (n=14,743); party_switch eval 111-115 0.652 (n=69); party_switch eval 116-118 0.078 (n=166); stable eval 111-115 0.231 (n=106,811); stable eval 116-118 0.235 (n=75,174)
[PASS] B11: Chamber / party switchers (one node id, train neighbourhood from another chamber or party) are a negligible or harmless slice of val/test positives | metric=0.014 (threshold per unit evaluating 116-118: switcher share of positives < 0.02, OR Recall@20 lost = share x (R@20 stable - switcher) < 0.005)
```

---
## [B-c69-o1] chart
```
title: sponsor_hist Recall@20 of Task-1 positives by cosponsor's node status vs the latest train Congress
subtitle: S3 folds, t0 = 0; chamber / party switchers keep one node id whose train history describes another role
encoding: {'x': 'cls', 'y': 'recall20', 'series': 'period', 'yfmt': '.0%', 'ylabel': 'share of positives in top 20'}
data (8 rows): cls | recall20 | period
  stable | 0.2310061698 | eval 111-115
  stable | 0.2346156916 | eval 116-118
  party_switch | 0.652173913 | eval 111-115
  party_switch | 0.078313253 | eval 116-118
  chamber_switch | 0.0802469136 | eval 111-115
  chamber_switch | 0.0594795539 | eval 116-118
  cold | 0.0 | eval 111-115
  cold | 0.0 | eval 116-118
```

---
## [B-c70-md] markdown
### B summary

All graph and node checks recorded in this section. FAIL does not always mean "do not split by
Congress": read the detail column for the remedy (allow within-Congress history up to t0, refit on
val, inductive features, filter edges by time, restrict subjects to a stable vocabulary, normalise
degree features). Checks tagged with a shared `quantity` are deduplicated in the decision section.
Every split-dependent check carries a `by_split` verdict (base splits, `+stream` variants, S1pool and,
where meaningful, S4). Distribution checks (B3, B8, B9) judge each split on failures in excess of chance
against one fixed 108-115 noise reference, so short sliding pools are not penalised for their size.
Every check keys every split in `ALL_SPLITS`; splits a check does not judge get None with a stated reason
(e.g. S5 random, or base splits in B1b / B1c, whose cold share is B1's INFO and is a protocol choice, not a
failure of the split). B11 and section D's D25 share the `node_identity_switch` key, so the decision
section sees one row and a conflict flag if they disagree.
B5b, B6b, B7, B10 and B12 are recorded as **requirements** (`requirement=True`): graph-construction rules every
split must follow (no subject features at t0, time-filter related links, referral-only committee edges,
time-masked cosponsor graph for eval AND training examples, target edges out of message passing). Each leakage path has its own quantity key, so none is hidden behind
another section's check; count split failures from `checks_summary(requirements=False)`.

---
## [B-c71-txt] text
```
id result req  metric                quantity                                                                                                                                           condition
 B1   INFO      0.3145            node_overlap                                                  Share of val/test Task-1 positive edges from legislators unseen in train (transductive-only model)
B1b   PASS      0.0071              cold_start                                       With eval-Congress history up to intro + t0 (+stream), few positives involve a legislator with no edge at all
B1c   PASS      0.0145                     NaN                                            Positives from train-cold legislators have a usable in-Congress neighbourhood (>= 5 edges) by intro + t0
 B2   FAIL      0.2172         sponsor_overlap                                            Few val/test bills have a sponsor with no sponsorship history in train, relative to train-to-train noise
 B3   PASS      2.0000             degree_dist                      Val/test bill and legislator degree distributions are no further from the train pool than train Congresses are from each other
 B4   PASS      0.0128                     NaN                                                        Val/test committee-referral edges point to committees seen in train (every by-Congress unit)
B5a   PASS      0.0161                     NaN                                                                Val/test bill-subject links use subject terms seen in train (every by-Congress unit)
B5b   INFO req  0.0096        subject_coverage             Requirement: do not use legislative subjects as node / edge features at t0 (not available at prediction time; coverage lags in 117-118)
B6a   INFO      0.0001                     NaN                                                                  Related-bill links from complete Congresses to later Congresses (cross-split leak)
B6b   FAIL req  0.4266   leakage_related_links                                                                                       Within-Congress related links are time-safe without filtering
 B7   INFO req  1.0000 leakage_committee_edges                                                                   Graph builder must keep only referral committee edges known by t0 (label leakage)
 B8   PASS      2.0000         graph_structure       Val/test graph structure (density, mean degree, homophily per chamber) close to the pool's train mean, relative to fixed train-to-train noise
 B9   PASS      0.1439                     NaN                                      Each split's val unit has a new-pair share similar to its own test unit (relative to train year-to-year swing)
B10   INFO req  0.5197      leakage_full_graph                                             Share of the eval Congress's final cosponsor graph that is NOT available at t0 (full-graph GNN leakage)
B12   FAIL req -0.0198  train_eval_input_shift                     Training inputs match eval inputs: train examples from as-of-(intro + t0) snapshots, target edges excluded from message passing
B11   PASS      0.0140    node_identity_switch Chamber / party switchers (one node id, train neighbourhood from another chamber or party) are a negligible or harmless slice of val/test positives

B split conditions: 8 pass, 1 fail, 2 info; requirements (all splits): 5 (2 need filtering)
```