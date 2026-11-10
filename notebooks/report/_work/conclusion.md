## Conclusion: the split we will use

### The short answer

**Yes: start with a simple split, then move to a better one.** Choose the simple split so that the better split is just "more of the same". The simple split below is the **last fold** of the better split (fold 117), as long as we also retrain on 108–116 before testing on 117. The code carries over unchanged. Numbers from the single split are for building and debugging. Model comparisons are made only on the full set of folds.

### Step 1: a simple split for building the pipeline and the first baselines

{{SIMPLE_SPLIT_DIAGRAM}}

| Role | Congresses | Years | Law-capable bills |
|---|---|---|---|
| Train | 108–115 | 2003–2018 | 82,644 |
| Validation | 116 | 2019–2020 | 14,340 |
| Test (after retraining on 108–116) | 117 | 2021–2022 | 15,242 |
| Locked final test (never used for tuning) | 118 | 2023–2024 | 16,565 |
| Excluded | 119 | 2025–now (in progress) | n/a |

Not every bill is scored. Bills whose label window runs past the end of the Congress are dropped, and for cosponsor prediction so are bills that gain no new cosponsor in the 30-day label window. That leaves about 4,000–5,000 scored bills per Congress for cosponsor prediction and about 10,000–14,000 for advancement (more in recent Congresses).

**Rules that come with it.** These apply to every split we use, now and later:

1. **Split by Congress, in time order.** Never split randomly.
2. **Build each example's graph as it looked at prediction time.** Use only edges and features dated on or before the bill's introduction date plus *t0* (0 or 7 days for cosponsor prediction, 0 for advancement). About half (52%) of a Congress's final cosponsor graph does not exist yet at that moment ([B10]).
   - For **cosponsor prediction**, this includes earlier cosponsorships from the current Congress.
   - For **advancement**, the main version uses only earlier Congresses plus the bill's own sponsor, referral committees and day-0 cosponsors.
3. **Drop bills whose label window runs past the end of their Congress.** Bills die at adjournment, so for these bills a "no" can never be confirmed.
4. **Use point-in-time facts.** Use each legislator's party as of that Congress, and draw candidates only from members in office at prediction time ([C10], [C17]).
5. **Never use fields written later.** That means legislative subjects, CRS summaries, later text versions, and final counts such as a bill's total number of cosponsors ([C11]). For advancement, also hide policy area at introduction, and in the main version drop related-bill links, which almost only connect bills of the same Congress.

**Why this split. The reasons use only the data, not any model results:**

- **Time order is required.** In a random split, **45%** of evaluation bills have a related or identically titled bill introduced *later* that sits in the training set ([C20]). A time-ordered split has none ([C19]). Knowledge-graph benchmarks show the same problem: Schlichtkrull et al. (2018), the R-GCN paper, following Toutanova and Chen (2015), note that FB15k and WN18 put a fact in training and its reverse in test, so much of the task becomes memorisation. FB15k-237 was built to remove those pairs.
- **Use whole Congresses, not slices of one Congress.** The closest prior work on cosponsorship (Russo et al., ACL 2023) splits *inside* each Congress: the first 60% of the period trains, the next 20% validates, and the last 20% tests. Applied to our advancement label, a within-Congress split fails. Among the last 20% of each Congress's bills, **81%** have a 180-day window that runs past adjournment, so their outcome is incomplete and they must be dropped ([C6b]). Cutting by time, as Russo et al. do, would be even worse, because the last 20% of a Congress is shorter than 180 days. Their label, active vs passive cosponsorship, has no future window, so they don't face this problem. A Congress boundary is a natural cut: every Congress starts with a fresh set of bills.
- **Test on 117 and lock 118.**
  - Both 117 and 118 differ from the training years in ways every forward-in-time split shares: more bills, a lower committee-report rate in the House (the Senate stays within its historical range), and more cosponsors signing on the first day ([A8], [A2], [A4]).
  - 118 also has fewer cosponsors per bill after day 0, and they sign later ([A3], [C3]). So report it separately.
  - Its records are still being maintained: **29%** were edited in the past year ([C14]). No labels are missing ([C24]), but we treat 118's labels as provisional.
  - We therefore test on 117 and keep the most recent complete Congress, 118, as a locked final test. One caveat: 118 already appeared once in this split analysis, as one of the baseline folds. It has never been used to tune a model, and it must stay that way.
- **Exclude 119.** It is still in session, so its labels are not final.
- **Enough data, and each chamber's majority party is represented.**
  - There are 82,644 training bills and about 15,000 each for validation and test.
  - In 116 and 117, the House majority party and the Senate majority party each also hold that chamber in at least one training Congress ([C8]).
  - The exact combination in 116 (Democratic House, Republican Senate) never appears in training. We found no sign that the combination matters beyond each chamber's majority ([C8b]).

**What the simple split cannot do.** Congresses differ from one another by more than random noise ([A2], [A4], [A8]). A single test Congress is one draw, so on its own it cannot show whether a small improvement is real. This is the reason to move on once the pipeline works.

### Step 2: a better split for comparing models, based on the baseline results

**Rolling "next Congress" folds.** Repeat the simple split once for each Congress *c* from 112 to 117:

1. Train on 108 to *c*−2.
2. Tune on *c*−1. For advancement, tune on the mean of the three preceding validation Congresses ([D17]).
3. Retrain on 108 to *c*−1 with the chosen settings.
4. Test on *c*.

Fold 117 is the simple split. Report every fold together with the average. At the very end, run the frozen final model once on **118**, and later check it on the finished part of **119**.

{{ROLLING_DIAGRAM}}

**Why. This is what the baseline runs showed:**

- **One test Congress is too noisy for advancement and imprecise for cosponsor prediction.** A real 0.02 AUROC difference in advancement prediction went unconfirmed in **3 of 7** Congresses when it was judged on a single Congress ([D9]). Six folds help, but by our power calculation they still resolve only fairly large differences in a single comparison: about 0.012–0.015 in Recall@20 and 0.017–0.036 in AUROC (more when several models are compared). So we decide each comparison with a paired t-test on the six per-fold differences, and treat differences the test does not resolve as ties.
- **The previous Congress is a good tuning set for cosponsor prediction.** It picks near-best models ([D11], [D21], [D23]) and predicts the test score level ([D1]). Model selection was tested only without the current-Congress history (next bullet), so it still has to be confirmed with it; the score-level check also passes with it ([D14]). For advancement it does not rank close models well, which is why tuning uses the mean of the three previous validation Congresses ([D17]).
- **The random split really does inflate scores.** On the same bills it adds **+0.075** to Recall@20 ([D7]), which confirms that rejecting it was right.
- **For cosponsor prediction, let the model see the current Congress up to the prediction time.** Earlier cosponsorships in the same Congress raise Recall@20 from **0.259 to 0.296**, higher in all 7 folds ([D13]). These figures use test Congresses 112–118; on the six development folds 112–117 the gain is +0.034. This also fixes the "new legislator" problem. Without it, about 18–20% of the correct answers are legislators the model never saw in training; with it, under 1% are ([B1], [R2]).
- **Keep old Congresses, but down-weight them.** For cosponsor prediction, recency weighting makes the growing window as good as a short sliding window ([D10]). For advancement this is still open, so we run the sliding window as a side check.
- **For advancement, report each Congress separately.** The AUROC level shifts from one Congress to the next ([D2b]), so a single averaged number hides real differences.

### When to switch

1. Build and debug the whole pipeline on the **simple split**: data loading, time-masked graphs, baselines and metrics.
2. Once the baselines reproduce, run the **same code on all six folds**. That costs only more training runs, not new code.
3. From then on, make every model comparison with a **paired test on the six per-fold differences**, reporting each fold and the average. Treat unresolved differences as ties. The simple-split numbers become the "fold 117" row.
4. Run **118 once**, at the end, with everything frozen.

### One paragraph for the paper

> We split the data by Congress in time order. For model development we use rolling-origin evaluation: for each Congress *c* from 112 to 117, we train on Congresses 108 to *c*−2, tune hyperparameters on *c*−1 (for bill advancement, on the mean of the three preceding validation Congresses), retrain on 108 to *c*−1 and test on *c*. We report every fold and decide model comparisons with a paired t-test on the six per-fold differences. Congress 118, which informed only our baseline protocol analysis, is scored once with the frozen configuration, and the ongoing 119th Congress is excluded. Each example's graph contains only information dated on or before its prediction time. For cosponsor prediction this includes earlier cosponsorships in the same Congress; for advancement it is limited to earlier Congresses and the bill's own edges. We use whole Congresses rather than a random or within-Congress split for two reasons. In a random split, 45% of evaluation bills have a related or identically titled bill introduced later that sits in the training set. In a within-Congress split, 81% of the last fifth of each Congress's bills have a 180-day advancement window that runs past adjournment. Our first pipeline runs use a single fold (train 108–115, validate 116, retrain on 108–116, test 117), the last of the six development folds.
