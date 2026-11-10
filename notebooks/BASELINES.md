# Baselines: cosponsor prediction and bill advancement

This file explains what the baselines notebook (`notebooks/baselines.ipynb`) trains, why each choice was made,
and what the results say. Code: the `notebooks/baselines/` package. Long-form design: `baselines/DESIGN.md`.
How to install and run: `BASELINES_RUN.md`. Tuning logs (validation only): `baselines_build/tuning/`.

Citations: L03-L12 are the CS224W lectures (`.claude/cs224w/lectures/`). Papers: Schlichtkrull et al. 2018
(R-GCN), Russo et al. 2023 (ACL, encoder + R-GCN for cosponsorship), Hamilton et al. 2017 (GraphSAGE),
He et al. 2020 (LightGCN), Wu et al. 2019 (SGC), Rossi et al. 2020 (SIGN), Ying et al. 2018 (PinSAGE),
Rendle et al. 2009 (BPR).

---

## 1. The two tasks and the protocol

**Task 1, cosponsor prediction (temporal link prediction).** For every law-capable bill (hr, s, hjres, sjres)
at prediction time t0 days after introduction (t0 = 0 primary, t0 = 7 secondary), rank the eligible legislators
by how likely they are to cosponsor in (t0, t0 + 30]. Eligible means: same chamber, in office at
introduction + t0, not the sponsor, and not already a cosponsor by t0. Metrics: Recall@10, Recall@20 and
NDCG@20 per bill over the **full** eligible pool (no sampled negatives), averaged over bills with at least
one positive and an uncensored window (L11: Recall@K).

**Task 2, bill advancement (node classification).** Is a law-capable bill reported out of committee within 180
days of introduction? Predicted at introduction (t0 = 0) with the policy area hidden. Bills whose 180-day window
passes the end of the Congress are dropped. Metrics: AUROC and average precision (positives are 3-9%).

**Split (from the split study, unchanged).** Rolling origin by Congress. Fold c trains on 108..c-2, validates
on c-1, refits on 108..c-1 with the chosen settings and tests on c, for c = 112..117. Simple mode is fold 117
(the notebook default). Congress 118 is a locked final test behind `LOCKED_118` (off, never used for tuning).
Congress 119 is excluded. Every hyperparameter was chosen on validation Congresses only.

**Deviation (the only one).** The GNNs and PairMLP use supervision Congresses **109**..c-2 (refit 109..c-1).
Congress 108 is history only. The reason: every graph is built from the H = 3 Congresses before the target
Congress, so a Congress-108 graph would have no history at all. That regime never occurs at test time. The
ported heuristics keep the split study's 108..c-2. The tabular references (TabGBDT, TabLogReg) use 109 by
default so that they are paired with the GNNs on the same rows (`RunConfig.tab_sup_start = 108` switches back).

**Protocols.**
- Task 1 uses **+stream** (the split study's recommendation). A bill may use its own Congress's events dated on
  or before its cutoff (introduction + t0, inclusive, day resolution), for example how often the candidate has
  cosponsored this sponsor's bills earlier in the same session.
- Task 2 uses **plain**: earlier Congresses plus the bill's own edges dated on or before the cutoff (sponsor,
  referral committees, day-0 cosponsors). No other bill of the same Congress is used.
- Never used anywhere: legislative subjects, CRS summaries, later text versions, the policy area for Task 2,
  final counts (`n_cosponsors` and so on). Party and eligible pools are point-in-time. `features.FORBIDDEN_PATTERNS`
  is asserted against every feature list.

## 2. Graph construction (why it is leak-free)

One heterogeneous graph per target Congress k (L05 p.66-67: one independent graph per split unit; L12:
seed-time semantics, every row only sees data from before its own timestamp).

| node type | what | features (all point-in-time) |
|---|---|---|
| `leg` | every legislator | party and caucus as of k, chamber, tenure in the window, state embedding, per-Congress sponsoring / cosponsoring rates, cross-party share, sponsor report rate |
| `hbill` | every bill of Congresses k-3..k-1 (finished) | bill type, chamber, final cosponsor count, advancement label, introduced-title TF-IDF/SVD-64 |
| `comm` | 40 full committees + `unk` | chamber, code embedding, referral volume, report rate |
| `tbill` | the law-capable bills of k (the targets) | allowlist `x_b`: type, chamber, day of Congress, sponsor attributes, day-0 cosponsors, referrals dated <= cutoff, introduced-title SVD |

- **History edges** (all finished before Congress k starts, each with a reverse edge): sponsors, original
  cosponsors, later cosponsors, referrals.
- **Target edges point into `tbill` only**: its sponsor, its cosponsors with lag <= t0, its referrals dated
  <= the cutoff. No edge leaves a target bill, so a target bill cannot see another target bill and cannot change
  any legislator's state. Label-window cosponsors are never edges.
- **History depth is constant**: every learned history feature uses the same window k-3..k-1 with half-life-1
  decay and is a per-Congress average, plus a `hist_depth` one-hot. Without this, Congress 109 would see one
  earlier Congress and 117 nine.
- **Pair features phi(b, l)** (Task 1, 22 columns): sponsor-to-candidate history, reciprocity, popularity, 2-hop
  walk mass, co-support with the bill's day-0 cosponsors, same caucus / state, and under +stream the same
  quantities from the bill's own Congress up to its cutoff (exact to the day). This is how +stream enters: the
  encoder graph never holds same-Congress events.
- **Why not PyG neighbour sampling.** `NeighborLoader` / `LinkNeighborLoader` (including temporal sampling)
  need pyg-lib or torch-sparse, which are not installed. Full-batch message passing on these graphs takes about
  0.1 s per layer, and per-target-Congress graphs give the same point-in-time semantics as temporal sampling.

**Checks that run every time** (section 2 of the notebook, `baselines_build/tests/`, 13 tests, all pass):
graph asserts on every graph (no `tbill` source, every edge dated before its cutoff), eligibility, a **stream
truncation test** (deleting every event after a cutoff day changes nothing for bills with that cutoff), with a
negative control (a deliberately leaky feature must fail it), a perturbation test with trained models,
the feature allowlist, history depth, point-in-time caucus, censored bills excluded from report-rate
denominators, label parity with the split notebook, and reproduction of the split study's reference bars
(fold 117: party+pop 0.1905, sponsor_hist 0.2615, rw2hop 0.2629, svd_r512_hl1 0.2832, +stream sponsor_hist
0.3071, eb_rate_t0safe 0.5598, all equal to `digest_D.md` to 5e-4).

**Known small leakage residuals** (none reaches the default models in a harmful way):
1. `leg_static` marks `in_office` and chamber from the Congress-k members table. A legislator who enters in the
   middle of the Congress, or switches chamber, therefore looks "in office" from day 0. This is a node attribute
   only. Eligible pools and labels use the exact point-in-time roster.
2. Cosponsorships that were withdrawn after the cutoff are dropped from the events (inherited from the split
   study). Whether a cosponsorship is later withdrawn is future information, so stream counts and the exclusion
   set are very slightly affected.
3. `phi_norm = "per_congress"` standardises each Congress with its own full-Congress pair distribution, which
   includes later bills' stream features. That makes it transductive, so it is off by default and marked as
   such in `config.py`.

## 3. Models and why they were picked

All three GNNs share the same input encoders, heads, losses, optimiser and selection rule, so the comparisons
are controlled. Each answers one question.

| model | question it answers | key design |
|---|---|---|
| **G1 HeteroSAGE** (main) | Does a learned, inductive, heterogeneous encoder beat strong counting heuristics? | 2 x `HeteroConv(SAGEConv, mean, root weight)`, one conv per relation (11), summed per destination (L04 p.23-24, L09 p.50-54); residual + LayerNorm (L04 skips, L05); DropEdge 0.1 on history edges |
| **G2 R-GCN** | Do relation-specific weights built from shared bases help? | Schlichtkrull et al. 2018, eq. 2 with per-relation mean, eq. 3 basis decomposition (4 bases); L09 p.25-28; the encoder family Russo et al. 2023 used |
| **G3 LightGCN-feat** | Is learned nonlinear message passing needed, or is fixed GCN diffusion enough? | parameter-free `A_hat^j X` with `A_hat = D^-1/2 A D^-1/2` (L11 p.42-51, He et al. 2020), applied to *features* (SGC / SIGN), hop blocks concatenated, then MLPs |
| **PairMLP** | How much do the pair features alone give? | G1 with 0 message-passing layers (same heads, loss, features) |
| **TabGBDT / TabLogReg** (Task 2) | How strong is a tabular model on the bill features `x_b`? | HistGradientBoosting (300 iterations, lr 0.05, 31 leaves, early stopping) / standardised logistic regression |

**Why inductive.** Every target bill is new, and 13% of Congress-117 positives come from legislators with no
cosponsorship in 108..116 (18-20% under the split study's definition, "unseen in the training Congresses").
ID-embedding models (MF, node2vec, plain LightGCN; L02, L11) cannot score them. So every model builds node states
from point-in-time features (PinSAGE's lesson, L11 p.58-59), and nothing is a free per-node ID vector.

**Why GraphSAGE mean + root weight (G1).** The root weight keeps a legislator's own party, chamber and tenure
separate from the neighbourhood (L04 p.18). Mean aggregation keeps a prolific cosponsor with 1,000 history edges
on the same scale as a freshman with 10. Two layers give the receptive field the task needs (legislator ->
bill -> legislator, committee -> bill), the depth both Schlichtkrull et al. and Russo et al. use.
LayerNorm, not BatchNorm, because degree statistics differ between Congress graphs.

**Shared heads.**
- Task 1: `s(b, l) = <q_b, k_l> / 8 + g(phi_bl)`. The bill tower reads [bill ; sponsor] (sponsor conditioning as in
  Russo et al. 2023), the legislator tower reads h_l. A two-tower dot product scores the whole pool with one
  matmul (L11 p.13-16). g is a 22 -> 32 -> 1 MLP over the pair features.
- Task 2: an MLP over [bill state ; sponsor state ; x_b] (L05 node head).

**Rejected models** (from `DESIGN.md` section 1):
- *Plain GCN* (L03): homogeneous, ties self and neighbour weights. The GCN family is still covered (G2 is the
  relational GCN, G3 uses the exact GCN normalisation).
- *GIN* (L06): injective sum aggregation buys WL-level structure discrimination that these feature-rich tasks do
  not need, and sums over heavy-tailed degrees tie the embedding scale to activity volume.
- *GAT / GATv2* (L04 p.26-35): more parameters and seed variance than the budget can tune. The pair features
  already give neighbour importance explicitly. Kept as an experiment (swap `SAGEConv` for `GATv2Conv`).
- *HGT* (L09 p.60-69): over-parameterised for 4 node types and about 7 supervision graphs, and normally used with
  HGSampling. The natural "stronger model" after these baselines.
- *ID-only LightGCN / NGCF / MF / node2vec / TransE-DistMult-ComplEx* (L02, L10, L11): transductive (see above).
  `svd_r512_hl1` represents the family as a reference.
- *PinSAGE as a system, temporal NeighborLoader, TGN / TGAT* (L11 p.58-75, L12): need pyg-lib / torch-sparse.
- *SEAL / NBFNet pair subgraphs (L07), graph transformers (L08)*: about 2M pairs per Congress, or quadratic attention.
- *Russo et al.'s full model*: needs BERT / Bi-LSTM over speeches and full text, and uses a within-Congress split
  that the split study rejects. We keep R-GCN, 2 layers, AdamW and the sponsor conditioning; the introduced-title
  TF-IDF/SVD replaces the text encoder.
- *Global BCE for Task 1* (L11 p.18-22): not personalised. *Focal loss / large pos_weight for Task 2*: the metrics
  are rank-based.

**Heuristic references** (ported verbatim from the split study): popularity, party+pop, sponsor_hist (the bar the
split study recommends), rw2hop (beta 0.3), sh_hl1, svd_r512_hl1 (plain only), and co-support with the day-0
cosponsors. Task 2: prior, chamber_maj, eb_rate_t0safe, eb_committee, and eb_rate (uses the policy area for every
bill, so it is a *leaky* upper reference, never a baseline).

## 4. Losses, negatives, optimiser, schedule, selection

- **Task 1 loss: multi-positive listwise softmax over the full eligible pool of each bill.**
  `L_b = -(1/|P_b|) sum_{p in P_b} [s_p - logaddexp(s_p, LSE_{n in N_b} s_n)]`, other positives excluded from each
  denominator, mean over bills (matching the per-bill Recall@K). This is BPR's per-query principle (L11 p.23-26)
  applied exactly to all ~100 (Senate) or ~440 (House) negatives, so there is no negative sampling at all. The
  pool already contains PinSAGE-style hard negatives (same party, same state; L11 p.68-71). BPR itself
  (`TrainConfig.loss = "bpr"`) was 0.003-0.005 Recall@20 worse on validation for every GNN.
- **Task 2 loss:** BCE with logits, no positive weighting (AUROC / AP are rank-based).
- **Optimiser:** AdamW (as in Russo et al. 2023), lr 3e-3 (G1, G2) or 1e-3 (G3, wider MLP inputs), betas
  (0.9, 0.999), weight decay 1e-4 (none on norms, biases, embeddings), gradient clipping at norm 1.0. The lr sits
  between Russo's 1e-4 (BERT fine-tuning) and Schlichtkrull's 1e-2 (tiny full-batch models).
- **Schedule:** 5% linear warmup, then cosine decay to 0.1 x lr over a fixed horizon of max_epochs x steps per
  epoch (20 epochs for Task 1, 30 for Task 2).
- **Batches:** G1 / G2 / PairMLP: one step = one encoder pass on one Congress graph plus 1024 (Task 1) or 2048
  (Task 2) of its bills. G3 has no per-step message passing, so its batches mix Congresses (256 / 1024 bills).
- **Recency weight** `w_k = 0.5^((k_last - k) / hl)` on supervision Congresses, rescaled to mean 1:
  G3 uses hl = 3; G1, G2 and PairMLP use none (see section 6 for why).
- **Early stopping and refit:** fit on 109..c-2, early-stop on Congress c-1 using a 3-epoch moving average of the
  validation metric (Recall@20 over the full pool / AUROC), patience 4 (Task 1) or 6 (Task 2). The best epoch is
  E*. Then refit from scratch on 109..c-1 with the same schedule horizon, stop after E* epochs, and score the test
  Congress once.
- **Seeds and threads:** `set_seed` seeds Python, NumPy and torch with deterministic algorithms. Results are
  reproducible for a fixed thread count, but **change with `THREADS`** (float reduction order). For example, G1
  Task 1 seed 0 gives validation 0.3771 / test 0.3930 with 8 threads and 0.3723 / 0.3911 with 4 threads.

## 5. Hyperparameters and the tuning evidence

Defaults (`config.APRIORI`) were chosen from the lectures, the papers and the split study, before any score of
this project was seen. A round-1 one-factor-at-a-time search on fold 117 (fit 109-115, validate 116 for Task 1;
mean AUROC over validation 114-116, each fit on 109..v-1, for Task 2) could adopt a change only if it gained at
least D_TOL = 0.005 Recall@20 or 0.01 AUROC. The test Congress was never looked at. Logs:

| file | what |
|---|---|
| `tuning/G1_HeteroSAGE_r1.json` | 18 Task-1 and 7 Task-2 factors for G1 |
| `tuning/G2_R_GCN_r1.json` (+ `rgcn_t*_fold117.json`) | 20 factors + 1 combination, 2-3 seeds for candidates |
| `tuning/G3_LightGCN_feat_r1.json` (+ `lightgcn_t*_fold117.json`) | about 40 fits, 2 seeds for candidates |
| `tuning/g2_r2_raw/` | interrupted round 2 for G2 (validation only, current code): APRIORI 3 seeds, 8 bases, d 64, dropout 0.1, avg_window 3, hl 3, block, 3 layers |
| `tuning/recheck_r1/` | post-fix re-check of G3's Task-2 dropout choice (current code) |

| setting | G1 | G2 | G3 | PairMLP | why / evidence |
|---|---|---|---|---|---|
| hidden size d | 128 | 128 | 128 | 128 | 64 and 256 equal within noise |
| layers / hops | 2 | 2 | K = 2, concat | 0 | receptive field leg -> bill -> leg; 1 layer -0.002, 3 layers no gain |
| dropout | 0.2 | 0.2 | 0.2 (T1), **0.5 (T2)** | 0.2 | G3 Task 2: the only adopted change (see below) |
| DropEdge | 0.1 | 0.2 | - | - | L05 augmentation; Schlichtkrull use 0.2-0.4; 0 / 0.2 / 0.4 all equal |
| bases | - | 4 | - | - | 2 / 8 / block equal on Task 1; 8 bases +0.005 AUROC on Task 2 (near miss) |
| lr | 3e-3 | 3e-3 | 1e-3 | 3e-3 | 1e-3 clearly worse for G1 / G2 |
| weight decay | 1e-4 | 1e-4 | 1e-4 | 1e-4 | 1e-3 / 1e-2 equal or worse |
| batch (bills) | 1024 / 2048 | 1024 / 2048 | 256 / 1024 | 1024 / 2048 | G3 Task 1 batch 1024: -0.010 (overfits before enough steps) |
| history window H | 3 | 3 | 3 | 3 | H = 2 worse (-0.003), H = 4 equal |
| recency half-life | none | none | 3 | none | see section 6 |
| Task-1 loss | softmax | softmax | softmax | softmax | BPR -0.003 to -0.005 |
| title SVD | on | on | on | on | off: equal on Task 2 |

**What the search showed.** The validation surface is flat for every GNN: the two or three seeds of the
defaults differ by 0.001-0.006, and no single factor gained more than that. Only G3's Task-2 dropout 0.5 was
adopted in round 1 (+0.004 AUROC, better in all 6 seed x Congress cells), under an extra rule that a replicated
consistent gain below D_TOL may be adopted. FROZEN = APRIORI everywhere else.

## 6. Changes after the round-1 search, and the post-fix re-check

Two code fixes went in on 2026-11-08 after the round-1 search. Neither changes a chosen setting.

1. **Recency weights were silently ignored.** The step loss was divided by the *weighted* bill count, so for the
   one-Congress steps of G1 / G2 / PairMLP the weight cancelled (`sup_half_life` had no effect). The loss is now
   divided by the unweighted count, so the knob works for every model. G1, G2 and PairMLP carry
   `sup_half_life = None`, which reproduces their round-1 behaviour exactly. G3 mixes Congresses in a batch and
   uses hl = 3, so its optimisation did change.
2. **Task 1 early stopping is smoothed** (`val_smooth = 3`, as Task 2 always was), because the one-epoch
   validation Recall@20 swings by more than D_TOL between epochs.

The round-1 tables in the notebook (sections 4-6) were measured with `val_smooth = 1` and the old normaliser;
they are kept as the record of the search. The re-check under the current code (validation only):

| model | Task 1, val 116 Recall@20, round 1 (old code) | Task 1, val 116, current code (seeds 0 / 1 / 2) | Task 2 re-check | decision |
|---|---|---|---|---|
| G1 HeteroSAGE | 0.3755 / 0.3767 (seeds 0 / 1) | 0.3771 / 0.3722 / 0.3747, mean 0.3746 | - | unchanged |
| G2 R-GCN | 0.3734 (3-seed mean) | 0.3732 / 0.3729 / 0.3767, mean 0.3743 (4 threads: mean 0.3713) | 8 bases vs 4: 0.7961 vs 0.7911 (+0.005, 2 seeds), still a near miss | unchanged |
| G3 LightGCN-feat | 0.3753 / 0.3746 | 0.3708 / 0.3712 / 0.3702, mean 0.3708 | dropout 0.5 vs 0.2: 0.8059 vs 0.8048 (+0.001, 2 seeds; round 1: +0.004) | unchanged |
| PairMLP | - | 0.3631 / 0.3697 / 0.3687, mean 0.3672 | - | unchanged |

Task-1 numbers are from the 3-seed notebook run (8 threads); Task-2 re-checks use the mean AUROC over validation
114-116 (`tuning/recheck_r1/`, `tuning/g2_r2_raw/`). G1 and G2 move by about one seed sd, as expected from
the smoothing change. G3 lost about 0.004 on validation, because it is the one model whose optimisation changed
with the normaliser fix, while its test score rose by about 0.01. That is a reminder that validation-to-test
transfer at this resolution is noisy. The interrupted round-2 search for G2 (`tuning/g2_r2_raw/`: d 64, dropout 0.1,
`avg_window` 3, hl 3, block decomposition, 3 layers, 12-epoch horizon) found nothing outside the APRIORI seed range
(0.3691-0.3750), and the user stopped tuning, so nothing was adopted. The round-2 knob `TrainConfig.avg_window`
(SWA-style tail averaging) is complete and tested but off by default.

## 7. Results (fold 117: validation 116, test 117)

Two runs of the notebook, both in simple mode with default settings and 8 threads, measured on 2026-11-09:
- the default single-seed run, `baselines_build/out/final_simple`, 12.8 minutes (re-run in the final review as
  `baselines_build/out/review_final`: 11.7 minutes, every number identical);
- a 3-seed run, `baselines_build/out/final_simple_3seeds`, cached in `baselines_results/reference_fold117_3seeds/`.
  Section 8 of the notebook prints it next to a single-seed run.

Below, learned models show the **mean +- sd over 3 seeds**, and the test CI is the 95% sponsor-cluster bootstrap
CI averaged over seeds. Heuristics are deterministic.

**Task 1 (t0 = 0, +stream unless marked plain; validation 116: 5,391 bills; test 117: 5,101 bills)**

| model | val Recall@20 | test Recall@20 [95% CI] | test Recall@10 | test NDCG@20 |
|---|---|---|---|---|
| party+pop (plain) | 0.2037 | 0.1905 [0.174, 0.208] | 0.1126 | 0.1152 |
| sponsor_hist (plain) | 0.2542 | 0.2615 [0.245, 0.279] | 0.1727 | 0.1662 |
| rw2hop (plain) | 0.2533 | 0.2629 [0.247, 0.281] | 0.1703 | 0.1649 |
| svd_r512_hl1 (plain) | 0.2757 | 0.2832 [0.265, 0.303] | 0.1906 | 0.1816 |
| sh_hl1 (plain) | 0.2775 | 0.2845 [0.267, 0.304] | 0.1923 | 0.1835 |
| co-support | 0.2686 | 0.2809 [0.264, 0.297] | 0.1855 | 0.1785 |
| sponsor_hist | 0.2884 | 0.3071 [0.291, 0.325] | 0.2035 | 0.1931 |
| rw2hop | 0.2863 | 0.3075 [0.291, 0.325] | 0.1986 | 0.1915 |
| PairMLP | 0.3672 +- 0.0035 | 0.3864 +- 0.0029 [0.369, 0.407] | 0.2514 | 0.2413 |
| G1 HeteroSAGE | 0.3746 +- 0.0024 | 0.3934 +- 0.0032 [0.374, 0.414] | 0.2541 | 0.2446 |
| G2 R-GCN | 0.3743 +- 0.0021 | 0.3935 +- 0.0047 [0.375, 0.415] | 0.2569 | 0.2453 |
| G3 LightGCN-feat | 0.3708 +- 0.0005 | 0.3997 +- 0.0007 [0.381, 0.420] | 0.2604 | 0.2487 |

**Task 2 (t0 = 0, plain, policy area hidden; test 117: 12,953 bills, 2.8% positive)**

| model | val AUROC | test AUROC [95% CI] | test AP |
|---|---|---|---|
| prior | 0.5000 | 0.5000 | 0.0281 |
| chamber_maj | 0.5350 | 0.5449 [0.511, 0.586] | 0.0307 |
| eb_rate_t0safe | 0.5688 | 0.5598 [0.526, 0.598] | 0.0358 |
| eb_committee | 0.7712 | 0.7258 [0.688, 0.761] | 0.0788 |
| eb_rate (leaky reference, uses policy area) | 0.6875 | 0.6924 | 0.0623 |
| TabLogReg (`x_b`) | 0.7706 | 0.7966 [0.765, 0.824] | 0.1281 |
| TabGBDT (`x_b`) | 0.7923 +- 0.0035 | 0.8201 +- 0.0036 [0.792, 0.846] | 0.1515 |
| PairMLP | 0.7923 +- 0.0034 | 0.8014 +- 0.0045 [0.768, 0.829] | 0.1350 |
| G1 HeteroSAGE | 0.8014 +- 0.0041 | 0.8048 +- 0.0050 [0.770, 0.833] | 0.1509 |
| G2 R-GCN | 0.8043 +- 0.0075 | 0.8095 +- 0.0022 [0.776, 0.838] | 0.1543 |
| G3 LightGCN-feat | 0.8013 +- 0.0074 | 0.8252 +- 0.0022 [0.797, 0.850] | 0.1618 |

Note: eb_rate_t0safe (0.5598) reproduces the split study's fold-117 value exactly (`digest_D.md`). It is
0.63-0.67 on folds 112-115 and 0.57 on 116: Congresses 116-117 are simply harder for the policy-area rate.

**Paired test differences over 3 seeds.** The table gives the seed-mean difference a - b, and the widest per-seed
CI. A difference is *real* if the seed mean is at least the MDE (0.01) **and** every seed's CI excludes 0.

| a - b | metric | mean (sd) | widest CI | real? |
|---|---|---|---|---|
| G1 - rw2hop (+stream) | Recall@20 | +0.086 (0.003) | [0.072, 0.100] | yes |
| G2 - rw2hop | Recall@20 | +0.086 (0.005) | [0.070, 0.103] | yes |
| G3 - rw2hop | Recall@20 | +0.092 (0.001) | [0.081, 0.104] | yes |
| G1 - PairMLP | Recall@20 | +0.007 (0.006) | [-0.006, 0.018] | no |
| G2 - PairMLP | Recall@20 | +0.007 (0.005) | [-0.003, 0.019] | no |
| G3 - PairMLP | Recall@20 | +0.013 (0.003) | [0.002, 0.022] | yes (barely) |
| G1 - G2 | Recall@20 | -0.000 (0.007) | [-0.010, 0.015] | no |
| G1 - G3 | Recall@20 | -0.006 (0.003) | [-0.015, 0.004] | no |
| G2 - G3 | Recall@20 | -0.006 (0.004) | [-0.018, 0.005] | no |
| G1 - eb_committee | AUROC | +0.079 (0.005) | [0.055, 0.111] | yes |
| G1 - TabGBDT | AUROC | -0.015 (0.006) | [-0.038, 0.005] | no |
| G2 - TabGBDT | AUROC | -0.011 (0.002) | [-0.029, 0.006] | no |
| G3 - TabGBDT | AUROC | +0.005 (0.006) | [-0.016, 0.028] | no |
| G1 - G3 | AUROC | -0.020 (0.004) | [-0.042, -0.002] | yes |
| G2 - G3 | AUROC | -0.016 (0.004) | [-0.036, 0.004] | no |

**New legislators** (Task 1 test, positives from legislators with no cosponsorship in 108..116: 3,814 of 30,129,
12.7%; the split study's 18-20% counts legislators unseen in the training Congresses of a plain split, a different
definition): hit@20 = 0.242 +- 0.026 (G1), 0.233 +- 0.037 (G2), 0.220 +- 0.017 (G3), 0.228 +- 0.017 (PairMLP),
against 0.05 for sponsor_hist and rw2hop (+stream) and 0 for every plain heuristic. On returning legislators all
learned models score 0.284-0.292.

**What the results mean.**
1. **Every GNN beats every heuristic by a wide margin on Task 1** (+0.086 to +0.092 Recall@20 over the best
   +stream heuristic, about 30% relative; the split study's learned-SVD bar 0.283 is also cleared). Most of that
   gain is already in PairMLP, i.e. in the learned combination of leak-free pair features and point-in-time node
   features.
2. **Message passing adds little on Task 1.** G1 and G2 are +0.007 above PairMLP (not real); only G3 clears the
   MDE (+0.013, barely). The three GNNs cannot be ranked against each other (pairwise differences <= 0.006, each
   CI includes 0). Validation actually ranks G3 last among the GNNs, so a validation-based choice would not have
   picked it, and its test advantage over PairMLP may not transfer.
3. **On Task 2 the graph is not expected to dominate, and it does not.** The signal is mostly in the bill's own
   features `x_b`: TabGBDT on `x_b` alone reaches 0.820, and neither G1 nor G2 beats it (both slightly below,
   within the CI). G3 is level with TabGBDT and beats G1 by 0.020 AUROC (real over 3 seeds). G1's Task-2
   validation AUROC peaks after 5-10 epochs and then falls while the training loss keeps dropping (overfitting),
   and the G1 search never tried the stronger regularisation that helped G3 (dropout 0.5). So "GNN < GBDT on
   Task 2" is an honest, explained result: a tabular model is the bar to beat, and a GNN should be judged by
   whether it adds anything on top of it.
4. **Single-seed results can mislead.** An earlier single-seed run at 4 threads showed G1 as much worse on new
   legislators (0.19 vs 0.24) and G3 as clearly best on Task 1. Neither claim survives 3 seeds.

**Rolling mode** was run for fold 116 (`--mode rolling --folds 116`, APRIORI settings, so every row is clean):
G1 0.3745, G3 0.3730, G2 0.3694, PairMLP 0.3666 Recall@20 (sponsor_hist +stream 0.2884); Task 2 G3 0.8054,
G2 0.8006, TabGBDT 0.7958, G1 0.7944 AUROC. That is about 12 minutes per fold with 4 threads alongside another
job (`baselines_build/out/final_rolling_f116`). An earlier quick (2-epoch) smoke test ran all six folds
(`out/challenger_rolling_quick`).

## 8. Limitations

- **One test Congress.** In the split study (check D9), for the worst relevant model difference a single test
  Congress failed to confirm its direction 43% of the time. Rolling mode (folds 112-117) is the stronger evidence;
  fold 118 stays locked.
- **Training noise is not in the CIs.** The bootstrap resamples bills, not training runs. Use 3 seeds (or the
  cached reference) before ranking learned models.
- **Settings were tuned on fold 117 only.** In rolling mode the default `ROLLING_CONFIG = "apriori"` uses the
  untuned defaults, so every fold is clean. `"frozen"` rows for folds <= 116 are tagged `future_tuned` and kept
  out of the means.
- **The three leakage residuals of section 2** (node-attribute `in_office`, withdrawn cosponsorships,
  transductive `per_congress` option).
- **Majority party** uses the split study's mid-Congress majority, known at the start of every Congress except
  for the 50-50 Senate tie-break.

## 9. Parameters to experiment with

Change one thing at a time and judge it on the validation column only. All names are fields in
`baselines/config.py` (`APRIORI` / `FROZEN` entries per model and task) or the notebook config cell.

| knob (config name) | default | suggested range | why it is worth trying |
|---|---|---|---|
| `SEEDS` | (0,) | (0, 1, 2) | needed before any learned-vs-learned claim |
| cold-start fix for G1 (`models/hetero_sage.py`, not implemented) | none | drop all history in-edges of 10-15% of legislators per step (PinSAGE-style augmentation); or gate the residual by `degree > 0` | G1 is weakest on new-legislator positives |
| `HeteroSAGEConfig.dropout` for G1 Task 2 (+ `TrainConfig.weight_decay`) | 0.2 (1e-4) | 0.3-0.5 (1e-3) | G1 Task 2 overfits after 5-10 epochs; 0.5 never tried for G1 |
| `RGCNConfig.num_bases` | 4 | 2-8 | 8 bases: +0.005 AUROC on Task 2, re-confirmed after the fixes |
| `RunConfig.H` | 3 | 2-4 | history window of graph and features |
| `HeteroSAGEConfig.layers`, `RGCNConfig.layers`, `LightGCNConfig.K` | 2 | 0-3 | receptive field (0 = PairMLP) |
| `*.d` | 128 | 64-256 | capacity (flat so far) |
| `TrainConfig.lr` | 3e-3 / 1e-3 (G3) | 1e-3 to 6e-3 | optimisation speed |
| `TrainConfig.loss` | "softmax" | "bpr" | listwise vs pairwise (L11) |
| `TrainConfig.sup_half_life` | None / 3 (G3) | None, 2-4 | recency weighting against drift (now active for every model) |
| `TrainConfig.batch_bills` | 1024 / 256 (G3) | 128-2048 | more updates before overfitting (G3 Task 1 batch 128: +0.002) |
| `TrainConfig.val_smooth`, `patience` | 3; 4 / 6 | 1-3; 4-8 | early-stopping stability |
| `TrainConfig.avg_window` | 0 (off) | 3-5 | SWA-style tail averaging (G2 Task 1: inside noise) |
| `LightGCNConfig.combine` | "concat" | "mean" | SIGN concat vs LightGCN layer average |
| `USE_TITLE` | True | False | value of the introduced title |
| `T0_LIST`, `T1_PROTOCOLS` | (0,), ("stream",) | (0, 7), ("stream", "plain") | secondary task; plain-protocol cold start |
| GAT | off | `GATv2Conv` in place of `SAGEConv` | learned neighbour weights (L04) |
| HGT | not implemented | - | the natural stronger model (L09) |
