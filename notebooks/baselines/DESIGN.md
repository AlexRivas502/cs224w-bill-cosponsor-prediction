# Baselines design (binding for all builders), revision 2

Repo root `R = /Users/alex_rivas/Documents/Stanford/CS224W/Final Project/cs224w-bill-cosponsor-prediction`
(quote it, the path has spaces). This file merges three expert proposals and the adversarial critique of
revision 1 into one plan. If something is not specified here, pick the simplest option that keeps the
leak-free guarantees of section 3 and the selection discipline of section 6, and write the choice into
`BASELINES.md`. Section 12 logs every decision that changed in revision 2 and every disagreement with
the critique.

Lecture citations use the course index (`.claude/cs224w/lectures/LNN-*.md`). Papers:
Schlichtkrull et al. 2018 (R-GCN, `1703.06103v4.pdf`), Russo et al. 2023 (ACL, `2023.acl-long.166.pdf`),
Hamilton et al. 2017 (GraphSAGE), He et al. 2020 (LightGCN), Wu et al. 2019 (SGC), Ying et al. 2018 (PinSAGE).

---------------------------------------------------------------------------------------------------
## 0. Decisions at a glance

| Item | Decision |
|---|---|
| GNN baselines | **G1 HeteroSAGE** (main), **G2 R-GCN** (basis decomposition), **G3 LightGCN-feat** (parameter-free GCN-normalised diffusion of inductive features, SGC/SIGN style) |
| Shared decoder | Task 1: sponsor-conditioned two-tower dot product **+** pair-feature MLP over leak-free `phi(b,l)`. Task 2: MLP over `[h_b ; h_sponsor ; x_b]` |
| Heuristic references | popularity, party+pop, sponsor_hist, rw2hop (plain and +stream, ported verbatim); sh_hl1; co-support (new); svd_r512_hl1 (plain only). Task 2: prior, chamber_maj, eb_rate_t0safe, eb_rate (leaky upper reference, footnoted), eb_committee (new, PA-free) |
| Learned non-GNN references | **PairMLP** = G1 with 0 message-passing layers (same head, loss and features). Task 2: **TabGBDT** (HistGradientBoosting) and **TabLogReg** on `x_b` |
| Task 1 protocol | **+stream** is primary. The encoder graph holds earlier Congresses only, and same-Congress signal enters only through `phi`, exact to the day. **plain** = the same models retrained without the stream columns of `phi` (G1 only by default) |
| Task 2 protocol | **plain**, t0 = 0, policy area hidden for every bill |
| History features | **Depth-consistent**: every learned history feature uses the same fixed window k-H..k-1 (H = 3) with hl-1 decay, normalised by the available depth, plus a `hist_depth` one-hot (section 3.3) |
| Loss | Task 1: full-pool multi-positive listwise softmax per bill (BPR is the option). Task 2: BCE, pos_weight 1 |
| Optimiser | AdamW, 5% warmup then cosine to 0.1 x lr over a fixed horizon, grad clip 1.0, early stopping on the validation-Congress metric. The refit keeps the same horizon and stops at E* |
| Split | Fold c: GNN supervision 109..c-2 (108 is history only; a documented deviation, section 2.1), validate c-1, refit 109..c-1, test c. Ported heuristics and eb_committee keep 108..c-2. Simple mode = fold 117. Rolling = 112..117. Locked 118 behind a flag. 119 never loaded as labels |
| Selection discipline | Frozen tuned configs carry `tuned_on`. Rolling mode uses a-priori defaults (`tuned_on = ()`) or a per-fold search. Rows whose test Congress is in, or before, `tuned_on` are tagged and excluded from rolling means (section 6.3) |

Why these three models. Together they answer three questions, each grounded in the lectures:
1. Does a learned inductive heterogeneous encoder beat strong counting heuristics? (G1 against the heuristics and PairMLP.)
2. Do relation-specific weights with shared bases help? (G2 against G1; L09 p.25-28.)
3. Is learned nonlinear message passing needed at all, or is fixed GCN diffusion enough? (G3 against G1; L11 LightGCN p.42-56.)

Proposal 1's "HeteroSAGE-Hybrid" is not a separate model. Its pair features are the shared decoder
component of every GNN (proposals 2 and 3 agreed), and its "GNN term off" arm is PairMLP.

---------------------------------------------------------------------------------------------------
## 1. Rejected models (keep a shortened version of this list in BASELINES.md)

- **Plain GCN (Kipf and Welling; L03, L04 p.22).** It is homogeneous and ties the self and neighbour weights. Its symmetric normalisation on a bipartite graph with degrees from 1 to 1000s is unstable. The GCN family is still covered: G2 is the relational GCN, G1 is a GCN with a separate root weight, and G3 uses exactly the GCN/LightGCN normalisation.
- **GIN (L06).** Injective sum aggregation buys WL-level structure discrimination, which these feature-rich node and link tasks do not need. Summing over heavy-tailed cosponsor degrees would tie the embedding scale to activity volume across Congresses.
- **GAT / GATv2 (L04 p.26-35).** Multi-head parameters and seed variance, and the 20-minute budget cannot cover tuning them. `phi` already supplies neighbour importance explicitly. Kept as a future ablation (swap SAGEConv for GATv2Conv in G1).
- **HGT (L09 p.60-69).** Over-parameterised for 4 node types and about 7 supervision graphs, and normally paired with HGSampling, which we cannot use. It is the natural "stronger model" after the baselines.
- **ID-only LightGCN / NGCF / MF / node2vec / TransE-DistMult-ComplEx (L02, L10, L11).** Transductive. 18-20% of plain positives are unseen legislators, and every target bill is new. svd_r512_hl1 represents this family as a reference.
- **PinSAGE as a system, temporal NeighborLoader, TGN/TGAT (L11 p.58-75, L12 p.51-52).** They need pyg-lib or torch-sparse. Per-target-Congress graphs plus exact-dated pair features give the same point-in-time semantics. PinSAGE's ideas are kept: features for inductiveness, and hard negatives, which the full pool already contains.
- **SEAL / NBFNet-style pair subgraphs (L07), graph transformers (L08).** About 2M pairs per Congress, or quadratic attention.
- **Russo et al.'s full model.** It needs BERT/Bi-LSTM over speeches and full text, and it uses a within-Congress split that the split study rejects. We keep R-GCN over sponsor/cosponsor relations, 2 layers, AdamW and [legislator ; bill ; sponsor] conditioning. TF-IDF-SVD of the introduced title replaces the text encoder.
- **Global BCE for Task 1 (L11 p.18-22).** It is not personalised. **Focal loss or a large pos_weight for Task 2:** the metrics are rank-based, and positives are 3-9%.

---------------------------------------------------------------------------------------------------
## 2. Task definitions (identical to the split notebook)

Port these verbatim, as functions, from `tools/split_notebook/cells/00_setup.py` into `baselines/data.py`:
`load`, `congress_start/end`, `build_cosponsor_events` (`cosp`, withdrawn rows dropped), `window_targets`,
`build_advancement_labels` (with its `censored` flag), `adv_cutoff`, `in_office_at`,
`_party_asof_table` (-> `members.party_code_asof`), the setup caucus assignment (`_dshare`, `CAUCUS_MIN_SHARE`
-> `members.caucus`), `_build_roster_asof` (-> `ROSTER_ASOF_ALL`), `majority_party` (-> `MAJORITY`),
`LAW_TYPES`, `WINDOW_DAYS = 30`, `ADV_WINDOW_DAYS = 180`, `ROSTER_REF_DAY = 15`. Do not import or exec the
split cells at runtime, because they print and compute drift statistics.

Port these from `40_splits.py` into `data.py` / `evaluate.py` / `models/heuristics.py`: `_LEG`, `_BILLIX`,
`_CD`, `_BL` (with `sp_caucus`, `spon`, `day`, `pm{t0}`, `sm{t0}`, `cluster`), `_CENS`, `_ROSTER`,
`_ROSTER_PIT`, `_boot_weights`, `_rank_metrics`, `_counts`, `_rownorm`, `_rw_parts`, `_rw_scores`,
`_ev_congress`, `_stream_hist`, `_stream_mats`, `_rw_stream`, `_ADV` (incl. `pa_hidden` with seed 1180
and `PA_MISS_T0`), `_grp`, `_t2_scores`, `_auc_ap`, and `_svd_factors` (r = 512, hl = 1). Give them public
names and keep a `# ported from 40_splits.py:<func>` comment. **RW_BETA = 0.3** for the main rw2hop.
The heuristic ports use the setup caucus and the 108..k-1 count history exactly as written, because the
reproduction asserts depend on them. The depth-consistent variants of section 3 are used only by
learned models.

Withdrawn cosponsorships are 0.09% of rows. Every component uses the setup `cosp` table, so pools and
features stay consistent. This is a documented, negligible approximation.

**Task 1 evaluation unit** (`T1Block`, one per (congress, chamber, t0)). It reproduces `_t1_eval`:
- bills: `_BL` rows of the Congress, uncensored (`_CENS[t0]`)
- candidates: `cand` = the columns of `_ROSTER_PIT[(c, ch)]`
- masks: `Excl` = sponsor, or cosponsored with lag <= t0, or not in office in the bill's `pm{t0}` month. `Pos` = lag in (t0, t0+30] and not `Excl`. Only bills with `Pos.any()` are kept.
- tie-break: `tieb = rng.random(M) * 0.5` with `rng = default_rng(t0 + 17)`, drawn in `_t1_eval`'s (congress, chamber) groupby order. Learned models add `1e-9 * tieb` before ranking.
- metrics over the full pool, with `Excl` set to -inf

The same blocks are the training examples, so the training population equals the evaluation population.

**Task 2 unit.** `_ADV` rows (law-capable, uncensored, congress 108..118), label `y`, `cluster` = sponsor.

### 2.1 Split, with one explicit deviation
SPLIT_DEFS S3 says "train 108..c-2". The GNNs and PairMLP use **supervision 109..c-2 / refit 109..c-1**
with 108 as message-passing and count history only, because a Congress-108 target graph would have
no earlier Congress at all (hist_depth 0, a regime that never occurs at test). This is the only change
to the split, and `BASELINES.md` must state it in a short "deviation" paragraph.
- Ported heuristics, svd and **eb_committee** train on 108..c-2 / 108..c-1 exactly as in the split study.
- **TabGBDT / TabLogReg** train on 109..c-2 by default, so they are paired with the GNNs on the same supervision rows. `tab_sup_start = 108` is a config switch. The report states which one was used (row field `sup`).
- Validation, refit and test Congresses are unchanged.

---------------------------------------------------------------------------------------------------
## 3. Leak-free and depth-consistent construction

### 3.1 Cutoffs and dates
- Bill cutoff `cut_b = introduced_date + t0`, inclusive, at day resolution. Cosponsorship date = `sponsorship_date` as day numbers (`_CD["day"]`).
- Referral date = `activity_ts`, localized to UTC, converted to America/New_York and normalised to a date (the 30_temporal.py convention; `load()` returns naive UTC).
- Referral rows: `activity.str.lower() == "referred to"` and `parent_committee_code.isna()`, i.e. full committees only (40 codes). Markup, Reported, Discharged and Hearings rows are read only by the Task 2 label function.

### 3.2 Graph per target Congress k (`graphs.build_graph(corpus, k, t0, H, feats)`)
There is one `HeteroData` per (k, t0, H, feature fit). It is cached and shared by both tasks when t0 = 0.
The same construction is used for supervision, validation and test Congresses (L05 p.66-67, one
independent graph per split unit; L12 p.43-52, seed-time semantics).

Node types:
- `leg`: all legislators in the global `_LEG` order (about 2.5k), so the index equals the legislator code
- `hbill`: every bill (all 8 bill types) of Congresses max(108, k-H)..k-1, with default H = 3
- `comm`: 40 full-committee codes in a global order, plus an `unk` row
- `tbill`: the law-capable bills of Congress k

History relations (all finished before k starts), each with a reverse `rev_*`:
- `(leg, sponsors, hbill)`
- `(leg, cosp_orig, hbill)`: `is_original` or lag 0
- `(leg, cosp_later, hbill)`
- `(hbill, referred, comm)`

Target relations, **incoming only**:
- `(leg, sponsors_t, tbill)`
- `(leg, cosp_t, tbill)`: lag <= t0, which is exactly the Excl-cosponsor set
- `(comm, referred_t, tbill)`: referral date <= cut_b

No relation has `tbill` as its source. So target bills cannot see each other (the Task 2 "plain" rule)
and cannot change `leg`/`comm` states. Label-window cosponsors and `Reported` actions are never edges.
Edge dates are stored on the graph object for `check_graph`.

### 3.3 Node features (`features.py`)
**Depth rule** (it fixes the 108 left-censoring): the data start at 108, so supervision Congress 109
would otherwise see 1 prior Congress while test 117 sees 9. Every learned history feature therefore:
- uses only the window W_k = max(108, k-H)..k-1 (H = `cfg.H`, default 3, the same H as the graph)
- weights Congress k' by `w = 0.5^(k-1-k')` (hl 1, the val pick of the split study)
- is reported as a **depth-normalised weighted rate**: `sum_k' w * count_k' / sum_k' in W_k w`, i.e. a per-Congress average, so the scale does not grow with depth
- `hist_depth = len(W_k)` in {1, 2, 3} enters as a one-hot on `leg` and `tbill` and as a `phi` column

Fits: standardisers are fitted on supervision blocks only. TF-IDF and SVD are fitted on titles of
Congresses <= the last supervision Congress of that fit (<= c-2 for the validation run, <= c-1 for the
refit). Each fitted object stores its Congress set.

**Point-in-time caucus for learned features** (`caucus_pit`):
- D/R: `party_code_asof` of Congress k
- other members (I, ID, L): the setup `_dshare` rule applied to cosponsorships from Congresses < k only (pooled over W_k). If there is no earlier record (a first-term independent), the value is `other`.

Residual convention: `MAJORITY[k, chamber]` is the setup mid-Congress majority. It is known at the start
of k for every Congress in range, apart from the 50-50 Senate tie-break, and it is used as is.
`BASELINES.md` documents this.

**History-bill labels and rates.** These use `build_advancement_labels()` with its `censored` flag:
- a censored hbill gets label 0 plus `adv_censored = 1`; a non-law-capable hbill gets 0 plus `not_law_capable = 1`
- every EB report rate (sponsor, committee, eb_committee) excludes censored bills from both the numerator and the denominator

Per node type:
- **leg** (as of k):
  - one-hots: `party_code_asof` (D/R/other), `caucus_pit` (D/R/other), Congress-k chamber (House/Senate/none), `hist_depth`
  - flags: delegate, in office in k, in majority (caucus_pit == MAJORITY), new member (no `members` row in W_k)
  - tenure: `tenure_win` = (# Congresses in W_k with a `members` row) / len(W_k). This replaces the unbounded log seniority.
  - state: an 8-d embedding (57 rows incl. `unk`)
  - depth-normalised rates over W_k:
    - log1p sponsored per Congress, log1p cosponsored per Congress
    - cross-caucus cosponsorship share (EB-shrunk to 0.5, m = 20)
    - sponsor EB report rate (law-capable, uncensored, shrunk to the chamber rate, m = 20)
  - legislators with no row in k keep their last known party and get in_office = 0
- **hbill** (finished bills, so these are legal):
  - bill_type one-hot (8), chamber, Congress lag one-hot (1..H), sponsor in majority
  - log1p final cosponsor count; cross-caucus cosponsor share
  - advancement label + `adv_censored` + `not_law_capable`
  - title SVD-64
  - no policy area (PA on history bills is an experiment knob, off by default)
- **comm**:
  - chamber; a learned 16-d code embedding (`unk` row for codes absent from supervision Congresses)
  - log1p referrals per Congress over W_k
  - EB report rate over W_k (uncensored only, m = 20, shrunk to the chamber rate)
- **tbill** = `x_b`, an allowlist; every value is known at cut_b:
  - bill_type one-hot (4), chamber, sponsor_is_by_request, day_of_congress / 730, session, `hist_depth`
  - sponsor: in majority, tenure_win, EB report rate, log1p sponsored per Congress (from the leg features)
  - log1p(# cosponsors with lag <= t0); their cross-caucus share and majority share (0 plus `has_cosp`)
  - # referral committees <= cut_b; max and mean comm EB rate over them (0 plus a flag when there are none)
  - title SVD-64: `bill_titles` rows with `title_type == "Official Title as Introduced"`, first row per bill, never `bills.title`. TF-IDF uses word 1-2-grams, min_df 5, sublinear tf, max_features 50k, then TruncatedSVD(64, random_state 0). `USE_TITLE = True`; the flag zeroes the block.
- **Forbidden everywhere for target bills**: policy_area(_harm), subjects, summaries, text versions, related bills, latest_action_*, n_cosponsors, n_committees, n_actions, and any count dated after the cutoff. `features.FORBIDDEN_PATTERNS` is asserted against every feature-name list.

### 3.4 Pair features `phi(b, l)` for Task 1 (`features.pair_features(corpus, block, t0, protocol, H)`)
`phi` is a dense `(nb, M, F)` float16 array per `T1Block`, cached to `.npy`. `s` = sponsor of b; `C_b` =
b's cosponsors with lag <= t0. All history columns use W_k with hl-1 weights and depth normalisation
(section 3.3). The unbounded `sh_all` and the 108..k-1 counts appear only in the heuristic ports.

History part:

| name | definition |
|---|---|
| `sh_w` | log1p Hw[s, l]; Hw = depth-normalised hl-1 sponsor->cosponsor counts over W_k |
| `recip_w` | log1p Hw[l, s] |
| `pop_w` | log1p popw[l] |
| `rw1_w`, `rw2_w` | `_rw_scores(_rw_parts(Hw))` parts, x 100 before log1p |
| `cosup_sum`, `cosup_max` | sum and max over m in C_b of log1p Cow[m, l]; Co = co-cosponsorship (B^T B over bill x leg incidence incl. the sponsor, diagonal zeroed), same weighting. 0 if C_b is empty |
| `same_caucus`, `same_state`, `same_both`, `cand_majority`, `cand_new`, `cand_delegate` | binary (caucus_pit) |
| `hist_depth_2`, `hist_depth_3` | one-hot (depth 1 is the base level) |

Stream part (Congress k events with day <= cut_b; **only when protocol = "stream"**). These are
naturally depth-free: the same day-of-Congress has the same meaning in every Congress.

| name | definition |
|---|---|
| `sh_st` | log1p Hst (exact daily, `_stream_mats`) |
| `pop_st` | log1p Pst (exact daily) |
| `recip_st` | l's Congress-k bills cosponsored by s, day <= cut_b (exact; `_stream_mats` with roles swapped) |
| `act30_st` | l's cosponsorships in (cut_b - 30, cut_b] (exact; difference of cumulative Pst rows) |
| `rw1_st`, `rw2_st` | `_rw_stream` parts (monthly snapshot before the first day of the cutoff month, so never later than the cutoff) |
| `cosup_st` | sum over m in C_b of log1p Co_k[m, l], Congress-k events before the cutoff-month start |

Standardisation: `phi_norm = "global"` (default) z-scores each column on the pooled supervision
blocks. Option `"per_congress"` z-scores within each Congress. Training blocks use the same function
with their own Congress as "current", so a feature means the same thing at train and test time.

**Drift check** (section 9, check 12): print the per-Congress mean of every standardised column of
`x_b`, `phi` and the leg features for 109..c. Warn if any column's test-Congress mean is more than 1 sd
from the supervision mean (and list columns above 0.5 sd). If a warning fires on validation 116, switch
that column family to `per_congress` and log the switch in the tuning JSON.

---------------------------------------------------------------------------------------------------
## 4. Models (`baselines/models/`)

Shared pieces live in `models/heads.py`:
- `TypeEncoder`: per node type, `Linear(F_type [+ emb] -> d) + LayerNorm + ReLU` (L09 p.53 type-specific pre-processing). Input dimensions are explicit, read from `meta`, so there are no lazy parameters.
- `LinkHead`:
  - `q_b = MLP([h_b ; h_s])`, d -> d -> 64
  - `k_l = MLP(h_l)`, d -> 64
  - `s(b,l) = <q_b, k_l>/8 + g(phi_bl)`, with `g = Linear(F -> 32) -> ReLU -> Linear(32 -> 1)`
  - flags `use_dot` and `use_pair` (ablations)
  - The factorised two-tower form scores the whole pool with one matmul (L11 p.7/15). Sponsor conditioning follows Russo et al. 2023 ([legislator ; bill ; sponsor]).
- `NodeHead`: `MLP([h_b ; h_s ; x_b])` -> 128 -> 1 logit (L05 node head; L09 entity classification).

Every model exposes `forward(graph) -> dict(leg=[NL,d], tbill=[NB,d])`,
`score_t1(emb, block, phi) -> [nb, M]` and `score_t2(emb, rows) -> [n]`.

### G1 HeteroSAGE (`models/hetero_sage.py`)
- 2 x `HeteroConv({rel: SAGEConv((d, d), d, aggr="mean", root_weight=True)}, aggr="sum")` over 11 relations: L04 GraphSAGE plus L09 p.50-54 two-stage aggregation. The explicit `(d, d)` input sizes avoid lazy init. If any lazy module is ever used, run one dry forward before building AdamW (`train.build_optimizer` asserts that no parameter is `UninitializedParameter`).
- Per layer and type: `h <- LayerNorm(h + Dropout(ReLU(conv(h))))`, a residual (L05 skip connections) with LayerNorm. BatchNorm is not used, because degree statistics differ per Congress graph.
- Mean aggregation keeps hubs on the scale of freshmen. The root weight preserves the legislator's own point-in-time attributes.
- Depth 2 is the needed receptive field, leg -> bill -> leg (L05: depth just above the needed receptive field; Russo App. Fig. 9 and Schlichtkrull also use 2).
- DropEdge 0.1 on history edges, during training only.
- `layers = 0` gives **PairMLP**.

### G2 R-GCN (`models/rgcn.py`)
- Once per graph (cached, not per step), build the homogeneous `edge_index`, `edge_type` and per-type node offsets in a fixed type order. Each step concatenates the TypeEncoder outputs in that order. Do not call `to_homogeneous()` per step. DropEdge is a boolean mask over the cached history edges.
- 2 x `RGCNConv(d, d, num_relations=11, num_bases=4, aggr="mean")`: Schlichtkrull eq. 2 with c_{v,r} = |N_v^r| and root W_0, and eq. 3 basis decomposition (L09 p.25-28).
- ReLU, dropout 0.2, residual and LayerNorm; DropEdge 0.2 on history edges. Same heads as G1 (a controlled encoder comparison).
- Assert that no homogeneous edge has a `tbill` source.
- If RGCNConv takes more than 0.5 s per forward+backward on the 117 graph, swap in a manual per-relation `scatter(reduce="mean")` loop, unit-tested against RGCNConv.
- Option, off: block-diagonal decomposition (L09 p.27).

### G3 LightGCN-feat (`models/lightgcn.py`)
Precomputed once per graph, with no learnable propagation:
- `A` = the symmetric history adjacency (leg-hbill, hbill-comm), weight 1, no self-loops
- `Ã = D^-1/2 A D^-1/2` (L11 p.45-51, L03)
- `X` = block-diagonal standardised type features
- `P_j = Ã^j X` for j = 0..K, by repeated sparse matmul (L11 p.47); keep only the `leg`/`comm` rows
- target bills, incoming only and parameter-free: `t_j = [mean P_j over sponsor ; over C_b ; over referral comms]`

Learned part:
- `z_v = MLP_type(concat_j LN(P_j[v]))`, 256 -> d = 128 (SIGN-style concat). Option `combine = "mean"` gives LightGCN layer averaging with alpha_j = 1/(K+1).
- `z_b = MLP([x_b ; concat_j t_j])`; same heads
- Default K = 2. Hop blocks are combined, not last-hop only, because Ã^j oscillates on a bipartite graph.
- Mini-batches mix Congresses (there is no encoder pass per step).

---------------------------------------------------------------------------------------------------
## 5. Losses, batching, optimisation (`train.py`)

**Task 1 loss.** For bill b with positives P_b and negatives N_b = pool \ Excl \ P_b:
`L_b = -(1/|P_b|) * sum_{p in P_b} [ s_p - logaddexp(s_p, LSE_{n in N_b} s_n) ]`.
- The batch mean is over bills, matching the per-bill Recall@K average. Other positives are excluded from each denominator.
- This is L11 p.23-26's per-query BPR principle applied exactly to all ~440 (House) / ~100 (Senate) negatives. The pool already contains PinSAGE-style hard negatives (L11 p.68-71).
- Option `loss="bpr"`: the per-bill mean over (p, n) pairs of softplus(s_n - s_p).

**Task 2 loss.** BCE with logits, pos_weight 1.

**Steps.**
- G1/G2 Task 1: one step = one encoder pass on G_k plus 1024 kept bills of k. The (k, batch) list is shuffled each epoch (about 35 steps on fold 117).
- G1/G2 Task 2: 2048 bills per step.
- G3: 256 (T1) and 1024 (T2) bills, mixed across Congresses.
- Supervision recency weight `w_k = 0.5^((k_last - k)/hl)` multiplies the per-bill loss; the weights are rescaled to mean 1 over the supervision bills and the step loss is divided by the *unweighted* bill count, so the weight is active for every model (before 2026-11-08 the step loss was divided by the weighted count, which cancelled the weight in one-Congress steps). Planned default hl = 3; after the round-1 search **G1, G2 and PairMLP use `None`** (identical to their round-1 behaviour) and **G3 uses 3**.

**Optimiser.**
- AdamW (as in Russo et al.), lr 3e-3 for G1/G2 and 1e-3 for G3, betas (0.9, 0.999), wd 1e-4. No decay on LayerNorm, bias or embedding parameters.
- The lr sits between Russo's 1e-4 (BERT fine-tuning) and Schlichtkrull's 1e-2 (tiny full-batch models). Clip gradients at norm 1.0.
- **Schedule horizon**: `T = max_epochs x steps_per_epoch` (max 20 for T1, 30 for T2), with 5% linear warmup and then cosine to 0.1 x lr over T. Early stopping on the validation metric (T1 Recall@20 over the full pool; T2 AUROC), both as a **3-epoch moving average** (`val_smooth = 3`; Task 1 was unsmoothed during the round-1 search and switched on 2026-11-08 because one-epoch Recall@20 swings by more than D_TOL), patience 4 / 6, keeping the best weights; E* = the best epoch.
- **Refit**: from scratch on 109..c-1 with the *same rule*. The horizon is `T_refit = max_epochs x refit_steps_per_epoch`, and training stops after E* epochs. So the refit follows the same LR trajectory, as a function of epoch fraction, that validation selected. It does not re-anneal over E*.

**Seeds and threads.** `set_seed(s)` seeds random, numpy and torch and calls
`torch.use_deterministic_algorithms(True, warn_only=True)`. `SEEDS = [0]` in simple mode, `[0, 1, 2]` for
the "final" setting (mean ± sd plus a seed-ensemble row). `THREADS = 8` for the user and 4 for agent
experiments. `DEVICE = "cpu"`; "mps" is allowed but not the reference.

---------------------------------------------------------------------------------------------------
## 6. Defaults, validation search and selection discipline

### 6.1 Two config sets
- `APRIORI`: the defaults in the table below, chosen from the lectures, the papers and the split study's per-fold validation picks (hl 1, beta 0.3). No Congress score of this project was used to choose them. `tuned_on = ()`.
- `FROZEN`: the output of the offline search for fold 117. It is written to `config.py` with `tuned_on = (116,)` for Task 1 and `(114, 115, 116)` for Task 2. Simple mode and locked 118 use FROZEN.

### 6.2 Offline search (orchestrator, fold 117, validation only)
- One factor at a time from APRIORI. A change is adopted only if it gains at least 0.005 Recall@20 (D_TOL = MDE/2) or 0.01 AUROC.
- Logs go to `tools/baselines_notebook/tuning/<model>_<task>.json`: config, val metric, best epoch, runtime and `tuned_on`. Test numbers are never logged.

| Model | APRIORI defaults | Search (T1 +stream, t0 = 0, val 116; fit 109..115) | ~runs |
|---|---|---|---|
| G1 | d 128, L 2, dropout 0.2, DropEdge 0.1, H 3, lr 3e-3, wd 1e-4, batch 1024, softmax, tower 64, pair 32, sup hl 3, phi_norm global, USE_TITLE | lr {1e-3, 3e-3}; d {64, 128}; L {1, 2, 3}; H {2, 3, 4}; loss {softmax, bpr}; dropout {0.1, 0.3}; sup hl {None, 3} | 10 |
| G2 | d 128, L 2, bases 4, dropout 0.2, DropEdge 0.2, H 3, lr 3e-3 | bases {2, 4, 8}; lr {3e-3, 1e-2}; L {2, 3} | 5 |
| G3 | K 2, concat, MLP 256 -> 128, dropout 0.2, lr 1e-3, batch 256 | K {1, 2, 3}; combine {concat, mean}; lr {3e-4, 1e-3, 3e-3} | 6 |
| Task 2 | as T1 plus NodeHead | G1 only: L {1, 2}; USE_TITLE {on, off}; scored on the mean of val 114, 115, 116, each fit on 109..v-1 (D17) | 4 |

Total: about 25 fits at about 1-2 min each with 4 threads. PairMLP inherits G1's setting with L = 0.
`H = all` is dropped from the grid because it breaks the depth rule.

### 6.3 Rolling mode and contamination (the fix for critique item 1)
- Default `rolling_config = "apriori"`: every fold 112..117 uses APRIORI with per-fold early stopping on c-1. All rows are clean. The rolling mean and the paired t-test use these rows. Fold 117 is reported twice, as `apriori` and as `frozen`, and is labelled as such.
- `rolling_config = "search"` (= `RUN_SEARCH = True`): the section 6.2 grid is rerun per fold on c-1 (T1) and on the mean of c-3..c-1 (T2, each fit on 109..v-1). This is clean and runs overnight.
- `rolling_config = "frozen"` is allowed only for exploration. Each row gets `selection` = `contaminated` if test ∈ `tuned_on` (T1 fold 116; T2 folds 114-116), `future_tuned` if max(`tuned_on`) > test (T1 folds 112-115; T2 folds 112-113), and `clean` otherwise. `report.rolling_summary` excludes every non-clean row from means and t-tests and prints them in a separate, clearly labelled block.
- Asserts:
  - `train.run_fold` raises if a config with `test ∈ tuned_on` is used unless `cfg.allow_contaminated` is set, and in that case it tags the row
  - `report.rolling_summary` asserts that every row entering a mean has `selection == "clean"`
  - fold 117 (FROZEN, tuned on <= 116) and locked 118 are always clean

---------------------------------------------------------------------------------------------------
## 7. Evaluation and reporting (`evaluate.py`, `report.py`)

- **Task 1.** Per-bill [Recall@10, Recall@20, NDCG@20] from `rank_metrics(S, Pos)` with `S[Excl] = -inf`, over the same `T1Block`s. Assert that bill codes are identical across models. t0 = 0 is primary; `T0_LIST = (0, 7)` adds a separately trained t0 = 7 model. Breakdowns: by chamber, and by new-legislator positives (pop over 108..k-1 = 0: hit@20 share).
- **Task 2.** AUROC, AP and AP lift from `auc_ap`, per fold, plus House and Senate rows. Never pool across folds.
- **CIs.** 95% cluster bootstrap by sponsor, B = 500. The weights come from `boot_weights(clusters, key)` with key = (fold, task, t0, "eval"), so models in one evaluation unit share the same resamples. A bill bootstrap is also reported.
- **Paired differences.** Each GNN against:
  - Task 1: the best same-protocol heuristic, co-support, and PairMLP
  - Task 2: TabGBDT
  - A difference counts as real only if it is at least the MDE (`config.D_MDE`: 0.01 Recall@20, 0.01 AUROC, the split study's values; `report.MDE` imports it) and the CI excludes 0; with several seeds, the seed-mean difference must reach the MDE and every seed's CI must exclude 0 (`report.paired_seed_summary`).
- **Rolling.** A per-fold table plus the mean over clean rows, and a paired t-test on per-fold differences.
- **Row schema** (`report.Row`): fold, role (val/test), task, t0, protocol, model, config_set (apriori/frozen/search), selection (clean/future_tuned/contaminated), sup (e.g. "109-116"), seed, metric, value, ci_low, ci_high, n_eval, runtime_s.
- `report.py` writes `results.csv` and prints pivot tables (models × metrics, with protocol and reference rows grouped).

**Reference bars** (fold 117 test, from digest_D.md; the notebook must reproduce them):

| Task 1, t0 = 0, Recall@20, 5,101 bills | plain | +stream |
|---|---|---|
| popularity | 0.1375 | 0.140 |
| party+pop | 0.1905 | 0.195 |
| sponsor_hist | 0.2615 | 0.3071 |
| rw2hop | 0.2629 | 0.308 |
| svd_r512_hl1 | 0.2832 | (plain only) |

- Validation 116 (fit 108..115): sponsor_hist 0.2542 plain / 0.2884 +stream; svd_r512_hl1 0.2757.
- Task 2 (12,953 bills): chamber_maj 0.5449, eb_rate_t0safe 0.5598, eb_rate 0.6924 (uses policy area; leaky upper reference).
- Context, 7-fold means: sponsor_hist 0.259 plain / 0.296 +stream; t0 = 7 0.241 / 0.275; eb_rate_t0safe 0.624.

New references:
- `cosup` (T1): `cosup_sum` (+ `cosup_st` under +stream) + 1e-6 · pop + tie-break, computed with the depth-consistent Cow. This is the honest bar for "it only used the day-0 cosponsors".
- `eb_committee` (T2): EB rate by (first referral committee <= cut_b, chamber, sponsor in majority), shrunk to chamber_maj with m = 20, fitted on 108..c-2 / 108..c-1 with uncensored rows only.
- `TabGBDT`: `HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, early_stopping=True, validation_fraction=0.1, random_state=seed)` on `x_b`.
- `TabLogReg`: `LogisticRegression(C=1.0, max_iter=2000)` on standardised `x_b`.

---------------------------------------------------------------------------------------------------
## 8. Package layout and interfaces

Imports go one way: `config -> data -> evaluate -> features -> graphs -> models.* -> train -> report`.
`evaluate` depends only on numpy/scipy and `data`. Every public function has a docstring. The cache lives
in `<repo>/data/cache/baselines/<key>.{parquet,npy,pt}`, keyed by a hash of the relevant config and
`FEATURE_VERSION`. Bump `FEATURE_VERSION` to "2" for the depth rule.

```python
# config.py
@dataclass
class RunConfig:
    mode: str = "simple"                      # "simple" -> [117]; "rolling" -> 112..117
    tasks: tuple = ("t1", "t2")
    t0_list: tuple = (0,)                     # (0, 7) roughly doubles Task-1 time (section 11)
    t1_protocols: tuple = ("stream",)         # add "plain"
    plain_models: tuple = ("hetero_sage",)
    models: tuple = ("hetero_sage", "rgcn", "lightgcn", "pair_mlp")
    heuristics: bool = True
    tabular_refs: bool = True
    tab_sup_start: int = 109                  # 108 = split-study convention
    seeds: tuple = (0,)
    device: str = "cpu"
    threads: int = 8
    locked_118: bool = False
    run_search: bool = False
    rolling_config: str = "apriori"           # "apriori" | "search" | "frozen" (exploration; tagged)
    allow_contaminated: bool = False
    use_title: bool = True
    quick: bool = False                       # smoke test: 2 epochs, Senate-only T1 blocks
    cache_dir: Path | None = None
    def folds(self) -> list[int]: ...
@dataclass
class TrainConfig: lr, weight_decay, batch_bills, max_epochs, patience, warmup_frac, clip, sup_half_life=3, loss="softmax"
@dataclass
class HeteroSAGEConfig: d=128, layers=2, dropout=0.2, drop_edge=0.1, H=3, tower=64, pair_hidden=32, phi_norm="global", train_t1=..., train_t2=..., tuned_on=()
@dataclass
class RGCNConfig: ..., num_bases=4, decomposition="basis", tuned_on=()
@dataclass
class LightGCNConfig: K=2, combine="concat", mlp=(256, 128), ..., tuned_on=()
APRIORI: dict[str, dict[str, Any]]; FROZEN: dict[str, dict[str, Any]]   # per (model, task)
SEARCH_SPACE: dict[str, dict[str, list]]
def fold_plan(c) -> dict   # {"sup_val": 109..c-2, "val": c-1, "sup_refit": 109..c-1, "test": c,
                           #  "heur_train_val": 108..c-2, "heur_train_test": 108..c-1, "t2_val_multi": [c-3, c-2, c-1]}
def selection_tag(tuned_on, test) -> str      # "clean" | "future_tuned" | "contaminated"

# data.py
def find_root(start: Path | None = None) -> Path           # walks up until "parsed/" exists
@dataclass
class Corpus: bills, cosp, adv, members, roster_all, majority, leg_index, billix, CD, BL, ADV, roster_pit, cens, titles, referrals
def load_corpus(cfg: RunConfig) -> Corpus
def caucus_pit(corpus, k) -> np.ndarray                    # per leg code; Congresses < k only
def hist_window(k, H) -> list[int]                         # max(108, k-H)..k-1
@dataclass
class T1Block: congress, chamber, t0, codes, cand, Pos, Excl, tieb, spon, has_sp
def t1_blocks(corpus, congress, t0) -> list[T1Block]
def t2_rows(corpus, congresses) -> pd.DataFrame

# evaluate.py
def rank_metrics(S, Pos) -> np.ndarray
def boot_weights(clusters, key, B=500) -> np.ndarray
def auc_ap(s, y, W) -> tuple
def summarize_t1(per_bill, codes, corpus, key) -> list[dict]
def summarize_t2(scores, rows, key) -> list[dict]
def paired_diff(per_bill_a, per_bill_b, clusters, key, metric_idx) -> dict

# features.py
FEATURE_VERSION: str; FORBIDDEN_PATTERNS: tuple
def fit_title_svd(corpus, max_congress, dim=64) -> TitleSVD
def windowed_counts(corpus, k, H, kind) -> scipy.sparse.csr_matrix   # depth-normalised, hl-1
def leg_features(corpus, k, H) -> tuple[np.ndarray, list[str]]
def hbill_features(corpus, k, H, title) -> ...
def comm_features(corpus, k, H) -> ...
def tbill_features(corpus, k, t0, H, title) -> tuple[np.ndarray, list[str]]
def pair_features(corpus, block, t0, protocol, H) -> tuple[np.ndarray, list[str]]
class Standardizer: fit(blocks, congresses) / transform; .fitted_on
def drift_table(feats_by_congress, sup_congresses, test_congress) -> pd.DataFrame

# graphs.py
def build_graph(corpus, k, t0, H, feats) -> HeteroData     # + .tbill_codes, edge dates, cached homogeneous view
def homogeneous_view(graph) -> HomView                     # edge_index, edge_type, offsets (built once)
def check_graph(graph, corpus, k, t0) -> None

# models/heuristics.py
def t1_heuristics(corpus, hist_congresses, eval_congress, t0, stream) -> tuple[dict, np.ndarray]
def t2_heuristics(corpus, train_congresses, eval_rows) -> dict
def t2_tabular(x_train, y_train, x_eval, seed) -> dict

# models/__init__.py
def build_model(name: str, cfg, meta: dict) -> nn.Module

# train.py
def set_seed(s: int) -> None
def build_optimizer(model, tcfg, total_steps) -> tuple[AdamW, LRScheduler]  # asserts no lazy params
@dataclass
class FitResult: model, best_epoch, val_curve, val_metric, seconds, horizon_steps
def fit(model_name, task, cfg, sup_congresses, val_congress | None, stop_epoch | None, data: FoldData) -> FitResult
def predict_t1(model, data, congress) -> dict; def predict_t2(...)
def run_fold(c, run_cfg, corpus) -> list[Row]              # val fit -> E* -> refit (same horizon) -> test; tags selection
def run_search(model_name, task, c, run_cfg, corpus) -> list[dict]

# report.py
def results_frame(rows) -> pd.DataFrame; def show_main(df); def show_paired(df); def show_timing(timer)
def rolling_summary(df) -> pd.DataFrame                    # clean rows only (asserted)
class Timer
```

`FoldData` bundles the cached graphs, blocks, `phi` and `x_b` per Congress for one fold and one feature
fit (the validation fit and the refit differ in title and standardisation statistics).

---------------------------------------------------------------------------------------------------
## 9. Sanity checks (all must pass; builders add them as asserts or test cells)

1. **Heuristic reproduction**, fold 117, t0 = 0, tolerance 5e-4 (the ports are verbatim), asserted in notebook section 3 before any GNN runs:
   - plain: popularity 0.1375, party+pop 0.1905, sponsor_hist 0.2615, rw2hop 0.2629 (n = 5,101); svd_r512_hl1 0.2832
   - +stream: sponsor_hist 0.3071, rw2hop 0.3080 (±0.001, because the digest rounds to 3 decimals)
   - val 116: sponsor_hist 0.2542 / 0.2884
   - Task 2: chamber_maj 0.5449, eb_rate_t0safe 0.5598, eb_rate 0.6924 (n = 12,953)
2. **Label parity** (`tools/baselines_notebook/tests/test_parity.py`): exec `00_setup.py` once and `assert_frame_equal` `window_targets(0/7, point_in_time=True)`, `build_advancement_labels()`, `ROSTER_ASOF_ALL` and `MAJORITY` against the package.
3. **Graph asserts** (`check_graph`, on every graph):
   - no edge type has `tbill` as its source (also in the cached homogeneous view)
   - every `cosp_t` lag <= t0, and every `referred_t` Eastern date <= cut_b
   - every hbill Congress is in W_k
   - no edge (l, b) with l in Pos_b
   - only "referred to" full-committee rows are used
4. **Stream truncation test**: for 3 cutoff days D in Congress 117 (early, mid, late), `phi` for the bills with cut_b == D, computed from events truncated to day <= D, must equal `phi` from the full table. Also assert max(event day used) <= cut_b via debug arrays. Plus a **perturbation test**: shift every Congress-117 event dated after each bill's cutoff by +1000 days; the scores of a trained G1 must be bit-identical (threads = 1).
5. **Allowlist**: no feature name matches `FORBIDDEN_PATTERNS`.
6. **Fit discipline**:
   - TF-IDF, SVD and standardisers store `fitted_on`; assert max <= the last supervision Congress
   - `fit()` raises if the validation Congress or any supervision Congress is >= the test Congress
   - tuning JSON files have no test-role entries
   - `selection_tag` is applied to every row; `rolling_summary` asserts that only clean rows enter means
7. **Population identity**: model and heuristic per-bill arrays share bill codes and `Excl.sum(1)`.
8. **Determinism**: two seed-0 runs with `threads = 1` give the same validation Recall@20 within 1e-6. With 8 threads the tolerance is 1e-4 (multithreaded scatter/index_add reductions are not bitwise reproducible).
9. **Cold start**: each model in `plain_models` (G1 by default) gets hit@20 > 0 on new-legislator positives under plain (svd is about 0 there). Under +stream, report the breakdown without asserting.
10. **Smoke and runtime**: `run_cells.py --quick` finishes in under 3 min. The full simple run prints a timing table and stays at or under about 20 min with 8 threads; measure it with 4 and 8 threads and report both. If it is over budget, drop `plain`, then G2's Task 2, in that order, and document it.
11. **Expected direction** (warnings): every GNN under +stream beats party+pop on validation, and G1 +stream >= sponsor_hist+stream - 0.01 on validation. If not, debug before tuning.
12. **Depth and drift**:
    - assert `hist_depth` is in {1, 2, 3} for every supervision, validation and test block, and that no learned feature reads Congresses outside W_k (`windowed_counts` records its Congress set)
    - print `drift_table` for 109..c (per-Congress means of the standardised columns); warn above 1 sd and list columns above 0.5 sd
    - assert `caucus_pit` uses no events from Congress >= k (truncation test: drop all Congress-k cosponsorships, and `caucus_pit(k)` is unchanged)
    - assert that censored hbills never enter an EB rate denominator

---------------------------------------------------------------------------------------------------
## 10. Notebook plan (`tools/baselines_notebook/cells/NN_*.py` -> `notebooks/02_baselines.ipynb`)

The notebook stays thin: each section has markdown (what, why, citation) and 1-5 calls into the package.
The working directory is `R/notebooks`; `find_root()` searches upward for `parsed/`.

| File | Section | Content |
|---|---|---|
| `00_config.py` | 0. Setup and config | Imports and the **single config cell**: `MODE`, `TASKS`, `T0_LIST`, `T1_PROTOCOLS`, `MODELS`, `SEEDS`, `DEVICE`, `THREADS`, `RUN_SEARCH`, `ROLLING_CONFIG = "apriori"`, `LOCKED_118 = False`, `USE_TITLE`, then `cfg = RunConfig(...)`. Markdown covers tasks, the split (including the 109 deviation), protocols and runtime |
| `10_data.py` | 1. Data and labels | `corpus = load_corpus(cfg)`; label and pool summary per Congress |
| `20_graphs.py` | 2. Leak-free graphs and features | Build and cache the fold's graphs, `phi` and `x_b`; `check_graph`; truncation test; allowlist; **depth and drift table**; text schematic of node and edge types with counts |
| `30_heuristics.py` | 3. Heuristic references | Both tasks; **reproduction asserts**; table |
| `40_hetero_sage.py` | 4. G1 HeteroSAGE | Why (L04, L09), architecture, validation curve, refit, test |
| `50_rgcn.py` | 5. G2 R-GCN | Why (Schlichtkrull, Russo); same flow |
| `60_lightgcn.py` | 6. G3 LightGCN-feat | Why (L11 LightGCN/SGC); same flow |
| `70_references.py` | 7. Learned non-GNN references | PairMLP (T1), TabGBDT and TabLogReg (T2) |
| `80_compare.py` | 8. Comparison | Main tables, paired bootstrap, new-legislator breakdown, timing |
| `90_rolling_locked.py` | 9. Rolling and locked 118 | Runs only if `MODE == "rolling"` or `LOCKED_118`. Per-fold table with `config_set`/`selection` columns, means and t-tests over clean rows; 118 printed separately |
| `95_experiments.py` | 10. What to experiment with | Knobs mapped to config fields: H, layers, d, lr, loss softmax/bpr, USE_TITLE, sup half-life, phi_norm, **t0 = 7 (`T0_LIST = (0, 7)`: about +8-10 min in simple mode with 8 threads)**, plain protocol, seeds, GATv2 swap, policy area on history bills, `tab_sup_start` |

`assemble.py` concatenates the cells into the ipynb (as in `_build`). `run_cells.py` runs them in one
namespace with environment overrides (`BASELINES_MODE`, `BASELINES_QUICK`, `BASELINES_THREADS`) and
writes `tools/baselines_notebook/out/<run>/{stdout.txt, results.csv, report.json}`.

---------------------------------------------------------------------------------------------------
## 11. Expected results and runtime (for sanity, not targets)

- **Task 1, fold 117, t0 = 0, +stream, Recall@20**: G1 about 0.32-0.36 (bars 0.3071 / 0.308); G2 and G3 within about 0.01 of G1; PairMLP 0.005-0.02 below G1. If the gap is below the MDE, report that message passing adds little beyond `phi`.
- **Task 1 plain, G1**: about 0.28-0.31 (bars 0.2615 / 0.2629 / svd 0.2832), with non-zero recall on new legislators.
- **Task 2 AUROC**: about 0.66-0.75 (bars 0.545 / 0.560); TabGBDT likely within ±0.02. Expect validation-to-test level shifts of 0.03-0.06 (D2b).

Runtime, simple mode, 8 threads (measured synthetic benchmark: 2-layer HeteroSAGE, 11 relations,
~1.3M edges, d = 128, 4 threads, about 0.2 s per forward+backward; G1 T1 is about 35 steps x <=20 epochs
x 2 fits, so about 4-5 min):

| stage | time |
|---|---|
| data | 1-2 min (cached) |
| graphs and `phi` | 2-4 min (cached) |
| heuristics and tabular | 1-2 min |
| G1 | 4-5 min |
| G2 | 4-5 min |
| G3 | 1-2 min |
| PairMLP | 1-2 min |
| total | about 15-20 min |

`T0_LIST = (0, 7)` adds about 8-10 min (Task-1 features and fits again; Task 2 unchanged), so about
25-30 min in total. Rolling with APRIORI is about 6x simple (overnight); `rolling_config = "search"`
is about 3x that.

---------------------------------------------------------------------------------------------------
## 12. Design decisions log (revision 2)

| # | Critique item | Decision | Rationale or disagreement |
|---|---|---|---|
| 1 | Rolling mode reused settings tuned on 116 (T1) and 114-116 (T2) | Fixed. Rolling defaults to APRIORI configs (`tuned_on = ()`) with per-fold early stopping only; `"search"` reruns the grid per fold on c-1; `"frozen"` rows are tagged contaminated or future_tuned and excluded from means and t-tests, with asserts in `run_fold` and `rolling_summary` (section 6.3) | Tagging only `contaminated` would still let folds 112-115 benefit from tuning on a later Congress, so `future_tuned` is excluded too. The residual is that APRIORI uses split-study per-fold val picks (hl 1, beta 0.3), each chosen on that fold's own c-1. That is allowed |
| 2 | Feature non-stationarity from left-censoring at 108 | Fixed. All learned history features use the fixed window W_k with hl-1 decay and depth normalisation; `hist_depth` one-hot; `tenure_win` replaces log seniority; `sh_all` and the 108..k-1 counts are dropped from `phi`; H = all is removed from the grid; sup half-life 3 is now the default; `phi_norm = per_congress` is an option; drift table plus check 12 | With H = 3, Congresses 109/110 still have depth 1/2. The one-hot lets the model condition on depth instead of dropping two supervision Congresses. `sup_min_depth` is not added: the drift table decides, and dropping 109-110 is listed as an experiment |
| 3 | Supervision starts at 109, not 108 | Kept for GNNs as an explicit deviation (section 2.1). Ported heuristics and eb_committee use 108..c-2; TabGBDT/LogReg default to 109 (paired with the GNNs), with `tab_sup_start = 108` and a `sup` row field | A Congress-108 target graph would have depth 0, a regime that never occurs at test |
| 4 | Caucus and majority used same-Congress information | Fixed for learned features: `caucus_pit` from Congresses < k, `other` for first-term independents. Setup caucus is kept for the heuristic ports. The MAJORITY mid-Congress convention is documented | MAJORITY is effectively known at the start of each Congress in 108-118, so the residual is negligible |
| 5 | Refit LR schedule was never validated | Fixed: the refit keeps the validation horizon (`max_epochs x refit steps/epoch`) and stops at E* | This is the same LR-vs-epoch-fraction curve that early stopping selected |
| 6 | Checks 8 and 9 were inconsistent | Check 9 is restricted to `plain_models`. Check 8 uses threads = 1 with 1e-6, or 1e-4 with 8 threads | none |
| 7 | Censored history bills in labels and rates | Fixed: censored bills are excluded from EB numerators and denominators; hbill label 0 plus an `adv_censored` flag | none |
| 8 | Implementation details | The homogeneous view is built once per graph; SAGEConv uses explicit `(d, d)` inputs, and `build_optimizer` asserts that no lazy parameters remain; t0 = 7 runtime is stated (+8-10 min) | t0 = 7 stays off by default to meet the 20-minute target. It is a single flag |
| - | Proposal 1's separate "Hybrid" model | Merged into the shared decoder | All GNNs get the same `phi`, so B1 against B3 would only be the PairMLP/GNN ablation |
| - | Proposal 2's collapsed legislator-only R-GCN graph and hard-negative curriculum | Not adopted | G2 shares G1's graph so that the encoder comparison is controlled. The full-pool softmax makes negative sampling unnecessary (a pool of about 440) |
