# %% [markdown]
# ## 5. G2 R-GCN (relation-specific weights with shared bases)
#
# **What it is, in plain language.** An R-GCN layer updates every node by averaging its neighbours *separately
# for each kind of edge*, passing each average through a weight matrix that belongs to that kind of edge, and
# adding a transformed copy of the node itself (Schlichtkrull et al. 2018, eq. 2; L09 p.25):
#
# `h_v' = W_0 h_v + sum_r (1 / |N_v^r|) sum_{u in N_v^r} W_r h_u`
#
# Here the relations r are the 11 edge types of our graph: who sponsored, who cosponsored at introduction
# ("original") or later, which committee a bill went to, their reverses, and the three incoming-only edges into
# the bill being predicted (its sponsor, its day-<=t0 cosponsors and its referral committees).
#
# **Why it is a good baseline here.**
# - It is the model family the closest prior work used: Russo et al. (2023) predict cosponsorship with an
#   encoder followed by an R-GCN over sponsor / cosponsor relations (2 layers, AdamW), and condition the score on
#   [legislator ; bill ; sponsor]. We keep the R-GCN and the sponsor conditioning, but replace their speech /
#   bill-text encoders with leak-free features (introduced title, point-in-time attributes) and use our
#   Congress-level split instead of their within-Congress split.
# - "Sponsored", "cosponsored at introduction" and "cosponsored later" are genuinely different signals. R-GCN gives
#   each its own transform. With 11 relations, free matrices would be 11 x 128 x 128 parameters per layer, and rare
#   relations would overfit, so we use the **basis decomposition** (eq. 3; L09 p.27-28): every W_r is a learned mix
#   `W_r = sum_b a_rb V_b` of only 4 shared matrices, so a relation costs 4 numbers plus its share of the bases.
# - It is **inductive**: nothing is an ID embedding. A legislator's state is built from point-in-time features and
#   the bills around them, so new legislators and every (new) target bill get representations.
# - Comparing G2 with G1 asks one controlled question: do relation-specific transforms built from a *shared*
#   basis, with a per-relation mean (c_{v,r} = |N_v^r|), help beyond one free SAGEConv per relation? The graph,
#   features, heads, loss, optimiser and schedule are identical to G1.
#
# **Architecture.** TypeEncoder (Linear + LayerNorm + ReLU per node type) -> 2 R-GCN layers (d = 128,
# 11 relations, 4 bases, per-relation mean), each followed by ReLU, dropout 0.2, a residual connection and a
# per-type LayerNorm (L05) -> the shared heads: Task 1 scores `<q_b, k_l> / 8 + g(phi_bl)` (sponsor-conditioned
# two-tower dot product plus a small MLP over the leak-free pair features); Task 2 is an MLP over
# [bill ; sponsor ; x_b]. Two layers give the receptive field the task needs (legislator -> bill -> legislator),
# the depth Schlichtkrull et al. and Russo et al. both use.
#
# **Regularisation.** DropEdge 0.2 on history edges during training (Schlichtkrull et al. use edge dropout 0.2
# for self-loops and 0.4 elsewhere in link prediction; L05 graph augmentation). The incoming target edges are
# never dropped, because they are the bill's only evidence.
#
# **Training.** Same as G1. Task 1: per-bill multi-positive listwise softmax over the full eligible pool (the
# per-query ranking principle of BPR, L11, with all ~100 / ~440 negatives). Task 2: BCE. AdamW (Russo et al.),
# lr 3e-3 (between Russo's 1e-4 for BERT fine-tuning and Schlichtkrull's 1e-2 for tiny full-batch models),
# wd 1e-4, 5% warmup then cosine, gradient clip 1. Early stopping on the validation Congress (Recall@20 /
# 3-epoch-averaged AUROC), then a refit on 109..c-1 for the selected number of epochs on the same schedule, then
# one evaluation on the test Congress.
#
# **Tuned settings (validation only; `baselines_build/tuning/G2_R_GCN_r1.json`).** We searched one factor at a
# time from the defaults above. Task 1 was fit on 109-115 and scored on validation 116 (+stream, t0 = 0,
# Recall@20). Task 2 used the mean AUROC over validation 114, 115 and 116. We first measured seed noise: the
# defaults give 0.3735 / 0.3704 / 0.3764 on Task 1 (sd about 0.003) and 0.7893 / 0.7934 on Task 2. A change had
# to beat the seed mean of the defaults by 0.005 Recall@20 or 0.01 AUROC, and promising single-seed results were
# re-run with more seeds. Nothing cleared that bar, so the defaults are kept (FROZEN = APRIORI, `tuned_on = (116,)`
# for Task 1 and `(114, 115, 116)` for Task 2). The test Congress was scored once, after this choice.
#
# | Task 1 factor tried (default) | val Recall@20 (seeds) | | Task 2 factor tried (default) | val AUROC (114-116) |
# |---|---|---|---|---|
# | **defaults** | **0.3734** (3) | | **defaults** | **0.7913** (2) |
# | bases 2 / 8 (4); block-diagonal | 0.3709 / 0.3697; 0.3749 | | bases 2 / 8 (4) | 0.7902 / 0.7965 (2) |
# | lr 1e-3 / 1e-2 (3e-3) | 0.3681 / 0.3734 | | lr 1e-3 / 1e-2 | 0.7917 / 0.7897 |
# | layers 1 / 3 (2) | 0.3712 / 0.3742 (2) | | layers 1 / 3 | 0.7895 / 0.7927 |
# | d 64 (128); d 64 + 30 epochs | 0.3735 (3); 0.3760 | | d 64 | 0.7925 |
# | dropout 0.1 / 0.4 (0.2) | 0.3754 / 0.3717 | | dropout 0.4 | 0.7909 |
# | DropEdge 0 / 0.4 (0.2) | 0.3731 / 0.3736 | | DropEdge 0.4 | 0.7888 |
# | weight decay 1e-2 (1e-4) | 0.3717 | | weight decay 1e-2 | 0.7893 |
# | BPR loss (softmax) | 0.3705 | | cosine horizon 12 epochs (30) | 0.7906 |
# | pair MLP 64 (32); H 2 (3); 10-epoch horizon | 0.3738 (3); 0.3728; 0.3713 | | | |
# | dropout 0.1 + pair MLP 64 + block | 0.3748 (2) | | | |
#
# *Dated note (2026-11-08):* these numbers were measured with Task-1 early stopping on the single-epoch metric.
# Under the current 3-epoch smoothing the defaults score 0.3691 / 0.3697 / 0.3750 on validation 116 (seeds 0-2,
# 4 threads, mean 0.3713; `tuning/g2_r2_raw/t1_A.jsonl`; 0.3732 / 0.3729 / 0.3767, mean 0.3743, in the 8-thread
# 3-seed notebook run), and 8 bases on Task 2 still gives +0.005 AUROC over 2 seeds
# (0.7961 vs 0.7911; `t2_C.jsonl`). The choice is unchanged.
#
# What this tells us: R-GCN is on a broad plateau. Its scores hardly depend on the number of bases, width,
# depth or regularisation. Most of its accuracy comes from the pair features and the node features: on the test
# Congress G2 beats its zero-layer counterpart PairMLP by only +0.007 Recall@20 (sd 0.005 over 3 seeds;
# below the 0.01 MDE, and the paired CI includes 0 for some seeds), so relation-specific message passing adds a little, not a lot. Lower lr and BPR (pairwise, L11) are clearly worse than the full-pool listwise
# softmax. One-layer R-GCN is slightly worse than two layers, which fits the 2-hop
# legislator -> bill -> legislator pattern. The closest near-miss is **8 bases on Task 2** (+0.005 AUROC,
# better on both seeds). It is a good first thing to try (`config.FROZEN[("rgcn", "t2")]`), but it is
# below the 0.01 bar. (During round 1 the recency weight `sup_half_life` had no effect because of the loss
# normaliser; this is fixed, and G2 keeps `sup_half_life = None`, its round-1 behaviour. Round-2 check: hl 3 gave
# 0.3717 on Task 1, inside seed noise.)
#
# **Implementation note.** `RGCNConfig.impl = "manual"` computes eq. 2 relation by relation in per-type
# coordinates. It equals PyG's `RGCNConv(aggr="mean")` on the homogeneous view to about 1e-6
# (`baselines.models.rgcn.check_equivalence`), and is about twice as fast on CPU. Set `impl = "pyg"` to use
# RGCNConv itself.

# %%
G2 = report.run_model_section("rgcn", cfg, corpus, timer)
RESULTS += G2
