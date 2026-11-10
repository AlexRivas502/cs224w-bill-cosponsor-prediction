# %% [markdown]
# ## 4. G1 HeteroSAGE (main GNN baseline)
#
# **In one sentence.** Each legislator, history bill, committee and target bill starts from its own
# point-in-time features. Two rounds of message passing then let a legislator absorb what kind of bills they
# sponsored and cosponsored in the last three Congresses, and let a target bill absorb who sponsored it, who
# cosponsored it by day t0, and which committees it was referred to. A small head turns these states into
# Task-1 rankings and Task-2 probabilities.
#
# **Why GraphSAGE (L04 p.23-24; Hamilton et al. 2017).** GraphSAGE is *inductive*. It learns how to aggregate
# neighbour features, not one free vector per node, so it can embed nodes it never saw in training. Every target
# bill is new, and about 13% of Task-1 positives in Congress 117 are legislators with no cosponsorship in
# 108..116. An ID-embedding model (MF, node2vec, plain LightGCN; L02, L11) cannot score those people at all.
# The update is `h_v <- W_root h_v + W_nbr mean_{u in N(v)} h_u`. The **root weight** keeps a legislator's own
# party, chamber and tenure separate from their neighbourhood (L04 p.18, "self-message"). The **mean**
# aggregator keeps a prolific cosponsor with 1,000 history edges on the same scale as a freshman with 10.
#
# **Why heterogeneous (L09 p.50-54).** Sponsoring, original cosponsorship, later cosponsorship and committee
# referral mean different things, and the four node types have different feature sets. Each relation therefore
# gets its own SAGEConv (relation-specific message, L09 p.50). The relations that point into a node type are
# summed (two-stage aggregation, L09 p.52; PyG `HeteroConv`). Per-type input encoders are L09 p.53's
# type-specific pre-processing.
#
# **Leak-free by construction (L05 p.66-67; L12 seed-time semantics).** There is one graph per target Congress
# k (section 2). History edges come from the finished Congresses k-3..k-1. Target bills only have *incoming*
# edges dated on or before their cutoff: sponsor, cosponsors with lag <= t0, and referrals. No edge leaves a
# target bill. So a bill cannot see other bills of its own Congress (the Task-2 "plain" rule) and cannot change
# any legislator's state, and label-window cosponsors are never edges. Same-Congress "+stream" signal enters
# Task 1 only through the exact-dated pair features `phi(b, l)`.
#
# **Architecture (DESIGN section 4).**
# - TypeEncoder: Linear + LayerNorm + ReLU per node type, plus state and committee embeddings, to d = 128.
# - 2 x HeteroConv(SAGEConv, mean, root weight), summed over the 11 relations. Each layer computes
#   `h <- LayerNorm(h + Dropout(ReLU(conv(h))))`, a residual skip connection (L05; L04 p.44-56 on why deep
#   GNNs need skips). We use LayerNorm rather than BatchNorm because degree statistics differ between
#   Congress graphs.
# - Two layers give the receptive field the task needs: legislator -> history bill -> legislator, and
#   committee -> history bill. Schlichtkrull et al. 2018 and Russo et al. 2023 also use 2 layers.
#   The last layer computes only the relations into `leg` and `tbill`, the only states the heads read.
# - DropEdge 0.1 on history edges, training only (L05 graph augmentation).
# - Task-1 head, shared by every model: `s(b, l) = <q_b, k_l> / 8 + g(phi_bl)`. The bill tower q_b reads
#   [bill ; sponsor] (sponsor conditioning as in Russo et al. 2023), and the legislator tower k_l reads h_l. A
#   two-tower dot product scores the whole pool with one matmul (L11 p.13-16). g is a small MLP over the
#   leak-free pair features (sponsor history, reciprocity, co-support, same-day stream counts).
#   Setting `layers = 0` removes message passing and gives exactly **PairMLP** (section 7). The G1 - PairMLP
#   gap is therefore the value of message passing alone.
# - Task-2 head: an MLP over [bill state ; sponsor state ; x_b], with policy area hidden.
#
# **Training (DESIGN section 5).**
# - Task-1 loss: a multi-positive listwise softmax over the *full* eligible pool of each bill. This is BPR's
#   per-query principle (L11 p.23-26), applied exactly to all ~100 (Senate) or ~440 (House) negatives. Those
#   negatives already contain PinSAGE-style hard negatives (same party, same state; L11 p.68-71). It matches
#   the per-bill Recall@K that we report better than a global binary loss (L11 p.18-22).
# - Task-2 loss: BCE.
# - AdamW (as in Russo et al.), lr 3e-3, weight decay 1e-4 (none on norms, biases or embeddings). 5% warmup,
#   then cosine decay to 0.1 x lr, gradient clipping at 1.0. Supervision Congresses are weighted equally
#   (`sup_half_life = None`; the recency weight is a knob, see section 10).
# - Model selection: early stopping on the validation Congress c-1 (Recall@20 for Task 1, AUROC for Task 2,
#   each as a 3-epoch moving average) chooses the epoch E*. We then refit from scratch on 109..c-1 with the same schedule horizon,
#   stop at E*, and score the test Congress once.
#
# **Tuned settings (validation only; `baselines_build/tuning/G1_HeteroSAGE_r1.json`).** We ran a one-factor-at-a-time
# search from the defaults above. Task 1 was fit on 109-115 and scored on validation 116 (+stream, t0 = 0, Recall@20).
# Task 2 used the mean AUROC over validation 114, 115 and 116. A change was adopted only if it gained at least
# 0.005 Recall@20 or 0.01 AUROC (half the minimum detectable effect). No change met that bar, so the defaults
# are kept (FROZEN = APRIORI, `tuned_on = (116,)` for Task 1 and `(114, 115, 116)` for Task 2).
# *Dated note (2026-11-08):* the numbers below were measured during the round-1 search, with Task-1 early stopping
# on the single-epoch metric (`val_smooth = 1`). The current code smooths over 3 epochs, which moves the scores by
# a few thousandths (re-check in `BASELINES.md`, section 6). No choice depends on this, because FROZEN = APRIORI.
#
# | Task 1 factor tried (default) | val Recall@20 | | Task 2 factor tried (default) | val AUROC (114-116) |
# |---|---|---|---|---|
# | defaults (seed 0 / seed 1) | **0.3755** / 0.3767 | | defaults | **0.7913** |
# | lr 1e-3 / 6e-3 (3e-3) | 0.3705 / 0.3760 | | layers 1 (2) | 0.7936 |
# | layers 1 / 3 (2) | 0.3738 / 0.3741 | | no title SVD | 0.7924 |
# | d 64 / 256 (128) | 0.3749 / 0.3758 | | lr 1e-3 | 0.7907 |
# | H 2 / 4 (3) | 0.3721 / 0.3751 | | dropout 0.3 | 0.7922 |
# | dropout 0.1 / 0.3 (0.2) | 0.3759 / 0.3727 | | weight decay 1e-3 | 0.7903 |
# | BPR loss (softmax) | 0.3720 | | d 64 | 0.7925 |
# | weight decay 1e-3 (1e-4) | 0.3753 | | batch 512 (2048) | 0.7914 |
# | DropEdge 0 / 0.2 (0.1) | 0.3744 / 0.3759 | | | |
# | pair MLP 128 (32); per-Congress phi z-score; patience 6 | 0.3742; 0.3741; 0.3755 | | | |
#
# What the search shows:
# - The validation surface is flat. The two seeds of the default config differ by 0.001, and nothing gains more
#   than that. The defaults from the lectures and papers (2 layers, mean SAGE, lr 3e-3, listwise softmax) sit on a
#   plateau, and the remaining error is not something these settings can fix.
# - Some changes are clearly worse: a smaller lr (1e-3), a shorter history window (H = 2), BPR in place of the
#   full-pool softmax, and heavier dropout. This supports the choices of the listwise loss (L11) and the
#   3-Congress window.
# - Task 1 with 1 layer is only 0.0017 below 2 layers, so most of the signal comes through the pair features
#   phi and the node features. Message passing adds little on top (compare with PairMLP in section 7).
# - During round 1 the recency weight `sup_half_life` had no effect for G1, because the step loss was divided by
#   the weighted bill count. This was fixed afterwards (the loss is now divided by the unweighted count), and G1
#   keeps `sup_half_life = None`, which is exactly its round-1 behaviour.
#
# **Cold start (new legislators).** On Task 1, G1 is only slightly better than its own zero-layer ablation PairMLP
# (3 seeds: +0.007 Recall@20, below the 0.01 MDE, and the CI includes 0 for some seeds). A first single-seed run
# (4 threads) suggested that G1 trades cold-start accuracy for returning legislators: hit@20 on positives from
# new legislators (no cosponsorship in 108..k-1, about 13% of positives) was 0.19 for G1 against 0.24 for
# PairMLP. With 3 seeds this does not hold up: new-legislator hit@20 is 0.24 +- 0.03 for G1, 0.23 for G2, 0.23 for
# PairMLP and 0.22 for G3, and the seed-to-seed spread (0.21-0.27 for G1) is larger than the gaps between models.
# There is still a plausible mechanism to watch: a freshman has no history in-edges, so all seven history-relation
# SAGE means are zero. Two cheap fixes we did **not** try (the search was stopped after round 1): drop all
# history in-edges for a random 10-15% of legislators per training step (simulated freshmen, PinSAGE-style
# augmentation, L11), or gate the residual update by a has-history indicator. Judge either on validation 116,
# over several seeds, with the new-legislator split of section 8.
#
# **Task 2: the graph is not expected to dominate.** Advancement is mostly explained by the bill's own features
# `x_b` (sponsor majority status, committees, day of Congress, title): TabGBDT on `x_b` alone gets about 0.82 test
# AUROC, and G1 (0.805 +- 0.005 over 3 seeds) and G2 (0.810) do not beat it (paired differences -0.015 and -0.011,
# CIs include 0). G3 is +0.005 above TabGBDT, also inside the CI. So on Task 2 message passing adds nothing measurable over a strong
# tabular model. G1's validation AUROC peaks after 5-10 epochs and then falls while the training loss keeps
# dropping, which is overfitting. The G1 search only tried dropout up to 0.3, while dropout 0.5 helped G3 on Task 2 in round 1,
# so stronger regularisation (dropout 0.5, weight decay 1e-3) is the first thing to try for G1 on Task 2.
#
# **What to look at.** The validation curve and E*, then validation and test against party+pop,
# sponsor_hist and rw2hop (+stream), co-support, and PairMLP for Task 1, and against eb_committee and TabGBDT
# for Task 2. With `"plain"` in `T1_PROTOCOLS`, G1 is also retrained without the stream columns of phi, which
# gives the plain-protocol comparison and the cold-start (new legislator) check.

# %%
G1 = report.run_model_section("hetero_sage", cfg, corpus, timer)
RESULTS += G1
