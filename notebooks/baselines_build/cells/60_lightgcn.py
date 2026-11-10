# %% [markdown]
# ## 6. G3 LightGCN-feat (fixed GCN diffusion of features, learned MLPs on top)
#
# **The question it answers.** Do we need *learned, nonlinear* message passing at all, or is a fixed GCN-style
# smoothing of the features enough? LightGCN (He et al. 2020; L11 p.42-56) showed for recommendation that the
# weight matrices and ReLUs of a GCN can be removed: propagate with the symmetric normalisation
# `A_hat = D^-1/2 A D^-1/2` (L03, L11 p.45), take `A_hat^K E` by K sparse multiplications (p.47), and combine the
# layers (p.48-51). SGC (Wu et al. 2019; L11 p.43) and SIGN make the same point for *node features*. G3 is the
# linear-propagation end of the GCN family; G1 (GraphSAGE) and G2 (R-GCN) are the learned ends.
#
# **Why features, not ID embeddings.** Plain LightGCN learns one embedding per node, which is transductive: every
# target bill is new, and many positives come from legislators with no history (see the new-legislator
# breakdown in section 8). So we diffuse the *standardised node features* (PinSAGE's lesson, L11 p.58-59):
# `X` is block-diagonal over node types, and `P_j = A_hat^j X` for j = 0..K is computed once per graph, in
# well under a second. In plain language: after one hop a legislator carries the (degree-normalised) average of
# the titles, outcomes and sizes of the bills they sponsored or cosponsored in the last 3 Congresses; after two
# hops, the profile of the colleagues they co-sponsor with and of the committees those bills went to.
#
# **Leak-freedom.** `A` holds only history edges (Congresses k-3..k-1, all finished before Congress k starts).
# A target bill reads its own incoming edges only, without parameters: the mean of `P_j` over its sponsor, over
# its cosponsors dated <= t0, and over its referral committees dated <= the cutoff. Target bills never feed
# anything back, so they cannot see each other (the Task-2 "plain" rule). The perturbation test in
# `tests/test_sanity.py` (`BASELINES_GNN=lightgcn`) checks that moving every post-cutoff event leaves the
# scores unchanged.
#
# **Architecture.** For legislators and committees, `z_v = LN(h0_v + MLP([h0_v ; LN(P_1 v) ; ... ; LN(P_K v)]))`,
# where `h0` is the shared TypeEncoder (the same as in G1, G2 and PairMLP). Target bills use the same form over
# `[x_b ; t_0 ; ... ; t_K]`. The MLP is 256 -> 128, with dropout 0.2 (Task 1) / 0.5 (Task 2). The heads are shared with every model
# (two-tower dot product + pair-feature MLP for Task 1, MLP over [bill ; sponsor ; x_b] for Task 2).
# K = 2 hops (leg -> bill -> leg is the receptive field the task needs, L05). Hop blocks are concatenated
# SIGN-style, each with its own LayerNorm, rather than taking the last hop only: on a bipartite graph `A_hat^j`
# alternates between the user and item sides (eigenvalue -1; L11 notes, HW1 Q1.5). The option
# `combine = "mean"` is the exact LightGCN layer average, alpha_j = 1/(K+1).
#
# **Training.** The losses are those of G1: full-pool listwise softmax for Task 1 (L11 BPR principle, every
# negative in the pool) and BCE for Task 2. Optimiser: AdamW, lr 1e-3, wd 1e-4, warmup + cosine, clip 1.0.
# The lower lr than G1 suits the larger MLP inputs. There is no message passing per step, so each mini-batch
# mixes bills from all supervision Congresses (256 bills for Task 1, 1024 for Task 2), and target-bill rows are
# encoded lazily (only the rows a batch needs). The usual protocol follows: fit on 109..c-2, early stopping on
# c-1, then refit on 109..c-1 for E* epochs on the same schedule and test on c.
#
# **Tuned settings (validation only; `baselines_build/tuning/lightgcn_t*_fold117.json` and
# `G3_LightGCN_feat_r1.json`).** About 40 validation fits over K {1, 2, 3}, combine {concat, mean}, lr {3e-4, 1e-3,
# 3e-3}, dropout {0.2-0.6}, weight decay {1e-4, 1e-3, 1e-2}, width {64, 128}, MLP depth, batch {64-1024}, loss
# {softmax, BPR}, recency weighting, per-Congress phi normalisation and a shorter schedule, with a second seed
# for the candidates. **Task 1** is flat: every change stays within about 0.004 Recall@20 of the defaults (seed
# noise is about 0.001-0.003), so the defaults are kept. Listwise softmax beats BPR (-0.005), and batches of 1024
# are clearly worse (-0.010), because Task 1 overfits after 3-6 epochs and smaller batches give more updates
# before that. **Task 2** adopts dropout 0.5 (+0.004 AUROC averaged over validation Congresses 114-116, better
# in all 6 seed x Congress cells), the only regulariser that helped consistently. Things worth trying yourself:
# `train.batch_bills` 128 for Task 1 (+0.002 on validation, about 1.5x slower), and K = 3 (no gain so far).
#
# *Post-fix re-check (2026-11-09).* G3 is the one model whose optimisation changed with the loss-normaliser fix
# (it mixes Congresses in a batch and uses half-life 3). Re-scored on validation 114-116 under the current code
# (2 seeds, `tuning/recheck_r1/g3_t2.jsonl`), dropout 0.5 vs 0.2 on Task 2 is 0.8059 vs 0.8048: still better on
# both seeds, but now by only +0.001. So the dropout choice no longer matters; it is kept so that nothing is
# re-tuned. The Task-1 defaults are unchanged.
#
# **How to read it.** If G3 is about as good as G1, learned message passing adds little beyond fixed
# smoothing of the features plus the pair features. If G1 wins clearly (by more than the MDE of 0.01
# Recall@20), the learned, relation-aware aggregation matters.

# %%
G3 = report.run_model_section("lightgcn", cfg, corpus, timer)
RESULTS += G3
