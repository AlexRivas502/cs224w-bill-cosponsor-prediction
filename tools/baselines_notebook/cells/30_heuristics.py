# %% [markdown]
# ## 3. Heuristic references
#
# These are the bars a learned model has to clear. They are ported verbatim from the split study (same pools,
# masks and tie-breaks), so their fold-117 scores must reproduce `report/_work/digest_D.md`; the cell asserts it.
#
# Task 1 (each ranks the full eligible pool; history = Congresses 108..k-1, plus the bill's own Congress up to
# its cutoff under +stream):
# - **popularity**: how often the candidate cosponsored anything. **party+pop**: same caucus as the sponsor
#   first, then popularity.
# - **sponsor_hist**: how often the candidate cosponsored this sponsor's bills (popularity breaks ties). This is
#   the bar the split study recommends.
# - **rw2hop**: two-step random walk from the sponsor over the sponsor -> cosponsor graph (beta = 0.3).
# - **sh_hl1** (sponsor history with half-life 1 Congress) and **svd_r512_hl1** (rank-512 truncated SVD of the
#   log sponsor x cosponsor matrix, a transductive embedding reference): plain only.
# - **cosup**: co-support with the bill's day-0 cosponsors (how often each candidate cosponsored together with
#   them), the honest bar for "it only used the day-0 cosponsors".
#
# Task 2: **prior**; **chamber_maj** (rate by chamber x sponsor-in-majority); **eb_rate_t0safe** (adds policy
# area, hidden for the share of bills that have none at introduction); **eb_committee** (rate by first referral
# committee x chamber x majority, no policy area); **eb_rate** uses the policy area for every bill, which is not
# available at t0 = 0: it is a *leaky upper reference*, shown but never a baseline.

# %%
with timer("heuristics"):
    for c in FOLDS:
        for t0 in cfg.t0_list:
            if "t1" in cfg.tasks:
                RESULTS += heuristics.t1_reference_results(corpus, c, t0, protocols=("stream", "plain"), H=cfg.H)
        if "t2" in cfg.tasks:
            RESULTS += heuristics.t2_reference_results(corpus, c)

# Reproduction of the split study's fold-117 bars (asserted when fold 117 at t0 = 0 is part of this run).
if 117 in FOLDS and 0 in cfg.t0_list and not cfg.quick:
    BARS = checks.check_reference_bars([r for r in RESULTS if r.kind != "gnn"], strict=True)
    report.show(BARS[["task", "protocol", "role", "model", "reference", "value", "diff", "n", "ok"]],
                "Reference bars (fold 117, t0 = 0) vs digest_D.md:", "{:.4f}")

# %%
DF = report.results_frame(RESULTS, corpus)
for c in cfg.folds()[-1:]:
    if "t1" in cfg.tasks:
        report.show(report.show_main(DF, "t1", "test", fold=c), f"\nTask 1, fold {c}, test Congress {c}, t0 = 0:")
    if "t2" in cfg.tasks:
        report.show(report.show_main(DF, "t2", "test", fold=c), f"\nTask 2, fold {c}, test Congress {c}:")
