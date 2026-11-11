# Predicting Bill Advancement and Cosponsorship with Graph Deep Learning on Congress.gov Data

**Project milestone (first draft of the final report)** · CS224W, Fall 2026
Andrew Lawlor (`adlawlor@stanford.edu`) · Alex Rivas (`alex502@stanford.edu`)

> **TL;DR.** We turned 20 years of U.S. Congress records (Congresses 108–118, about 154,000 bills and 1.9 million cosponsorships) into a leak-free temporal graph. On it we set up two prediction tasks: **will a bill advance out of committee**, and **who will cosponsor it next**. This milestone establishes strong baselines for both: simple heuristics, tabular models, and three GNNs (HeteroSAGE, R-GCN, LightGCN). The GNNs beat every heuristic by about 30% on cosponsor prediction. On bill advancement, a tabular model on the bill's own features is as strong as the GNNs. The next phase of the project is to make the graph earn its keep.

---

## 1. Why this problem

Legislative staff, advocacy groups and government-affairs platforms (Quorum, FiscalNote, GovTrack) ask two questions every day: *which bills are going to move?* and *whom should we ask for support next?* Both answers are relational. In our data, 76.6% of cosponsorships are same-party, and 54.7% come from a legislator who had already backed the same sponsor earlier in that Congress. That makes Congress a natural, public, timestamped graph benchmark in the spirit of RelBench [5].

## 2. Data

### Sources

All data comes from the **Congress.gov API** [8]: legislators, committees, and every bill with its sponsor, cosponsors (and the date each signed on), committee referrals and actions. For the bill records we used the API's bulk-download equivalent, the **BILLSTATUS** files published on GovInfo [9]. Figure 1 shows the resulting tables.

![Figure 1](docs/figures/fig_schema_compact.png)

*Figure 1: Relational schema for Congresses 108–118 (row counts per table; arrows point from foreign key to primary key).*

### What we used, and how we cleaned it

| Step | What we did | Why |
|---|---|---|
| Time range | Congresses 108–118 (2003–2024); 119 excluded | 119 is still in session, so its outcomes are unknown |
| Bills | Law-capable types only: HR, S, HJRES, SJRES (128,791 bills) | Simple and concurrent resolutions cannot become law |
| Parsing | Converted the API data from XML into Parquet tables (173,437 XML files) | Columnar tables, partitioned by Congress, are fast to load and join |
| Cosponsorships | 1,771 withdrawn cosponsorships removed (1.92M remain) | A withdrawn signature is not support |
| Point-in-time facts | Describe each legislator as they were *during that Congress* (their party and chamber then, not today). Only members actually in office on the prediction date count as possible cosponsors | Legislators switch parties, change chambers and leave office; using today's values would leak information from the future |
| Censoring | Drop bills whose label window runs past the end of their Congress | Bills die at adjournment, so their outcome can never be observed |
| Leaky fields | Never use legislative subjects, CRS summaries, later text versions, final counts (e.g. total cosponsors) or, for advancement, the policy area | These are written or updated *after* introduction and encode the outcome |

## 3. Evaluation protocol (shared by both tasks)

Both tasks predict **at the moment a bill is introduced** (prediction time *t0* = 0; cosponsor prediction also at *t0* = 7 days later), using only information dated on or before that moment. Two design choices make this honest: a temporal split, and graphs built as of the prediction time.

### A temporal split by Congress

A random split leaks the future: 45% of evaluation bills would have a related bill introduced *later* sitting in the training set. A within-Congress split, as in Russo et al. [1], leaves most late-session bills without an observable outcome. So we split **by whole Congresses, in time order**.

**Each row of Figure 2 is one fold.** Fold *c* is an honest forecast of Congress *c* that uses only earlier Congresses:
- Train on Congresses up to *c*−2.
- Tune hyperparameters and pick the stopping epoch on Congress *c*−1 (validation).
- Retrain on everything up to *c*−1 with those settings, and score Congress *c* (test) once.

**Why Congress 108 is history only.** Each Congress's graph is built from the bills of the three Congresses before it. Congress 108 is the first in our data, so it has no earlier Congresses: a training example from 108 would see an empty history, which never happens at test time. Its bills and edges are still used as history for 109–111; they are just never used as training examples.

**Why fold 117 is the simple split.** This milestone reports one fold: train on 109–115, validate on 116, retrain on 109–116, test on 117. It is the most recent fold whose test Congress is complete and settled. It is also exactly the last of the six development folds (112–117), so the simple split is not a separate setup: moving to the full protocol means running the same code on the other five folds and comparing models on the average.

**The locked test (Congress 118).** Congress 118 is the most recent completed Congress, and we hold it back. It is never used to tune or choose anything. It is scored **once**, at the very end, with the final model frozen (train up to 116, validate on 117, retrain up to 117, test on 118). Keeping one Congress untouched protects the final number from the many small choices made while developing on folds 112–117. Congress 119 is excluded entirely because it is still in session.

![Figure 2](docs/figures/fig_rolling_split.png)

*Figure 2: Rolling-origin evaluation folds.*

### Graphs built as of the prediction time

Message passing over the full graph would let a bill "see" cosponsorships that happen after its prediction time. For every target Congress *k* we therefore build one graph (Figure 3):

- **History part.** Legislators, committees and the bills of the three previous Congresses, with edges in both directions.
- **Target part.** Each new bill of Congress *k* is attached only through **incoming** edges dated on or before its cutoff: its sponsor, its cosponsors so far, and its committee referrals. No edge leaves a target bill, so target bills cannot see each other or the future.
- **Current-Congress signal.** For cosponsor prediction, what happened earlier in the same Congress (for example, how often a candidate has backed this sponsor this session) enters as leak-free pair features, computed exactly up to each bill's cutoff day.

Automated tests delete every event after a cutoff and confirm that no input changes, and a deliberately leaky feature is caught by the same test.

![Figure 3](docs/figures/fig_leakfree_graph.png)

*Figure 3: Leak-free graph for target Congress k. History edges run both ways; target bills only receive edges dated before their cutoff; labels are never edges.*

## 4. Models and training (shared by both tasks)

### Three GNN encoders

All three turn each node's own features (party, chamber, activity, bill type, …) into an embedding by repeatedly mixing in information from its neighbours, two hops deep. They differ in *how* they mix:

- **HeteroSAGE** (GraphSAGE [10] applied per relation): each node averages its neighbours separately for each kind of edge (sponsor, cosponsor, referral, …) and combines the results with its own features. It works from features rather than IDs, so it handles nodes never seen in training.
- **R-GCN** [3]: each kind of edge gets its own learned transformation. To keep the model small, the 11 edge types share 4 basis matrices and each type learns how to mix them.
- **LightGCN-feat** [11]: no learning inside the propagation. Node features are spread along the graph with a fixed normalized average, and an MLP learns from the result. It tests whether *learned* message passing is needed at all.

We also train **PairMLP**: the same model with message passing switched off. The only difference between PairMLP and the GNNs is the graph, so comparing against it isolates what message passing adds. We did not use **GIN**: its extra expressive power matters for classifying whole graphs, not for feature-rich node and link prediction. **HGT** is our candidate for the improved model.

All models share the same input encoders and output heads (Figure 4).

![Figure 4](docs/figures/fig_model_pipeline.png)

*Figure 4: Shared pipeline. One encoder per model; a node head for bill advancement and a link head (two-tower dot product plus a pair-feature MLP) for cosponsor prediction.*

### Shared training recipe

| Choice | Setting | Reason |
|---|---|---|
| Optimizer | AdamW, lr 3e-3 (1e-3 for LightGCN), weight decay 1e-4, gradient clipping at 1.0 | Standard for small GNNs; as in [1] |
| Schedule | 5% warmup, then cosine decay | Stable training on a few large graphs |
| Architecture | 2 layers, hidden size 128, dropout 0.2, DropEdge 0.1 | Two hops cover legislator → bill → legislator; deeper did not help |
| Model selection | Early stopping on the validation Congress (3-epoch moving average), then retrain on train + validation for the same number of epochs | Uses validation only; test is scored once |
| Robustness | 3 seeds; paired bootstrap confidence intervals over sponsors | Single-seed differences below about 0.01 are noise |
| Hyperparameters | One round of one-factor-at-a-time search on validation only | The validation surface was flat: only one change (LightGCN dropout 0.5 for advancement) beat seed noise |

Graphs are trained full-batch, one graph per Congress, and the whole baseline notebook runs in about 12 minutes on a laptop CPU.

## 5. Bill advancement

### 5.1 Task and metrics

**Question (node classification):** will a law-capable bill be **reported out of committee within 180 days** of introduction? Most bills never get that far: only 5.6% of the 110,802 uncensored bills do, and 2.8% in our test Congress (117).

**Metrics: AUROC (headline) and average precision (AP).**
- **Accuracy is useless here.** A model that predicts "no" for every bill is 97% accurate.
- **AUROC** is the probability that a random advancing bill is scored above a random stalled one. It needs no decision threshold, and it does not depend on the base rate, which varies between Congresses (from about 3% to 9%). That makes it the fairest number for comparing models across Congresses. Random = 0.5.
- **AP** focuses on the top of the ranking, which is what a user acting on the top predictions cares about. Random AP equals the positive rate (0.028), so it shows how far above chance a model is.

### 5.2 Baselines and why they fit this task

| Model | Why it is a sensible baseline for advancement |
|---|---|
| Prior (random) | The floor: what AUROC and AP look like with no information |
| Committee report rate | Committees are the gatekeepers: a bill only advances if its committee reports it. A committee's historical report rate is the obvious domain heuristic |
| Logistic regression, gradient-boosted trees (bill features) | Advancement is a property of a single bill: who sponsors it, whether they are in the majority, which committees it goes to, how many original cosponsors it has. A strong tabular model on those features tells us how much signal exists **without any graph**; gradient-boosted trees are the standard baseline in relational benchmarks [5] |
| PairMLP | The neural model without message passing, so any GNN gain over it comes from the graph |
| HeteroSAGE | A bill's chances depend on its sponsor's and committees' track records; HeteroSAGE pulls those in by averaging over the bill's neighbours, and stays inductive for new sponsors |
| R-GCN | "Sponsored by" and "referred to" carry very different meaning for advancement; R-GCN learns a separate transformation per edge type |
| LightGCN-feat | Smooths the sponsors' and committees' history signals over the graph cheaply, to test whether a learned propagation is needed |
| Kumo Relational (in progress) | A relational foundation model [6, 7] that learns from the database directly, a reference point for what automatic relational learning achieves |

**Loss:** binary cross-entropy, no class weights. AUROC and AP depend only on the ranking, so reweighting the rare positives does not help.

### 5.3 Results

Test Congress 117 has 12,953 bills, 2.8% positive. Learned models show the mean over 3 seeds; brackets are 95% bootstrap CIs.

| Model | Validation AUROC (116) | Test AUROC (117) | Test AP |
|---|---|---|---|
| Prior (random) | 0.500 | 0.500 | 0.028 |
| Committee report rate (heuristic) | 0.771 | 0.726 [0.688, 0.761] | 0.079 |
| Logistic regression (bill features) | 0.771 | 0.797 [0.765, 0.824] | 0.128 |
| Gradient-boosted trees (bill features) | 0.792 | 0.820 [0.792, 0.846] | 0.152 |
| PairMLP (no message passing) | 0.792 | 0.801 [0.768, 0.829] | 0.135 |
| HeteroSAGE | 0.801 | 0.805 [0.770, 0.833] | 0.151 |
| R-GCN | 0.804 | 0.810 [0.776, 0.838] | 0.154 |
| **LightGCN-feat** | 0.801 | **0.825** [0.797, 0.850] | **0.162** |
| Kumo Relational | *to be added* | | |

**What this shows.** The learned models clearly beat the heuristics (+0.08 AUROC over the committee report rate). But most of the signal sits in the bill's own features: who sponsors it, which committees it goes to, and how many original cosponsors it has. Gradient-boosted trees on those features alone reach 0.820. HeteroSAGE and R-GCN do not beat them, and LightGCN only ties them (+0.005, within noise). For this task, a tabular model is the bar the graph has to clear.

### 5.4 Kumo Relational baseline (in progress)

*Placeholder for the Kumo Relational foundation-model baseline [6, 7]* 
For a fair comparison it must use the same definitions as the table above:
- **Label:** reported out of committee within 180 days.
- **Population:** HR, S, HJRES and SJRES bills.
- **Split:** train on Congresses up to 115, validate on 116, test on 117.
- **Censoring:** censored bills dropped, leaving 12,953 test bills.
- **Prediction time:** the bill's introduction date, with no later information.

Results and per-bill predictions (for paired CIs) will be added here.

## 6. Cosponsor prediction

### 6.1 Task and metrics

**Question (temporal link prediction):** given a new bill, rank every legislator who could still sign on (same chamber, in office, not the sponsor, not already a cosponsor) by how likely they are to **cosponsor within the next 30 days**.

**Metric: Recall@20 (headline), plus Recall@10 and NDCG@20.**
- **It matches the use case.** Recall@20 is the share of a bill's true new cosponsors that appear in the model's top 20, a realistic shortlist for staff deciding whom to contact.
- **It treats bills equally.** It divides by the number of true cosponsors, so a bill with 1 new cosponsor counts as much as a bill with 30. Precision would be dominated by the popular bills.
- **It ranks the full pool.** We score all ~435 House members (or ~100 senators) for every bill, never a sampled subset, so the number reflects the real difficulty.
- **NDCG@20** additionally rewards putting the true cosponsors near the top of the list.

### 6.2 Baselines and why they fit this task

| Model | Why it is a sensible baseline for cosponsor prediction |
|---|---|
| Party + popularity | 76.6% of cosponsorships are same-party, and some legislators simply cosponsor far more than others |
| Sponsor history | The strongest single signal: 54.7% of cosponsorships come from someone who already backed the same sponsor earlier in the Congress |
| 2-hop random walk | "Friends of friends": legislators who cosponsor alongside the sponsor's usual supporters are likely next |
| Learned embeddings (SVD) | Classic collaborative filtering from recommender systems: legislator–bill matrix factorization. It cannot score legislators it never saw |
| "+ current Congress" variants | The same heuristics, also counting what happened earlier in the same session, up to the bill's cutoff |
| PairMLP | Learns how to combine the leak-free pair features (sponsor history, co-support, same party/state, same-session activity) with legislator features, without message passing |
| HeteroSAGE | Inductive: builds legislators from features and neighbours, so it can rank the 13% of test cosponsors who are new legislators |
| R-GCN | Sponsoring, cosponsoring and sharing a committee are different relationships; R-GCN weighs each separately |
| LightGCN-feat | Comes from recommender systems, where users and items form a bipartite graph, just like legislators and bills |

**Loss:** a listwise softmax over each bill's **full** eligible pool. It optimizes the ranking directly, needs no negative sampling, and the pool already contains hard negatives (same party, same state). The pairwise BPR loss was 0.003–0.005 Recall@20 worse.

### 6.3 Results

Prediction at introduction (*t0* = 0). Test Congress 117 has 5,101 bills with at least one new cosponsor. Learned models show the mean over 3 seeds.

| Model | Val Recall@20 (116) | Test Recall@20 (117) | Test Recall@10 | Test NDCG@20 |
|---|---|---|---|---|
| Party + popularity | 0.204 | 0.191 | 0.113 | 0.115 |
| Sponsor history (earlier Congresses only) | 0.254 | 0.262 | 0.173 | 0.166 |
| Learned embeddings (SVD) | 0.276 | 0.283 | 0.191 | 0.182 |
| Sponsor history (+ current Congress) | 0.288 | 0.307 | 0.204 | 0.193 |
| 2-hop random walk (+ current Congress) | 0.286 | 0.308 | 0.199 | 0.192 |
| PairMLP (no message passing) | 0.367 | 0.386 | 0.251 | 0.241 |
| HeteroSAGE | 0.375 | 0.393 | 0.254 | 0.245 |
| R-GCN | 0.374 | 0.394 | 0.257 | 0.245 |
| **LightGCN-feat** | 0.371 | **0.400** | **0.260** | **0.249** |

**What this shows.**
- **Every GNN beats the best heuristic by about 0.09 Recall@20**, roughly 30% better. In practice, the model's top 20 contains about 40% of a bill's future cosponsors, against 31% for the best heuristic.
- **Most of the gain is already in PairMLP.** It comes from learning to combine leak-free pair features with point-in-time legislator features.
- **Message passing adds little so far.** Only LightGCN clearly beats PairMLP (+0.013 over 3 seeds); the three GNNs cannot be ranked against each other.
- **New legislators** (13% of test cosponsors) are found in the top 20 about 23% of the time by the learned models, against 5% for the heuristics. That is the payoff of inductive features.

## 7. Next steps: how can we make message passing matter?

On both tasks, almost all of the gain over heuristics comes from features the model sees *without* the graph: the bill's own features for advancement, the hand-built pair features for cosponsor prediction. The central question for the rest of the project is how to get more out of the graph itself.

### Techniques to try

1. **Attention over relations and neighbours.** HGT or GATv2 let a bill weigh *which* cosponsors and committees matter instead of averaging them all equally.
2. **Time inside message passing.** Encode how long ago each edge happened (time encodings as in TGAT, or recency-decayed messages), so recent collaborations count more than old ones.
3. **A direct legislator–legislator graph.** Add co-cosponsorship edges between legislators. Today a collaboration is two hops away (legislator → bill → legislator), which averaging dilutes.
4. **Link-prediction-aware GNNs.** Plain GNNs score a bill and a legislator separately and then compare them, so they struggle to recover pair signals like "has this legislator backed this sponsor before". Labeling-trick methods (SEAL, NBFNet) encode the pair inside the message passing.
5. **Hybrid node representations.** Add learned ID embeddings for returning legislators on top of their features, keeping the features for new legislators.
6. **Harder training signal.** Hard-negative mining and curriculum learning, as in PinSAGE, for cosponsor prediction.
7. **Richer leak-free node features.** Better embeddings of the introduced title (only the version at introduction), and the policy areas of *finished* history bills.
8. **Stacking and multi-task learning.** Feed GNN embeddings into gradient-boosted trees for advancement, and train both tasks jointly on the shared encoder.

### Ablations and other tests

- **Remove the pair features** from the GNNs: can message passing recover them from the graph alone?
- **Remove edge types** one at a time (committees, history bills, cosponsor edges) to see which relations carry signal.
- **Remove the current-Congress signal**, and vary the history window (2, 3 or 4 previous Congresses).
- **Answer the research question:** how much does the first week of support help? Compare predicting at day 0 with predicting at day 7.
- **Break results down** by chamber, party, majority vs minority sponsors, and new vs returning legislators.
- **Run the full protocol:** six rolling folds (112–117) with paired tests, then score the locked Congress 118 once.
- **Add the Kumo Relational baseline** for bill advancement (Section 5.4).

## Code

- `data_processing/`: data download (`bulk_bill_download.sh`, `fetch_congress.py`) and the XML-to-Parquet ETL (`parsers.py`).
- `notebooks/01_split_analysis.ipynb`: evidence for the split.
- `notebooks/02_baselines.ipynb`: every baseline in this report. Run instructions are in `docs/BASELINES_RUN.md`, and every model, loss and hyperparameter choice is documented in `docs/BASELINES.md`.

## References

[1] Russo, G., et al. (2023) Helping a friend or supporting a cause? Disentangling active and passive cosponsorship in the U.S. Congress. *ACL 2023*, pp. 2952–2969.
[2] Sotoudeh, S., et al. (2026) A network-based measure of cosponsorship influence on bill passage in the United States House of Representatives. *EPJ Data Science*.
[3] Schlichtkrull, M., et al. (2018) Modeling relational data with graph convolutional networks. *The Semantic Web*, pp. 593–607.
[4] Fey, M., et al. (2023) Relational deep learning: Graph representation learning on relational databases. *arXiv:2312.04615*.
[5] Robinson, J., et al. (2024) RelBench: A benchmark for deep learning on relational databases. *arXiv:2407.20060*.
[6] Hudovernik, V., et al. (2026) KumoRFM-2: Scaling foundation models for relational learning. *arXiv:2604.12596*.
[7] NVIDIA (2026) Kumo Relational model card, version 1.0.0.
[8] Library of Congress. Congress.gov API. https://github.com/LibraryOfCongress/api.congress.gov
[9] U.S. Government Publishing Office. GovInfo bulk data repository. https://www.govinfo.gov/bulkdata
[10] Hamilton, W., Ying, R. & Leskovec, J. (2017) Inductive representation learning on large graphs. *NeurIPS*.
[11] He, X., et al. (2020) LightGCN: Simplifying and powering graph convolution network for recommendation. *SIGIR*.
