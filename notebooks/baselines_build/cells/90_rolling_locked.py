# %% [markdown]
# ## 9. Rolling folds and the locked test Congress 118
#
# Runs only when `MODE = "rolling"` or `LOCKED_118 = True`. Rolling mode reports every fold 112-117 and the mean
# over **clean** rows: a row is clean only if its settings were not tuned on its test Congress or on any later
# Congress (`selection` column). With `ROLLING_CONFIG = "apriori"` every row is clean; `"frozen"` rows of folds
# <= 116 are tagged and listed separately. The paired t-test compares per-fold scores with sponsor_hist (Task 1)
# and eb_rate_t0safe (Task 2). Congress 118 is printed on its own and never enters a mean or a decision.

# %%
if cfg.mode == "rolling":
    _dev = DF[DF.fold.isin(cfg.folds())]
    for _task, _base in (("t1", "sponsor_hist"), ("t2", "eb_rate_t0safe")):
        if _task not in cfg.tasks:
            continue
        _m = "Recall@20" if _task == "t1" else "AUROC"
        _per = _dev[(_dev.task == _task) & (_dev.role == "test") & (_dev.metric == _m)].pivot_table(
            index=["t0", "protocol", "model", "config_set", "selection"], columns="fold", values="value")
        report.show(_per, f"\n{_task} {_m} per test fold:")
        _rs = report.rolling_summary(_dev[_dev.task == _task], baseline=_base)
        report.show(_rs[_rs.metric == _m], f"\n{_task}: mean over clean folds, paired t-test vs {_base}:")
    _bad = report.tainted_rows(_dev)
    if len(_bad):
        report.show(_bad[["fold", "task", "model", "config_set", "selection", "metric", "value"]].drop_duplicates(),
                    "\nRows excluded from means (tuned on the test Congress or a later one):")
if cfg.locked_118:
    _l = DF[(DF.fold == 118) & (DF.role == "test")]
    for _task in cfg.tasks:
        report.show(report.show_main(_l, _task, "test", fold=118), f"\nLOCKED test Congress 118, {_task}:")
if cfg.mode != "rolling" and not cfg.locked_118:
    print("simple mode without the locked test: nothing to do here (set MODE = 'rolling' or LOCKED_118 = True).")
