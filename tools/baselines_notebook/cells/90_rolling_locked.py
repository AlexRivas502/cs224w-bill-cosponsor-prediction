# %% [markdown]
# ## 9. Rolling folds and the locked test
# Only runs with `MODE = "rolling"` or `LOCKED_118 = True`. Rolling mode prints each test fold and the mean over
# folds (paired t-test against the best heuristic). Congress 118 is printed on its own.

# %%
if cfg.mode == "rolling":
    _dev = DF[DF.fold.isin(cfg.folds())]
    for _task, _base, _m in (("t1", "sponsor_hist", "Recall@20"), ("t2", "eb_rate_t0safe", "AUROC")):
        if _task in cfg.tasks:
            _per = _dev[(_dev.task == _task) & (_dev.role == "test") & (_dev.metric == _m)].pivot_table(
                index=["t0", "protocol", "model", "config_set", "selection"], columns="fold", values="value")
            report.show(_per, f"\n{_task} {_m} per test fold:")
            _rs = report.rolling_summary(_dev[_dev.task == _task], baseline=_base)
            report.show(_rs[_rs.metric == _m], f"\n{_task}: mean over folds vs {_base} (folds tuned on test data excluded):")
if cfg.locked_118:
    _l = DF[(DF.fold == 118) & (DF.role == "test")]
    for _task in cfg.tasks:
        report.show(report.show_main(_l, _task, "test", fold=118), f"\nLocked test Congress 118, {_task}:")
if cfg.mode != "rolling" and not cfg.locked_118:
    print("simple mode: nothing to do here")
