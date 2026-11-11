# %% [markdown]
# # Choosing a train / validation / test split
#
# > **Disclaimer:** this entire notebook was created with Claude (Anthropic's AI assistant) for exploration purposes only.
#
# **Goal:** decide how to split the Congress data (Congresses 108–118) for the two tasks in the proposal:
#
# 1. **Cosponsor prediction** (temporal link prediction): for a bill at time *t* (day 0 or day 7 after
#    introduction), rank the eligible legislators who will cosponsor it in *(t, t+30]*.
# 2. **Bill advancement** (node classification): will a law-capable bill (HR, S, HJRES, SJRES) be
#    reported out of committee within 180 days of introduction?
#
# The default idea is a **by-Congress temporal split** (train on earlier Congresses, validate on the
# next, test on the last). That split is only trustworthy if a set of conditions hold: similar label
# rates, similar feature distributions, enough node overlap for the graph model, no leakage, fully
# observed labels, and stable baseline scores across folds. Each section below tests a group of those
# conditions and records a **check** (pass / fail / info). The last section collects every check and
# compares the candidate splits.
#
# **Run it:** JupyterLab, kernel with `pandas`, `pyarrow`, `numpy`, `scipy` (see
# `notebooks/requirements.txt`). Charts use D3 v7, vendored at `notebooks/vendor/d3.v7.min.js` (no
# internet needed). Run the cells top to bottom; charts appear as interactive SVGs (hover for values).

# %%
# --- Setup: imports, paths, constants -------------------------------------------------------------
import json
import os
import uuid
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 30)
pd.set_option("display.float_format", lambda v: f"{v:,.4f}")


def _find_root():
    """Repo root = first parent (from the working dir) that contains data/parsed/."""
    here = Path.cwd().resolve()
    for p in [here, *here.parents]:
        if (p / "data" / "parsed").is_dir():
            return p
    raise FileNotFoundError("Couldn't find the data/parsed/ folder; run the notebook from inside the repo.")


ROOT = _find_root()
PARSED = ROOT / "data" / "parsed"
VENDOR_D3 = ROOT / "notebooks" / "vendor" / "d3.v7.min.js"

CONGRESSES = list(range(108, 119))   # complete Congresses used for modeling
IN_PROGRESS = 119                    # still in session: labels censored, excluded from splits
LAW_TYPES = ["hr", "s", "hjres", "sjres"]
WINDOW_DAYS = 30                     # cosponsor prediction horizon
ADV_WINDOW_DAYS = 180                # advancement horizon
PARTY_CODE = {"Republican": "R", "Democratic": "D", "Independent": "I",
              "Independent Democrat": "ID", "Libertarian": "L"}
CHAMBER_OF_ORIGIN = {"House": "House of Representatives", "Senate": "Senate"}


def congress_start(c):
    """Congress c convenes on Jan 3 of 1789 + 2(c-1)."""
    return pd.Timestamp(year=1789 + 2 * (int(c) - 1), month=1, day=3)


def congress_end(c):
    return congress_start(int(c) + 1)


print(f"repo: {ROOT}\nparsed tables: {len(list(PARSED.glob('*.parquet')))}")

# %%
# --- Data loading ---------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def load(table, columns=None):
    """Read data/parsed/<table>.parquet. `congress` becomes int; date-like object columns -> datetime."""
    cols = list(columns) if columns else None
    df = pd.read_parquet(PARSED / f"{table}.parquet", columns=cols)
    if "congress" in df.columns:
        df["congress"] = df["congress"].astype(int)
    for c in df.columns:
        if df[c].dtype == object and (c.endswith("_date") or c == "action_date"):
            df[c] = pd.to_datetime(df[c], errors="coerce")
        elif str(df[c].dtype).startswith("datetime64") and getattr(df[c].dt, "tz", None) is not None:
            df[c] = df[c].dt.tz_convert(None)
    return df


bills = load("bills").copy()
bills["law_capable"] = bills["bill_type"].isin(LAW_TYPES)
bills["chamber"] = bills["origin_chamber"].map(CHAMBER_OF_ORIGIN)
bills["day_of_congress"] = (bills["introduced_date"] - bills["congress"].map(congress_start)).dt.days
# Harmonised CRS policy area: categories absent from at least one complete Congress (taxonomy changes, e.g.
# a retired category) and missing labels fold into one bucket, so every section uses the same taxonomy.
_pa = bills[bills.law_capable & bills.congress.isin(CONGRESSES)].assign(pa=lambda d: d.policy_area.fillna("(missing)"))
_pa_tab = pd.crosstab(_pa.pa, _pa.congress)
PA_UNSTABLE = sorted(set(_pa_tab.index[(_pa_tab == 0).any(axis=1)]) | {"(missing)"})
bills["policy_area_harm"] = bills.policy_area.fillna("(missing)").where(
    lambda s: ~s.isin(PA_UNSTABLE), "(unstable/missing)")

# Data snapshot date = latest record modification (bills.update_date), normalised to the day. Used for record
# ages in the in-progress Congress (the only one observed before CRS and committees catch up).
SNAPSHOT_DATE = bills["update_date"].max().normalize()
SNAPSHOT_LATEST_INTRO = bills["introduced_date"].max()
# Policy area missing at prediction time (proxy): share of the in-progress Congress's law-capable bills without a
# CRS policy area, by age at SNAPSHOT_DATE. PA_MISS_T0 = the youngest bucket (<= 7 days), with its n.
PA_AGE_BUCKETS = [(0, 7, "0-7d"), (8, 30, "8-30d"), (31, 90, "31-90d"), (91, 180, "91-180d"), (181, 10**6, ">180d")]
_b_ip = bills[bills.law_capable & (bills.congress == IN_PROGRESS)]
_age_ip = (SNAPSHOT_DATE - _b_ip.introduced_date).dt.days
PA_MISSING_AT_T0 = pd.DataFrame([dict(age_bucket=lab, age_lo=lo, age_hi=min(hi, int(_age_ip.max())),
                                      n=int(m.sum()), n_missing=int(_b_ip.policy_area[m].isna().sum()),
                                      share_missing=float(_b_ip.policy_area[m].isna().mean()) if m.any() else np.nan)
                                 for lo, hi, lab in PA_AGE_BUCKETS for m in [_age_ip.between(lo, hi)]]
                                ).set_index("age_bucket")
PA_MISS_T0 = float(PA_MISSING_AT_T0.loc["0-7d", "share_missing"])
PA_MISS_T0_N = int(PA_MISSING_AT_T0.loc["0-7d", "n"])

members = load("members").copy()
members["party_code"] = members["party"].map(PARTY_CODE).fillna(members["party"])   # LATEST party (legacy)

# Member terms (year granularity) and party history: fill missing chambers and derive the party AS OF each
# Congress. As-of rule: the party-history interval containing Jan 1 of the Congress's second year
# (start_year <= first year and end_year >= second year); members not in office then (joined in year 2 or
# left in year 1) take the latest interval containing their term start. Fallback: latest party.
_terms = load("member_terms").copy()
_terms["y0"] = 1789 + 2 * (_terms.congress - 1)
_ch_fill = (_terms.sort_values("start_year").drop_duplicates(["bioguide_id", "congress"], keep="last")
            .set_index(["bioguide_id", "congress"])["chamber"])
_n_ch_missing = int(members.chamber.isna().sum())
members["chamber"] = members["chamber"].fillna(
    pd.Series(list(zip(members.bioguide_id, members.congress))).map(_ch_fill).set_axis(members.index))


def _party_asof_table():
    """(bioguide_id, congress) -> party code as of that Congress, from member_party_history."""
    ph = load("member_party_history", ("bioguide_id", "party_code", "start_year", "end_year"))
    t = (_terms.sort_values("start_year").drop_duplicates(["bioguide_id", "congress"])
         [["bioguide_id", "congress", "y0", "start_year"]].rename(columns={"start_year": "t0"}))
    h = t.merge(ph.rename(columns={"start_year": "h0", "end_year": "h1"}), on="bioguide_id")
    h1 = h.h1.fillna(9999)
    h["prio"] = np.select([(h.h0 <= h.y0) & (h1 >= h.y0 + 1), (h.h0 <= h.t0) & (h1 >= h.t0)], [2, 1], 0)
    h = h[h.prio > 0].sort_values(["prio", "h0"]).drop_duplicates(["bioguide_id", "congress"], keep="last")
    return h.set_index(["bioguide_id", "congress"])["party_code"]


_pasof = _party_asof_table()
members["party_code_asof"] = (pd.Series(list(zip(members.bioguide_id, members.congress))).map(_pasof)
                              .set_axis(members.index).fillna(members["party_code"]))
_n_switch = int((members.party_code_asof != members.party_code).sum())

print(f"bills: {len(bills):,} ({bills.law_capable.sum():,} law-capable); "
      f"members rows: {len(members):,}; congresses: {bills.congress.min()}-{bills.congress.max()}")
print(f"members: chamber filled from member_terms for {_n_ch_missing - int(members.chamber.isna().sum())}/"
      f"{_n_ch_missing} rows; party_code_asof differs from latest party_code in {_n_switch} member-Congress rows; "
      f"policy areas folded into '(unstable/missing)': {len(PA_UNSTABLE)}")
print(f"snapshot (max update_date): {SNAPSHOT_DATE.date()} (latest introduced_date {SNAPSHOT_LATEST_INTRO.date()}); "
      f"policy area missing among {IN_PROGRESS}th law-capable bills by age at snapshot: "
      + ", ".join(f"{r.Index} {r.share_missing:.0%} (n={r.n})" for r in PA_MISSING_AT_T0.itertuples())
      + f" -> PA_MISS_T0 = {PA_MISS_T0:.1%} (n={PA_MISS_T0_N}, small-sample proxy)")

# %%
# --- Shared task definitions (every section uses these, so labels are identical everywhere) -------
def build_cosponsor_events():
    """One row per active (non-withdrawn) cosponsorship with its lag from introduction (days)."""
    c = load("bill_cosponsors", ("bill_id", "bioguide_id", "party", "sponsorship_date", "is_original",
                                 "withdrawn_date", "congress"))
    c = c[c["withdrawn_date"].isna()].merge(
        bills[["bill_id", "introduced_date", "sponsor_bioguide_id", "sponsor_party", "chamber",
               "bill_type", "law_capable"]], on="bill_id", how="inner")
    c["lag_days"] = (c["sponsorship_date"] - c["introduced_date"]).dt.days
    return c


cosp = build_cosponsor_events()


def window_targets(t0, horizon=WINDOW_DAYS, law_capable_only=True, point_in_time=False, include_delegates=True):
    """Per bill: number of new cosponsors in (t0, t0+horizon] and the size of the eligible pool.

    Eligible = members of the bill's chamber, minus the sponsor and anyone who had already cosponsored by
    day t0. point_in_time=False (default, legacy): pool = everyone who served in that chamber that Congress.
    point_in_time=True: pool = members in office in the month of introduced_date + t0 (clipped into the bill's
    Congress); adds column `pool_month`. include_delegates=True (default; Task-1 pool): the pool comes from
    ROSTER_ASOF_ALL (voting members + Delegates / Resident Commissioners, who can cosponsor); False: the
    voting-only ROSTER_ASOF (pre-change behaviour). Bills whose window runs past the end of the Congress are
    flagged `censored` (exclude them from evaluation)."""
    b = bills[bills.law_capable] if law_capable_only else bills
    b = b[b.congress.isin(CONGRESSES + [IN_PROGRESS])][["bill_id", "congress", "chamber", "introduced_date"]].copy()
    pos = cosp[(cosp.lag_days > t0) & (cosp.lag_days <= t0 + horizon)].groupby("bill_id").size()
    seen = cosp[cosp.lag_days <= t0].groupby("bill_id").size()
    b["n_pos"] = b["bill_id"].map(pos).fillna(0).astype(int)
    b["n_seen"] = b["bill_id"].map(seen).fillna(0).astype(int)
    if point_in_time:
        ref = (b["introduced_date"] + pd.Timedelta(days=t0)).clip(
            upper=b["congress"].map(congress_end) - pd.Timedelta(days=ROSTER_REF_DAY + 5))
        b["pool_month"] = ref.dt.to_period("M").dt.to_timestamp()
        pool = (ROSTER_ASOF_ALL if include_delegates else ROSTER_ASOF).groupby(["month", "congress", "chamber"])["bioguide_id"].size()
        b["pool"] = pool.reindex(pd.MultiIndex.from_arrays([b.pool_month, b.congress, b.chamber])).to_numpy(float)
        b["eligible"] = (b["pool"] - 1 - b["n_seen"]).clip(lower=0)
    else:
        pool = members.dropna(subset=["chamber"]).groupby(["congress", "chamber"])["bioguide_id"].nunique()
        b["pool"] = [pool.get((c, ch), np.nan) for c, ch in zip(b.congress, b.chamber)]
        b["eligible"] = b["pool"] - 1 - b["n_seen"]
    b["censored"] = b["introduced_date"] + pd.Timedelta(days=t0 + horizon) > b["congress"].map(congress_end)
    return b


def adv_cutoff(c):
    """Last introduction date whose 180-day advancement window ends inside Congress c (later = censored)."""
    return congress_end(c) - pd.Timedelta(days=ADV_WINDOW_DAYS)


def build_advancement_labels(window=ADV_WINDOW_DAYS):
    """Per law-capable bill: reported out of committee within `window` days; `censored` if the window
    runs past the end of the Congress (unpassed bills expire at adjournment).

    Alternative label (label_alt): reported OR passed the chamber of origin within `window` days, which also
    counts bills that advance without a committee report (suspension / unanimous consent).
    `passed_origin_date` = first 'Passed/agreed to in <origin chamber>' action. (Named passed_origin_date, not
    passed_date, so it does not collide with section A's local `passed_date` merge.)"""
    a = load("bill_actions", ("bill_id", "action_date", "action_type", "text", "congress"))
    rep = a[a.action_type.isin(["Committee", "Calendars"])
            & a.text.str.contains(r"\bReported\b", regex=True, na=False)
            & ~a.text.str.contains("Ordered to be", na=False)]
    first_rep = rep.groupby("bill_id")["action_date"].min()
    pas = a[a.text.str.match(r"Passed/agreed to in (House|Senate)", na=False)]
    pas = pas.assign(chamber=pas.text.str.extract(r"in (House|Senate)", expand=False).map(CHAMBER_OF_ORIGIN))
    first_pass = pas.groupby(["bill_id", "chamber"])["action_date"].min()
    b = bills[bills.law_capable][["bill_id", "congress", "chamber", "introduced_date", "sponsor_party",
                                  "policy_area", "policy_area_harm"]].copy()
    b["reported_date"] = b["bill_id"].map(first_rep)
    days = (b["reported_date"] - b["introduced_date"]).dt.days
    b["label"] = days.between(0, window).astype(int)
    b["censored"] = b["introduced_date"] + pd.Timedelta(days=window) > b["congress"].map(congress_end)
    b["passed_origin_date"] = first_pass.reindex(pd.MultiIndex.from_arrays([b.bill_id, b.chamber])).to_numpy()
    pdays = (b["passed_origin_date"] - b["introduced_date"]).dt.days
    b["label_alt"] = ((b["label"] == 1) | pdays.between(0, window)).astype(int)
    return b


# --- Who is in office when, and which party they caucus with -----------------------------------------
NONVOTING_TYPES = ("Delegate", "Resident Commissioner")   # member_terms.member_type of non-voting members


@lru_cache(maxsize=None)
def _activity_span():
    """First / last dated activity (sponsorship or cosponsorship) per (congress, bioguide_id)."""
    act = pd.concat([cosp[["congress", "bioguide_id", "sponsorship_date"]].set_axis(["congress", "bioguide_id", "d"], axis=1),
                     bills[["congress", "sponsor_bioguide_id", "introduced_date"]].set_axis(["congress", "bioguide_id", "d"], axis=1)])
    return act.dropna().groupby(["congress", "bioguide_id"])["d"].agg(first="min", last="max")


def in_office_at(date, voting_only=True, slack_days=30):
    """Members in office on `date`: DataFrame (bioguide_id, congress, chamber, state_code, district).

    member_terms has year granularity, so a term of the Congress containing `date` counts if
    start_year <= year(date) <= end_year. Then, using activity dates (first / last sponsorship or
    cosponsorship in that Congress; activity is not chamber-specific):
      * a member with terms in both chambers keeps the earlier one (the switch happens later that year);
      * seats with more members than they hold (1 per House district, 2 per Senate state) keep the members
        whose activity span is closest to `date` (ties: the later arrival);
      * other terms starting or ending mid-Congress in year(date) need activity within +-slack_days of date.
    Approximate by construction (year-granular terms). voting_only drops Delegates / Resident Commissioners."""
    date = pd.Timestamp(date)
    y = date.year
    c = 1 + (y - 1789) // 2 - (1 if (y % 2 == 1 and date < pd.Timestamp(year=y, month=1, day=3)) else 0)
    t = _terms[(_terms.congress == c) & (_terms.start_year <= y) & (_terms.end_year.fillna(9999) >= y)]
    if voting_only:
        t = t[~t.member_type.isin(NONVOTING_TYPES)]
    t = t.sort_values("start_year", kind="stable").drop_duplicates("bioguide_id").copy()
    t["seat"] = t.state_code + "-" + np.where(t.chamber == "Senate", "S", t.district.fillna(0).astype(int).astype(str))
    span = _activity_span().reindex(pd.MultiIndex.from_arrays([t.congress, t.bioguide_id]))
    first, last = span["first"].set_axis(t.index), span["last"].set_axis(t.index)
    t["dist"] = (pd.concat([(first - date).dt.days, (date - last).dt.days], axis=1).max(axis=1)
                 .clip(lower=0).fillna(1e9))
    t["first"] = first
    cap = np.where(t.chamber == "Senate", 2, 1)
    over = t.groupby(["chamber", "seat"])["bioguide_id"].transform("size").to_numpy() > cap
    t["rank"] = (t.sort_values(["dist", "first"], ascending=[True, False]).groupby(["chamber", "seat"]).cumcount()
                 .reindex(t.index))
    edge = (((t.start_year == y) & (y != t.y0)) | ((t.end_year == y) & (y != t.y0 + 2))).to_numpy()
    keep = np.where(over, t["rank"].to_numpy() < cap, ~edge | (t["dist"].to_numpy() <= slack_days))
    return t.loc[keep, ["bioguide_id", "congress", "chamber", "state_code", "district"]]


def congress_midpoint(c):
    """Reference date for point-in-time rosters: one year after the Congress convenes."""
    return congress_start(c) + pd.Timedelta(days=365)


# Caucus: D/R members caucus with their as-of party. Others (I, ID, L) are assigned from the data: the party
# whose bills they cosponsor more often in that Congress (share of D vs R sponsors > 0.5); exact ties -> 'other'.
CAUCUS_MIN_SHARE = 0.5
_dshare = (cosp[cosp.sponsor_party.isin(["D", "R"])].assign(isd=lambda d: d.sponsor_party == "D")
           .groupby(["bioguide_id", "congress"])["isd"].mean())
_mk = pd.MultiIndex.from_arrays([members.bioguide_id, members.congress])
members["caucus"] = np.where(members.party_code_asof.isin(["D", "R"]), members.party_code_asof,
                             np.select([_dshare.reindex(_mk).to_numpy() > CAUCUS_MIN_SHARE,
                                        _dshare.reindex(_mk).to_numpy() < 1 - CAUCUS_MIN_SHARE], ["D", "R"], "other"))

# Point-in-time roster: members in office on day ROSTER_REF_DAY of each month (in_office_at), from the start of
# the first complete Congress to the snapshot. One row per (month, member).
#   ROSTER_ASOF_ALL: voting members + Delegates / Resident Commissioners (column `voting` = False for them). This is
#     the Task-1 eligible pool (delegates can cosponsor; they cast most positives outside the voting-only pool, C17)
#     and the default of window_targets(point_in_time=True) and roster_asof(date).
#   ROSTER_ASOF: voting members only (unchanged; use for seat / majority counts, which exclude delegates).
# roster_asof(date, include_delegates=True) returns the rows of the month containing `date`.
ROSTER_REF_DAY = 15
_ROSTER_COLS = ["month", "congress", "chamber", "bioguide_id", "caucus", "party_code_asof", "state_code", "district"]


def _build_roster_asof():
    """ROSTER_ASOF_ALL (all members, with `voting`). Voting rows come from in_office_at(voting_only=True) so the
    voting-only roster is exactly the pre-change ROSTER_ASOF; delegates are added from voting_only=False."""
    months = pd.date_range(congress_start(CONGRESSES[0]).to_period("M").to_timestamp(), SNAPSHOT_DATE, freq="MS")
    cau = members.drop_duplicates(["bioguide_id", "congress"]).set_index(["bioguide_id", "congress"])
    nonvoting = set(_terms.loc[_terms.member_type.isin(NONVOTING_TYPES), "bioguide_id"])
    parts = []
    for m in months:
        d = m + pd.Timedelta(days=ROSTER_REF_DAY - 1)
        v = in_office_at(d).assign(voting=True)
        a = in_office_at(d, voting_only=False)
        dl = a[a.bioguide_id.isin(nonvoting) & ~a.bioguide_id.isin(v.bioguide_id)].assign(voting=False)
        parts += [v.assign(month=m), dl.assign(month=m)]
    r = pd.concat(parts, ignore_index=True)
    k = pd.MultiIndex.from_arrays([r.bioguide_id, r.congress])
    r["caucus"] = k.map(cau["caucus"]).fillna("other")
    r["party_code_asof"] = k.map(cau["party_code_asof"])
    return r[_ROSTER_COLS + ["voting"]].sort_values(["month", "voting"], ascending=[True, False], kind="stable").reset_index(drop=True)


ROSTER_ASOF_ALL = _build_roster_asof()
ROSTER_ASOF = ROSTER_ASOF_ALL.loc[ROSTER_ASOF_ALL.voting, _ROSTER_COLS].reset_index(drop=True)


def roster_asof(date, include_delegates=True):
    """Members in office in the month containing `date`: rows of ROSTER_ASOF_ALL (default, Task-1 pool, with
    `voting`) or of the voting-only ROSTER_ASOF (include_delegates=False)."""
    m = pd.Timestamp(date).to_period("M").to_timestamp()
    r = ROSTER_ASOF_ALL if include_delegates else ROSTER_ASOF
    return r[r.month == m]

# Senate ties (50-50 after caucusing) are broken by the Vice President, whose party is not in the parsed
# data (committee_history has no chairs). Documented constant: VP party = President's party.
VP_PARTY = {108: "R", 109: "R", 110: "R", 111: "D", 112: "D", 113: "D", 114: "D", 115: "R", 116: "R",
            117: "D", 118: "D", 119: "R"}


@lru_cache(maxsize=None)
def _majority_party_table():
    """Majority party per (congress, chamber), point-in-time at the Congress midpoint.

    Counts voting members in office at congress_midpoint(c) (in_office_at; delegates excluded), each with
    the party AS OF that Congress (members.party_code_asof) and independents assigned to their caucus
    (members.caucus, data-derived). Ties are broken by the Vice President's party (Senate only, VP_PARTY),
    cross-checked against the data: the party sponsoring most committee-reported bills in that chamber.
    Columns: congress, chamber, majority, seats_D, seats_R, seats_other, n_voting, tie_broken_by."""
    rows = []
    cau = members.drop_duplicates(["bioguide_id", "congress"]).set_index(["bioguide_id", "congress"])["caucus"]
    for c in sorted(members.congress.unique()):
        io = in_office_at(congress_midpoint(c))
        io = io[io.congress == c]
        io = io.assign(caucus=pd.MultiIndex.from_arrays([io.bioguide_id, io.congress]).map(cau)).fillna({"caucus": "other"})
        for ch, g in io.groupby("chamber"):
            n = g.caucus.value_counts()
            d, r = int(n.get("D", 0)), int(n.get("R", 0))
            row = dict(congress=int(c), chamber=ch, seats_D=d, seats_R=r, seats_other=int(len(g) - d - r),
                       n_voting=int(len(g)))
            if d != r:
                row.update(majority="D" if d > r else "R", tie_broken_by="none (seat count)")
            else:
                vp = VP_PARTY.get(int(c)) if ch == "Senate" else None
                rep = adv[(adv.congress == c) & (adv.chamber == ch) & (adv.label == 1) & adv.sponsor_party.isin(["D", "R"])]
                dat = rep.sponsor_party.value_counts(normalize=True)
                dat_p = dat.idxmax() if len(dat) else None
                chk = (f"reported-bill sponsors {dat_p} {dat.max():.0%} "
                       f"({'agrees' if dat_p == vp else 'DISAGREES'})" if dat_p else "no reported bills")
                row.update(majority=vp or dat_p, tie_broken_by=(f"VP party ({chk})" if vp else f"data: {chk}"))
            rows.append(row)
    cols = ["congress", "chamber", "majority", "seats_D", "seats_R", "seats_other", "n_voting", "tie_broken_by"]
    return pd.DataFrame(rows)[cols]


def majority_party():
    """Majority party per (congress, chamber); see _majority_party_table for the derivation. Returns a copy
    with columns congress, chamber, majority, seats_D, seats_R, seats_other, n_voting, tie_broken_by."""
    return _majority_party_table().copy()


adv = build_advancement_labels()
MAJORITY = majority_party()
print(f"cosponsorship events: {len(cosp):,}; advancement labels: {len(adv):,} "
      f"({adv[~adv.censored].label.mean():.1%} positive among uncensored; alternative label_alt = reported OR "
      f"passed origin chamber within {ADV_WINDOW_DAYS} d: {adv[~adv.censored].label_alt.mean():.1%})")
_rs = ROSTER_ASOF[ROSTER_ASOF.congress.isin(CONGRESSES)].groupby(["month", "chamber"]).size().groupby("chamber").agg(["min", "median", "max"])
print(f"ROSTER_ASOF: {ROSTER_ASOF.month.nunique()} months, voting members in office per month (complete Congresses): "
      + "; ".join(f"{ch[:5]} min {r['min']:.0f} / median {r['median']:.0f} / max {r['max']:.0f}" for ch, r in _rs.iterrows()))
_rd = ROSTER_ASOF_ALL[~ROSTER_ASOF_ALL.voting & ROSTER_ASOF_ALL.congress.isin(CONGRESSES)].groupby("month").size()
print(f"ROSTER_ASOF_ALL (Task-1 pool, default of window_targets(point_in_time=True) / roster_asof()): adds delegates and "
      f"resident commissioners, {_rd.min()}-{_rd.max()} per month (median {_rd.median():.0f}); ROSTER_ASOF stays voting-only")
_ind = members[~members.party_code_asof.isin(["D", "R"]) & members.congress.isin(CONGRESSES + [IN_PROGRESS])].copy()
_ind["dsh"] = _dshare.reindex(pd.MultiIndex.from_arrays([_ind.bioguide_id, _ind.congress])).to_numpy()
_indg = _ind.groupby(["name", "party_code_asof", "caucus"]).agg(c0=("congress", "min"), c1=("congress", "max"),
                                                                 s0=("dsh", "min"), s1=("dsh", "max")).reset_index()
print("Non-D/R members -> data-derived caucus (share of their cosponsorships on D-sponsored bills; near 0.5 = weak "
      "evidence): " + "; ".join(f"{r.name.split(',')[0]} {r.party_code_asof} {r.c0}" + (f"-{r.c1}" if r.c1 > r.c0 else "")
                                + f" ({r.s0:.2f}" + (f"-{r.s1:.2f}" if r.c1 > r.c0 else "") + f")->{r.caucus}"
                                for r in _indg.itertuples()))
print("Majority per chamber (point-in-time at mid-Congress, caucus-adjusted; ties -> tie_broken_by):")
print(MAJORITY.assign(chamber=MAJORITY.chamber.str[:5]).pivot(index="chamber", columns="congress",
      values="majority").to_string())
_ties = MAJORITY[MAJORITY.tie_broken_by != "none (seat count)"]
print("ties: " + ("; ".join(f"{r.congress} {r.chamber}: {r.seats_D}-{r.seats_R} -> {r.majority} by {r.tie_broken_by}"
                            for r in _ties.itertuples()) or "none"))

# Shared party-regime labels (one definition for A15b, C7/C8, D26): REGIME[c] = '<House majority>H/<Senate majority>S',
# e.g. 'RH/RS', from MAJORITY (majority_party()). REGIME_HS[c] = (house, senate); regime_match(a, b) = number of
# chambers (0-2) whose majority party is the same in Congresses a and b.
_REG_HS = MAJORITY.pivot(index="congress", columns="chamber", values="majority")
REGIME_HS = {int(c): (r["House of Representatives"], r["Senate"]) for c, r in _REG_HS.iterrows()}
REGIME = {c: f"{h}H/{s}S" for c, (h, s) in REGIME_HS.items()}


def regime_match(a, b):
    """Number of chambers (0, 1, 2) with the same majority party in Congresses a and b (from REGIME_HS)."""
    return int(sum(x == y for x, y in zip(REGIME_HS[int(a)], REGIME_HS[int(b)])))


_same_reg = [c for c in CONGRESSES if REGIME.get(c) == REGIME.get(IN_PROGRESS)]
print("REGIME (House/Senate majority): " + ", ".join(f"{c} {r}" for c, r in REGIME.items())
      + f" | complete Congresses with {IN_PROGRESS}'s regime ({REGIME.get(IN_PROGRESS)}): {_same_reg}")

# %%
# --- Candidate splits (single source of truth: every section evaluates exactly these pools) -----------
# Each split expands into evaluation units via iter_eval_units(split): one dict per (fold, role) with
# the Congresses to fit on and the Congress (or window) evaluated. Tuning units fit on `train`; test
# units refit on train + val. Definitions follow builder D's backtest.
FOLD_C = list(range(112, CONGRESSES[-1] + 1))      # rolling / sliding folds: test c, val c-1
SLIDE_K = 3                                        # sliding window length (Congresses)
S4_CUTS = (0.6, 0.8)                               # within-Congress: train <60% | val 60-80% | test >=80%
S5_FRACS = (0.70, 0.15, 0.15)                      # random bill split (leaky reference)


def _rng_label(cs):
    cs = sorted(cs)
    return f"{cs[0]}-{cs[-1]}" if len(cs) > 1 else str(cs[0])


SPLIT_DEFS = {
    "S1a": dict(name="Expanding by Congress (test 117)", kind="expanding",
                train=list(range(108, 116)), val=[116], test=[117],
                desc="train 108-115 | val 116 | test 117 (test refit on 108-116)"),
    "S1b": dict(name="Expanding by Congress (test 118)", kind="expanding",
                train=list(range(108, 117)), val=[117], test=[118],
                desc="train 108-116 | val 117 | test 118 (test refit on 108-117)"),
    "S2": dict(name=f"Sliding window k={SLIDE_K}", kind="sliding", k=SLIDE_K, folds=FOLD_C,
               desc=f"for test c: train c-{SLIDE_K + 1}..c-2 | val c-1 | test c (refit on c-{SLIDE_K}..c-1)"),
    "S3": dict(name="Rolling origin, expanding", kind="rolling", folds=FOLD_C,
               desc="for test c: train 108..c-2 | val c-1 | test c (refit on 108..c-1); "
                    f"c in {FOLD_C[0]}-{FOLD_C[-1]}"),
    "S4": dict(name="Within-Congress temporal", kind="within", folds=[116, 117, 118], cuts=S4_CUTS,
               desc="prior Congresses + bills of c introduced before its 60% date quantile | "
                    "val 60-80% | test last 20% (only events dated before the cutoff are used)"),
    "S5": dict(name="Random bill split (leaky reference)", kind="random", fracs=S5_FRACS,
               desc="law-capable bills of 108-118 shuffled 70/15/15; ignores time entirely"),
}


def iter_eval_units(split=None):
    """Yield evaluation units of one split (key of SPLIT_DEFS) or of all splits (split=None).

    Each unit: dict(split, fold, role in {'val','test'}, train (list of whole Congresses to fit on),
    eval_congress (int, or None for S5), train_label, within). `within` is None except for S4, where it is
    (c, train_q, eval_lo_q, eval_hi_q): also fit on bills of Congress c introduced before its train_q date
    quantile and evaluate bills of c between the eval_lo_q and eval_hi_q quantiles. S5 units carry
    `random=(fracs, role)` and train=all CONGRESSES (the rows are split at random, not by Congress)."""
    keys = list(SPLIT_DEFS) if split is None else [split]
    for s in keys:
        d = SPLIT_DEFS[s]
        k = d["kind"]
        if k == "expanding":
            v, t = d["val"][0], d["test"][0]
            units = [("val", s, list(d["train"]), v), ("test", s, list(d["train"]) + [v], t)]
        elif k in ("rolling", "sliding"):
            units = []
            for c in d["folds"]:
                tv = list(range(CONGRESSES[0], c - 1)) if k == "rolling" else list(range(c - 1 - d["k"], c - 1))
                tt = list(range(CONGRESSES[0], c)) if k == "rolling" else list(range(c - d["k"], c))
                units += [("val", f"c={c}", tv, c - 1), ("test", f"c={c}", tt, c)]
        elif k == "within":
            q1, q2 = d["cuts"]
            for c in d["folds"]:
                prior = list(range(CONGRESSES[0], c))
                yield dict(split=s, fold=f"c={c}", role="val", train=prior, eval_congress=c,
                           train_label=f"{_rng_label(prior)} + {c}<{q1:.0%}", within=(c, q1, q1, q2))
                yield dict(split=s, fold=f"c={c}", role="test", train=prior, eval_congress=c,
                           train_label=f"{_rng_label(prior)} + {c}<{q2:.0%}", within=(c, q2, q2, 1.0))
            continue
        elif k == "random":
            for role, lab in (("val", f"random {d['fracs'][0]:.0%}"),
                              ("test", f"random {d['fracs'][0] + d['fracs'][1]:.0%}")):
                yield dict(split=s, fold="random", role=role, train=list(CONGRESSES), eval_congress=None,
                           train_label=lab, within=None, random=(d["fracs"], role))
            continue
        else:
            raise ValueError(f"unknown split kind {k}")
        for role, fold, tr, ev in units:
            yield dict(split=s, fold=fold, role=role, train=tr, eval_congress=ev, train_label=_rng_label(tr),
                       within=None)


print(pd.DataFrame([(k, v["name"], v["desc"], sum(1 for _ in iter_eval_units(k))) for k, v in SPLIT_DEFS.items()],
                   columns=["split", "name", "definition", "eval units"]).to_string(index=False))

# Protocol variants (section D): same Congress pools as a base split, different protocol. They are candidate
# splits for the decision section (SPLIT_RESULTS has rows for them) but are NOT yielded by iter_eval_units(),
# whose consumers only know the five base kinds. "+stream": the eval Congress's own events dated
# <= introduced_date + t0 of each scored bill are added to the history (temporal-GNN deployment).
SPLIT_VARIANTS = {
    "S1a+stream": dict(base="S1a", stream=True, desc="S1a + eval-Congress events dated <= introduced_date + t0"),
    "S1b+stream": dict(base="S1b", stream=True, desc="S1b + eval-Congress events dated <= introduced_date + t0"),
    "S3+stream": dict(base="S3", stream=True, desc="S3 rolling folds + eval-Congress events dated <= introduced_date + t0"),
    "S1pool": dict(base="S1a", stream=False, train=list(range(108, 116)), val=[116], test=[117, 118],
                   desc="train 108-115 | val 116 | test 117+118 pooled (refit on 108-116 once)"),
}
ALL_SPLITS = list(SPLIT_DEFS) + list(SPLIT_VARIANTS)   # every candidate a check's by_split may refer to
BY_CONGRESS_KINDS = ("expanding", "sliding", "rolling")


def split_kind(split):
    """Kind of a split or variant ('expanding', 'sliding', 'rolling', 'within', 'random')."""
    return SPLIT_DEFS[SPLIT_VARIANTS[split]["base"] if split in SPLIT_VARIANTS else split]["kind"]


def split_eval_congresses(split, roles=("val", "test")):
    """Congresses evaluated (in the given roles) by a split or variant. S4: its folds; S5: all CONGRESSES."""
    if split in SPLIT_VARIANTS and "test" in SPLIT_VARIANTS[split]:
        v = SPLIT_VARIANTS[split]
        return sorted(set(v["val"] * ("val" in roles) + v["test"] * ("test" in roles)))
    base = SPLIT_VARIANTS[split]["base"] if split in SPLIT_VARIANTS else split
    if SPLIT_DEFS[base]["kind"] == "random":
        return list(CONGRESSES)
    return sorted({u["eval_congress"] for u in iter_eval_units(base) if u["role"] in roles})


# Shared S5 draw (bill_id -> 'train'/'val'/'test'): law-capable bills of CONGRESSES in `bills` order, seed 5,
# cut at S5_FRACS. Identical to section D's original local draw, so C and D use the same random split.
_s5b = bills.loc[bills.law_capable & bills.congress.isin(CONGRESSES), "bill_id"].to_numpy()
S5_ROLE = pd.Series(pd.cut(np.random.default_rng(5).random(len(_s5b)),
                           [0, S5_FRACS[0], S5_FRACS[0] + S5_FRACS[1], 1.0001],
                           labels=["train", "val", "test"]).astype(str), index=_s5b, name="s5_role")

# Shared S4 cut dates: S4_CUT_DATES[c][q] = introduction-date quantile q of Congress c's law-capable bills, floored
# to the day (q = 1.0 -> day after the Congress ends). Rule: fit on bills introduced < cut[q1];
# val = cut[q1] <= introduced < cut[q2]; test = introduced >= cut[q2] (and < cut[1.0]).
_EPOCH = np.datetime64("1970-01-01", "D")


def _s4_cut(c, q):
    if q >= 1.0:
        return congress_end(c) + pd.Timedelta(days=1)
    d = bills.loc[bills.law_capable & (bills.congress == c), "introduced_date"].dropna()
    day = int(np.quantile(d.to_numpy().astype("datetime64[D]").astype(np.int64), q))
    return pd.Timestamp(_EPOCH + np.timedelta64(day, "D"))


S4_CUT_DATES = {c: {q: _s4_cut(c, q) for q in (*S4_CUTS, 1.0)} for c in SPLIT_DEFS["S4"]["folds"]}
print("variants (SPLIT_VARIANTS, not in iter_eval_units): " + "; ".join(f"{k}: {v['desc']}" for k, v in SPLIT_VARIANTS.items()))
print(f"S5_ROLE: {S5_ROLE.value_counts().to_dict()} | S4_CUT_DATES: " + "; ".join(
    f"{c}: {d[S4_CUTS[0]].date()} / {d[S4_CUTS[1]].date()}" for c, d in S4_CUT_DATES.items()))

# %%
# --- Drift statistics -----------------------------------------------------------------------------
def psi(expected, actual, bins=10):
    """Population Stability Index of `actual` vs `expected` (numeric). <0.1 stable, 0.1-0.25 moderate, >0.25 shift."""
    e, a = np.asarray(expected, float), np.asarray(actual, float)
    e, a = e[~np.isnan(e)], a[~np.isnan(a)]
    edges = np.unique(np.quantile(e, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    pe = np.histogram(e, edges)[0] / len(e) + 1e-6
    pa = np.histogram(a, edges)[0] / len(a) + 1e-6
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def psi_categorical(expected, actual):
    """PSI for categorical values (same thresholds as psi)."""
    pe = pd.Series(expected).value_counts(normalize=True)
    pa = pd.Series(actual).value_counts(normalize=True)
    idx = pe.index.union(pa.index)
    pe, pa = pe.reindex(idx, fill_value=0) + 1e-6, pa.reindex(idx, fill_value=0) + 1e-6
    return float(np.sum((pa - pe) * np.log(pa / pe)))


def jsd(p, q):
    """Jensen-Shannon divergence (base 2, in [0, 1]) between two count/probability vectors."""
    p, q = np.asarray(p, float), np.asarray(q, float)
    p, q = p / p.sum(), q / q.sum()
    m = (p + q) / 2
    kl = lambda x, y: np.sum(np.where(x > 0, x * np.log2(np.where(x > 0, x, 1) / np.where(y > 0, y, 1)), 0))
    return float((kl(p, m) + kl(q, m)) / 2)


def ks(a, b):
    """Two-sample Kolmogorov-Smirnov statistic and p-value."""
    r = stats.ks_2samp(np.asarray(a, float), np.asarray(b, float))
    return float(r.statistic), float(r.pvalue)


def smd(a, b):
    """Standardized mean difference (|d| < 0.1 negligible, > 0.2 notable)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt((np.nanvar(a) + np.nanvar(b)) / 2)
    return float((np.nanmean(b) - np.nanmean(a)) / s) if s > 0 else 0.0


def psi_level(v):
    return "stable" if v < 0.1 else "moderate" if v < 0.25 else "shift"


# --- Shared "different from train noise" rule ------------------------------------------------------------
# Every section judges a per-Congress quantity against the SAME fixed reference set REL_REF (the S1a train
# Congresses 108-115), whatever split or pool it is looking at, so verdicts are comparable across sections.
REL_REF = list(SPLIT_DEFS["S1a"]["train"])
REL_ALPHA = 0.05


def rel_to_train(values, train=None, side="two", log=False, alpha=REL_ALPHA, cands=None):
    """Judge per-Congress values against the spread of a fixed reference set of train Congresses.

    values: Series indexed by Congress. For a distance (PSI, JSD, KS, AUROC drop), give each reference Congress
    its leave-one-out distance to the other reference Congresses and each candidate its distance to the
    reference pool. train: reference Congresses (default REL_REF; keep the default unless you have a reason).
    side: 'two' | 'upper' (only large values are bad, e.g. distances) | 'lower'. log: judge log(values)
    (rates, counts, ratios; values <= 0 become NaN). cands: Congresses to judge (default: every other Congress
    in `values`).

    Returns (tab, info). tab has one row per Congress (reference rows judged by nested leave-one-out against the
    other reference Congresses, role='ref'; candidates role='cand') with columns
      value, z (vs reference mean / SD, comparable to the old z > 2 rules), t_pi (t statistic of the
      prediction interval for one new Congress: (x - mean) / (sd * sqrt(1 + 1/n))), pi_lo / pi_hi (interval on
      the original scale), fail (outside the (1 - alpha) t prediction interval, df = n - 1), trend_pred,
      trend_z (studentised residual from a linear trend over the reference Congresses: separates a steady
      trend from a break; df = n - 2), trend_fail, in_range (inside the reference min-max).
    info: n, mean, sd, tcrit, alpha, side, log, ref, fa_count / fa_rate (reference Congresses that fail the
      rule against the other n - 1 = per-Congress false-alarm rate of the rule), k (number of candidates),
      fwer_loo = 1 - (1 - fa_rate)^k (chance that at least one of k unchanged candidates fails, from the
      nested LOO), fwer_nominal = 1 - (1 - alpha)^k, rule (text for record_check threshold)."""
    train = list(REL_REF if train is None else train)
    v = pd.Series(values, dtype=float)
    f = (lambda s: np.log(s.where(s > 0))) if log else (lambda s: s)
    inv = np.exp if log else (lambda s: s)
    fv = f(v)
    ref = fv.reindex(train).dropna()
    cands = [c for c in v.index if c not in train] if cands is None else list(cands)
    n = len(ref)
    q = 1 - alpha / 2 if side == "two" else 1 - alpha

    def judge(x, r, cx):
        m, s, k = r.mean(), r.std(ddof=1), len(r)
        tc = stats.t.ppf(q, k - 1)
        half = tc * s * np.sqrt(1 + 1 / k)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (x - m) / (s * np.sqrt(1 + 1 / k))
            z = (x - m) / s
            xs = np.asarray(r.index, float)
            sxx = ((xs - xs.mean()) ** 2).sum()
            b1 = ((xs - xs.mean()) * (r.values - r.mean())).sum() / sxx
            pred = r.mean() + b1 * (cx - xs.mean())
            sres = np.sqrt(((r.values - (r.mean() + b1 * (xs - xs.mean()))) ** 2).sum() / (k - 2))
            tz = (x - pred) / (sres * np.sqrt(1 + 1 / k + (cx - xs.mean()) ** 2 / sxx))
        tct = stats.t.ppf(q, k - 2)
        flag = {"two": lambda a, c: np.abs(a) > c, "upper": lambda a, c: a > c, "lower": lambda a, c: a < -c}[side]
        return dict(z=z, t_pi=t, pi_lo=float(inv(m - half)), pi_hi=float(inv(m + half)),
                    fail=bool(flag(t, tc)) if np.isfinite(x) else None, trend_pred=float(inv(pred)),
                    trend_z=float(tz), trend_fail=bool(flag(tz, tct)) if np.isfinite(x) else None)

    rows = []
    for c in ref.index:
        rows.append(dict(congress=c, role="ref", value=v[c], **judge(ref[c], ref.drop(c), float(c))))
    for c in cands:
        x = fv.get(c, np.nan)
        rows.append(dict(congress=c, role="cand", value=v.get(c, np.nan), **judge(x, ref, float(c))))
    tab = pd.DataFrame(rows).set_index("congress")
    inr = tab.value.between(v.reindex(ref.index).min(), v.reindex(ref.index).max())
    tab["in_range"] = inr.astype(object).where(tab.role == "cand", None)   # None for reference rows
    fa = tab[tab.role == "ref"]
    fa_count = int((fa.fail == True).sum())   # noqa: E712 (None-safe)
    fa_rate = fa_count / n if n else np.nan
    k = int((tab.role == "cand").sum())
    tail = {"two": "two-sided", "upper": "upper tail", "lower": "lower tail"}[side]
    info = dict(n=n, mean=float(inv(ref.mean())), sd=float(ref.std(ddof=1)), tcrit=float(stats.t.ppf(q, n - 1)),
                alpha=alpha, side=side, log=log, ref=_rng_label(ref.index), fa_count=fa_count, fa_rate=fa_rate,
                fa_ids=[int(c) for c in fa.index[fa.fail == True]], k=k,   # noqa: E712
                fwer_loo=float(1 - (1 - fa_rate) ** k) if n else np.nan, fwer_nominal=float(1 - (1 - alpha) ** k),
                rule=f"outside {1 - alpha:.0%} t prediction interval of ref {_rng_label(ref.index)} "
                     f"({tail}{', log scale' if log else ''}; n={n})")
    return tab, info


def rel_detail(tab, info, fmt=".3f"):
    """One-line summary of rel_to_train output for record_check(detail=...)."""
    cand = tab[tab.role == "cand"]
    return (f"ref {info['ref']} PI [{cand.pi_lo.iloc[0] if len(cand) else np.nan:{fmt}}, "
            f"{cand.pi_hi.iloc[0] if len(cand) else np.nan:{fmt}}] ({info['rule']}); "
            + ", ".join(f"{c}={r.value:{fmt}} (t={r.t_pi:+.1f}, trend_z={r.trend_z:+.1f}{', FAIL' if r.fail else ''})"
                        for c, r in cand.iterrows())
            + f"; nested-LOO false alarms {info['fa_count']}/{info['n']} (FWER over {info['k']} candidates "
              f"~{info['fwer_loo']:.0%}, nominal {info['fwer_nominal']:.0%})")


REL_SPLIT_MODES = ("cands", "excess", "any")
REL_SPLIT_MODE = "cands"   # default mode of rel_by_split for rel_to_train tabs (frames with a `role` column)


def _rel_verdicts(tab, col):
    """Per-Congress verdicts (bool, None dropped) and per-Congress false-alarm rates from a rel_to_train tab."""
    s = tab[col] if isinstance(tab, pd.DataFrame) else pd.Series(tab)
    s = s[[x is not None and not (isinstance(x, float) and np.isnan(x)) for x in s]]
    role = tab["role"] if isinstance(tab, pd.DataFrame) and "role" in tab.columns else None
    return s, role


def rel_split_table(tab, col="fail", splits=None, roles=("val", "test"), alpha=REL_ALPHA):
    """Per-split breakdown of per-Congress verdicts (rel_to_train's tab, or any frame / Series by Congress).

    One row per split: n_eval (Congresses judged), n_fail, n_cand / n_cand_fail (candidate Congresses only,
    role 'cand'; for inputs without `role` every Congress counts as a candidate), expected = sum over the judged
    Congresses of their per-Congress false-alarm rate, binom_p = P(X >= n_fail) for X ~ Poisson-binomial with those
    rates, and the three verdicts: pass_any (no judged Congress fails; legacy), pass_cands (no candidate Congress
    fails; None if the split evaluates no candidate), pass_excess (binom_p >= 0.05).
    Per-Congress false-alarm rate = max(nested-LOO fa_rate of the tab (share of reference rows failing `col`),
    alpha): the same empirical rate applies to candidates (judged against the reference) and to reference
    Congresses (judged by nested LOO); floored at the nominal alpha because n = 8 cannot show a rate of 0.
    For inputs without reference rows (plain Series of threshold verdicts) the rate is alpha."""
    s, role = _rel_verdicts(tab, col)
    splits = [k for k in ALL_SPLITS if split_kind(k) in BY_CONGRESS_KINDS] if splits is None else splits
    if role is not None and (role == "ref").any():
        ref_v = [bool(x) for x in s[s.index.isin(role.index[role == "ref"])]]
        fa = float(np.mean(ref_v)) if ref_v else alpha
    else:
        fa = alpha
    rate = max(fa, alpha)
    rows = []
    for k in splits:
        ev = [c for c in split_eval_congresses(k, roles) if c in s.index]
        fails = [bool(s[c]) for c in ev]
        is_c = [True if role is None else role.get(c) == "cand" for c in ev]
        n_fail, n_c = int(sum(fails)), int(sum(is_c))
        n_cf = int(sum(f for f, ic in zip(fails, is_c) if ic))
        pmf = np.array([1.0])
        for _ in ev:   # Poisson-binomial (identical rates -> binomial), exact by convolution
            pmf = np.convolve(pmf, [1 - rate, rate])
        p = float(pmf[n_fail:].sum()) if ev else np.nan
        rows.append(dict(split=k, n_eval=len(ev), n_fail=n_fail, n_cand=n_c, n_cand_fail=n_cf,
                         fa_rate=rate, expected=rate * len(ev), binom_p=p,
                         pass_any=(not any(fails)) if ev else None,
                         pass_cands=(n_cf == 0) if n_c else None,
                         pass_excess=bool(p >= 0.05) if ev else None))
    return pd.DataFrame(rows).set_index("split")


def rel_by_split(tab, col="fail", splits=None, roles=("val", "test"), mode=None):
    """Map per-Congress verdicts (rel_to_train's tab, or any frame / Series indexed by Congress) onto candidate
    splits for record_check(by_split=...). Values: True (holds) / False (fails) / None (not judged).

    mode (see rel_split_table for the numbers behind each):
      'cands'  (default for rel_to_train tabs, i.e. inputs with a `role` column): a split fails if any CANDIDATE
               Congress (role 'cand', 116-118) it evaluates fails; reference Congresses (109-115, evaluated only by
               S2/S3 folds) are not judged, because their nested-LOO verdict uses a reference that contains later
               Congresses and would make 7-fold splits fail more by construction. A split that evaluates no
               candidate gets None.
      'excess' : fails if the number of failing evaluated Congresses (reference + candidates) exceeds chance:
               Poisson-binomial p < 0.05 given per-Congress false-alarm rates. Low power for splits that evaluate
               only 2 Congresses (S1a / S1b need both to fail), so use it as a cross-check.
      'any'    (legacy; default for plain Series / frames without `role`, e.g. deterministic threshold verdicts such
               as coverage < 95%): fails if any evaluated Congress fails.
    Default splits: by-Congress splits and their variants (S4 / S5 reuse Congresses for training, so a
    per-Congress verdict does not apply; give their entries explicitly if your check covers them)."""
    has_role = isinstance(tab, pd.DataFrame) and "role" in tab.columns
    mode = (REL_SPLIT_MODE if has_role else "any") if mode is None else mode
    if mode not in REL_SPLIT_MODES:
        raise ValueError(f"mode must be one of {REL_SPLIT_MODES}")
    t = rel_split_table(tab, col=col, splits=splits, roles=roles)
    v = t[f"pass_{mode}"].astype(object)
    return {k: (None if v[k] is None or (isinstance(v[k], float) and np.isnan(v[k])) else bool(v[k])) for k in t.index}


# %%
# --- Check registry: every section records its conditions here; the final section summarizes -------
CHECKS = []
# Owner section per shared quantity: when several sections test the same quantity, record_check(...,
# quantity=key) lets checks_summary() count it once (the owner's verdict; disagreements are flagged).
QUANTITY_OWNERS = {
    # A: labels / targets and covariates
    "label_rate_t1": "A", "label_rate_t2": "A", "cosp_density": "A", "orig_share": "A", "majority_or": "A",
    "covariate_psi": "A", "drift_gradual": "A", "label_validity_t2": "A", "label_rate_t2_alt": "A",
    "label_count_t2": "A", "label_rate_t2_chamber": "A", "majority_or_regime": "A",
    "majority_or_regime_effect": "A", "same_party_share": "A", "bill_volume": "A", "topic_mix_jsd": "A",
    "joint_covariate_shift": "A", "model_calibration_t2": "A", "model_auroc_transfer_t2": "A", "pool_distance": "A",
    "deployment_proxy": "A", "drift_type": "A", "cand_fail_count": "A",
    "concept_drift_t1": "A", "model_auroc_transfer_t1": "A", "regime_similarity": "A",
    # B: graph structure
    "node_overlap": "B", "cold_start": "B", "degree_dist": "B", "graph_structure": "B", "sponsor_overlap": "B",
    "node_identity_switch": "B",   # party / chamber switchers (B11; D25 tests the same quantity from the model side)
    "train_eval_input_shift": "B",  # B12 (construction requirement: train vs eval graph inputs built alike)
    # C: temporal structure, censoring, leakage
    "subject_coverage": "C", "annotation_coverage": "C", "lag_dist": "C", "party_regime": "C", "censoring": "C",
    "label_completeness": "C", "task2_t0": "C",
    # leakage, one quantity per leakage path (the broad "leakage" key is kept only for backward compatibility:
    # it hid B's leakage FAILs behind C's INFO; tag new checks with a per-path key)
    "leakage_snapshot_fields": "C", "leakage_related_links": "B", "leakage_committee_edges": "B",
    "leakage_full_graph": "B", "leakage": "C",
    # D: backtest
    "score_stability": "D", "val_test_gap": "D", "learned_model_selection": "D", "embedding_staleness": "D",
    "regime_weighting": "D",
}

# Check kinds. The decision section RANKS splits on 'validity' checks only; 'drift' checks choose remedies.
#   validity    : does the split give an honest, usable estimate? (leakage, val predicts test, enough eval Congresses,
#                 score stability). A FAIL here is a reason to prefer another split.
#   drift       : does the evaluation / deployment target differ from train? Every forward split fails these by
#                 design (that is the split exposing drift); a FAIL selects remedies (recency / regime weighting,
#                 recalibration, per-Congress reporting), not a different split.
#   requirement : construction rule every split must follow (same as requirement=True).
CHECK_KINDS = ("validity", "drift", "requirement")
# Default kind for checks that pass kind=None, by quantity (clear-cut cases only; record_check(kind=...) overrides,
# and `kind_source` says which was used). Checks with neither stay kind=None ('unclassified' in the scores).
QUANTITY_KIND = {
    **{q: "drift" for q in (
        "label_rate_t1", "label_rate_t2", "cosp_density", "orig_share", "majority_or", "covariate_psi",
        "drift_gradual", "label_rate_t2_alt", "label_count_t2", "label_rate_t2_chamber", "majority_or_regime",
        "majority_or_regime_effect", "same_party_share", "bill_volume", "topic_mix_jsd", "joint_covariate_shift",
        "model_calibration_t2", "model_auroc_transfer_t2", "model_auroc_transfer_t1", "concept_drift_t1",
        "pool_distance", "deployment_proxy", "drift_type", "cand_fail_count", "regime_similarity", "degree_dist",
        "graph_structure", "lag_dist", "party_regime", "subject_coverage", "annotation_coverage")},
    **{q: "validity" for q in (
        "leakage_snapshot_fields", "leakage_related_links", "leakage_committee_edges", "leakage_full_graph",
        "leakage", "score_stability", "val_test_gap", "learned_model_selection", "censoring", "label_completeness",
        "cold_start")},
    # sponsor / node-identity turnover: the eval graph differs from train (choose remedies, not a split)
    **{q: "drift" for q in ("sponsor_overlap", "node_identity_switch")},
    "train_eval_input_shift": "requirement",
}


def _py(v):
    """numpy scalars -> plain Python (keeps by_split values JSON-friendly)."""
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    return v


def record_check(section, check_id, condition, passed, metric=None, threshold=None, detail="", quantity=None,
                 by_split=None, requirement=False, kind=None, na_reason=None, t0_task2=None):
    """passed: True / False, or None for informational checks. Re-running a cell replaces its check.
    quantity: optional key (see QUANTITY_OWNERS) naming the measured quantity, so checks of the same
    quantity in different sections are deduplicated by checks_summary().
    by_split: optional dict split -> verdict for checks whose outcome depends on the split. Keys are names in
    ALL_SPLITS (SPLIT_DEFS + SPLIT_VARIANTS); values True (holds for that split) / False (fails) / None (not
    judged), or a number (the split's metric; then give `passed` per the threshold). Give EVERY split in
    ALL_SPLITS a key: splits you do not judge get an explicit None plus a reason in `na_reason` (omitted splits are
    stored in `by_split_omitted`, counted as not judged, and flagged on the printed line). rel_by_split() builds it
    from per-Congress verdicts. Leave by_split=None for split-independent checks.
    na_reason: why splits with a None verdict are not judged: one string for all of them, or dict split -> str.
    kind: 'validity' | 'drift' | 'requirement' (CHECK_KINDS). Default: 'requirement' if requirement=True, else
    QUANTITY_KIND[quantity] if listed, else None (unclassified). Only 'validity' checks rank splits.
    requirement=True (or kind='requirement'): a construction rule every split must follow (e.g. "filter related
    links to t0"), not a property one split satisfies better than another. Listed by checks_requirements(),
    deduplicated separately, excluded from checks_by_split(), never counted as split failures.
    t0_task2: for Task-2 checks, the prediction time (days after introduction) the check assumes, e.g. 0 / 7
    (no policy area) or 30 (policy area available; bills reported by t0 excluded). None = not a Task-2 check."""
    global CHECKS
    if kind is not None and kind not in CHECK_KINDS:
        raise ValueError(f"record_check {check_id}: kind must be one of {CHECK_KINDS} or None, got {kind!r}")
    requirement = bool(requirement) or kind == "requirement"
    if not requirement and kind is None and QUANTITY_KIND.get(quantity) == "requirement":
        requirement, kind = True, "requirement"   # quantity default: keep the requirement flag consistent
        kind_source = "quantity default"
    elif requirement:
        kind_source = "explicit" if kind == "requirement" or kind is None else "requirement=True overrides " + kind
        kind = "requirement"
    elif kind is not None:
        kind_source = "explicit"
    elif quantity in QUANTITY_KIND:
        kind, kind_source = QUANTITY_KIND[quantity], "quantity default"
    else:
        kind_source = "unset"
    omitted, no_reason = [], []
    if by_split is not None:
        by_split = {str(k): _py(v) for k, v in dict(by_split).items()}
        unknown = [k for k in by_split if k not in ALL_SPLITS]
        if unknown:
            print(f"  (record_check {check_id}: unknown split keys {unknown}; known: {ALL_SPLITS})")
        omitted = [k for k in ALL_SPLITS if k not in by_split]
        if isinstance(na_reason, dict):
            na_reason = {str(k): str(v) for k, v in na_reason.items()}
            no_reason = [k for k, v in by_split.items() if v is None and k not in na_reason]
        elif not na_reason:
            no_reason = [k for k, v in by_split.items() if v is None]
    CHECKS = [c for c in CHECKS if c["id"] != check_id]
    CHECKS.append(dict(section=section, id=check_id, condition=condition, passed=passed,
                       metric=metric, threshold=threshold, detail=detail, quantity=quantity, by_split=by_split,
                       requirement=requirement, kind=kind, kind_source=kind_source, na_reason=na_reason,
                       by_split_omitted=omitted, t0_task2=t0_task2))
    mark = {True: "PASS", False: "FAIL", None: "INFO"}[passed] + ("/REQ" if requirement else "")
    m = "" if metric is None else f" | metric={metric:.4g}" if isinstance(metric, (int, float, np.floating)) else f" | {metric}"
    bs = ""
    if by_split:
        bad = [k for k, v in by_split.items() if v is False]
        bs = f" [fails for: {', '.join(bad)}]" if bad else ""
        if omitted or no_reason:
            bs += f" [not judged, no reason: {', '.join(omitted + no_reason)}]"
    print(f"[{mark}] {check_id}: {condition}{m}" + (f" (threshold {threshold})" if threshold is not None else "") + bs)


_CHECK_COLS = ["section", "id", "condition", "passed", "metric", "threshold", "detail", "quantity", "by_split",
               "requirement", "kind", "kind_source", "na_reason", "by_split_omitted", "t0_task2"]


def checks_summary(dedup=True, requirements=True, kinds=None):
    """DataFrame of recorded checks. dedup=True keeps one row per `quantity` (checks without a quantity are
    kept as-is; requirement and non-requirement checks of one quantity are kept apart): the owner section's
    check if present (QUANTITY_OWNERS), else the first recorded. Added columns: `dup_ids` (other checks of that
    quantity), `hidden_fail_ids` (hidden checks that FAIL), and `conflict` = True when the hidden checks'
    pass/fail verdicts disagree with each other or with the kept check, including a kept INFO (None) hiding a
    FAIL. When conflict is True, read hidden_fail_ids before counting the row as a pass / info.
    requirements: True (all rows), False (drop requirement=True rows: use this to count split failures),
    'only' (just the construction requirements; same as checks_requirements()).
    kinds: None (all) or an iterable of kinds to keep after dedup ('validity', 'drift', 'requirement'; use None
    inside the iterable for unclassified checks).
    `by_split` holds the per-split dict given to record_check (None if split-independent); see
    checks_by_split() for a long table and checks_split_scores() for per-split n_judged / n_fail."""
    df = pd.DataFrame(CHECKS, columns=_CHECK_COLS)
    df["requirement"] = df.requirement.fillna(False).astype(bool)
    if requirements == "only":
        df = df[df.requirement]
    elif requirements is False:
        df = df[~df.requirement]
    if dedup and not df.empty:
        df["_key"] = (df.quantity.where(df.quantity.notna(), "id:" + df.id)
                      + np.where(df.requirement & df.quantity.notna(), "|req", ""))
        df["_own"] = (df.section == df.quantity.map(QUANTITY_OWNERS)).astype(int)
        out = df.sort_values("_own", ascending=False, kind="stable").drop_duplicates("_key")
        kept = dict(zip(out._key, out.id))
        dup, hid, conf = {}, {}, {}
        for k, g in df.groupby("_key", sort=False):
            others = g[g.id != kept[k]]
            own_v = g.loc[g.id == kept[k], "passed"].iloc[0]
            dup[k] = ", ".join(others.id)
            hid[k] = ", ".join(others.id[others.passed.map(lambda x: x is False or x is np.False_)])
            vs = {x for x in g.passed if x is not None}
            conf[k] = len(vs) > 1 or (own_v is None and bool(hid[k]))
        out["dup_ids"] = out._key.map(dup)
        out["hidden_fail_ids"] = out._key.map(hid)
        out["conflict"] = out._key.map(conf)
        df = out.sort_index().drop(columns=["_key", "_own"])
    if kinds is not None:
        ks = set(kinds)
        df = df[df.kind.map(lambda k: (k if isinstance(k, str) else None) in ks)]
    return df


def checks_requirements():
    """Construction requirements (record_check(..., requirement=True)), one row per check (not deduplicated)."""
    return checks_summary(dedup=False, requirements="only")


def _na_reason_for(reason, split):
    if isinstance(reason, dict):
        return reason.get(split)
    return reason or None


def checks_by_split(dedup=True, kinds=None, include_omitted=False):
    """Long table of per-split verdicts: one row per (check, split) from by_split. Columns section, id,
    quantity, kind, passed (overall), split, value (as recorded), split_passed (True/False/None; None when the
    value is a metric or not judged), judged (split_passed is True/False), na_reason (why not judged, if given;
    '(omitted)' for splits missing from by_split). Checks without by_split and requirement checks are omitted
    (they apply to every split alike). include_omitted=True adds a row (judged=False) for every split in ALL_SPLITS
    missing from a check's by_split, so per-split counts never treat 'no entry' as 'not failing'."""
    cols = ["section", "id", "quantity", "kind", "passed", "split", "value", "split_passed", "judged", "na_reason"]
    rows = []
    for r in checks_summary(dedup=dedup, requirements=False, kinds=kinds).itertuples():
        bsd = r.by_split or {}
        for k, v in bsd.items():
            sp = v if isinstance(v, bool) or v is None else None
            rows.append(dict(section=r.section, id=r.id, quantity=r.quantity, kind=r.kind, passed=r.passed, split=k,
                             value=v, split_passed=sp, judged=isinstance(sp, bool),
                             na_reason=None if isinstance(sp, bool) else _na_reason_for(r.na_reason, k)))
        if include_omitted and bsd:
            for k in ALL_SPLITS:
                if k not in bsd:
                    rows.append(dict(section=r.section, id=r.id, quantity=r.quantity, kind=r.kind, passed=r.passed,
                                     split=k, value=None, split_passed=None, judged=False, na_reason="(omitted)"))
    return pd.DataFrame(rows, columns=cols)


def checks_split_scores(kinds=("validity",), dedup=True):
    """Per-split scoreboard over split-dependent checks of the given kinds (default: validity only, the ones that
    should rank splits). One row per split in ALL_SPLITS: n_checks (split-dependent checks of these kinds),
    n_judged (True/False verdicts for this split), n_fail, n_pass, n_na_reason (None with a reason),
    n_not_judged (None without a reason, metric-only values, or omitted from by_split), fail_rate = n_fail /
    n_judged. Compare splits on n_fail AND n_judged: a split that was not judged has not passed.
    kinds=None scores every kind; include None in the iterable for unclassified checks.
    Prints a warning listing split-dependent checks with kind=None (unless None is among `kinds`), so
    unclassified checks are not dropped from the ranking without notice."""
    if kinds is not None and None not in set(kinds):
        un = checks_summary(dedup=dedup, requirements=False, kinds=(None,))
        un = un[un.by_split.map(lambda d: isinstance(d, dict) and len(d) > 0)]
        if len(un):
            print(f"  WARNING checks_split_scores: {len(un)} split-dependent check(s) have kind=None and are NOT "
                  f"scored: {', '.join(un.id)}. Classify them (record_check(kind=...) or QUANTITY_KIND) or pass "
                  f"kinds including None to score them.")
    lb = checks_by_split(dedup=dedup, kinds=kinds, include_omitted=True)
    rows = []
    for s in ALL_SPLITS:
        g = lb[lb.split == s]
        nj = int(g.judged.sum())
        nf = int((g.split_passed == False).sum())  # noqa: E712 (object column)
        nr = int((~g.judged & g.na_reason.notna() & (g.na_reason != "(omitted)")).sum())
        rows.append(dict(split=s, n_checks=int(g.id.nunique()), n_judged=nj, n_fail=nf, n_pass=nj - nf,
                         n_na_reason=nr, n_not_judged=int(len(g) - nj - nr),
                         fail_rate=(nf / nj if nj else np.nan),
                         fail_ids=", ".join(g.id[g.split_passed == False])))  # noqa: E712
    return pd.DataFrame(rows).set_index("split")


# %%
# --- D3 charts ------------------------------------------------------------------------------------
# Charts are SVGs drawn by D3 v7 in the notebook output. `show_d3` embeds the data as JSON and runs a
# JS function body with (el, data, d3, opts) in scope. D3 itself is injected once per session from the
# vendored file (fallback: CDN). Palette: reference categorical palette (fixed order, never cycled).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
PARTY_COLORS = {"D": "#2a78d6", "R": "#e34948", "I": "#7a7a7a"}
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

_HARNESS = os.environ.get("NB_HARNESS_OUT")   # set only by the offline test harness
_D3_EMITTED = False
HARNESS_CHARTS = []

# Shared JS: tooltip + axis styling, available to every chart as `viz` inside the function body.
_VIZ_JS = r"""
const viz = {
  ink: '%INK%', ink2: '%INK2%', grid: '%GRID%', palette: %PALETTE%,
  tooltip(el) {
    const t = d3.select(el).style('position', 'relative').append('div')
      .style('position', 'absolute').style('pointer-events', 'none').style('background', '#fff')
      .style('border', '1px solid #c9c8c2').style('border-radius', '4px').style('padding', '4px 8px')
      .style('font', '12px sans-serif').style('color', '%INK%').style('opacity', 0);
    return {
      show(html, event) {
        const [x, y] = d3.pointer(event, el);
        t.html(html).style('left', (x + 12) + 'px').style('top', (y - 10) + 'px').style('opacity', 1);
      },
      hide() { t.style('opacity', 0); },
    };
  },
  axis(g) {
    g.selectAll('path,line').attr('stroke', '%INK2%');
    g.selectAll('text').attr('fill', '%INK2%').style('font', '11px sans-serif');
    return g;
  },
  title(svg, text, sub) {
    svg.append('text').attr('x', 0).attr('y', 14).attr('fill', '%INK%')
      .style('font', '600 13px sans-serif').text(text);
    if (sub) svg.append('text').attr('x', 0).attr('y', 30).attr('fill', '%INK2%')
      .style('font', '11px sans-serif').text(sub);
  },
  legend(svg, items, x, y) {
    const g = svg.append('g').attr('transform', `translate(${x},${y})`);
    let dx = 0;
    items.forEach(([label, color]) => {
      const it = g.append('g').attr('transform', `translate(${dx},0)`);
      it.append('rect').attr('width', 10).attr('height', 10).attr('rx', 2).attr('fill', color);
      it.append('text').attr('x', 14).attr('y', 9).attr('fill', '%INK2%').style('font', '11px sans-serif').text(label);
      dx += 24 + label.length * 6.5;
    });
    return g;
  },
};
""".replace("%INK%", INK).replace("%INK2%", INK_2).replace("%GRID%", GRID).replace("%PALETTE%", json.dumps(PALETTE))


def _to_jsonable(data):
    if isinstance(data, pd.DataFrame):
        df = data.copy()
        for c in df.columns:
            if str(df[c].dtype).startswith("datetime"):
                df[c] = df[c].dt.strftime("%Y-%m-%d")
        return json.loads(df.to_json(orient="records"))
    return json.loads(json.dumps(data, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def show_d3(js_body, data, height=320, width=720, opts=None, name=None):
    """Render a D3 chart. `js_body` is JS run with (el, data, d3, opts, viz) in scope; `el` is the
    container div. Draw into d3.select(el).append('svg')... and keep text in ink colors."""
    global _D3_EMITTED
    cid = "d3_" + uuid.uuid4().hex[:10]
    payload = dict(data=_to_jsonable(data), opts=dict(opts or {}, width=width, height=height))
    func = f"function(el, data, d3, opts) {{\n{_VIZ_JS}\n{js_body}\n}}"
    if _HARNESS is not None:
        HARNESS_CHARTS.append(dict(id=cid, name=name or cid, func=func, payload=payload))
        print(f"<chart {name or cid}: {len(payload['data']) if isinstance(payload['data'], list) else 1} records>")
        return
    from IPython.display import HTML, display
    d3_tag = ""
    if not _D3_EMITTED and VENDOR_D3.exists():
        d3_tag = f"<script>{VENDOR_D3.read_text()}</script>"
        _D3_EMITTED = True
    html = f"""{d3_tag}<div id="{cid}" style="width:{width}px;min-height:{height}px"></div>
<script>(function() {{
  const payload = {json.dumps(payload)};
  const draw = {func};
  function go() {{ const el = document.getElementById("{cid}"); el.innerHTML = '';
                   try {{ draw(el, payload.data, window.d3, payload.opts); }}
                   catch (e) {{ el.innerHTML = '<pre style="color:#c00">' + e + '</pre>'; }} }}
  if (window.d3) go();
  else {{ const s = document.createElement('script'); s.src = 'https://cdn.jsdelivr.net/npm/d3@7';
         s.onload = go; document.head.appendChild(s); }}
}})();</script>"""
    display(HTML(html))


def reset_d3():
    """Re-inject D3 on the next chart (use after reloading the browser page mid-session)."""
    global _D3_EMITTED
    _D3_EMITTED = False


# --- Ready-made charts (use these when they fit; write custom js_body with show_d3 otherwise) -------
_LINE_JS = r"""
const {width, height, x, y, series, title, subtitle, yfmt, ylabel, xlabel, yzero} = opts;
const m = {top: subtitle ? 44 : 30, right: 120, bottom: 36, left: 56};
const svg = d3.select(el).append('svg').attr('width', width).attr('height', height);
viz.title(svg, title || '', subtitle);
const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
const W = width - m.left - m.right, H = height - m.top - m.bottom;
const keys = series ? Array.from(new Set(data.map(d => d[series]))) : ['all'];
const color = d3.scaleOrdinal(keys, viz.palette);
const xs = d3.scaleLinear(d3.extent(data, d => +d[x]), [0, W]);
const yext = d3.extent(data, d => +d[y]);
const ys = d3.scaleLinear([yzero === false ? yext[0] : Math.min(0, yext[0]), yext[1]], [H, 0]).nice();
g.append('g').call(d3.axisLeft(ys).ticks(5).tickSize(-W).tickFormat(d3.format(yfmt || ',~g')))
  .call(a => a.selectAll('line').attr('stroke', viz.grid)).call(a => a.select('.domain').remove())
  .call(a => a.selectAll('text').attr('fill', viz.ink2).style('font', '11px sans-serif'));
g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(xs).ticks(Math.min(12, data.length)).tickFormat(d3.format('d'))).call(viz.axis);
if (ylabel) g.append('text').attr('x', -m.left + 4).attr('y', -8).attr('fill', viz.ink2).style('font', '11px sans-serif').text(ylabel);
if (xlabel) g.append('text').attr('x', W).attr('y', H + 32).attr('text-anchor', 'end').attr('fill', viz.ink2).style('font', '11px sans-serif').text(xlabel);
const tip = viz.tooltip(el);
keys.forEach(k => {
  const rows = data.filter(d => !series || d[series] === k).sort((a, b) => a[x] - b[x]);
  g.append('path').datum(rows).attr('fill', 'none').attr('stroke', color(k)).attr('stroke-width', 2)
    .attr('d', d3.line().defined(d => d[y] !== null && !isNaN(d[y])).x(d => xs(+d[x])).y(d => ys(+d[y])));
  g.selectAll(null).data(rows.filter(d => d[y] !== null)).join('circle').attr('cx', d => xs(+d[x])).attr('cy', d => ys(+d[y]))
    .attr('r', 4).attr('fill', color(k)).attr('stroke', '#fff').attr('stroke-width', 1.5)
    .on('mousemove', (e, d) => tip.show(`<b>${series ? k + ' · ' : ''}${d[x]}</b><br>${d3.format(yfmt || ',~g')(d[y])}`, e))
    .on('mouseleave', () => tip.hide());
  if (series) {
    const last = rows[rows.length - 1];
    if (last) g.append('text').attr('x', xs(+last[x]) + 8).attr('y', ys(+last[y]) + 4).attr('fill', viz.ink2)
      .style('font', '11px sans-serif').text(k);
  }
});
if (series && keys.length > 1) viz.legend(svg, keys.map(k => [String(k), color(k)]), m.left, height - 10);
"""


def line_chart(df, x, y, series=None, title="", subtitle="", yfmt=",~g", ylabel="", xlabel="",
               yzero=True, height=300, width=720, name=None):
    """Lines over x (usually congress), one per `series` value. Hover points for values."""
    cols = [x, y] + ([series] if series else [])
    d = df[cols].copy()
    if series:
        d[series] = d[series].astype(str)
    show_d3(_LINE_JS, d, height, width, dict(x=x, y=y, series=series, title=title, subtitle=subtitle,
                                             yfmt=yfmt, ylabel=ylabel, xlabel=xlabel, yzero=yzero), name=name)


_BAR_JS = r"""
const {width, height, x, y, series, title, subtitle, yfmt, ylabel} = opts;
const m = {top: subtitle ? 44 : 30, right: 16, bottom: series ? 52 : 36, left: 56};
const svg = d3.select(el).append('svg').attr('width', width).attr('height', height);
viz.title(svg, title || '', subtitle);
const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
const W = width - m.left - m.right, H = height - m.top - m.bottom;
const xsVals = Array.from(new Set(data.map(d => String(d[x]))));
const keys = series ? Array.from(new Set(data.map(d => String(d[series])))) : ['all'];
const color = d3.scaleOrdinal(keys, viz.palette);
const x0 = d3.scaleBand(xsVals, [0, W]).paddingInner(0.2).paddingOuter(0.1);
const x1 = d3.scaleBand(keys, [0, x0.bandwidth()]).padding(0.08);
const ylo = Math.min(0, d3.min(data, d => +d[y]) || 0), yhi = Math.max(0, d3.max(data, d => +d[y]) || 0);
const ys = d3.scaleLinear([ylo, yhi > ylo ? yhi : ylo + 1], [H, 0]).nice();
g.append('g').call(d3.axisLeft(ys).ticks(5).tickSize(-W).tickFormat(d3.format(yfmt || ',~g')))
  .call(a => a.selectAll('line').attr('stroke', viz.grid)).call(a => a.select('.domain').remove())
  .call(a => a.selectAll('text').attr('fill', viz.ink2).style('font', '11px sans-serif'));
g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(x0)).call(viz.axis);
if (ylo < 0) g.append('line').attr('x1', 0).attr('x2', W).attr('y1', ys(0)).attr('y2', ys(0)).attr('stroke', viz.ink2);
if (ylabel) g.append('text').attr('x', -m.left + 4).attr('y', -8).attr('fill', viz.ink2).style('font', '11px sans-serif').text(ylabel);
const tip = viz.tooltip(el);
g.selectAll(null).data(data).join('rect')
  .attr('x', d => x0(String(d[x])) + (series ? x1(String(d[series])) : 0))
  .attr('width', series ? x1.bandwidth() : x0.bandwidth())
  .attr('y', d => ys(Math.max(0, +d[y]))).attr('height', d => Math.abs(ys(0) - ys(+d[y])))
  .attr('rx', 2).attr('fill', d => color(series ? String(d[series]) : 'all'))
  .on('mousemove', (e, d) => tip.show(`<b>${d[x]}${series ? ' · ' + d[series] : ''}</b><br>${d3.format(yfmt || ',~g')(d[y])}`, e))
  .on('mouseleave', () => tip.hide());
if (series && keys.length > 1) viz.legend(svg, keys.map(k => [k, color(k)]), m.left, height - 10);
"""


def bar_chart(df, x, y, series=None, title="", subtitle="", yfmt=",~g", ylabel="", height=300,
              width=720, name=None):
    """Vertical bars (grouped by `series` if given). Hover bars for values."""
    cols = [x, y] + ([series] if series else [])
    show_d3(_BAR_JS, df[cols], height, width, dict(x=x, y=y, series=series, title=title,
                                                   subtitle=subtitle, yfmt=yfmt, ylabel=ylabel), name=name)


_HEAT_JS = r"""
const {width, height, x, y, v, title, subtitle, fmt, domain, diverging} = opts;
const m = {top: subtitle ? 48 : 34, right: 70, bottom: 40, left: 130};
const svg = d3.select(el).append('svg').attr('width', width).attr('height', height);
viz.title(svg, title || '', subtitle);
const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
const W = width - m.left - m.right, H = height - m.top - m.bottom;
const xv = Array.from(new Set(data.map(d => String(d[x])))), yv = Array.from(new Set(data.map(d => String(d[y]))));
const xs = d3.scaleBand(xv, [0, W]).padding(0.04), ys = d3.scaleBand(yv, [0, H]).padding(0.04);
const ext = domain || d3.extent(data, d => +d[v]);
const color = diverging
  ? d3.scaleDiverging([ext[0], 0, ext[1]], t => d3.interpolateRgbBasis(['#2a78d6', '#f2f1ee', '#e34948'])(t))
  : d3.scaleSequential(ext, d3.interpolateRgb('#eef4fc', '#0f3f7a'));
const tip = viz.tooltip(el);
g.selectAll(null).data(data).join('rect').attr('x', d => xs(String(d[x]))).attr('y', d => ys(String(d[y])))
  .attr('width', xs.bandwidth()).attr('height', ys.bandwidth()).attr('rx', 2)
  .attr('fill', d => d[v] === null ? '#f7f7f5' : color(+d[v]))
  .on('mousemove', (e, d) => tip.show(`<b>${d[y]} · ${d[x]}</b><br>${d[v] === null ? 'n/a' : d3.format(fmt || '.3f')(d[v])}`, e))
  .on('mouseleave', () => tip.hide());
if (xv.length * yv.length <= 200) g.selectAll(null).data(data.filter(d => d[v] !== null)).join('text')
  .attr('x', d => xs(String(d[x])) + xs.bandwidth() / 2).attr('y', d => ys(String(d[y])) + ys.bandwidth() / 2 + 4)
  .attr('text-anchor', 'middle').style('font', '10px sans-serif')
  .attr('fill', d => { const c = d3.hsl(color(+d[v])); return c.l < 0.55 ? '#fff' : viz.ink; })
  .text(d => d3.format(fmt || '.3f')(d[v]));
g.append('g').attr('transform', `translate(0,${H})`).call(d3.axisBottom(xs)).call(viz.axis);
g.append('g').call(d3.axisLeft(ys)).call(viz.axis);
"""


def heatmap(df, x, y, value, title="", subtitle="", fmt=".3f", domain=None, diverging=False,
            height=360, width=720, name=None):
    """Matrix of cells colored by `value` (sequential blue, or diverging blue-gray-red around 0)."""
    d = df[[x, y, value]].copy()
    d[value] = d[value].astype(float).where(d[value].notna(), None)
    show_d3(_HEAT_JS, d, height, width, dict(x=x, y=y, v=value, title=title, subtitle=subtitle, fmt=fmt,
                                             domain=domain, diverging=diverging), name=name)


print("setup complete: load(), cosp, adv (label, label_alt), adv_cutoff(), window_targets(point_in_time=), "
      "in_office_at(), ROSTER_ASOF / roster_asof(), majority_party(), REGIME / REGIME_HS / regime_match(), SNAPSHOT_DATE, PA_MISSING_AT_T0 / PA_MISS_T0, "
      "SPLIT_DEFS / SPLIT_VARIANTS / iter_eval_units(), S5_ROLE, S4_CUT_DATES, drift stats, rel_to_train() / "
      "rel_detail() / rel_by_split(mode=) / rel_split_table(), record_check(by_split=, kind=, na_reason=, t0_task2=) / "
      "checks_summary(kinds=) / checks_requirements() / checks_by_split() / checks_split_scores(), ROSTER_ASOF_ALL, "
      "show_d3() / line_chart() / bar_chart() / heatmap()")
