"""Data loading, task labels, eligibility pools and folds.

Every definition here is ported from the split notebook (`notebooks/_build/cells/00_setup.py` and the
engine block of `40_splits.py`) so that labels, censoring, point-in-time pools, caucus and majority are
identical; `baselines_build/tests/test_parity.py` asserts it. Ported names keep a comment with their
origin. Nothing here is learned.
"""
from __future__ import annotations

import hashlib
import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RunConfig

# --- Constants (00_setup.py) ------------------------------------------------------------------------
CONGRESSES = list(range(108, 119))
IN_PROGRESS = 119
LAW_TYPES = ["hr", "s", "hjres", "sjres"]
ALL_BILL_TYPES = ["hr", "s", "hjres", "sjres", "hres", "sres", "hconres", "sconres"]
WINDOW_DAYS = 30
ADV_WINDOW_DAYS = 180
ROSTER_REF_DAY = 15
CAUCUS_MIN_SHARE = 0.5
NONVOTING_TYPES = ("Delegate", "Resident Commissioner")
PARTY_CODE = {"Republican": "R", "Democratic": "D", "Independent": "I",
              "Independent Democrat": "ID", "Libertarian": "L"}
CHAMBER_OF_ORIGIN = {"House": "House of Representatives", "Senate": "Senate"}
CHAMBERS = ("House of Representatives", "Senate")
VP_PARTY = {108: "R", 109: "R", 110: "R", 111: "D", 112: "D", 113: "D", 114: "D", 115: "R", 116: "R",
            117: "D", 118: "D", 119: "R"}
PA_AGE_BUCKETS = [(0, 7, "0-7d"), (8, 30, "8-30d"), (31, 90, "31-90d"), (91, 180, "91-180d"), (181, 10**6, ">180d")]
INTMAX = np.iinfo(np.int64).max
D_BIG = 10 ** 6
DATA_VERSION = "1"
_TOKENS = itertools.count(1)


def find_root(start: Path | None = None) -> Path:
    """Repo root = first directory at or above `start` (default: the working directory, then this package's
    folder) containing parsed/."""
    for here in [Path(start or Path.cwd()).resolve(), Path(__file__).resolve().parent]:
        for p in [here, *here.parents]:
            if (p / "parsed").is_dir():
                return p
    raise FileNotFoundError("Couldn't find the parsed/ folder; run the notebook from inside the repo.")


def congress_start(c) -> pd.Timestamp:
    """Congress c convenes on Jan 3 of 1789 + 2(c-1)."""  # ported from 00_setup.py:congress_start
    return pd.Timestamp(year=1789 + 2 * (int(c) - 1), month=1, day=3)


def congress_end(c) -> pd.Timestamp:
    """Start of Congress c+1."""  # ported from 00_setup.py:congress_end
    return congress_start(int(c) + 1)


def adv_cutoff(c) -> pd.Timestamp:
    """Last introduction date whose 180-day advancement window ends inside Congress c."""  # 00_setup.py:adv_cutoff
    return congress_end(c) - pd.Timedelta(days=ADV_WINDOW_DAYS)


def congress_midpoint(c) -> pd.Timestamp:
    """Reference date for point-in-time majorities: one year after the Congress convenes."""
    return congress_start(c) + pd.Timedelta(days=365)


def to_day(dates) -> np.ndarray:
    """Day numbers since the epoch (int64) of a datetime Series / array; NaT -> INTMAX."""
    s = pd.Series(pd.to_datetime(pd.Series(dates).to_numpy()))
    v = s.to_numpy().astype("datetime64[D]").astype(np.int64)
    return np.where(s.notna().to_numpy(), v, INTMAX)


def mstart_day(dates) -> np.ndarray:
    """Day number of the first day of each date's month."""  # ported from 40_splits.py:_mstart_day
    return pd.to_datetime(dates).dt.to_period("M").dt.to_timestamp().values.astype("datetime64[D]").astype(np.int64)


class Loader:
    """Cached reader of `parsed/<table>.parquet` with the setup's type conventions (00_setup.py:load)."""

    def __init__(self, parsed: Path):
        self.parsed = Path(parsed)
        self._cache: dict = {}

    def __call__(self, table: str, columns: tuple | None = None) -> pd.DataFrame:
        """Read a table: `congress` -> int, date-like object columns -> datetime, tz-aware -> naive UTC."""
        key = (table, columns)
        if key not in self._cache:
            cols = list(columns) if columns else None
            df = pd.read_parquet(self.parsed / f"{table}.parquet", columns=cols)
            if "congress" in df.columns:
                df["congress"] = df["congress"].astype(int)
            for c in df.columns:
                if df[c].dtype == object and (c.endswith("_date") or c == "action_date"):
                    df[c] = pd.to_datetime(df[c], errors="coerce")
                elif str(df[c].dtype).startswith("datetime64") and getattr(df[c].dt, "tz", None) is not None:
                    df[c] = df[c].dt.tz_convert(None)
            self._cache[key] = df
        return self._cache[key]


# --- Setup definitions --------------------------------------------------------------------------------
def prepare_bills(load: Loader) -> tuple[pd.DataFrame, dict]:
    """bills with law_capable, chamber, day_of_congress, policy_area_harm; plus snapshot info (PA_MISS_T0)."""
    # ported from 00_setup.py (data loading cell)
    bills = load("bills").copy()
    bills["law_capable"] = bills["bill_type"].isin(LAW_TYPES)
    bills["chamber"] = bills["origin_chamber"].map(CHAMBER_OF_ORIGIN)
    bills["day_of_congress"] = (bills["introduced_date"] - bills["congress"].map(congress_start)).dt.days
    _pa = bills[bills.law_capable & bills.congress.isin(CONGRESSES)].assign(pa=lambda d: d.policy_area.fillna("(missing)"))
    _pa_tab = pd.crosstab(_pa.pa, _pa.congress)
    pa_unstable = sorted(set(_pa_tab.index[(_pa_tab == 0).any(axis=1)]) | {"(missing)"})
    bills["policy_area_harm"] = bills.policy_area.fillna("(missing)").where(
        lambda s: ~s.isin(pa_unstable), "(unstable/missing)")
    snapshot = bills["update_date"].max().normalize()
    b_ip = bills[bills.law_capable & (bills.congress == IN_PROGRESS)]
    age = (snapshot - b_ip.introduced_date).dt.days
    tab = pd.DataFrame([dict(age_bucket=lab, age_lo=lo, age_hi=min(hi, int(age.max())), n=int(m.sum()),
                             n_missing=int(b_ip.policy_area[m].isna().sum()),
                             share_missing=float(b_ip.policy_area[m].isna().mean()) if m.any() else np.nan)
                        for lo, hi, lab in PA_AGE_BUCKETS for m in [age.between(lo, hi)]]).set_index("age_bucket")
    info = dict(snapshot_date=snapshot, pa_unstable=pa_unstable, pa_missing_at_t0=tab,
                pa_miss_t0=float(tab.loc["0-7d", "share_missing"]), pa_miss_t0_n=int(tab.loc["0-7d", "n"]))
    return bills, info


def party_asof_table(load: Loader, terms: pd.DataFrame) -> pd.Series:
    """(bioguide_id, congress) -> party code as of that Congress."""  # ported from 00_setup.py:_party_asof_table
    ph = load("member_party_history", ("bioguide_id", "party_code", "start_year", "end_year"))
    t = (terms.sort_values("start_year").drop_duplicates(["bioguide_id", "congress"])
         [["bioguide_id", "congress", "y0", "start_year"]].rename(columns={"start_year": "t0"}))
    h = t.merge(ph.rename(columns={"start_year": "h0", "end_year": "h1"}), on="bioguide_id")
    h1 = h.h1.fillna(9999)
    h["prio"] = np.select([(h.h0 <= h.y0) & (h1 >= h.y0 + 1), (h.h0 <= h.t0) & (h1 >= h.t0)], [2, 1], 0)
    h = h[h.prio > 0].sort_values(["prio", "h0"]).drop_duplicates(["bioguide_id", "congress"], keep="last")
    return h.set_index(["bioguide_id", "congress"])["party_code"]


def prepare_members(load: Loader) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(members with party_code, filled chamber and party_code_asof; member_terms with y0)."""
    # ported from 00_setup.py (data loading cell)
    members = load("members").copy()
    members["party_code"] = members["party"].map(PARTY_CODE).fillna(members["party"])
    terms = load("member_terms").copy()
    terms["y0"] = 1789 + 2 * (terms.congress - 1)
    ch_fill = (terms.sort_values("start_year").drop_duplicates(["bioguide_id", "congress"], keep="last")
               .set_index(["bioguide_id", "congress"])["chamber"])
    members["chamber"] = members["chamber"].fillna(
        pd.Series(list(zip(members.bioguide_id, members.congress))).map(ch_fill).set_axis(members.index))
    pasof = party_asof_table(load, terms)
    members["party_code_asof"] = (pd.Series(list(zip(members.bioguide_id, members.congress))).map(pasof)
                                  .set_axis(members.index).fillna(members["party_code"]))
    return members, terms


def build_cosponsor_events(load: Loader, bills: pd.DataFrame) -> pd.DataFrame:
    """One row per active (non-withdrawn) cosponsorship with its lag from introduction (days)."""
    # ported from 00_setup.py:build_cosponsor_events
    c = load("bill_cosponsors", ("bill_id", "bioguide_id", "party", "sponsorship_date", "is_original",
                                 "withdrawn_date", "congress"))
    c = c[c["withdrawn_date"].isna()].merge(
        bills[["bill_id", "introduced_date", "sponsor_bioguide_id", "sponsor_party", "chamber",
               "bill_type", "law_capable"]], on="bill_id", how="inner")
    c["lag_days"] = (c["sponsorship_date"] - c["introduced_date"]).dt.days
    return c


def build_advancement_labels(load: Loader, bills: pd.DataFrame, window: int = ADV_WINDOW_DAYS) -> pd.DataFrame:
    """Per law-capable bill: reported out of committee within `window` days (`label`), `censored` if the window
    runs past the Congress end, plus label_alt (reported OR passed origin chamber)."""
    # ported from 00_setup.py:build_advancement_labels
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


def activity_span(cosp: pd.DataFrame, bills: pd.DataFrame) -> pd.DataFrame:
    """First / last dated (co)sponsorship per (congress, bioguide_id)."""  # ported from 00_setup.py:_activity_span
    act = pd.concat([cosp[["congress", "bioguide_id", "sponsorship_date"]].set_axis(["congress", "bioguide_id", "d"], axis=1),
                     bills[["congress", "sponsor_bioguide_id", "introduced_date"]].set_axis(["congress", "bioguide_id", "d"], axis=1)])
    return act.dropna().groupby(["congress", "bioguide_id"])["d"].agg(first="min", last="max")


def in_office_at(date, terms: pd.DataFrame, span: pd.DataFrame, voting_only: bool = True,
                 slack_days: int = 30) -> pd.DataFrame:
    """Members in office on `date` (bioguide_id, congress, chamber, state_code, district); year-granular terms
    refined with activity dates (see the setup docstring)."""  # ported from 00_setup.py:in_office_at
    date = pd.Timestamp(date)
    y = date.year
    c = 1 + (y - 1789) // 2 - (1 if (y % 2 == 1 and date < pd.Timestamp(year=y, month=1, day=3)) else 0)
    t = terms[(terms.congress == c) & (terms.start_year <= y) & (terms.end_year.fillna(9999) >= y)]
    if voting_only:
        t = t[~t.member_type.isin(NONVOTING_TYPES)]
    t = t.sort_values("start_year", kind="stable").drop_duplicates("bioguide_id").copy()
    t["seat"] = t.state_code + "-" + np.where(t.chamber == "Senate", "S", t.district.fillna(0).astype(int).astype(str))
    sp = span.reindex(pd.MultiIndex.from_arrays([t.congress, t.bioguide_id]))
    first, last = sp["first"].set_axis(t.index), sp["last"].set_axis(t.index)
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


def setup_dshare(cosp: pd.DataFrame) -> pd.Series:
    """(bioguide_id, congress) -> share of a member's cosponsorships on D-sponsored (of D/R-sponsored) bills."""
    # ported from 00_setup.py (_dshare)
    return (cosp[cosp.sponsor_party.isin(["D", "R"])].assign(isd=lambda d: d.sponsor_party == "D")
            .groupby(["bioguide_id", "congress"])["isd"].mean())


def assign_caucus(members: pd.DataFrame, dshare: pd.Series) -> pd.Series:
    """Setup caucus per members row: D/R as-of party, others by their same-Congress D share (>0.5 D, <0.5 R)."""
    # ported from 00_setup.py (members["caucus"]); uses the member's own Congress (heuristic ports only)
    mk = pd.MultiIndex.from_arrays([members.bioguide_id, members.congress])
    v = dshare.reindex(mk).to_numpy()
    return pd.Series(np.where(members.party_code_asof.isin(["D", "R"]), members.party_code_asof,
                              np.select([v > CAUCUS_MIN_SHARE, v < 1 - CAUCUS_MIN_SHARE], ["D", "R"], "other")),
                     index=members.index)


_ROSTER_COLS = ["month", "congress", "chamber", "bioguide_id", "caucus", "party_code_asof", "state_code", "district"]


def build_roster_asof(members: pd.DataFrame, terms: pd.DataFrame, span: pd.DataFrame,
                      snapshot_date: pd.Timestamp) -> pd.DataFrame:
    """ROSTER_ASOF_ALL: members in office on day 15 of each month (voting + delegates, column `voting`)."""
    # ported from 00_setup.py:_build_roster_asof
    months = pd.date_range(congress_start(CONGRESSES[0]).to_period("M").to_timestamp(), snapshot_date, freq="MS")
    cau = members.drop_duplicates(["bioguide_id", "congress"]).set_index(["bioguide_id", "congress"])
    nonvoting = set(terms.loc[terms.member_type.isin(NONVOTING_TYPES), "bioguide_id"])
    parts = []
    for m in months:
        d = m + pd.Timedelta(days=ROSTER_REF_DAY - 1)
        v = in_office_at(d, terms, span).assign(voting=True)
        a = in_office_at(d, terms, span, voting_only=False)
        dl = a[a.bioguide_id.isin(nonvoting) & ~a.bioguide_id.isin(v.bioguide_id)].assign(voting=False)
        parts += [v.assign(month=m), dl.assign(month=m)]
    r = pd.concat(parts, ignore_index=True)
    k = pd.MultiIndex.from_arrays([r.bioguide_id, r.congress])
    r["caucus"] = k.map(cau["caucus"]).fillna("other")
    r["party_code_asof"] = k.map(cau["party_code_asof"])
    return r[_ROSTER_COLS + ["voting"]].sort_values(["month", "voting"], ascending=[True, False], kind="stable").reset_index(drop=True)


def majority_party(members: pd.DataFrame, terms: pd.DataFrame, span: pd.DataFrame, adv: pd.DataFrame) -> pd.DataFrame:
    """Majority party per (congress, chamber) at the Congress midpoint, caucus-adjusted, VP tie-break."""
    # ported from 00_setup.py:_majority_party_table
    rows = []
    cau = members.drop_duplicates(["bioguide_id", "congress"]).set_index(["bioguide_id", "congress"])["caucus"]
    for c in sorted(members.congress.unique()):
        io = in_office_at(congress_midpoint(c), terms, span)
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


# --- Corpus -------------------------------------------------------------------------------------------
@dataclass
class Corpus:
    """Everything the package needs, built once by `load_corpus`. Names mirror the split notebook:
    `leg_index` = _LEG, `billix` = _BILLIX, `CD` = _CD, `BL` = _BL, `ADV` = _ADV, `roster_pit` = _ROSTER_PIT,
    `cens` = _CENS. `titles` (introduced official title per bill code) and `referrals` (full-committee
    referral rows with Eastern dates) feed the learned models only."""
    root: Path
    cache_dir: Path
    load: Loader
    bills: pd.DataFrame
    info: dict
    members: pd.DataFrame
    terms: pd.DataFrame
    cosp: pd.DataFrame
    adv: pd.DataFrame
    roster_all: pd.DataFrame
    majority: pd.DataFrame
    leg_index: pd.Index
    billix: pd.Index
    CD: dict
    BL: pd.DataFrame
    ADV: pd.DataFrame
    NG: dict
    roster: dict
    roster_pit: dict
    cens: dict
    dshare: pd.Series
    titles: pd.Series = None
    referrals: pd.DataFrame = None
    committees: pd.Index = None
    timings: dict = field(default_factory=dict)

    @property
    def NL(self) -> int:
        """Number of coded legislators (len(leg_index))."""
        return len(self.leg_index)

    def maj(self, congress: int, chamber: str) -> str:
        """Majority party (setup MAJORITY) of a Congress and chamber."""
        return self._maj.get((int(congress), chamber))

    def __post_init__(self):
        self._maj = {(int(r.congress), r.chamber): r.majority for r in self.majority.itertuples()}
        self.token = next(_TOKENS)   # unique per object (dataclasses.replace -> new token); used by caches


def _cache_key(parsed: Path, extra: str = "") -> str:
    h = hashlib.sha1()
    for p in sorted(parsed.glob("*.parquet")):
        st = p.stat()
        h.update(f"{p.name}:{st.st_mtime_ns}:{st.st_size}".encode())
    h.update((DATA_VERSION + extra).encode())
    return h.hexdigest()[:12]


def default_cache_dir(root: Path) -> Path:
    """R/derived/cache/baselines (created on demand)."""
    d = root / "derived" / "cache" / "baselines"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_corpus(cfg: RunConfig | None = None, root: Path | None = None) -> Corpus:
    """Build the Corpus: setup tables (bills, members, cosp, adv, ROSTER_ASOF_ALL, MAJORITY) and the 40_splits
    engine arrays. ROSTER_ASOF_ALL is cached as Parquet under the cache dir, keyed by the parsed files."""
    cfg = cfg or RunConfig()
    t_all = time.time()
    root = Path(root) if root else find_root()
    parsed = root / "parsed"
    cache = Path(cfg.cache_dir) if cfg.cache_dir else default_cache_dir(root)
    cache.mkdir(parents=True, exist_ok=True)
    load = Loader(parsed)
    tm = {}
    t = time.time()
    bills, info = prepare_bills(load)
    info["data_key"] = _cache_key(parsed)
    members, terms = prepare_members(load)
    cosp = build_cosponsor_events(load, bills)
    dshare = setup_dshare(cosp)
    members["caucus"] = assign_caucus(members, dshare)
    adv = build_advancement_labels(load, bills)
    span = activity_span(cosp, bills)
    tm["tables"] = time.time() - t
    t = time.time()
    rpath = cache / f"roster_asof_all_{info['data_key']}.parquet"
    if rpath.exists():
        roster_all = pd.read_parquet(rpath)
        roster_all["month"] = pd.to_datetime(roster_all["month"]).astype("datetime64[us]")
    else:
        roster_all = build_roster_asof(members, terms, span, info["snapshot_date"])
        roster_all.to_parquet(rpath)
    majority = majority_party(members, terms, span, adv)
    tm["roster_majority"] = time.time() - t
    t = time.time()
    corpus = _engine(root, cache, load, bills, info, members, terms, cosp, adv, roster_all, majority, dshare)
    corpus.titles = _titles(load, bills, corpus.billix)
    corpus.referrals, corpus.committees = _referrals(load, corpus.billix)
    tm["engine"] = time.time() - t
    tm["total"] = time.time() - t_all
    corpus.timings = tm
    return corpus


def _engine(root, cache, load, bills, info, members, terms, cosp, adv, roster_all, majority, dshare) -> Corpus:
    """Integer-coded arrays of the 40_splits.py backtest engine (_LEG, _CD, _BL, _CENS, _ROSTER(_PIT), _ADV)."""
    # ported from 40_splits.py (D.1 engine block)
    LEG = pd.Index(pd.unique(pd.concat([members.bioguide_id, cosp.bioguide_id, bills.sponsor_bioguide_id.dropna()])))
    BILLIX = pd.Index(bills.bill_id)
    cdate = cosp["sponsorship_date"]
    CD = dict(leg=LEG.get_indexer(cosp.bioguide_id), spon=LEG.get_indexer(cosp.sponsor_bioguide_id),
              cong=cosp.congress.to_numpy(), bill=BILLIX.get_indexer(cosp.bill_id),
              lag=cosp.lag_days.to_numpy(dtype=float),
              day=np.where(cdate.notna(), cdate.values.astype("datetime64[D]").astype(np.int64), INTMAX),
              orig=(cosp.is_original.fillna(False).astype(bool) | (cosp.lag_days == 0)).to_numpy())
    CAU = members.drop_duplicates(["bioguide_id", "congress"]).set_index(["bioguide_id", "congress"])["caucus"]
    BL = bills[bills.law_capable & bills.congress.isin(CONGRESSES)][
        ["bill_id", "congress", "chamber", "sponsor_bioguide_id", "sponsor_party", "introduced_date"]].copy()
    BL["sp_caucus"] = pd.MultiIndex.from_arrays([BL.sponsor_bioguide_id, BL.congress]).map(CAU)
    BL["sp_caucus"] = BL.sp_caucus.where(BL.sp_caucus.isin(["D", "R"]), BL.sponsor_party)
    BL["code"] = BILLIX.get_indexer(BL.bill_id)
    BL["spon"] = LEG.get_indexer(BL.sponsor_bioguide_id.fillna("?"))
    BL["day"] = BL.introduced_date.values.astype("datetime64[D]").astype(np.int64)
    for t0 in (0, 7):
        ref = BL.introduced_date + pd.Timedelta(days=t0)
        BL[f"sm{t0}"] = mstart_day(ref)
        BL[f"pm{t0}"] = mstart_day(ref.clip(upper=BL.congress.map(congress_end) - pd.Timedelta(days=ROSTER_REF_DAY + 5)))
    BL = BL.set_index("code")
    BL["cluster"] = np.where(BL.spon >= 0, BL.spon, -1 - np.arange(len(BL)))
    # _CENS: window_targets(t0)["censored"] (the flag does not depend on the pool definition)
    lc = bills[bills.law_capable & bills.congress.isin(CONGRESSES + [IN_PROGRESS])]
    cens = {t0: pd.Series((lc.introduced_date + pd.Timedelta(days=t0 + WINDOW_DAYS) > lc.congress.map(congress_end)).to_numpy(),
                          index=BILLIX.get_indexer(lc.bill_id), name="censored") for t0 in (0, 7)}
    mm = members.dropna(subset=["chamber"]).drop_duplicates(["congress", "chamber", "bioguide_id"])
    ROSTER = {k: (LEG.get_indexer(g.bioguide_id), g.caucus.to_numpy()) for k, g in mm.groupby(["congress", "chamber"])}
    RA = roster_all[roster_all.congress.isin(CONGRESSES)].assign(
        leg=lambda d: LEG.get_indexer(d.bioguide_id), mday=lambda d: d.month.values.astype("datetime64[D]").astype(np.int64))
    assert set(RA.chamber.unique()) <= set(mm.chamber.unique()), "ROSTER_ASOF chamber labels differ from members"
    ROSTER_PIT = {}
    for (c, ch), g in RA[RA.leg >= 0].groupby(["congress", "chamber"]):
        cand = pd.unique(np.r_[ROSTER.get((c, ch), (np.array([], int),))[0], g.leg.to_numpy()])
        cau = pd.MultiIndex.from_arrays([LEG[cand], np.full(len(cand), c)]).map(CAU)
        cau = np.asarray(pd.Series(cau).fillna("other"))
        md = np.sort(g.mday.unique())
        cp = pd.Series(np.arange(len(cand)), index=cand)
        io = np.zeros((len(md), len(cand)), bool)
        io[np.searchsorted(md, g.mday.to_numpy()), cp.reindex(g.leg).to_numpy()] = True
        ROSTER_PIT[(c, ch)] = (cand, cau, md, io)
    # _ADV (Task 2 rows), ported from 40_splits.py
    ADV = adv[~adv.censored & adv.congress.isin(CONGRESSES)].merge(majority[["congress", "chamber", "majority"]],
                                                                   on=["congress", "chamber"], how="left")
    ADV["code"] = BILLIX.get_indexer(ADV.bill_id)
    ADV["sp_caucus"] = BL.sp_caucus.reindex(ADV.code).to_numpy()
    ADV["maj"] = (ADV.sp_caucus == ADV.majority).astype(int)
    ADV["policy_area"] = ADV.policy_area.fillna("(none)")
    ADV["day"] = ADV.introduced_date.values.astype("datetime64[D]").astype(np.int64)
    ADV["y"] = ADV.label.astype(float)
    ADV["cluster"] = BL.cluster.reindex(ADV.code).to_numpy()
    ADV["g_ch"] = pd.factorize(ADV.chamber)[0]
    ADV["g1"] = pd.factorize(ADV.chamber + "|" + ADV.maj.astype(str))[0]
    ADV["g2"] = pd.factorize(ADV.policy_area + "|" + ADV.chamber + "|" + ADV.maj.astype(str))[0]
    ADV["g3"] = pd.factorize(ADV.policy_area + "|" + ADV.chamber)[0]
    ADV["pa_hidden"] = np.random.default_rng(1180).random(len(ADV)) < info["pa_miss_t0"]
    ADV = ADV.reset_index(drop=True)
    NG = {k: int(ADV[k].max()) + 1 for k in ("g_ch", "g1", "g2", "g3")}
    return Corpus(root=root, cache_dir=cache, load=load, bills=bills, info=info, members=members, terms=terms,
                  cosp=cosp, adv=adv, roster_all=roster_all, majority=majority, leg_index=LEG, billix=BILLIX,
                  CD=CD, BL=BL, ADV=ADV, NG=NG, roster=ROSTER, roster_pit=ROSTER_PIT, cens=cens, dshare=dshare)


def _titles(load: Loader, bills: pd.DataFrame, billix: pd.Index) -> pd.Series:
    """Official title as introduced (first row per bill) indexed by bill code; never bills.title."""
    t = load("bill_titles", ("bill_id", "title_type", "title"))
    t = t[t.title_type == "Official Title as Introduced"].drop_duplicates("bill_id", keep="first")
    s = pd.Series(t.title.fillna("").to_numpy(), index=billix.get_indexer(t.bill_id))
    s = s[s.index >= 0]
    return s.reindex(np.arange(len(billix))).fillna("")


def _referrals(load: Loader, billix: pd.Index) -> tuple[pd.DataFrame, pd.Index]:
    """Full-committee 'referred to' rows: bill code, committee code index, Eastern-time day number."""
    pq = load.parsed / "bill_committees.parquet"
    bc = pd.read_parquet(pq, columns=["bill_id", "committee_code", "parent_committee_code", "activity", "activity_ts",
                                      "chamber", "congress"])
    bc = bc[(bc.activity.str.lower() == "referred to") & bc.parent_committee_code.isna()].copy()
    ts = bc.activity_ts
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize("UTC")
    et = ts.dt.tz_convert("America/New_York").dt.tz_localize(None).dt.normalize()
    comms = pd.Index(sorted(bc.committee_code.unique()))
    out = pd.DataFrame({"code": billix.get_indexer(bc.bill_id), "comm": comms.get_indexer(bc.committee_code),
                        "day": to_day(et.reset_index(drop=True)), "congress": bc.congress.astype(int).to_numpy(),
                        "comm_chamber": bc.chamber.to_numpy()})
    out = out[out.code >= 0].reset_index(drop=True)
    return out, comms


# --- Labels for both tasks --------------------------------------------------------------------------
def window_targets(corpus: Corpus, t0: int, horizon: int = WINDOW_DAYS, law_capable_only: bool = True,
                   point_in_time: bool = False, include_delegates: bool = True) -> pd.DataFrame:
    """Per bill: new cosponsors in (t0, t0+horizon] (n_pos), already seen (n_seen), pool size, eligible and the
    `censored` flag (window past the Congress end)."""  # ported from 00_setup.py:window_targets
    bills, cosp = corpus.bills, corpus.cosp
    b = bills[bills.law_capable] if law_capable_only else bills
    b = b[b.congress.isin(CONGRESSES + [IN_PROGRESS])][["bill_id", "congress", "chamber", "introduced_date"]].copy()
    pos = cosp[(cosp.lag_days > t0) & (cosp.lag_days <= t0 + horizon)].groupby("bill_id").size()
    seen = cosp[cosp.lag_days <= t0].groupby("bill_id").size()
    b["n_pos"] = b["bill_id"].map(pos).fillna(0).astype(int)
    b["n_seen"] = b["bill_id"].map(seen).fillna(0).astype(int)
    if point_in_time:
        ra = corpus.roster_all
        roster = ra if include_delegates else ra[ra.voting]
        ref = (b["introduced_date"] + pd.Timedelta(days=t0)).clip(
            upper=b["congress"].map(congress_end) - pd.Timedelta(days=ROSTER_REF_DAY + 5))
        b["pool_month"] = ref.dt.to_period("M").dt.to_timestamp()
        pool = roster.groupby(["month", "congress", "chamber"])["bioguide_id"].size()
        b["pool"] = pool.reindex(pd.MultiIndex.from_arrays([b.pool_month, b.congress, b.chamber])).to_numpy(float)
        b["eligible"] = (b["pool"] - 1 - b["n_seen"]).clip(lower=0)
    else:
        pool = corpus.members.dropna(subset=["chamber"]).groupby(["congress", "chamber"])["bioguide_id"].nunique()
        b["pool"] = [pool.get((c, ch), np.nan) for c, ch in zip(b.congress, b.chamber)]
        b["eligible"] = b["pool"] - 1 - b["n_seen"]
    b["censored"] = b["introduced_date"] + pd.Timedelta(days=t0 + horizon) > b["congress"].map(congress_end)
    return b


@dataclass
class T1Block:
    """One Task-1 evaluation / training unit: the kept bills of one (congress, chamber, t0).

    codes: bill codes (rows); cand: legislator codes of the point-in-time candidate columns (M); cparty: setup
    caucus of each candidate (heuristic ports); Pos / Excl: [nb, M] bool; tieb: [M] tie-break; spon: sponsor
    leg code per bill (-1 = none); has_sp: sponsor present among the candidates; sp_caucus: sponsor caucus
    (setup convention); day: introduction day number; sm: cutoff-month start day (stream snapshots)."""
    congress: int
    chamber: str
    t0: int
    codes: np.ndarray
    cand: np.ndarray
    cparty: np.ndarray
    Pos: np.ndarray
    Excl: np.ndarray
    tieb: np.ndarray
    spon: np.ndarray
    has_sp: np.ndarray
    sp_caucus: np.ndarray
    day: np.ndarray
    sm: np.ndarray
    sp_col: np.ndarray = None

    @property
    def nb(self) -> int:
        """Number of bills in the block."""
        return len(self.codes)

    @property
    def M(self) -> int:
        """Number of candidate columns."""
        return len(self.cand)

    @property
    def cut(self) -> np.ndarray:
        """Bill cutoff day = introduction day + t0 (inclusive)."""
        return self.day + self.t0

    def subset(self, rows: np.ndarray) -> "T1Block":
        """Block restricted to the given row positions (same candidates and tie-break)."""
        r = np.asarray(rows)
        return T1Block(self.congress, self.chamber, self.t0, self.codes[r], self.cand, self.cparty, self.Pos[r],
                       self.Excl[r], self.tieb, self.spon[r], self.has_sp[r], self.sp_caucus[r], self.day[r],
                       self.sm[r], None if self.sp_col is None else self.sp_col[r])


_BLOCK_CACHE: dict = {}


def t1_blocks(corpus: Corpus, congress: int, t0: int) -> list[T1Block]:
    """Task-1 blocks of one Congress (House then Senate), exactly the bills, pools, masks and tie-breaks of
    40_splits.py:_t1_eval with pit=True for an evaluation unit of that single Congress."""
    key = (corpus.token, int(congress), int(t0))
    if key in _BLOCK_CACHE:
        return _BLOCK_CACHE[key]
    CD, BL, NL = corpus.CD, corpus.BL, corpus.NL
    eb = BL[BL.congress == congress]
    eb = eb[~corpus.cens[t0].reindex(eb.index).fillna(True).to_numpy(bool)]
    rows = np.isin(CD["bill"], eb.index.to_numpy())
    r_bill, r_leg, r_lag = CD["bill"][rows], CD["leg"][rows], CD["lag"][rows]
    rng = np.random.default_rng(t0 + 17)
    out = []
    for (c, ch), g in eb.groupby(["congress", "chamber"]):
        if (c, ch) not in corpus.roster_pit:
            continue
        cand, cparty, mdays, inoff = corpus.roster_pit[(c, ch)]
        M, nb = len(cand), len(g)
        colpos = np.full(NL, -1)
        colpos[cand] = np.arange(M)
        rowpos = pd.Index(g.index).get_indexer(r_bill)
        sel = rowpos >= 0
        rp, cp, lg = rowpos[sel], colpos[r_leg[sel]], r_lag[sel]
        win = (lg > t0) & (lg <= t0 + WINDOW_DAYS)
        Pos = np.zeros((nb, M), bool)
        Excl = np.zeros((nb, M), bool)
        Pos[rp[win & (cp >= 0)], cp[win & (cp >= 0)]] = True
        seen = (lg <= t0) & (cp >= 0)
        Excl[rp[seen], cp[seen]] = True
        spv = g.spon.to_numpy()
        sp = colpos[np.where(spv >= 0, spv, 0)]
        has_sp = (spv >= 0) & (sp >= 0)
        Excl[np.arange(nb)[has_sp], sp[has_sp]] = True
        mi = np.minimum(np.searchsorted(mdays, g[f"pm{t0}"].to_numpy()), len(mdays) - 1)
        okm = mdays[mi] == g[f"pm{t0}"].to_numpy()
        Excl |= okm[:, None] & ~inoff[mi]
        Pos &= ~Excl
        keep = Pos.sum(1) > 0
        tieb = rng.random(len(cand)) * 0.5
        if not keep.any():
            continue
        Pos, Excl, g = Pos[keep], Excl[keep], g[keep]
        spv = g.spon.to_numpy()
        spcol = colpos[np.where(spv >= 0, spv, 0)]
        spcol = np.where(spv >= 0, spcol, -1)
        out.append(T1Block(int(c), ch, int(t0), g.index.to_numpy(), cand, cparty, Pos, Excl, tieb, spv,
                           spv >= 0, g.sp_caucus.to_numpy(), g.day.to_numpy(), g[f"sm{t0}"].to_numpy(), spcol))
    _BLOCK_CACHE[key] = out
    return out


def t2_rows(corpus: Corpus, congresses) -> pd.DataFrame:
    """Task-2 rows (_ADV: law-capable, uncensored) of the given Congresses, in _ADV order."""
    return corpus.ADV[corpus.ADV.congress.isin(list(congresses))]


def hist_window(k: int, H: int) -> list[int]:
    """History window W_k = max(108, k-H)..k-1 used by graphs and learned features."""
    return list(range(max(CONGRESSES[0], int(k) - int(H)), int(k)))


def caucus_pit(corpus: Corpus, k: int, H: int = 3) -> np.ndarray:
    """Point-in-time caucus per legislator code for Congress k ('D' / 'R' / 'other').

    D/R: party_code_asof of Congress k (legislators without a Congress-k row keep their last earlier row, else
    'other'; later Congresses are never read). Others (I, ID, L): the setup D-share rule applied to their cosponsorships from the
    window W_k only (Congresses < k); no earlier record -> 'other'."""
    party = party_pit(corpus, k)
    out = np.where(np.isin(party, ["D", "R"]), party, "other").astype(object)
    other = np.flatnonzero(~np.isin(party, ["D", "R"]))
    if len(other):
        W = hist_window(k, H)
        c = corpus.cosp
        m = c.congress.isin(W) & c.sponsor_party.isin(["D", "R"])
        sub = c.loc[m, ["bioguide_id", "sponsor_party"]]
        sub = sub[sub.bioguide_id.isin(corpus.leg_index[other])]
        ds = (sub.sponsor_party == "D").groupby(sub.bioguide_id).mean()
        v = ds.reindex(corpus.leg_index[other]).to_numpy()
        out[other] = np.select([v > CAUCUS_MIN_SHARE, v < 1 - CAUCUS_MIN_SHARE], ["D", "R"], "other")
    return out


def party_pit(corpus: Corpus, k: int) -> np.ndarray:
    """As-of party code per legislator code for Congress k: its Congress-k members row, else its latest row
    before k, else 'other'. Strictly point-in-time: rows of later Congresses are never read (legislators who
    first serve after k get 'other'; they are not in office in k and have no Congress-k edges)."""
    m = corpus.members.drop_duplicates(["bioguide_id", "congress"])
    out = np.full(corpus.NL, "other", dtype=object)
    before = m[m.congress <= k].sort_values("congress").drop_duplicates("bioguide_id", keep="last")
    out[corpus.leg_index.get_indexer(before.bioguide_id)] = before.party_code_asof.to_numpy()
    return out


def label_summary(corpus: Corpus, congresses, t0: int = 0) -> pd.DataFrame:
    """Per Congress: law-capable bills, Task-1 evaluable bills (uncensored, >= 1 positive) and positives at t0,
    median eligible pool (House / Senate), Task-2 rows and positive rate."""
    rows = []
    for k in congresses:
        bl = t1_blocks(corpus, k, t0)
        pools = {b.chamber[:5]: int(np.median((~b.Excl).sum(1))) for b in bl}
        t2 = t2_rows(corpus, [k])
        rows.append(dict(congress=k, law_capable=int((corpus.BL.congress == k).sum()),
                         t1_bills=int(sum(b.nb for b in bl)), t1_positives=int(sum(b.Pos.sum() for b in bl)),
                         pool_house=pools.get("House"), pool_senate=pools.get("Senat"),
                         t2_bills=len(t2), t2_pos_rate=float(t2.y.mean()) if len(t2) else np.nan))
    return pd.DataFrame(rows).set_index("congress")
