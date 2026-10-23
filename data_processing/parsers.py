# /// script
# requires-python = ">=3.10,<3.14"
# dependencies = ["pyspark>=4.0,<4.1", "defusedxml>=0.7"]
# ///
"""
ETL parsers: Congress data -> Parquet tables for BigQuery.

Sources
  data/billstatus_bulk/<congress>/<type>/BILLSTATUS-*.xml   GovInfo BILLSTATUS bulk download
  data/congress_api/...                                     Congress.gov API JSON (fetch_congress.py)

Output
  data/parsed/<table>.parquet/     one Parquet dataset per table, Hive-partitioned
                                   (congress=118/part-*.parquet) when the table has a congress
  data/parsed/_staging/billstatus_raw.parquet/
                                   every XML parsed once into one nested row per bill; the
                                   bill tables are derived from it, so re-running a single
                                   table doesn't re-parse 170k XML files

Usage (run from your own terminal; uv installs PySpark, Java 17 must be installed):
  uv run data/parsers.py --list                        # show available tables
  uv run data/parsers.py                               # build every table
  uv run data/parsers.py --table bills                 # one table
  uv run data/parsers.py --table bill_cosponsors bill_actions
  uv run data/parsers.py --table api                   # all API tables (or: billstatus)
  uv run data/parsers.py --congresses 116 117 118      # only these partitions
  uv run data/parsers.py --rebuild-staging             # after re-downloading the bulk XML

Partitioned tables are rewritten one congress partition at a time (dynamic partition
overwrite), so `--congresses 118` replaces congress=118 and leaves the others alone. Each
partition is written as a single file (up to --max-records-per-file rows), which keeps
the file count low for BigQuery.
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

DATA_DIR = Path(__file__).resolve().parent
BULK_DIR = DATA_DIR / "billstatus_bulk"
API_DIR = DATA_DIR / "congress_api"
DEFAULT_OUT = DATA_DIR / "parsed"

# Spark's Python workers must use this interpreter (the one uv set up with defusedxml).
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

import defusedxml.ElementTree as ET  # noqa: E402  (bulk files are downloaded data: refuse XXE/entity bombs)
from pyspark.sql import Column, DataFrame, SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql.types import (  # noqa: E402
    ArrayType,
    BooleanType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

# ---------------------------------------------------------------------------
# Staging schema: one nested row per BILLSTATUS file
# ---------------------------------------------------------------------------

STR, INT, BOOL = StringType(), IntegerType(), BooleanType()


def struct(*fields):
    return StructType([StructField(name, dtype, True) for name, dtype in fields])


def array(dtype):
    return ArrayType(dtype, True)


PERSON = [
    ("bioguide_id", STR),
    ("full_name", STR),
    ("first_name", STR),
    ("last_name", STR),
    ("party", STR),
    ("state", STR),
    ("district", INT),
]

BILLSTATUS_SCHEMA = struct(
    ("congress", INT),
    ("bill_type", STR),
    ("bill_number", INT),
    ("title", STR),
    ("introduced_date", STR),
    ("origin_chamber", STR),
    ("policy_area", STR),
    ("update_date", STR),
    ("update_date_including_text", STR),
    ("latest_action_date", STR),
    ("latest_action_text", STR),
    ("sponsors", array(struct(*PERSON, ("is_by_request", STR)))),
    (
        "cosponsors",
        array(
            struct(
                *PERSON,
                ("sponsorship_date", STR),
                ("is_original", BOOL),
                ("withdrawn_date", STR),
            )
        ),
    ),
    (
        "committees",
        array(
            struct(
                ("committee_code", STR),
                ("name", STR),
                ("chamber", STR),
                ("committee_type", STR),
                ("parent_code", STR),
                ("activities", array(struct(("name", STR), ("date", STR)))),
            )
        ),
    ),
    ("subjects", array(STR)),
    (
        "actions",
        array(
            struct(
                ("action_date", STR),
                ("action_time", STR),
                ("action_type", STR),
                ("action_code", STR),
                ("source_code", STR),
                ("source_name", STR),
                ("text", STR),
                ("committee_codes", array(STR)),
                (
                    "recorded_votes",
                    array(
                        struct(
                            ("roll_number", INT),
                            ("chamber", STR),
                            ("congress", INT),
                            ("session_number", INT),
                            ("date", STR),
                            ("url", STR),
                        )
                    ),
                ),
            )
        ),
    ),
    (
        "related_bills",
        array(
            struct(
                ("congress", INT),
                ("bill_type", STR),
                ("bill_number", INT),
                ("title", STR),
                ("relationships", array(struct(("type", STR), ("identified_by", STR)))),
            )
        ),
    ),
    (
        "titles",
        array(
            struct(
                ("title_type", STR),
                ("title", STR),
                ("chamber", STR),
                ("text_version", STR),
            )
        ),
    ),
    (
        "summaries",
        array(
            struct(
                ("version_code", STR),
                ("action_date", STR),
                ("action_desc", STR),
                ("update_date", STR),
                ("text", STR),
            )
        ),
    ),
    (
        "text_versions",
        array(struct(("version_type", STR), ("date", STR), ("urls", array(STR)))),
    ),
    ("laws", array(struct(("law_type", STR), ("law_number", STR)))),
    (
        "cbo_estimates",
        array(
            struct(
                ("pub_date", STR), ("title", STR), ("description", STR), ("url", STR)
            )
        ),
    ),
    ("source_file", STR),
)


# ---------------------------------------------------------------------------
# XML -> staging row (runs inside Spark's Python workers)
# ---------------------------------------------------------------------------


def _t(el, path):
    """Stripped text at `path`, or None."""
    value = el.findtext(path) if el is not None else None
    value = value.strip() if value else None
    return value or None


def _i(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _person(el):
    return (
        _t(el, "bioguideId"),
        _t(el, "fullName"),
        _t(el, "firstName"),
        _t(el, "lastName"),
        _t(el, "party"),
        _t(el, "state"),
        _i(_t(el, "district")),
    )


def _activities(el):
    return [(_t(a, "name"), _t(a, "date")) for a in el.findall("activities/item")]


def parse_billstatus(path, content):
    """One BILLSTATUS XML file -> [staging row] (a tuple in BILLSTATUS_SCHEMA order)."""
    b = ET.fromstring(content).find("bill")
    if b is None:
        return []

    committees = []
    for c in b.findall("committees/item"):
        code, chamber = _t(c, "systemCode"), _t(c, "chamber")
        committees.append(
            (code, _t(c, "name"), chamber, _t(c, "type"), None, _activities(c))
        )
        for s in c.findall("subcommittees/item"):
            committees.append(
                (
                    _t(s, "systemCode"),
                    _t(s, "name"),
                    chamber,
                    "Subcommittee",
                    code,
                    _activities(s),
                )
            )

    actions = []
    for a in b.findall("actions/item"):
        votes = [
            (
                _i(_t(v, "rollNumber")),
                _t(v, "chamber"),
                _i(_t(v, "congress")),
                _i(_t(v, "sessionNumber")),
                _t(v, "date"),
                _t(v, "url"),
            )
            for v in a.findall("recordedVotes/recordedVote")
        ]
        actions.append(
            (
                _t(a, "actionDate"),
                _t(a, "actionTime"),
                _t(a, "type"),
                _t(a, "actionCode"),
                _t(a, "sourceSystem/code"),
                _t(a, "sourceSystem/name"),
                _t(a, "text"),
                [_t(c, "systemCode") for c in a.findall("committees/item")],
                votes,
            )
        )

    related = [
        (
            _i(_t(r, "congress")),
            (_t(r, "type") or "").lower() or None,
            _i(_t(r, "number")),
            _t(r, "title"),
            [
                (_t(d, "type"), _t(d, "identifiedBy"))
                for d in r.findall("relationshipDetails/item")
            ],
        )
        for r in b.findall("relatedBills/item")
    ]

    source = path.split("billstatus_bulk/", 1)[-1]
    return [
        (
            # A few legacy-format files (mostly reserved bill numbers) use billType/billNumber.
            _i(_t(b, "congress")),
            (_t(b, "type") or _t(b, "billType") or "").lower() or None,
            _i(_t(b, "number") or _t(b, "billNumber")),
            _t(b, "title"),
            _t(b, "introducedDate"),
            _t(b, "originChamber"),
            _t(b, "policyArea/name"),
            _t(b, "updateDate"),
            _t(b, "updateDateIncludingText"),
            _t(b, "latestAction/actionDate"),
            _t(b, "latestAction/text"),
            [(*_person(s), _t(s, "isByRequest")) for s in b.findall("sponsors/item")],
            [
                (
                    *_person(c),
                    _t(c, "sponsorshipDate"),
                    _t(c, "isOriginalCosponsor") == "True",
                    _t(c, "sponsorshipWithdrawnDate"),
                )
                for c in b.findall("cosponsors/item")
            ],
            committees,
            [_t(s, "name") for s in b.findall("subjects/legislativeSubjects/item")],
            actions,
            related,
            [
                (
                    _t(x, "titleType"),
                    _t(x, "title"),
                    _t(x, "chamberName"),
                    _t(x, "billTextVersionName"),
                )
                for x in b.findall("titles/item")
            ],
            [
                (
                    _t(s, "versionCode"),
                    _t(s, "actionDate"),
                    _t(s, "actionDesc"),
                    _t(s, "updateDate"),
                    _t(s, "cdata/text") or _t(s, "text"),
                )
                for s in b.findall("summaries/summary")
            ],
            [
                (
                    _t(v, "type"),
                    _t(v, "date"),
                    [_t(u, "url") for u in v.findall("formats/item")],
                )
                for v in b.findall("textVersions/item")
            ],
            [(_t(x, "type"), _t(x, "number")) for x in b.findall("laws/item")],
            [
                (_t(x, "pubDate"), _t(x, "title"), _t(x, "description"), _t(x, "url"))
                for x in b.findall("cboCostEstimates/item")
            ],
            source,
        )
    ]


# ---------------------------------------------------------------------------
# Context: Spark session, congress filter, cached sources
# ---------------------------------------------------------------------------


class Context:
    def __init__(self, spark, congresses, out_dir, rebuild_staging):
        self.spark = spark
        self.congresses = congresses
        self.out_dir = out_dir
        self.staging = out_dir / "_staging" / "billstatus_raw.parquet"
        self.rebuild_staging = rebuild_staging
        self._billstatus = None

    def billstatus(self) -> DataFrame:
        """Nested BILLSTATUS rows for the requested congresses (parsed once, then reused)."""
        if self._billstatus is None:
            present = {
                int(p.name.split("=")[1]) for p in self.staging.glob("congress=*")
            }
            missing = (
                self.congresses
                if self.rebuild_staging
                else [c for c in self.congresses if c not in present]
            )
            if missing:
                self._parse_xml_to_staging(missing)
            self._billstatus = (
                self.spark.read.parquet(str(self.staging))
                .where(F.col("congress").isin(self.congresses))
                .persist()
            )
        return self._billstatus

    def _parse_xml_to_staging(self, congresses):
        globs = [
            str(BULK_DIR / str(c) / "*" / "*.xml")
            for c in congresses
            if (BULK_DIR / str(c)).is_dir()
        ]
        if not globs:
            raise SystemExit(
                f"No BILLSTATUS XML found under {BULK_DIR} for congresses {congresses}"
            )
        log(f"staging: parsing BILLSTATUS XML for congresses {congresses} (one-time)")
        sc = self.spark.sparkContext
        files = sc.binaryFiles(",".join(globs), minPartitions=sc.defaultParallelism * 8)
        rows = files.flatMap(lambda kv: parse_billstatus(kv[0], kv[1]))
        df = self.spark.createDataFrame(rows, BILLSTATUS_SCHEMA, verifySchema=False)
        (
            df.repartition("congress")
            .write.mode("overwrite")
            .partitionBy("congress")
            .parquet(str(self.staging))
        )

    def api_json(self, pattern) -> DataFrame:
        """Read API JSON files (one JSON document per file) with the source path attached."""
        # Expand the glob here: given a single wildcard path, Spark first probes it as a
        # streaming-sink directory and logs a harmless FileNotFoundException stack trace.
        files = sorted(str(f) for f in API_DIR.glob(pattern))
        if not files:
            raise SystemExit(
                f"No API files match {API_DIR / pattern}; run fetch_congress.py first"
            )
        return (
            self.spark.read.option("multiLine", True)
            .json(files)
            .withColumn("_file", F.col("_metadata.file_path"))
        )


# ---------------------------------------------------------------------------
# Table registry
# ---------------------------------------------------------------------------


@dataclass
class Table:
    name: str
    source: str  # "billstatus" or "api"
    partitioned: bool  # partitioned by congress?
    doc: str
    build: Callable[[Context], DataFrame]


TABLES: dict[str, Table] = {}


def table(name, source, partitioned=True, doc=""):
    def register(fn):
        TABLES[name] = Table(name, source, partitioned, doc, fn)
        return fn

    return register


def bill_id(congress="congress", bill_type="bill_type", number="bill_number") -> Column:
    """Stable bill key, e.g. '118-hr-1234'."""
    return F.concat_ws(
        "-",
        F.col(congress).cast("string"),
        F.col(bill_type),
        F.col(number).cast("string"),
    )


def opt(df: DataFrame, path: str, dtype: str = "string") -> Column:
    """Column at a dotted path, or NULL if no file had that field (JSON schema inference)."""
    schema = df.schema
    for part in path.split("."):
        field = next((f for f in getattr(schema, "fields", []) if f.name == part), None)
        if field is None:
            return F.lit(None).cast(dtype)
        schema = field.dataType
    return F.col(path).cast(dtype)


def congress_start_year(congress: Column) -> Column:
    return F.lit(1789) + (congress - 1) * 2


# --- BILLSTATUS tables -------------------------------------------------------


@table(
    "bills",
    "billstatus",
    doc="One row per bill: metadata, sponsor, law, latest action, counts.",
)
def build_bills(ctx):
    b = ctx.billstatus()
    sponsor, law = F.col("sponsors")[0], F.col("laws")[0]
    return b.select(
        bill_id().alias("bill_id"),
        "congress",
        "bill_type",
        "bill_number",
        "title",
        F.col("introduced_date").cast("date").alias("introduced_date"),
        "origin_chamber",
        "policy_area",
        sponsor["bioguide_id"].alias("sponsor_bioguide_id"),
        sponsor["party"].alias("sponsor_party"),
        sponsor["state"].alias("sponsor_state"),
        sponsor["district"].alias("sponsor_district"),
        (sponsor["is_by_request"] == "Y").alias("sponsor_is_by_request"),
        law["law_type"].alias("law_type"),
        law["law_number"].alias("law_number"),
        F.col("latest_action_date").cast("date").alias("latest_action_date"),
        "latest_action_text",
        F.size("cosponsors").alias("n_cosponsors"),
        F.size("committees").alias("n_committees"),
        F.size("subjects").alias("n_subjects"),
        F.size("actions").alias("n_actions"),
        F.size("related_bills").alias("n_related_bills"),
        F.size("summaries").alias("n_summaries"),
        F.size("text_versions").alias("n_text_versions"),
        F.col("update_date").cast("timestamp").alias("update_date"),
        "source_file",
    )


@table(
    "bill_cosponsors",
    "billstatus",
    doc="One row per (bill, cosponsor), with sponsorship date.",
)
def build_bill_cosponsors(ctx):
    c = ctx.billstatus().select(
        bill_id().alias("bill_id"), "congress", F.explode("cosponsors").alias("c")
    )
    return c.select(
        "bill_id",
        "congress",
        F.col("c.bioguide_id").alias("bioguide_id"),
        F.col("c.full_name").alias("full_name"),
        F.col("c.party").alias("party"),
        F.col("c.state").alias("state"),
        F.col("c.district").alias("district"),
        F.col("c.sponsorship_date").cast("date").alias("sponsorship_date"),
        F.col("c.is_original").alias("is_original"),
        F.col("c.withdrawn_date").cast("date").alias("withdrawn_date"),
    )


@table(
    "bill_committees",
    "billstatus",
    doc="One row per (bill, committee or subcommittee, activity) - referrals, markups, reports.",
)
def build_bill_committees(ctx):
    c = ctx.billstatus().select(
        bill_id().alias("bill_id"), "congress", F.explode("committees").alias("c")
    )
    c = c.select("bill_id", "congress", "c.*").withColumn(
        "a", F.explode_outer("activities")
    )
    return c.select(
        "bill_id",
        "congress",
        "committee_code",
        F.col("name").alias("committee_name"),
        "chamber",
        "committee_type",
        F.col("parent_code").alias("parent_committee_code"),
        F.col("a.name").alias("activity"),
        F.col("a.date").cast("timestamp").alias("activity_ts"),
    )


@table(
    "bill_subjects",
    "billstatus",
    doc="One row per (bill, CRS legislative subject term).",
)
def build_bill_subjects(ctx):
    return ctx.billstatus().select(
        bill_id().alias("bill_id"), "congress", F.explode("subjects").alias("subject")
    )


@table(
    "bill_actions",
    "billstatus",
    doc="One row per bill action, in source order (action_seq).",
)
def build_bill_actions(ctx):
    a = ctx.billstatus().select(
        bill_id().alias("bill_id"),
        "congress",
        F.posexplode("actions").alias("action_seq", "a"),
    )
    return a.select(
        "bill_id",
        "congress",
        "action_seq",
        F.col("a.action_date").cast("date").alias("action_date"),
        F.col("a.action_time").alias("action_time"),
        F.col("a.action_type").alias("action_type"),
        F.col("a.action_code").alias("action_code"),
        F.col("a.source_code").alias("source_system_code"),
        F.col("a.source_name").alias("source_system"),
        F.col("a.text").alias("text"),
        F.col("a.committee_codes").alias("committee_codes"),
    )


@table(
    "bill_recorded_votes",
    "billstatus",
    doc="One row per roll-call vote referenced by a bill action (links bills to House/Senate votes).",
)
def build_bill_recorded_votes(ctx):
    a = ctx.billstatus().select(
        bill_id().alias("bill_id"),
        "congress",
        F.posexplode("actions").alias("action_seq", "a"),
    )
    v = a.select(
        "bill_id", "congress", "action_seq", F.explode("a.recorded_votes").alias("v")
    )
    return v.select(
        "bill_id",
        "congress",
        "action_seq",
        F.col("v.chamber").alias("chamber"),
        F.col("v.session_number").alias("session_number"),
        F.col("v.roll_number").alias("roll_number"),
        F.col("v.date").cast("timestamp").alias("vote_ts"),
        F.col("v.url").alias("url"),
    )


@table(
    "bill_related", "billstatus", doc="One row per (bill, related bill, relationship)."
)
def build_bill_related(ctx):
    r = ctx.billstatus().select(
        bill_id().alias("bill_id"), "congress", F.explode("related_bills").alias("r")
    )
    r = r.withColumn("d", F.explode_outer("r.relationships"))
    return r.select(
        "bill_id",
        "congress",
        bill_id("r.congress", "r.bill_type", "r.bill_number").alias("related_bill_id"),
        F.col("r.title").alias("related_title"),
        F.col("d.type").alias("relationship"),
        F.col("d.identified_by").alias("identified_by"),
    )


@table(
    "bill_titles",
    "billstatus",
    doc="One row per title (short, official, display) per bill.",
)
def build_bill_titles(ctx):
    t = ctx.billstatus().select(
        bill_id().alias("bill_id"), "congress", F.explode("titles").alias("t")
    )
    return t.select(
        "bill_id", "congress", "t.title_type", "t.title", "t.chamber", "t.text_version"
    )


@table(
    "bill_summaries",
    "billstatus",
    doc="One row per CRS summary version (HTML text). Written as bills advance: leakage risk.",
)
def build_bill_summaries(ctx):
    s = ctx.billstatus().select(
        bill_id().alias("bill_id"), "congress", F.explode("summaries").alias("s")
    )
    return s.select(
        "bill_id",
        "congress",
        "s.version_code",
        F.col("s.action_date").cast("date").alias("action_date"),
        "s.action_desc",
        F.col("s.update_date").cast("timestamp").alias("update_date"),
        "s.text",
    )


@table(
    "bill_text_versions",
    "billstatus",
    doc="One row per bill text version, with download URLs.",
)
def build_bill_text_versions(ctx):
    v = ctx.billstatus().select(
        bill_id().alias("bill_id"), "congress", F.explode("text_versions").alias("v")
    )
    return v.select(
        "bill_id",
        "congress",
        "v.version_type",
        F.col("v.date").cast("timestamp").alias("version_ts"),
        "v.urls",
    )


@table(
    "bill_cbo_estimates",
    "billstatus",
    doc="One row per CBO cost estimate attached to a bill.",
)
def build_bill_cbo_estimates(ctx):
    e = ctx.billstatus().select(
        bill_id().alias("bill_id"), "congress", F.explode("cbo_estimates").alias("e")
    )
    return e.select(
        "bill_id",
        "congress",
        F.col("e.pub_date").cast("timestamp").alias("pub_ts"),
        "e.title",
        "e.description",
        "e.url",
    )


# --- Congress.gov API tables -------------------------------------------------


@table("congresses", "api", doc="One row per Congress: name and years.")
def build_congresses(ctx):
    c = ctx.api_json("congresses.json").select(F.explode("congresses").alias("c"))
    return c.select(
        F.regexp_extract("c.name", r"^(\d+)", 1).cast("int").alias("congress"),
        F.col("c.name").alias("name"),
        F.col("c.startYear").cast("int").alias("start_year"),
        F.col("c.endYear").cast("int").alias("end_year"),
    )


@table(
    "congress_sessions",
    "api",
    doc="One row per (Congress, chamber, session) with start/end dates.",
)
def build_congress_sessions(ctx):
    c = ctx.api_json("congresses.json").select(F.explode("congresses").alias("c"))
    s = c.select(
        F.regexp_extract("c.name", r"^(\d+)", 1).cast("int").alias("congress"),
        F.explode("c.sessions").alias("s"),
    )
    return s.select(
        "congress",
        F.col("s.chamber").alias("chamber"),
        opt(s, "s.number", "int").alias("session_number"),
        F.col("s.type").alias("session_type"),
        F.col("s.startDate").cast("date").alias("start_date"),
        opt(s, "s.endDate", "date").alias("end_date"),
    )


@table(
    "members",
    "api",
    doc="One row per (Congress, legislator, chamber served that Congress): party, state, district.",
)
def build_members(ctx):
    m = ctx.api_json("members/by-congress/*.json").select(
        F.regexp_extract("source", r"(\d+)$", 1).cast("int").alias("congress"),
        F.explode("members").alias("m"),
    )
    start = congress_start_year(F.col("congress"))
    # A term counts if it overlaps the Congress's two years [start, start + 2).
    served = F.filter(
        "m.terms.item",
        lambda t: (t["startYear"].cast("int") < start + 2)
        & (F.coalesce(t["endYear"].cast("int"), F.lit(9999)) > start),
    )
    m = m.withColumn(
        "chamber",
        F.explode_outer(F.array_distinct(F.transform(served, lambda t: t["chamber"]))),
    )
    return m.select(
        "congress",
        F.col("m.bioguideId").alias("bioguide_id"),
        F.col("m.name").alias("name"),
        F.col("m.partyName").alias("party"),
        F.col("m.state").alias("state"),
        opt(m, "m.district", "int").alias("district"),
        "chamber",
    )


@table(
    "legislators",
    "api",
    partitioned=False,
    doc="One row per legislator (biography). Not partitioned: a legislator spans many Congresses.",
)
def build_legislators(ctx):
    d = ctx.api_json("members/details/*.json")
    latest_party = F.element_at(
        F.sort_array(
            F.transform(
                "partyHistory",
                lambda p: F.struct(
                    p["startYear"].alias("y"),
                    p["partyName"].alias("party"),
                    p["partyAbbreviation"].alias("abbr"),
                ),
            )
        ),
        -1,
    )
    return d.select(
        F.col("bioguideId").alias("bioguide_id"),
        F.col("firstName").alias("first_name"),
        opt(d, "middleName").alias("middle_name"),
        F.col("lastName").alias("last_name"),
        opt(d, "suffixName").alias("suffix_name"),
        opt(d, "nickName").alias("nick_name"),
        F.col("directOrderName").alias("direct_order_name"),
        opt(d, "honorificName").alias("honorific"),
        opt(d, "birthYear", "int").alias("birth_year"),
        opt(d, "deathYear", "int").alias("death_year"),
        F.col("currentMember").alias("current_member"),
        F.col("state").alias("state"),
        opt(d, "district", "int").alias("district"),
        latest_party["party"].alias("latest_party"),
        latest_party["abbr"].alias("latest_party_code"),
        opt(d, "officialWebsiteUrl").alias("official_website"),
        opt(d, "depiction.imageUrl").alias("image_url"),
        F.col("updateDate").cast("timestamp").alias("update_date"),
    )


@table(
    "member_terms",
    "api",
    doc="One row per (legislator, Congress, chamber) term from member details.",
)
def build_member_terms(ctx):
    t = ctx.api_json("members/details/*.json").select(
        F.col("bioguideId").alias("bioguide_id"), F.explode("terms").alias("t")
    )
    return t.select(
        F.col("t.congress").cast("int").alias("congress"),
        "bioguide_id",
        F.col("t.chamber").alias("chamber"),
        F.col("t.memberType").alias("member_type"),
        F.col("t.stateCode").alias("state_code"),
        opt(t, "t.district", "int").alias("district"),
        F.col("t.startYear").cast("int").alias("start_year"),
        opt(t, "t.endYear", "int").alias("end_year"),
    )


@table(
    "member_party_history",
    "api",
    partitioned=False,
    doc="One row per party affiliation period per legislator.",
)
def build_member_party_history(ctx):
    p = ctx.api_json("members/details/*.json").select(
        F.col("bioguideId").alias("bioguide_id"), F.explode("partyHistory").alias("p")
    )
    return p.select(
        "bioguide_id",
        F.col("p.partyAbbreviation").alias("party_code"),
        F.col("p.partyName").alias("party"),
        F.col("p.startYear").cast("int").alias("start_year"),
        opt(p, "p.endYear", "int").alias("end_year"),
    )


@table("committees", "api", doc="One row per (Congress, committee or subcommittee).")
def build_committees(ctx):
    c = ctx.api_json("committees/by-congress/*.json").select(
        F.regexp_extract("source", r"(\d+)$", 1).cast("int").alias("congress"),
        F.explode("committees").alias("c"),
    )
    return c.select(
        "congress",
        F.col("c.systemCode").alias("committee_code"),
        F.col("c.name").alias("name"),
        F.col("c.chamber").alias("chamber"),
        F.col("c.committeeTypeCode").alias("committee_type"),
        opt(c, "c.parent.systemCode").alias("parent_committee_code"),
        F.col("c.updateDate").cast("timestamp").alias("update_date"),
    )


@table(
    "committee_history",
    "api",
    partitioned=False,
    doc="One row per committee name period (official name, start/end dates).",
)
def build_committee_history(ctx):
    d = ctx.api_json("committees/details/*.json")
    d = d.withColumn("h", F.explode_outer("history"))
    return d.select(
        F.col("systemCode").alias("committee_code"),
        F.regexp_extract("_file", r"details/([a-z]+)-", 1).alias("chamber"),
        F.col("type").alias("committee_type"),
        F.col("isCurrent").alias("is_current"),
        F.col("h.officialName").alias("official_name"),
        F.col("h.libraryOfCongressName").alias("name"),
        F.col("h.startDate").cast("timestamp").alias("start_ts"),
        opt(d, "h.endDate", "timestamp").alias("end_ts"),
    )


@table(
    "house_votes",
    "api",
    doc="One row per House roll-call vote (API covers the 118th Congress on).",
)
def build_house_votes(ctx):
    v = ctx.api_json("house-votes/*/*/*.json")
    leg_type, leg_num = opt(v, "vote.legislationType"), opt(v, "vote.legislationNumber")
    return v.select(
        F.col("vote.congress").cast("int").alias("congress"),
        F.col("vote.sessionNumber").cast("int").alias("session_number"),
        F.col("vote.rollCallNumber").cast("int").alias("roll_call"),
        F.col("vote.startDate").cast("timestamp").alias("vote_ts"),
        F.col("vote.voteType").alias("vote_type"),
        F.col("vote.result").alias("result"),
        opt(v, "detail.voteQuestion").alias("vote_question"),
        F.when(
            leg_type.isNotNull(),
            F.concat_ws(
                "-", F.col("vote.congress").cast("string"), F.lower(leg_type), leg_num
            ),
        ).alias("bill_id"),
        leg_type.alias("legislation_type"),
        leg_num.alias("legislation_number"),
        opt(v, "vote.amendmentType").alias("amendment_type"),
        opt(v, "vote.amendmentNumber").alias("amendment_number"),
        opt(v, "vote.amendmentAuthor").alias("amendment_author"),
        F.col("vote.sourceDataURL").alias("source_url"),
    )


@table(
    "house_vote_members",
    "api",
    doc="One row per (House roll-call vote, member): how they voted.",
)
def build_house_vote_members(ctx):
    v = ctx.api_json("house-votes/*/*/*.json").select(
        F.col("vote.congress").cast("int").alias("congress"),
        F.col("vote.sessionNumber").cast("int").alias("session_number"),
        F.col("vote.rollCallNumber").cast("int").alias("roll_call"),
        F.explode("results").alias("r"),
    )
    return v.select(
        "congress",
        "session_number",
        "roll_call",
        F.col("r.bioguideID").alias("bioguide_id"),
        F.col("r.voteCast").alias("vote_cast"),
        F.col("r.voteParty").alias("party"),
        F.col("r.voteState").alias("state"),
    )


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_table(ctx, t: Table, df: DataFrame, compression, max_records):
    path = ctx.out_dir / f"{t.name}.parquet"
    if t.partitioned:
        # One shuffle partition per congress -> one file per congress=N directory.
        df = df.where(F.col("congress").isin(ctx.congresses)).repartition("congress")
        writer = df.write.mode("overwrite").partitionBy("congress")
    else:
        writer = df.coalesce(1).write.mode("overwrite")
    (
        writer.option("compression", compression)
        .option("maxRecordsPerFile", max_records)
        .parquet(str(path))
    )

    files = list(path.rglob("*.parquet"))
    rows = ctx.spark.read.parquet(str(path)).count()
    size_mb = sum(f.stat().st_size for f in files) / 1e6
    log(
        f"  {t.name}: {rows:,} rows, {len(files)} files, {size_mb:,.1f} MB -> {path.relative_to(DATA_DIR)}"
    )


def make_spark(driver_memory):
    return (
        SparkSession.builder.master("local[*]")
        .appName("congress-etl")
        .config("spark.driver.memory", driver_memory)
        # Only replace the congress partitions being written.
        .config("spark.sql.sources.partitionOverwriteMode", "dynamic")
        # BigQuery reads TIMESTAMP_MICROS; Spark's default INT96 is legacy.
        .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
        .config("spark.sql.session.timeZone", "UTC")
        # Spark 4 enables ANSI mode by default, which fails the job on one bad date
        # string; with it off, unparseable values become NULL instead.
        .config("spark.sql.ansi.enabled", "false")
        .config("spark.sql.shuffle.partitions", "64")
        .config(
            "spark.hadoop.mapreduce.fileoutputcommitter.marksuccessfuljobs", "false"
        )
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def resolve_tables(names):
    if not names or names == ["all"]:
        return list(TABLES.values())
    chosen = []
    for name in names:
        if name in ("billstatus", "api"):
            chosen += [t for t in TABLES.values() if t.source == name]
        elif name in TABLES:
            chosen.append(TABLES[name])
        else:
            raise SystemExit(f"Unknown table '{name}'. Use --list to see the options.")
    return list({t.name: t for t in chosen}.values())  # de-duplicate, keep order


def bulk_congresses():
    return sorted(
        int(p.name) for p in BULK_DIR.iterdir() if p.is_dir() and p.name.isdigit()
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--table",
        "-t",
        nargs="+",
        default=["all"],
        help="table name(s), 'billstatus', 'api', or 'all' (default)",
    )
    p.add_argument("--list", action="store_true", help="list tables and exit")
    p.add_argument(
        "--congresses",
        type=int,
        nargs="+",
        help=f"default: every congress folder in data/{BULK_DIR.name}/",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--rebuild-staging",
        action="store_true",
        help="re-parse the BILLSTATUS XML even if staging already has those congresses",
    )
    p.add_argument(
        "--compression",
        default="snappy",
        choices=["snappy", "zstd", "gzip"],
        help="Parquet codec (snappy is the safest for BigQuery)",
    )
    p.add_argument("--max-records-per-file", type=int, default=5_000_000)
    p.add_argument("--driver-memory", default="8g")
    args = p.parse_args()

    if args.list:
        for t in TABLES.values():
            part = "congress" if t.partitioned else "-"
            print(f"{t.name:22} {t.source:10} partition={part:9} {t.doc}")
        return

    tables = resolve_tables(args.table)
    congresses = sorted(args.congresses or bulk_congresses())
    spark = make_spark(args.driver_memory)
    spark.sparkContext.setLogLevel("WARN")
    ctx = Context(spark, congresses, args.out.resolve(), args.rebuild_staging)
    log(
        f"congresses {congresses[0]}-{congresses[-1]}; tables: {', '.join(t.name for t in tables)}"
    )
    try:
        for t in tables:
            started = time.time()
            log(f"{t.name} ...")
            write_table(
                ctx, t, t.build(ctx), args.compression, args.max_records_per_file
            )
            log(f"  done in {time.time() - started:.0f}s")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
