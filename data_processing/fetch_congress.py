#!/usr/bin/env python3
"""
Fetch the Congress.gov API data that the BILLSTATUS bulk download lacks, as JSON.

Bills (details, cosponsors, committees, subjects, actions, ...) come from the
bulk files in data/billstatus_bulk/. This script adds:

  default phases
    congresses        data/congress_api/congresses.json                       (1 call)
    members           data/congress_api/members/by-congress/<c>.json          (~3 calls/congress)
    committees        data/congress_api/committees/by-congress/<c>.json       (~1-2 calls/congress)
  optional phases
    member_details    data/congress_api/members/details/<bioguideId>.json     (1 call/member, ~2k)
                      covers every member in the lists above AND every sponsor/
                      cosponsor bioguideId found in the bulk BILLSTATUS files
    committee_details data/congress_api/committees/details/<chamber>-<code>.json (1 call/committee)
    house_votes       data/congress_api/house-votes/<c>/<session>/<roll>.json (118th Congress onward only)

By default the congress range matches the folders in data/billstatus_bulk/.

Every list call uses limit=250 (the API maximum). A sliding one-hour window
(persisted in data/congress_api/_state/) keeps us under 5,000 requests/hour; on HTTP 429
the script sleeps until the window frees up and retries. Re-running skips files
that already exist, so it is safe to interrupt and resume.

API key: set CONGRESS_API_KEY, or put it in secrets.ini next to this file.

Usage:
  python3 fetch_congress.py                                   # default phases, bulk congress range
  python3 fetch_congress.py --phases all                      # everything
  python3 fetch_congress.py --phases member_details committee_details
  python3 fetch_congress.py --congresses 117 118 --refresh    # re-download existing files
"""
import argparse
import configparser
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import deque
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent  # Final Project/data
DEFAULT_OUT = HERE / "congress_api"
BULK_DIR = HERE / "billstatus_bulk"
SECRETS_FILE = HERE / "secrets.ini"
API_ROOT = os.environ.get("CONGRESS_API_ROOT", "https://api.congress.gov/v3/")

PAGE_SIZE = 250              # API maximum per call
WINDOW_SECONDS = 3600        # api.data.gov limits are a rolling hour
DEFAULT_HOURLY_LIMIT = 4900  # 5,000 minus headroom for manual testing
MIN_RATE_LIMIT_SLEEP = 300   # minimum sleep after a 429
MAX_RETRIES = 5

DEFAULT_PHASES = ["congresses", "members", "committees"]
OPTIONAL_PHASES = ["member_details", "committee_details", "house_votes"]
ALL_PHASES = DEFAULT_PHASES + OPTIONAL_PHASES


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def log(msg):
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}", flush=True)


class FetchError(Exception):
    pass


def load_api_key():
    key = os.environ.get("CONGRESS_API_KEY", "").strip()
    if not key and SECRETS_FILE.exists():
        cfg = configparser.ConfigParser()
        cfg.read(SECRETS_FILE)
        key = cfg.get("cdg_api", "api_auth_key", fallback="").strip()
    if not key or key.startswith("PASTE"):
        sys.exit(f"No API key. Set CONGRESS_API_KEY or edit {SECRETS_FILE}.\n"
                 "Get a free key at https://api.congress.gov/sign-up/")
    return key


def write_json(path, obj):
    """Atomic write so an interrupted run never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1))
    tmp.replace(path)


class RateLimiter:
    """Sliding one-hour window, persisted so restarts don't overshoot."""

    def __init__(self, state_file, hourly_limit):
        self.state_file = state_file
        self.limit = hourly_limit
        stamps = json.loads(state_file.read_text()) if state_file.exists() else []
        self.stamps = deque(sorted(stamps))
        self._prune()

    def _prune(self):
        cutoff = time.time() - WINDOW_SECONDS
        while self.stamps and self.stamps[0] <= cutoff:
            self.stamps.popleft()

    def save(self):
        self._prune()
        write_json(self.state_file, list(self.stamps))

    def used(self):
        self._prune()
        return len(self.stamps)

    def acquire(self):
        self._prune()
        if len(self.stamps) >= self.limit:
            self._sleep(self.stamps[0] + WINDOW_SECONDS - time.time() + 1,
                        f"local budget of {self.limit}/hour reached")
            self._prune()
        self.stamps.append(time.time())
        if len(self.stamps) % 25 == 0:
            self.save()

    def wait_for_reset(self, retry_after=None):
        """Called when the server says we're out of quota."""
        if retry_after and str(retry_after).isdigit():
            wait = int(retry_after)
        elif self.stamps:
            wait = self.stamps[0] + WINDOW_SECONDS - time.time() + 1
        else:
            wait = 15 * 60  # quota used elsewhere; we can't see that window
        self._sleep(max(wait, MIN_RATE_LIMIT_SLEEP), "server reports rate limit exceeded")

    def _sleep(self, seconds, reason):
        self.save()
        seconds = max(0, seconds)
        resume = datetime.fromtimestamp(time.time() + seconds)
        log(f"Rate limit: {reason}. Sleeping {seconds / 60:.1f} min (until {resume:%H:%M:%S}).")
        time.sleep(seconds)


class Client:
    def __init__(self, api_key, limiter, error_log):
        self.key = api_key
        self.limiter = limiter
        self.error_log = error_log
        self.requests_made = 0

    def get(self, path, params=None):
        """GET one page. Returns parsed JSON, or None on 404."""
        query = urllib.parse.urlencode({"format": "json", **(params or {})})
        url = urllib.parse.urljoin(API_ROOT, path) + "?" + query
        attempt = 0
        while True:
            self.limiter.acquire()
            self.requests_made += 1
            req = urllib.request.Request(url, headers={
                "X-Api-Key": self.key, "Accept": "application/json",
                "User-Agent": "cs224w-congress-fetcher/2.0"})
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    body = resp.read()
                    remaining = resp.headers.get("X-RateLimit-Remaining")
                data = json.loads(body)
                if remaining == "0":
                    self.limiter.wait_for_reset()
                if self.requests_made % 100 == 0:
                    log(f"  {self.requests_made} requests this run; "
                        f"{self.limiter.used()}/{self.limiter.limit} in the last hour; "
                        f"server remaining={remaining}")
                return data
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    self.limiter.wait_for_reset(e.headers.get("Retry-After"))
                    continue
                if e.code == 404:
                    return None
                if e.code in (401, 403):
                    detail = e.read()[:300].decode(errors="replace")
                    sys.exit(f"HTTP {e.code} for {path}: {detail}\n"
                             "Either the API key is invalid, or (in Apple Claude Code) "
                             "api.congress.gov is not on the domain allowlist.")
                if e.code >= 500 and attempt < MAX_RETRIES:
                    attempt += 1
                    time.sleep(2 ** attempt)
                    continue
                raise FetchError(f"HTTP {e.code} for {url}")
            except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
                if attempt < MAX_RETRIES:
                    attempt += 1
                    time.sleep(2 ** attempt)
                    continue
                raise FetchError(f"{type(e).__name__}: {e} for {url}")

    def fetch_all(self, path, list_key, params=None):
        """Fetch every page of a list endpoint. Returns (items, API-reported count)."""
        items, expected, offset = [], None, 0
        while True:
            data = self.get(path, {**(params or {}), "limit": PAGE_SIZE, "offset": offset})
            if data is None:
                break
            page = data.get(list_key) or []
            items.extend(page)
            pagination = data.get("pagination") or {}
            expected = pagination.get("count", expected)
            offset += PAGE_SIZE
            has_next = pagination.get("next") or (not pagination and len(page) == PAGE_SIZE)
            if not page or not has_next:
                break
        if expected is not None and len(items) < expected:
            log(f"  WARNING {path}: got {len(items)} of {expected} reported items")
        return items, expected

    def record_error(self, path, err):
        self.error_log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.error_log, "a") as f:
            f.write(json.dumps({"path": path, "error": str(err), "at": now_iso()}) + "\n")
        log(f"  ERROR {path}: {err} (will retry on next run)")


def list_file(path, list_key, items, expected, params=None):
    return {"source": path, "params": params or {}, "fetched_at": now_iso(),
            "api_count": expected, "count": len(items), list_key: items}


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------

def phase_congresses(client, out, congresses, refresh):
    target = out / "congresses.json"
    if target.exists() and not refresh:
        return
    items, expected = client.fetch_all("congress", "congresses")
    write_json(target, list_file("congress", "congresses", items, expected))
    log(f"congresses: {len(items)} stored")


def phase_members(client, out, congresses, refresh):
    for c in congresses:
        target = out / "members" / "by-congress" / f"{c}.json"
        if target.exists() and not refresh:
            continue
        path = f"member/congress/{c}"
        # currentMember=false is required for complete data on past congresses (MemberEndpoint.md)
        params = {"currentMember": "false"}
        try:
            items, expected = client.fetch_all(path, "members", params)
        except FetchError as e:
            client.record_error(path, e)
            continue
        write_json(target, list_file(path, "members", items, expected, params))
        log(f"members c{c}: {len(items)} stored")


def phase_committees(client, out, congresses, refresh):
    for c in congresses:
        target = out / "committees" / "by-congress" / f"{c}.json"
        if target.exists() and not refresh:
            continue
        path = f"committee/{c}"
        try:
            items, expected = client.fetch_all(path, "committees")
        except FetchError as e:
            client.record_error(path, e)
            continue
        write_json(target, list_file(path, "committees", items, expected))
        log(f"committees c{c}: {len(items)} stored")


BIOGUIDE_RE = re.compile(rb"<bioguideId>([A-Z]\d{6})</bioguideId>")


def bulk_bioguide_ids(congresses):
    """Every sponsor/cosponsor/amendment-sponsor ID in the bulk BILLSTATUS files."""
    ids = set()
    for c in congresses:
        for f in (BULK_DIR / str(c)).glob("*/*.xml"):
            ids.update(m.decode() for m in BIOGUIDE_RE.findall(f.read_bytes()))
    return ids


def phase_member_details(client, out, congresses, refresh):
    ids = set()
    for c in congresses:
        f = out / "members" / "by-congress" / f"{c}.json"
        if f.exists():
            ids.update(m["bioguideId"] for m in json.loads(f.read_text())["members"]
                       if m.get("bioguideId"))
    n_listed = len(ids)
    if BULK_DIR.exists():
        log(f"member_details: scanning bulk BILLSTATUS files for bioguide IDs...")
        ids |= bulk_bioguide_ids(congresses)
    log(f"member_details: {len(ids)} members ({len(ids) - n_listed} only seen in bulk bill files)")
    target_dir = out / "members" / "details"
    todo = sorted(i for i in ids if refresh or not (target_dir / f"{i}.json").exists())
    log(f"member_details: {len(todo)} to fetch")
    for bid in todo:
        path = f"member/{bid}"
        try:
            data = client.get(path)
        except FetchError as e:
            client.record_error(path, e)
            continue
        if data is None:
            client.record_error(path, "404 not found")
            continue
        write_json(target_dir / f"{bid}.json", data.get("member", data))


def phase_committee_details(client, out, congresses, refresh):
    committees = {}  # (chamber, code) -> name
    for c in congresses:
        f = out / "committees" / "by-congress" / f"{c}.json"
        if not f.exists():
            continue
        for item in json.loads(f.read_text())["committees"]:
            chamber = (item.get("chamber") or "").lower()
            for entry in [item, *(item.get("subcommittees") or [])]:
                if entry.get("systemCode") and chamber:
                    committees[(chamber, entry["systemCode"])] = entry.get("name")
    if not committees:
        log("committee_details: no committee lists found; run the committees phase first")
        return
    target_dir = out / "committees" / "details"
    todo = sorted(k for k in committees
                  if refresh or not (target_dir / f"{k[0]}-{k[1]}.json").exists())
    log(f"committee_details: {len(committees)} committees, {len(todo)} to fetch")
    for chamber, code in todo:
        path = f"committee/{chamber}/{code}"
        try:
            data = client.get(path)
        except FetchError as e:
            client.record_error(path, e)
            continue
        if data is None:
            client.record_error(path, "404 not found")
            continue
        write_json(target_dir / f"{chamber}-{code}.json", data.get("committee", data))


def phase_house_votes(client, out, congresses, refresh):
    for c in congresses:
        if c < 118:
            continue  # the API only has House roll-call votes from the 118th Congress on
        for session in (1, 2):
            path = f"house-vote/{c}/{session}"
            try:
                votes, _ = client.fetch_all(path, "houseRollCallVotes")
            except FetchError as e:
                client.record_error(path, e)
                continue
            vote_dir = out / "house-votes" / str(c) / str(session)
            todo = [v for v in votes if v.get("rollCallNumber") is not None and
                    (refresh or not (vote_dir / f"{v['rollCallNumber']}.json").exists())]
            log(f"house_votes c{c} s{session}: {len(votes)} votes, {len(todo)} to fetch")
            for v in todo:
                roll = v["rollCallNumber"]
                mpath = f"house-vote/{c}/{session}/{roll}/members"
                try:
                    pages, offset = [], 0
                    # member results are nested, so page manually instead of fetch_all
                    while True:
                        data = client.get(mpath, {"limit": PAGE_SIZE, "offset": offset})
                        block = (data or {}).get("houseRollCallVoteMemberVotes") or {}
                        results = block.get("results") or []
                        pages.extend(results)
                        if len(results) < PAGE_SIZE or not (data.get("pagination") or {}).get("next"):
                            break
                        offset += PAGE_SIZE
                except FetchError as e:
                    client.record_error(mpath, e)
                    continue
                write_json(vote_dir / f"{roll}.json",
                           {"vote": v, "detail": {k: val for k, val in block.items() if k != "results"},
                            "results": pages, "fetched_at": now_iso()})


PHASE_FUNCS = {
    "congresses": phase_congresses,
    "members": phase_members,
    "committees": phase_committees,
    "member_details": phase_member_details,
    "committee_details": phase_committee_details,
    "house_votes": phase_house_votes,
}


def bulk_congresses():
    if not BULK_DIR.exists():
        return []
    return sorted(int(p.name) for p in BULK_DIR.iterdir() if p.is_dir() and p.name.isdigit())


def resolve_phases(phases):
    if phases == ["all"]:
        return ALL_PHASES
    if phases == ["default"]:
        return DEFAULT_PHASES
    unknown = set(phases) - set(ALL_PHASES)
    if unknown:
        sys.exit(f"Unknown phases: {sorted(unknown)}. Choose from {ALL_PHASES}, 'default' or 'all'.")
    return sorted(phases, key=ALL_PHASES.index)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--congresses", type=int, nargs="+",
                   help=f"default: the congresses present in data/{BULK_DIR.name}/")
    p.add_argument("--phases", nargs="+", default=["default"],
                   help=f"'default' ({' '.join(DEFAULT_PHASES)}), 'all', or any of {ALL_PHASES}")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output directory")
    p.add_argument("--refresh", action="store_true", help="re-download files that already exist")
    p.add_argument("--hourly-limit", type=int, default=DEFAULT_HOURLY_LIMIT)
    args = p.parse_args()

    congresses = args.congresses or bulk_congresses()
    if not congresses:
        sys.exit(f"No congresses given and none found in {BULK_DIR}. Pass --congresses.")
    phases = resolve_phases(args.phases)
    out = args.out.resolve()

    limiter = RateLimiter(out / "_state" / "request_log.json", args.hourly_limit)
    client = Client(load_api_key(), limiter, out / "_state" / "errors.jsonl")
    log(f"congresses {congresses[0]}-{congresses[-1]} ({len(congresses)}), phases {phases}, out {out}")
    try:
        for phase in phases:
            PHASE_FUNCS[phase](client, out, congresses, args.refresh)
    except KeyboardInterrupt:
        log("Interrupted. Finished files are saved; re-run to continue.")
    finally:
        limiter.save()
        log(f"{client.requests_made} requests made this run.")


if __name__ == "__main__":
    main()
