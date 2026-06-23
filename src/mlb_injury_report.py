#!/usr/bin/env python3
"""
MLB injury report scraper for StateSpace.

Fetches all 30 MLB team injury pages from MLB.com and outputs normalized JSON.

Install:
    pip install requests beautifulsoup4

Run:
    python3 scripts/mlb_injury_report.py --pretty

Save:
    python3 scripts/mlb_injury_report.py --pretty --output data/mlb_injuries.json

Single team:
    python3 scripts/mlb_injury_report.py --team mets --pretty
    python3 scripts/mlb_injury_report.py --team NYM --pretty
"""

# flake8: noqa: E402,E501

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.mlb_injuries import persist_mlb_injury_snapshot
from src.data.unified_db import DB_PATH


BASE_URL = "https://www.mlb.com"
INJURY_INDEX_URL = "https://www.mlb.com/injury-report"

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/126.0.0.0 Safari/537.36"
)

DEFAULT_HEADERS = {
    # Browser-like UA avoids failures caused by Python's default requests UA.
    # Do not use this to evade hard bot challenges, CAPTCHA, or access controls.
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.mlb.com/",
    "Connection": "keep-alive",
}


@dataclass(frozen=True)
class TeamInfo:
    team: str
    abbr: str
    slug: str


TEAMS: list[TeamInfo] = [
    TeamInfo("Blue Jays", "TOR", "blue-jays"),
    TeamInfo("Orioles", "BAL", "orioles"),
    TeamInfo("Rays", "TB", "rays"),
    TeamInfo("Red Sox", "BOS", "red-sox"),
    TeamInfo("Yankees", "NYY", "yankees"),
    TeamInfo("Guardians", "CLE", "guardians"),
    TeamInfo("Royals", "KC", "royals"),
    TeamInfo("Tigers", "DET", "tigers"),
    TeamInfo("Twins", "MIN", "twins"),
    TeamInfo("White Sox", "CWS", "white-sox"),
    TeamInfo("Angels", "LAA", "angels"),
    TeamInfo("Astros", "HOU", "astros"),
    TeamInfo("Athletics", "ATH", "athletics"),
    TeamInfo("Mariners", "SEA", "mariners"),
    TeamInfo("Rangers", "TEX", "rangers"),
    TeamInfo("Braves", "ATL", "braves"),
    TeamInfo("Marlins", "MIA", "marlins"),
    TeamInfo("Mets", "NYM", "mets"),
    TeamInfo("Nationals", "WSH", "nationals"),
    TeamInfo("Phillies", "PHI", "phillies"),
    TeamInfo("Brewers", "MIL", "brewers"),
    TeamInfo("Cardinals", "STL", "cardinals"),
    TeamInfo("Cubs", "CHC", "cubs"),
    TeamInfo("Pirates", "PIT", "pirates"),
    TeamInfo("Reds", "CIN", "reds"),
    TeamInfo("D-backs", "ARI", "d-backs"),
    TeamInfo("Dodgers", "LAD", "dodgers"),
    TeamInfo("Giants", "SF", "giants"),
    TeamInfo("Padres", "SD", "padres"),
    TeamInfo("Rockies", "COL", "rockies"),
]


POSITION_TOKEN = r"RHP|LHP|SP|RP|P|C|1B|2B|3B|SS|LF|CF|RF|OF|INF|IF|UTIL|DH"

PLAYER_HEADER_RE = re.compile(
    rf"^(?P<position>(?:{POSITION_TOKEN})(?:/(?:{POSITION_TOKEN}))*)\s+"
    rf"(?P<player>.+)$",
    flags=re.I,
)

INJURY_HEADING_RE = re.compile(
    r"^(?:LATEST\s+)?INJUR(?:Y|IES)(?:\s+(?:UPDATES?|NEWS))?$",
    flags=re.I,
)

TRANSACTIONS_HEADING_RE = re.compile(
    r"^(?:LATEST\s+)?TRANSACTIONS$",
    flags=re.I,
)

INJURY_BLOCK_RE = re.compile(
    r"^(?P<header>.*?)\s+"
    r"Injury\s*:\s*(?P<injury>.*?)\s+"
    r"(?:IL\s+[Dd]ate\s*:\s*(?P<il_date>.*?)\s+)?"
    r"Expected return\s*:\s*(?P<expected_return>.*?)\s+"
    r"Status\s*:\s*(?P<status>.*)$",
    flags=re.I | re.S,
)

UPDATED_RE = re.compile(
    r"\(\s*(?:Last\s+)?updated\s*:?\s*(?P<updated>[^)]+?)\s*\)",
    flags=re.I,
)


def normalize_labels(text: str) -> str:
    """
    Normalize MLB's inconsistent label spacing:
      Injury :
      IL date :
      Expected return :
      Status :
    """
    text = re.sub(r"\bInjury\s+:", "Injury:", text, flags=re.I)
    text = re.sub(r"\bIL\s+[Dd]ate\s+:", "IL date:", text, flags=re.I)
    text = re.sub(r"\bExpected return\s+:", "Expected return:", text, flags=re.I)
    text = re.sub(r"\bStatus\s+:", "Status:", text, flags=re.I)
    return clean_text(text)


@dataclass
class InjuryItem:
    team: str
    team_abbr: str
    team_slug: str
    player: str
    position: Optional[str]
    injury: Optional[str]
    il_date: Optional[str]
    expected_return: Optional[str]
    status: Optional[str]
    updated: Optional[str]
    source_url: str
    fetched_at_utc: str


def clean_text(value: Optional[str]) -> str:
    if not value:
        return ""
    value = value.replace("\xa0", " ")
    return re.sub(r"\s+", " ", value).strip()


def injury_url_for_team(team: TeamInfo) -> str:
    return f"{BASE_URL}/news/{team.slug}-injuries-and-roster-moves"


def make_session(user_agent: str = DEFAULT_USER_AGENT) -> requests.Session:
    session = requests.Session()

    headers = dict(DEFAULT_HEADERS)
    headers["User-Agent"] = user_agent
    session.headers.update(headers)

    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=0.7,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def fetch_soup(session: requests.Session, url: str, timeout: int = 20) -> BeautifulSoup:
    response = session.get(url, timeout=timeout)

    if response.status_code == 403:
        raise RuntimeError(
            f"403 from MLB for {url}. The script uses a normal browser-like "
            "User-Agent, but it should not attempt to bypass hard access controls."
        )

    response.raise_for_status()
    return BeautifulSoup(response.text, "html.parser")


def is_injury_heading(text: str) -> bool:
    return bool(INJURY_HEADING_RE.match(clean_text(text)))


def is_transactions_heading(text: str) -> bool:
    return bool(TRANSACTIONS_HEADING_RE.match(clean_text(text)))


def find_injury_heading(soup: BeautifulSoup):
    return soup.find(
        lambda tag: tag.name in {"h2", "h3", "h4"}
        and is_injury_heading(tag.get_text(" ", strip=True))
    )


def get_page_title(soup: BeautifulSoup) -> Optional[str]:
    h1 = soup.find("h1")
    if h1:
        title = clean_text(h1.get_text(" ", strip=True))
        return title or None
    return None


def strip_more_link_text(text: str) -> str:
    text = normalize_labels(text)
    text = re.sub(r"\s*More\s*>>\s*", " ", text, flags=re.I)
    text = re.sub(r"\s*•\s*More\s+.*?injury news\s*$", "", text, flags=re.I)
    return clean_text(text)


def extract_injury_blocks_from_paragraphs(soup: BeautifulSoup) -> list[str]:
    """
    Preferred parser.

    MLB team injury pages usually put one player injury item in one paragraph:
        POS Player
        Injury: ...
        IL date: ...
        Expected return: ...
        Status: ...
    """
    start = find_injury_heading(soup)

    if not start:
        return []

    blocks: list[str] = []

    for node in start.find_all_next(["h2", "h3", "h4", "p", "li"]):
        text = normalize_labels(node.get_text(" ", strip=True))

        if not text:
            continue

        if node.name in {"h2", "h3", "h4"} and is_transactions_heading(text):
            break

        if node.name not in {"p", "li"}:
            continue

        if "Injury:" not in text:
            continue

        if "Expected return:" not in text:
            continue

        if "Status:" not in text:
            continue

        blocks.append(strip_more_link_text(text))

    return blocks


def extract_section_text_fallback(soup: BeautifulSoup) -> str:
    """
    Fallback if paragraph extraction breaks due to MLB markup changes.
    Pulls raw text between the injury heading and transactions heading.
    """
    page_text = soup.get_text("\n", strip=True)
    lines = [clean_text(line) for line in page_text.splitlines()]
    lines = [line for line in lines if line]

    start_idx = None
    end_idx = None

    for idx, line in enumerate(lines):
        if is_injury_heading(line):
            start_idx = idx + 1
            break

    if start_idx is None:
        return ""

    for idx in range(start_idx, len(lines)):
        if is_transactions_heading(lines[idx]):
            end_idx = idx
            break

    section_lines = lines[start_idx:end_idx]
    return "\n".join(section_lines)


def combine_split_position_lines(lines: list[str]) -> list[str]:
    """
    In fallback mode, BeautifulSoup can split:
        CF
        Daulton Varsho
    into two lines. Combine them into:
        CF Daulton Varsho
    """
    combined: list[str] = []
    idx = 0

    position_only_re = re.compile(
        rf"^(?:{POSITION_TOKEN})(?:/(?:{POSITION_TOKEN}))*$", re.I
    )

    while idx < len(lines):
        current = clean_text(lines[idx])

        if (
            position_only_re.match(current)
            and idx + 1 < len(lines)
            and not lines[idx + 1].startswith(
                ("Injury:", "IL date:", "Expected return:", "Status:")
            )
        ):
            combined.append(f"{current} {clean_text(lines[idx + 1])}")
            idx += 2
            continue

        combined.append(current)
        idx += 1

    return combined


def extract_injury_blocks_from_text(section_text: str) -> list[str]:
    """
    Fallback parser that turns line-based article text into player blocks.
    """
    if not section_text:
        return []

    raw_lines = [clean_text(line) for line in section_text.splitlines()]
    lines = [line for line in raw_lines if line]
    lines = combine_split_position_lines(lines)

    blocks: list[str] = []
    current: list[str] = []

    for line in lines:
        if line.startswith("• More "):
            break

        if PLAYER_HEADER_RE.match(line):
            if current:
                block = strip_more_link_text(" ".join(current))
                if (
                    "Injury:" in block
                    and "Expected return:" in block
                    and "Status:" in block
                ):
                    blocks.append(block)
            current = [line]
            continue

        if current:
            current.append(line)

    if current:
        block = strip_more_link_text(" ".join(current))
        if "Injury:" in block and "Expected return:" in block and "Status:" in block:
            blocks.append(block)

    return blocks


def extract_injury_blocks(soup: BeautifulSoup) -> list[str]:
    blocks = extract_injury_blocks_from_paragraphs(soup)

    if blocks:
        return blocks

    section_text = extract_section_text_fallback(soup)
    return extract_injury_blocks_from_text(section_text)


def normalize_player_name(player: str) -> str:
    player = clean_text(player)

    # Remove prospect notes appended to player names.
    # Example: "Ricky Tiedemann (Blue Jays' No. 4 prospect)"
    player = re.sub(
        r"\s+\([^)]*\bprospect\b[^)]*\)\s*$",
        "",
        player,
        flags=re.I,
    )

    # Remove occasional dangling bullets or "More" artifacts.
    player = re.sub(r"\s*More\s*>>\s*$", "", player, flags=re.I)

    return clean_text(player)


def parse_injury_block(
    team: TeamInfo,
    source_url: str,
    block: str,
    fetched_at_utc: str,
) -> Optional[InjuryItem]:
    block = strip_more_link_text(block)

    match = INJURY_BLOCK_RE.match(block)

    if not match:
        logging.debug("Could not parse block for %s: %s", team.team, block[:300])
        return None

    header = clean_text(match.group("header"))
    injury = clean_text(match.group("injury"))
    il_date = clean_text(match.group("il_date"))
    expected_return = clean_text(match.group("expected_return"))
    status = clean_text(match.group("status"))

    updated = None
    updated_match = UPDATED_RE.search(status)

    if updated_match:
        updated = clean_text(updated_match.group("updated"))
        status = clean_text(UPDATED_RE.sub("", status))

    position = None
    player = header

    header_match = PLAYER_HEADER_RE.match(header)
    if header_match:
        position = header_match.group("position").upper()
        player = header_match.group("player")

    player = normalize_player_name(player)

    if not player:
        logging.debug(
            "Skipping empty player parse for %s block: %s", team.team, block[:300]
        )
        return None

    return InjuryItem(
        team=team.team,
        team_abbr=team.abbr,
        team_slug=team.slug,
        player=player,
        position=position,
        injury=injury or None,
        il_date=il_date or None,
        expected_return=expected_return or None,
        status=status or None,
        updated=updated,
        source_url=source_url,
        fetched_at_utc=fetched_at_utc,
    )


def parse_team_page(
    team: TeamInfo,
    source_url: str,
    soup: BeautifulSoup,
    fetched_at_utc: str,
) -> list[InjuryItem]:
    blocks = extract_injury_blocks(soup)

    if not blocks:
        page_title = get_page_title(soup)
        headings = [
            clean_text(h.get_text(" ", strip=True))
            for h in soup.find_all(["h1", "h2", "h3", "h4"])
        ]
        logging.warning(
            "No injury blocks found for %s. title=%r headings=%r url=%s",
            team.team,
            page_title,
            headings[:20],
            source_url,
        )
        return []

    injuries: list[InjuryItem] = []

    for block in blocks:
        item = parse_injury_block(
            team=team,
            source_url=source_url,
            block=block,
            fetched_at_utc=fetched_at_utc,
        )
        if item:
            injuries.append(item)

    return injuries


def select_teams(team_filters: Optional[list[str]]) -> list[TeamInfo]:
    if not team_filters:
        return TEAMS

    normalized_filters = {
        value.strip().lower().replace("_", "-")
        for raw in team_filters
        for value in raw.split(",")
        if value.strip()
    }

    selected: list[TeamInfo] = []

    for team in TEAMS:
        aliases = {
            team.team.lower(),
            team.abbr.lower(),
            team.slug.lower(),
            team.team.lower().replace(" ", "-"),
        }

        if aliases & normalized_filters:
            selected.append(team)

    missing = normalized_filters - {
        alias
        for team in TEAMS
        for alias in {
            team.team.lower(),
            team.abbr.lower(),
            team.slug.lower(),
            team.team.lower().replace(" ", "-"),
        }
    }

    if missing:
        logging.warning("Unknown team filter(s): %s", ", ".join(sorted(missing)))

    return selected


def collect_mlb_injuries(
    teams: Iterable[TeamInfo],
    delay_seconds: float,
    user_agent: str,
    *,
    require_complete: bool = False,
) -> list[InjuryItem]:
    teams = list(teams)
    session = make_session(user_agent=user_agent)
    fetched_at_utc = datetime.now(timezone.utc).isoformat()

    all_injuries: list[InjuryItem] = []
    failed_teams: list[str] = []

    for idx, team in enumerate(teams):
        url = injury_url_for_team(team)

        try:
            soup = fetch_soup(session, url)
            injuries = parse_team_page(
                team=team,
                source_url=url,
                soup=soup,
                fetched_at_utc=fetched_at_utc,
            )
            logging.info("%s: parsed %d injuries", team.team, len(injuries))
            if not injuries:
                failed_teams.append(team.abbr)
            all_injuries.extend(injuries)

        except requests.HTTPError as exc:
            logging.warning("%s: HTTP error for %s: %s", team.team, url, exc)
            failed_teams.append(team.abbr)

        except requests.RequestException as exc:
            logging.warning("%s: request failed for %s: %s", team.team, url, exc)
            failed_teams.append(team.abbr)

        except RuntimeError as exc:
            logging.warning("%s: %s", team.team, exc)
            failed_teams.append(team.abbr)

        if delay_seconds > 0 and idx < len(teams) - 1:
            time.sleep(delay_seconds)

    if require_complete and failed_teams:
        raise RuntimeError(
            "Incomplete MLB injury collection; failed teams: "
            + ", ".join(sorted(set(failed_teams)))
        )
    return all_injuries


def write_json(data: list[dict], output_path: Optional[Path], pretty: bool) -> None:
    json_text = json.dumps(
        data,
        ensure_ascii=False,
        indent=2 if pretty else None,
        separators=None if pretty else (",", ":"),
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json_text + "\n", encoding="utf-8")
        logging.info("Wrote %d records to %s", len(data), output_path)
    else:
        print(json_text)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch MLB injury report data as JSON."
    )

    parser.add_argument(
        "--team",
        action="append",
        help="Team filter. Accepts name, slug, or abbreviation. Can be repeated or comma-separated.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output JSON path.",
    )

    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON.",
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=0.25,
        help="Delay between team page requests in seconds. Default: 0.25",
    )

    parser.add_argument(
        "--user-agent",
        default=DEFAULT_USER_AGENT,
        help="User-Agent header to send with requests.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=DB_PATH,
        help="Unified SQLite database path. Defaults to data/unified.db.",
    )
    parser.add_argument(
        "--no-db",
        action="store_true",
        help="Do not persist the completed snapshot to SQLite.",
    )
    parser.add_argument(
        "--allow-filtered-db",
        action="store_true",
        help=(
            "Allow a --team filtered run to become the latest database snapshot. "
            "Use only for isolated testing databases."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    teams = select_teams(args.team)

    if not teams:
        logging.error("No valid teams selected.")
        return 2

    filtered_run = bool(args.team)
    persist_to_db = not args.no_db and (not filtered_run or args.allow_filtered_db)
    if filtered_run and not args.no_db and not args.allow_filtered_db:
        logging.warning(
            "Filtered team run will not update the shared injury database. "
            "Use --allow-filtered-db only with an isolated --db-path."
        )

    try:
        injuries = collect_mlb_injuries(
            teams=teams,
            delay_seconds=args.delay,
            user_agent=args.user_agent,
            require_complete=persist_to_db,
        )
    except RuntimeError as exc:
        logging.error("%s", exc)
        return 1

    payload = [asdict(item) for item in injuries]

    write_json(
        data=payload,
        output_path=args.output,
        pretty=args.pretty,
    )

    if not injuries:
        logging.warning("No injuries parsed.")
        return 1

    if persist_to_db:
        metadata = persist_mlb_injury_snapshot(
            payload,
            db_path=args.db_path,
            source_path=args.output,
            expected_team_count=len(teams),
        )
        logging.info(
            "Stored snapshot %s in %s (%d teams, %d records)",
            metadata["snapshot_id"],
            args.db_path,
            metadata["team_count"],
            metadata["record_count"],
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
