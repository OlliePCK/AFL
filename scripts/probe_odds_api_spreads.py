"""Probe: does The Odds API return AFL spreads for the Australian region?

Answers:
  1. Is the spreads market available for aussierules_afl / region=au?
  2. If yes, which bookmakers publish lines?
  3. What's the quota cost of fetching both h2h + spreads?
  4. What does the raw JSON look like -- how do we parse the line value?

Doesn't save anything. One-shot read.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Manually parse .env (this is a probe script, not worth importing dotenv)
env_path = PROJECT_ROOT / ".env"
if env_path.exists():
    for line in env_path.read_text().splitlines():
        if "=" in line and not line.strip().startswith("#"):
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())

api_key = os.environ.get("ODDS_API_KEY")
if not api_key:
    print("FATAL: ODDS_API_KEY not set")
    sys.exit(1)

url = "https://api.the-odds-api.com/v4/sports/aussierules_afl/odds"

# Request BOTH h2h and spreads so we can see which books offer what
print("=" * 80)
print("Probing: GET /v4/sports/aussierules_afl/odds?markets=h2h,spreads&regions=au")
print("=" * 80)

resp = requests.get(url, params={
    "apiKey": api_key,
    "regions": "au",
    "markets": "h2h,spreads",
    "oddsFormat": "decimal",
})
print(f"\nHTTP {resp.status_code}")
print(f"Quota used:      {resp.headers.get('x-requests-used')}")
print(f"Quota remaining: {resp.headers.get('x-requests-remaining')}")
print(f"Last cost:       {resp.headers.get('x-requests-last')}")

if resp.status_code != 200:
    print(f"\nERROR body:\n{resp.text[:500]}")
    sys.exit(1)

data = resp.json()
print(f"\nGames returned: {len(data)}")

if not data:
    print("No games in window (off-season?).")
    sys.exit(0)

# Inspect the first game in depth
g = data[0]
print(f"\n--- First match: {g.get('home_team')} vs {g.get('away_team')} "
      f"@ {g.get('commence_time')} ---")
bookmakers = g.get("bookmakers", [])
print(f"Bookmakers: {len(bookmakers)}")

has_h2h_count = 0
has_spreads_count = 0
example_spread_by_book = {}

for bm in bookmakers:
    key = bm.get("key", "?")
    markets = {m["key"]: m for m in bm.get("markets", [])}
    has_h2h = "h2h" in markets
    has_spreads = "spreads" in markets
    has_h2h_count += int(has_h2h)
    has_spreads_count += int(has_spreads)

    line_str = ""
    if has_spreads:
        outcomes = markets["spreads"].get("outcomes", [])
        # points = line magnitude, price = decimal odds on that line
        parts = [f"{o.get('name')} {o.get('point'):+.1f}@{o.get('price'):.2f}"
                 for o in outcomes if "point" in o]
        line_str = "  |  " + " / ".join(parts) if parts else ""
        # Store a detailed example per book
        if key not in example_spread_by_book:
            example_spread_by_book[key] = outcomes

    mark_str = []
    if has_h2h:
        mark_str.append("h2h")
    if has_spreads:
        mark_str.append("spreads")
    print(f"  {key:>20s}  [{'+'.join(mark_str):<15s}]{line_str}")

print(f"\nBookmakers offering h2h:     {has_h2h_count}/{len(bookmakers)}")
print(f"Bookmakers offering spreads: {has_spreads_count}/{len(bookmakers)}")

# Show raw JSON for one spread market
if example_spread_by_book:
    book, outcomes = next(iter(example_spread_by_book.items()))
    print(f"\n--- Raw spread outcomes for '{book}' ---")
    print(json.dumps(outcomes, indent=2))

# Coverage across all games
print("\n--- Coverage across all games ---")
total_books, games_with_spreads = 0, 0
per_book_coverage: dict[str, int] = {}
for game in data:
    any_spread = False
    for bm in game.get("bookmakers", []):
        total_books += 1
        markets = {m["key"] for m in bm.get("markets", [])}
        if "spreads" in markets:
            any_spread = True
            per_book_coverage[bm["key"]] = per_book_coverage.get(bm["key"], 0) + 1
    if any_spread:
        games_with_spreads += 1

print(f"Games with at least one spread price: {games_with_spreads}/{len(data)}")
print(f"\nPer-bookmaker spread coverage (games with spreads):")
for k, n in sorted(per_book_coverage.items(), key=lambda x: -x[1]):
    print(f"  {k:>20s}  {n}/{len(data)}")
