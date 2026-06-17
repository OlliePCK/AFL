"""
Weather data for AFL matches.

Source: Visual Crossing API (free tier: 1000 requests/day).
Fetches historical weather by date + venue coordinates.
Caches results to CSV to avoid re-fetching.
"""
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path

from src.config import DATA_RAW, DATA_PROCESSED
from src.utils import setup_logging

log = setup_logging()

CACHE_PATH = DATA_PROCESSED / "weather_historical.csv"

# AFL venue coordinates (lat, lon) — covers all venues in the dataset.
# For roofed/indoor venues, weather still matters for travel/conditions context,
# but rain_mm is less relevant (flagged via ROOFED_VENUES).
VENUE_COORDINATES = {
    "M.C.G.":                   (-37.8200, 144.9834),
    "Docklands":                (-37.8165, 144.9475),
    "Marvel Stadium":           (-37.8165, 144.9475),  # Same as Docklands
    "Adelaide Oval":            (-34.9156, 138.5961),
    "Gabba":                    (-27.4858, 153.0381),
    "Carrara":                  (-28.0067, 153.3667),
    "Perth Stadium":            (-31.9512, 115.8891),
    "Optus Stadium":            (-31.9512, 115.8891),  # Same as Perth Stadium
    "S.C.G.":                   (-33.8917, 151.2247),
    "Subiaco":                  (-31.9444, 115.8308),
    "Sydney Showground":        (-33.8447, 151.0697),
    "Kardinia Park":            (-38.1581, 144.3547),
    "GMHBA Stadium":            (-38.1581, 144.3547),  # Same as Kardinia Park
    "Football Park":            (-34.8800, 138.4964),
    "Bellerive Oval":           (-42.8775, 147.3764),
    "York Park":                (-41.4281, 147.1383),
    "University of Tasmania Stadium": (-41.4281, 147.1383),  # Same as York Park
    "Manuka Oval":              (-35.3178, 149.1347),
    "UNSW Canberra Oval":       (-35.3178, 149.1347),  # Same as Manuka
    "Marrara Oval":             (-12.4083, 130.8736),
    "Stadium Australia":        (-33.8475, 151.0633),
    "Cazaly's Stadium":         (-16.9203, 145.7531),
    "Eureka Stadium":           (-37.5622, 143.8503),
    "Mars Stadium":             (-37.5622, 143.8503),  # Same as Eureka
    "Traeger Park":             (-23.7000, 133.8667),
    "Norwood Oval":             (-34.9219, 138.6328),
    "Wellington":               (-41.2728, 174.7842),
    "Adelaide Hills":           (-35.0275, 138.7100),
    "Adelaide Arena at Jiangwan Stadium": (31.2814, 121.5147),
    "Jiangwan Stadium":         (31.2814, 121.5147),
    "Barossa Park":             (-34.5667, 138.9500),
    "Blacktown":                (-33.7700, 150.9069),
    "Riverway Stadium":         (-19.2833, 146.7333),
    "Hands Oval":               (-37.8267, 140.7833),
}

# Venues with retractable/closed roofs — rain doesn't affect play
ROOFED_VENUES = {"Docklands", "Marvel Stadium"}


def _get_api_key() -> str | None:
    return os.environ.get("VISUAL_CROSSING_API_KEY")


def fetch_match_weather(lat: float, lon: float, date_str: str,
                        api_key: str) -> dict | None:
    """Fetch weather for a single location + date from Visual Crossing.

    Args:
        lat, lon: Venue coordinates
        date_str: "YYYY-MM-DD" format
        api_key: Visual Crossing API key

    Returns:
        dict with temp_c, rain_mm, wind_kmh, humidity, conditions
        or None on failure
    """
    import requests

    url = (
        f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
        f"/{lat},{lon}/{date_str}/{date_str}"
        f"?unitGroup=metric&include=days&key={api_key}&contentType=json"
    )

    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        day = data["days"][0]
        return {
            "temp_c": day.get("temp"),
            "temp_max_c": day.get("tempmax"),
            "temp_min_c": day.get("tempmin"),
            "rain_mm": day.get("precip", 0) or 0,
            "wind_kmh": day.get("windspeed", 0) or 0,
            "wind_gust_kmh": day.get("windgust", 0) or 0,
            "humidity": day.get("humidity"),
            "conditions": day.get("conditions", ""),
        }
    except Exception as e:
        log.warning(f"Weather fetch failed for {lat},{lon} on {date_str}: {e}")
        return None


def load_weather_cache() -> pd.DataFrame:
    """Load cached weather data."""
    if CACHE_PATH.exists():
        return pd.read_csv(CACHE_PATH)
    return pd.DataFrame()


def fetch_all_weather(df: pd.DataFrame, rate_limit_pause: float = 0.5) -> pd.DataFrame:
    """Fetch weather for all matches in the dataset, using cache.

    Fetches missing entries only. Respects API rate limits.
    Saves cache after every 50 fetches for resilience.

    Args:
        df: Master dataset with 'venue', 'date' columns
        rate_limit_pause: Seconds between API calls

    Returns:
        Weather DataFrame with one row per (venue, date)
    """
    api_key = _get_api_key()
    if not api_key:
        log.warning("No VISUAL_CROSSING_API_KEY set — cannot fetch weather")
        return load_weather_cache()

    # Load existing cache
    cache = load_weather_cache()
    cached_keys = set()
    if not cache.empty:
        cached_keys = set(zip(cache["venue"], cache["date_str"]))

    # Build list of (venue, date, lat, lon) to fetch
    to_fetch = []
    for _, row in df.iterrows():
        venue = row["venue"]
        if venue not in VENUE_COORDINATES:
            continue
        date_str = pd.to_datetime(row["date"]).strftime("%Y-%m-%d")
        if (venue, date_str) in cached_keys:
            continue
        lat, lon = VENUE_COORDINATES[venue]
        to_fetch.append((venue, date_str, lat, lon))

    # Deduplicate (same venue+date for multiple matches unlikely but possible)
    to_fetch = list(set(to_fetch))

    if not to_fetch:
        log.info(f"Weather cache is complete ({len(cached_keys)} entries)")
        return load_weather_cache()

    log.info(f"Fetching weather for {len(to_fetch)} venue-dates "
             f"(cached: {len(cached_keys)}, API limit: ~1000/day)")

    new_rows = []
    fetched = 0
    for venue, date_str, lat, lon in to_fetch:
        weather = fetch_match_weather(lat, lon, date_str, api_key)
        if weather:
            weather["venue"] = venue
            weather["date_str"] = date_str
            weather["lat"] = lat
            weather["lon"] = lon
            new_rows.append(weather)
            fetched += 1
        else:
            # Store a row with NaN so we don't re-fetch failures
            new_rows.append({"venue": venue, "date_str": date_str,
                             "lat": lat, "lon": lon})

        # Rate limiting
        time.sleep(rate_limit_pause)

        # Save checkpoint every 50 fetches
        if fetched % 50 == 0 and fetched > 0:
            checkpoint = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
            checkpoint.to_csv(CACHE_PATH, index=False)
            log.info(f"  Checkpoint: {fetched}/{len(to_fetch)} fetched")

    # Final save
    all_weather = pd.concat([cache, pd.DataFrame(new_rows)], ignore_index=True)
    all_weather.to_csv(CACHE_PATH, index=False)
    log.info(f"Weather fetch complete: {fetched} new, {len(all_weather)} total cached")

    return all_weather


def merge_weather(df: pd.DataFrame) -> pd.DataFrame:
    """Merge cached weather data into a match DataFrame.

    Adds: temp_c, rain_mm, wind_kmh, wind_gust_kmh, humidity, is_wet, wind_strong, is_roofed
    """
    weather = load_weather_cache()
    if weather.empty:
        log.info("No weather cache found — skipping weather merge")
        return df

    df = df.copy()
    df["_date_str"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    weather_cols = ["date_str", "venue", "temp_c", "rain_mm", "wind_kmh",
                    "wind_gust_kmh", "humidity"]
    available = [c for c in weather_cols if c in weather.columns]
    weather_merge = weather[available].drop_duplicates(subset=["venue", "date_str"])

    df = df.merge(
        weather_merge,
        left_on=["venue", "_date_str"],
        right_on=["venue", "date_str"],
        how="left",
    )
    df = df.drop(columns=["_date_str", "date_str"], errors="ignore")

    # Derived features
    df["is_roofed"] = df["venue"].isin(ROOFED_VENUES).astype(int)
    if "rain_mm" in df.columns:
        # For roofed venues, rain doesn't affect play
        effective_rain = df["rain_mm"].copy()
        effective_rain[df["is_roofed"] == 1] = 0
        df["is_wet"] = (effective_rain > 1.0).astype(int)
    if "wind_kmh" in df.columns:
        df["wind_strong"] = (df["wind_kmh"] > 25).astype(int)

    matched = df["temp_c"].notna().sum() if "temp_c" in df.columns else 0
    log.info(f"Weather merge: {matched}/{len(df)} matches with weather data")

    return df
