"""
Nominatim Tool: query OpenStreetMap for real places.
Free, no API key. Rate limit: 1 req/sec (we enforce 1.1s gap).
"""
import sys
import time
import random
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import log_action

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "Lookking/1.0 (S8 academic project; karim@uir.ac.ma)"
TIMEOUT = 10
DEFAULT_COUNTRY = "ma"  # Morocco

# Keyword → OSM category mapping
CATEGORY_KEYWORDS = {
    "restaurant": "restaurant",
    "sushi": "restaurant",
    "pizza": "restaurant",
    "moroccan": "restaurant",
    "traditional": "restaurant",
    "cafe": "cafe",
    "coffee": "cafe",
    "spa": "spa",
    "gym": "gym",
    "fitness": "gym",
    "barbershop": "hairdresser",
    "barber": "hairdresser",
    "hairdresser": "hairdresser",
    "hotel": "hotel",
    "pharmacy": "pharmacy",
    "bar": "bar",
    "club": "nightclub",
    "bakery": "bakery",
    "supermarket": "supermarket",
    "bank": "bank",
}

# Moroccan cities — used to detect city in query
KNOWN_CITIES = [
    "rabat", "casablanca", "marrakech", "marrakesh", "fes", "fez",
    "tangier", "agadir", "meknes", "oujda", "tetouan", "kenitra",
    "essaouira", "el jadida", "nador", "safi",
]

_last_call_time = 0.0


def _rate_limit():
    """Enforce 1.1s minimum gap between Nominatim calls."""
    global _last_call_time
    elapsed = time.time() - _last_call_time
    if elapsed < 1.1:
        time.sleep(1.1 - elapsed)
    _last_call_time = time.time()


def parse_query(query: str) -> tuple[str, str]:
    """Extract (category_term, city) from a free-text query."""
    q = query.lower()

    # Strip MODE prefix and refinement tag if present
    if q.startswith("[mode:"):
        q = q.split("] ", 1)[-1] if "] " in q else q
    if "refinement:" in q:
        # Use everything from refinement onward as the main signal
        parts = q.split("refinement:", 1)
        original = parts[0].strip().strip(",").strip()
        refined = parts[1].strip()
        q = f"{original} {refined}"

    # Detect city
    city = ""
    for c in KNOWN_CITIES:
        if c in q:
            city = c.capitalize()
            break

    # Detect category keyword
    category = ""
    for kw, osm in CATEGORY_KEYWORDS.items():
        if kw in q:
            category = osm
            break

    # If no category keyword, fall back to first significant word
    if not category:
        words = [w for w in q.split() if len(w) > 3 and w not in KNOWN_CITIES]
        if words:
            category = words[0]

    return category, city


def query_nominatim(search_term: str, city: str = "", limit: int = 8) -> list[dict]:
    """Call Nominatim. Returns list of result dicts. Empty list on error."""
    full_q = f"{search_term} {city}".strip()
    _rate_limit()
    try:
        r = requests.get(
            NOMINATIM_URL,
            params={
                "q": full_q,
                "format": "json",
                "limit": limit,
                "countrycodes": DEFAULT_COUNTRY,
                "addressdetails": 1,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
        log_action("NominatimTool", "query_ok", {"q": full_q}, f"{len(data)} results")
        return data
    except Exception as e:
        log_action("NominatimTool", "query_error", {"q": full_q}, str(e))
        return []


def format_results(results: list[dict]) -> str:
    """Format Nominatim results as text. Includes Google Maps + OSM links."""
    if not results:
        return ""

    lines = []
    for i, p in enumerate(results, 1):
        name = p.get("name") or p.get("display_name", "").split(",")[0].strip()
        addr = p.get("address", {}) or {}
        city = (
            addr.get("city") or addr.get("town") or addr.get("village")
            or addr.get("county") or "Unknown"
        )
        osm_type = p.get("type", "place")
        osm_class = p.get("class", "amenity")
        category = osm_type if osm_class == "amenity" else f"{osm_class}/{osm_type}"

        # Coordinates
        lat = p.get("lat", "")
        lon = p.get("lon", "")

        # Importance from Nominatim (0-1) → synthetic rating (3.5–5.0)
        importance = float(p.get("importance", 0.5) or 0.5)
        rating = round(3.5 + min(importance, 1.0) * 1.5, 1)
        price = "$" * (1 + int(importance * 3))

        # Links — Telegram auto-detects URLs
        gmaps_link = f"https://www.google.com/maps?q={lat},{lon}"
        osm_link = f"https://www.openstreetmap.org/?mlat={lat}&mlon={lon}&zoom=18"

        lines.append(
            f"• {name} | {category} | {city} | "
            f"Rating: {rating}/5 (synthetic) | "
            f"Price: {price} | Source: OpenStreetMap (live)\n"
            f"  📍 Map: {gmaps_link}\n"
            f"  🗺️ OSM: {osm_link}"
        )
    return "\n".join(lines)


def nominatim_search(query: str, limit: int = 8) -> str:
    """High-level: parse query, hit Nominatim, format. Returns text or '' on empty."""
    category, city = parse_query(query)
    if not category:
        return ""

    results = query_nominatim(category, city, limit=limit)
    return format_results(results)
