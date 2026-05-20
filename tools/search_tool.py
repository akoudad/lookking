"""
Search tool: hybrid search for places + leads.
- Places: Nominatim (OpenStreetMap, live) → CSV fallback if Nominatim empty.
- Leads:  CSV (synthetic — no OSM equivalent for lead generation).
Returns top candidates as formatted text.
"""
import sys
from pathlib import Path
import pandas as pd
from crewai.tools import tool

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.logger import log_action
from tools.nominatim_tool import nominatim_search

# Prefer real-data CSV (from OSM crawl); fall back to synthetic if not present
_PLACES_REAL = Path(__file__).parent.parent / "data" / "places_real.csv"
_PLACES_SYNTH = Path(__file__).parent.parent / "data" / "places.csv"
PLACES_CSV = _PLACES_REAL if _PLACES_REAL.exists() else _PLACES_SYNTH
LEADS_CSV  = Path(__file__).parent.parent / "data" / "leads.csv"


def _load(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def search_places_fn(query: str, city: str = "", category: str = "", limit: int = 8) -> str:
    df = _load(PLACES_CSV)

    if city:
        # Tolerant city match (real CSV has multi-script names)
        city_l = city.lower()
        mask = df["city"].astype(str).str.lower().str.contains(city_l, na=False)
        df = df[mask] if mask.any() else df

    if category:
        cat_l = category.lower()
        cat_mask = df["category"].astype(str).str.lower().str.contains(cat_l, na=False)
        if "subcategory" in df.columns:
            sub_mask = df["subcategory"].astype(str).str.lower().str.contains(cat_l, na=False)
            mask = cat_mask | sub_mask
        else:
            mask = cat_mask
        df = df[mask] if mask.any() else df

    # Keyword filter on description
    if "description" in df.columns:
        keywords = [w for w in query.lower().split() if len(w) > 3]
        if keywords:
            kw_mask = df["description"].astype(str).str.lower().apply(
                lambda d: any(kw in d for kw in keywords)
            )
            filtered = df[kw_mask]
            if len(filtered) >= 3:
                df = filtered

    if "rating" in df.columns:
        df = df.sort_values("rating", ascending=False)
    df = df.head(limit)

    if df.empty:
        return "No candidates found matching the criteria."

    results = []
    for _, row in df.iterrows():
        # Field-by-field with safe defaults — works for both schemas
        name     = row.get("name", "Unknown")
        cat      = row.get("category", "")
        subcat   = row.get("subcategory", "") or row.get("cuisine", "")
        city_v   = row.get("city", "Unknown")
        rating   = row.get("rating", 4.0)
        niche    = row.get("niche", "casual")
        price    = "$" * int(row.get("price_level", 2))

        line = f"• {name} | {cat}"
        if subcat:
            line += f" ({subcat})"
        line += f" | {city_v} | Rating: {rating}/5 | Niche: {niche} | Price: {price}"

        # Real-data extras: distance only if synthetic schema has it
        if "distance_km" in df.columns and pd.notna(row.get("distance_km")):
            line += f" | {row['distance_km']}km"
        if "is_open" in df.columns:
            line += f" | {'Open' if row.get('is_open') else 'Closed'}"

        # If lat/lon present (real CSV), add map links
        lat = row.get("lat", None)
        lon = row.get("lon", None)
        if pd.notna(lat) and pd.notna(lon):
            line += (
                f"\n  📍 Map: https://www.google.com/maps?q={lat},{lon}"
                f"\n  🗺️ OSM: https://www.openstreetmap.org/?mlat={lat}&mlon={lon}&zoom=18"
            )

        results.append(line)
    output = "\n".join(results)
    log_action("SearchTool", "search_places",
               {"query": query, "city": city, "category": category},
               output)
    return output


def search_leads_fn(query: str, city: str = "", category: str = "", limit: int = 8) -> str:
    df = _load(LEADS_CSV)

    if city:
        mask = df["city"].str.lower() == city.lower()
        df = df[mask] if mask.any() else df

    if category:
        mask = df["category"].str.lower().str.contains(category.lower())
        df = df[mask] if mask.any() else df

    df = df.sort_values("rating", ascending=False).head(limit)

    if df.empty:
        return "No leads found matching the criteria."

    results = []
    for _, row in df.iterrows():
        gaps = []
        if not row["has_website"]:
            gaps.append("no website")
        if not row["has_instagram"]:
            gaps.append("no instagram")
        if not row["has_video_content"]:
            gaps.append("no video")
        gap_str = ", ".join(gaps) if gaps else "has all channels"

        results.append(
            f"• {row['business_name']} | {row['category']} | "
            f"{row['city']} | Rating: {row['rating']}/5 | "
            f"Niche: {row['niche']} | Gaps: {gap_str} | "
            f"Reviews: {row['review_count']}"
        )
    output = "\n".join(results)
    log_action("SearchTool", "search_leads", {"query": query, "city": city, "category": category}, output)
    return output


@tool("Search Places")
def search_places(query: str) -> str:
    """
    Search for real places (restaurants, spas, gyms, barbershops, cafes, hotels).
    Primary source: OpenStreetMap (Nominatim, live data, free).
    Fallback: real-data CSV (places_real.csv from OSM crawl) if Nominatim empty.
    Input: natural language query.
    Returns: list of matching places with name, category, city, rating, price,
    Google Maps + OpenStreetMap links.
    """
    # 1. Try live OSM first (cap at 3 to keep token usage down)
    try:
        live = nominatim_search(query, limit=3)
        if live and live.strip():
            log_action("SearchPlaces", "source_osm", {"query": query}, live[:300])
            return live
    except Exception as e:
        log_action("SearchPlaces", "osm_error", {"query": query}, str(e))

    # 2. Fall back to real-data CSV (also capped at 3)
    log_action("SearchPlaces", "source_csv_fallback", {"query": query}, "using CSV")
    return search_places_fn(query, limit=3)


@tool("Search Leads")
def search_leads(query: str) -> str:
    """
    Search the business leads database for potential clients.
    Input: description of the service offered and target business type.
    Returns: list of business leads with contact gaps (no website, no instagram, etc.).
    """
    return search_leads_fn(query, limit=3)
