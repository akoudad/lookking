"""
Collect REAL place data from OpenStreetMap (Nominatim).
Free, no API key, 1 req/sec rate limit (we enforce 1.2s).

Strategy:
  For each (city, category) pair → Nominatim search → 30-50 results per combo.
  Total: ~9 cities × 10 categories × ~30 results = ~2700 candidates,
  ~1500 after dedup.

Output: data/places_real.csv with name, category, city, lat, lon, address, niche, description.
"""
import csv
import sys
import time
from collections import Counter
from pathlib import Path

import requests

OUT_CSV = Path(__file__).parent / "places_real.csv"
USER_AGENT = "Lookking/1.0 (academic project; karim@uir.ac.ma)"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"

CITIES = [
    "Rabat", "Casablanca", "Marrakech", "Agadir", "Fes",
    "Tangier", "Meknes", "Oujda", "Tetouan",
]

# Search term → our normalized category
SEARCH_TERMS = {
    "restaurant":  "restaurant",
    "cafe":        "cafe",
    "bar":         "bar",
    "fast food":   "fast_food",
    "bakery":      "bakery",
    "hairdresser": "hairdresser",
    "barbershop":  "hairdresser",
    "pharmacy":    "pharmacy",
    "spa":         "spa",
    "hammam":      "spa",
    "gym":         "gym",
    "fitness":     "gym",
    "hotel":       "hotel",
    "supermarket": "supermarket",
}

RATE_LIMIT_S = 1.2
PER_QUERY_LIMIT = 30


def _sleep():
    time.sleep(RATE_LIMIT_S)


def nominatim_search(term: str, city: str, limit: int = 30) -> list[dict]:
    q = f"{term} {city}"
    try:
        r = requests.get(
            NOMINATIM_URL,
            params={
                "q": q,
                "format": "json",
                "limit": limit,
                "countrycodes": "ma",
                "addressdetails": 1,
                "extratags": 1,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=15,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"    ! Error: {e}")
        return []


def extract_record(p: dict, search_term: str, normalized_cat: str, city: str) -> dict | None:
    """Normalize Nominatim result → flat record. Skip generic admin/area results."""
    name = p.get("name") or p.get("display_name", "").split(",")[0].strip()
    if not name or len(name) < 2:
        return None

    # Skip if it's just an administrative area
    osm_class = p.get("class", "")
    osm_type  = p.get("type", "")
    if osm_class in ("boundary", "place") and osm_type in ("administrative", "city", "town", "suburb"):
        return None

    addr = p.get("address", {}) or {}
    actual_city = (
        addr.get("city") or addr.get("town") or addr.get("village")
        or addr.get("county") or city
    )

    lat = p.get("lat", "")
    lon = p.get("lon", "")
    importance = float(p.get("importance", 0.5) or 0.5)

    extratags = p.get("extratags", {}) or {}
    cuisine     = extratags.get("cuisine", "")
    hours       = extratags.get("opening_hours", "")
    website     = extratags.get("website") or extratags.get("contact:website", "")
    phone       = extratags.get("phone") or extratags.get("contact:phone", "")
    wheelchair  = extratags.get("wheelchair", "")

    # Heuristic niche
    name_lower = name.lower()
    niche = "casual"
    if any(w in name_lower for w in ["royal", "palace", "grand", "luxury", "premium", "elite", "riad"]):
        niche = "luxury"
    elif any(w in name_lower for w in ["budget", "cheap", "quick", "express", "fast"]):
        niche = "budget"

    # Synthetic rating + price from importance (no real ratings in OSM)
    rating = round(3.5 + min(importance, 1.0) * 1.5, 1)
    price_level = {"luxury": 3, "casual": 2, "budget": 1}[niche]

    # Description for DL training
    bits = [f"{name} - {niche} {normalized_cat} in {actual_city}"]
    if cuisine:
        bits.append(f"cuisine: {cuisine}")
    if hours:
        bits.append("has opening hours")
    if website:
        bits.append("has website")
    if phone:
        bits.append("has phone")
    if wheelchair == "yes":
        bits.append("wheelchair accessible")
    bits.append(f"rated {rating}/5")
    description = ", ".join(bits)

    return {
        "name":        name,
        "category":    normalized_cat,
        "subcategory": cuisine or search_term,
        "niche":       niche,
        "city":        actual_city,
        "lat":         lat,
        "lon":         lon,
        "rating":      rating,
        "price_level": price_level,
        "has_website": bool(website),
        "has_phone":   bool(phone),
        "opening_hours": hours,
        "cuisine":     cuisine,
        "address":     addr.get("road", ""),
        "osm_class":   osm_class,
        "osm_type":    osm_type,
        "importance":  importance,
        "description": description,
    }


def main():
    print(f"Collecting real places from OpenStreetMap Nominatim")
    print(f"Cities: {len(CITIES)}  ×  Search terms: {len(SEARCH_TERMS)}\n")

    all_records: list[dict] = []
    seen: set = set()

    for city in CITIES:
        print(f"--- {city} ---")
        for term, category in SEARCH_TERMS.items():
            print(f"  {term:<13}", end=" ", flush=True)
            results = nominatim_search(term, city, limit=PER_QUERY_LIMIT)
            kept = 0
            for p in results:
                rec = extract_record(p, term, category, city)
                if not rec:
                    continue
                # Dedup by (name+city+category)
                key = (rec["name"].lower(), rec["city"].lower(), rec["category"])
                if key in seen:
                    continue
                seen.add(key)
                all_records.append(rec)
                kept += 1
            print(f"→ {kept:3d} new places")
            _sleep()
        print()

    if not all_records:
        print("! No data collected. Check internet.")
        sys.exit(1)

    print(f"[+] Total unique places: {len(all_records)}\n")

    # Write CSV
    fields = list(all_records[0].keys())
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in all_records:
            writer.writerow(row)

    print(f"[+] Saved → {OUT_CSV}")

    # Stats
    by_cat = Counter(r["category"] for r in all_records)
    by_city = Counter(r["city"] for r in all_records)
    by_niche = Counter(r["niche"] for r in all_records)
    print("\nBy category:")
    for k, v in by_cat.most_common():
        print(f"  {k:<13} {v}")
    print("\nBy city (top 10):")
    for k, v in by_city.most_common(10):
        print(f"  {k:<13} {v}")
    print("\nBy niche:")
    for k, v in by_niche.most_common():
        print(f"  {k:<13} {v}")


if __name__ == "__main__":
    main()
