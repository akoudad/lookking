"""
Build training data V2 — uses REAL places from places_real.csv as candidates.
Generates (query, candidate, label) triples with rule-based labels:
  HIGH   = same category + same city
  MEDIUM = same category, different city OR partial niche/feature match
  LOW    = different category (clear mismatch)

Output: data/training_data_v2.csv
"""
import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

random.seed(42)

DATA_DIR = Path(__file__).parent
PLACES_REAL_CSV = DATA_DIR / "places_real.csv"
OUT_CSV = DATA_DIR / "training_data_v2.csv"

# City normalisation — many OSM names include arabic + tifinagh
CITY_NORMALIZE = {
    "fès": "Fes", "fes": "Fes", "fas": "Fes",
    "casablanca": "Casablanca",
    "rabat": "Rabat",
    "marrakech": "Marrakech", "marrakesh": "Marrakech",
    "agadir": "Agadir",
    "tanger": "Tangier", "tangier": "Tangier",
    "meknès": "Meknes", "meknes": "Meknes",
    "oujda": "Oujda",
    "tétouan": "Tetouan", "tetouan": "Tetouan",
}


def normalize_city(name: str) -> str:
    """Strip arabic/tifinagh + multilingual junk; map to canonical name."""
    if not isinstance(name, str):
        return ""
    # keep only ascii letters + spaces
    ascii_part = re.sub(r"[^A-Za-zÀ-ÿ\s]", " ", name).strip()
    first_word = ascii_part.split()[0] if ascii_part else ""
    canonical = CITY_NORMALIZE.get(first_word.lower(), first_word.capitalize() or "Unknown")
    return canonical


# Query templates — natural-sounding things a user would type
TEMPLATES_PLACES = {
    "restaurant": [
        "good restaurant in {city}",
        "restaurant {niche} {city}",
        "where to eat in {city}",
        "find me a restaurant {city} open now",
        "{cuisine} restaurant in {city}",
        "best restaurant {city} good reviews",
        "i want to eat in {city}",
    ],
    "cafe": [
        "coffee shop in {city}",
        "cafe {city} morning",
        "cafe near me {city}",
        "cozy cafe in {city}",
        "{niche} cafe {city}",
        "place for coffee in {city}",
    ],
    "bar": [
        "bar in {city}",
        "good bar {city} nightlife",
        "place for drinks in {city}",
        "{niche} bar {city}",
    ],
    "fast_food": [
        "fast food in {city}",
        "quick bite {city}",
        "fast food near me {city}",
        "burger place {city}",
    ],
    "bakery": [
        "bakery in {city}",
        "fresh bread {city}",
        "bakery {city} morning",
        "{niche} bakery {city}",
    ],
    "hairdresser": [
        "barber in {city}",
        "haircut {city}",
        "hairdresser {city}",
        "{niche} barber in {city}",
    ],
    "pharmacy": [
        "pharmacy {city}",
        "pharmacy open now {city}",
        "where can i find a pharmacy in {city}",
    ],
    "spa": [
        "spa in {city}",
        "{niche} spa {city}",
        "hammam {city}",
        "relax wellness {city}",
        "massage {city}",
    ],
    "gym": [
        "gym {city}",
        "{niche} gym {city}",
        "fitness center {city}",
        "where to work out in {city}",
        "training club {city}",
    ],
    "hotel": [
        "hotel {city}",
        "{niche} hotel {city}",
        "place to stay {city}",
        "where to sleep in {city}",
        "{niche} accommodation {city}",
    ],
    "supermarket": [
        "supermarket {city}",
        "grocery {city}",
        "where to buy food in {city}",
    ],
}

# Niche modifiers for queries
NICHES = ["luxury", "cheap", "casual", "good", "budget", ""]


def fmt_query(template: str, city: str, niche: str = "", cuisine: str = "") -> str:
    """Fill template; trim awkward double-spaces or empty modifiers."""
    q = template.format(city=city, niche=niche or "", cuisine=cuisine or "")
    q = re.sub(r"\s+", " ", q).strip()
    return q


def candidate_text(row: pd.Series) -> str:
    """Build the candidate description that the model sees as input."""
    name = row["name"]
    cat  = row["category"]
    city = row["city"]
    niche = row["niche"]
    rating = row.get("rating", 4.0)
    cuisine = row.get("cuisine", "") or ""
    has_web = row.get("has_website", False)
    has_phone = row.get("has_phone", False)

    bits = [f"{name} - {niche} {cat} in {city}"]
    if cuisine:
        bits.append(f"cuisine: {cuisine}")
    bits.append(f"rated {rating}/5")
    if has_web:
        bits.append("website available")
    if has_phone:
        bits.append("phone available")
    return ", ".join(bits)


def main():
    print("Loading real places...")
    df = pd.read_csv(PLACES_REAL_CSV)

    # Normalise city names
    df["city"] = df["city"].apply(normalize_city)
    print(f"  {len(df)} places loaded across {df['city'].nunique()} cities, "
          f"{df['category'].nunique()} categories")

    by_cat: dict[str, list] = defaultdict(list)
    by_cat_city: dict[tuple, list] = defaultdict(list)
    for _, row in df.iterrows():
        by_cat[row["category"]].append(row)
        by_cat_city[(row["category"], row["city"])].append(row)

    all_categories = list(by_cat.keys())
    all_cities = sorted(df["city"].unique())

    training: list[dict] = []

    # ---- HIGH MATCHES ----
    # Query asks for category X in city Y → candidate is real X in Y
    print("\nGenerating HIGH matches...")
    for cat, places in by_cat.items():
        if cat not in TEMPLATES_PLACES:
            continue
        for place in places:
            city = place["city"]
            if city == "Unknown":
                continue
            niche = place["niche"] if place["niche"] in ("luxury", "budget", "casual") else ""
            cuisine = place.get("cuisine", "")
            templates = TEMPLATES_PLACES[cat]
            # 2 templates per place → diversity
            for tpl in random.sample(templates, min(2, len(templates))):
                # choose niche modifier matching place niche occasionally
                use_niche = random.choice([niche, "", random.choice(NICHES)])
                query = fmt_query(tpl, city, use_niche, cuisine)
                training.append({
                    "query": query,
                    "candidate": candidate_text(place),
                    "label": "high",
                    "category": cat,
                    "city": city,
                })

    # ---- MEDIUM MATCHES ----
    # Same category but different city, OR niche mismatch
    print("Generating MEDIUM matches...")
    for cat, places in by_cat.items():
        if cat not in TEMPLATES_PLACES or len(places) < 3:
            continue
        cities_in_cat = list({p["city"] for p in places if p["city"] != "Unknown"})
        if len(cities_in_cat) < 2:
            continue
        templates = TEMPLATES_PLACES[cat]
        # pick 30 medium examples per category
        for _ in range(min(30, len(places))):
            query_city = random.choice(cities_in_cat)
            # candidate from a DIFFERENT city, same category
            other_cities = [c for c in cities_in_cat if c != query_city]
            if not other_cities:
                continue
            cand_city = random.choice(other_cities)
            cands_in_other = by_cat_city.get((cat, cand_city), [])
            if not cands_in_other:
                continue
            place = random.choice(cands_in_other)
            tpl = random.choice(templates)
            use_niche = random.choice(NICHES)
            query = fmt_query(tpl, query_city, use_niche)
            training.append({
                "query": query,
                "candidate": candidate_text(place),
                "label": "medium",
                "category": cat,
                "city": query_city,
            })

    # ---- LOW MATCHES ----
    # Different category entirely
    print("Generating LOW matches...")
    for cat, places in by_cat.items():
        if cat not in TEMPLATES_PLACES:
            continue
        templates = TEMPLATES_PLACES[cat]
        other_cats = [c for c in all_categories if c != cat]
        cities_in_cat = list({p["city"] for p in places if p["city"] != "Unknown"})
        if not cities_in_cat:
            continue
        for _ in range(min(40, len(places))):
            query_city = random.choice(cities_in_cat)
            tpl = random.choice(templates)
            use_niche = random.choice(NICHES)
            query = fmt_query(tpl, query_city, use_niche)
            # candidate from a wrong category
            wrong_cat = random.choice(other_cats)
            wrong_places = by_cat.get(wrong_cat, [])
            if not wrong_places:
                continue
            place = random.choice(wrong_places)
            training.append({
                "query": query,
                "candidate": candidate_text(place),
                "label": "low",
                "category": cat,
                "city": query_city,
            })

    random.shuffle(training)

    out_df = pd.DataFrame(training)
    out_df.to_csv(OUT_CSV, index=False)
    print(f"\n[+] Saved → {OUT_CSV}")
    print(f"    Total: {len(out_df)} pairs")
    print(f"    Label distribution:")
    for lbl, count in Counter(out_df["label"]).most_common():
        print(f"      {lbl:<8} {count}")
    print(f"    Categories covered: {out_df['category'].nunique()}")
    print(f"    Cities covered:    {out_df['city'].nunique()}")


if __name__ == "__main__":
    main()
