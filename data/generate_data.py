"""
Generates mock data for Lookking:
  - places.csv       : searchable places database
  - leads.csv        : business leads database
  - training_data.csv: query-candidate pairs for DL training (3 classes)
"""
import random
import pandas as pd
from pathlib import Path

random.seed(42)
OUT = Path(__file__).parent


# ---------------------------------------------------------------------------
# 1. PLACES
# ---------------------------------------------------------------------------
PLACES_RAW = [
    # (name, category, subcategory, niche)
    ("Sushi Garden",       "restaurant", "sushi",     "casual"),
    ("Tokyo Nights",       "restaurant", "sushi",     "luxury"),
    ("Wasabi Express",     "restaurant", "sushi",     "budget"),
    ("Dar Zitoun",         "restaurant", "moroccan",  "luxury"),
    ("Cafe Atlas",         "restaurant", "moroccan",  "casual"),
    ("La Trattoria",       "restaurant", "italian",   "luxury"),
    ("Pizza Roma",         "restaurant", "italian",   "budget"),
    ("Burger Palace",      "restaurant", "burger",    "budget"),
    ("Le Grill",           "restaurant", "french",    "luxury"),
    ("Noodle House",       "restaurant", "asian",     "casual"),
    ("Cafe Central",       "cafe",       "coffee",    "casual"),
    ("Patisserie Royale",  "cafe",       "pastry",    "luxury"),
    ("Cafe Darna",         "cafe",       "coffee",    "budget"),
    ("Royal Spa",          "spa",        "massage",   "luxury"),
    ("Zen Retreat",        "spa",        "wellness",  "casual"),
    ("Hammam Palace",      "spa",        "hammam",    "luxury"),
    ("Urban Spa",          "spa",        "beauty",    "casual"),
    ("Budget Hammam",      "spa",        "hammam",    "budget"),
    ("Classic Cuts",       "barbershop", "haircut",   "casual"),
    ("The Barber Room",    "barbershop", "beard",     "casual"),
    ("Elite Barber",       "barbershop", "styling",   "luxury"),
    ("Quick Cut",          "barbershop", "haircut",   "budget"),
    ("FitLife Gym",        "gym",        "fitness",   "casual"),
    ("Iron Works",         "gym",        "weights",   "casual"),
    ("Yoga Studio",        "gym",        "yoga",      "luxury"),
    ("CrossFit Zone",      "gym",        "crossfit",  "casual"),
    ("Premium Fitness",    "gym",        "fitness",   "luxury"),
    ("Grand Palace Hotel", "hotel",      "5-star",    "luxury"),
    ("City Center Inn",    "hotel",      "3-star",    "casual"),
    ("Budget Stay",        "hotel",      "budget",    "budget"),
]

CITIES = ["Rabat", "Casablanca", "Marrakech", "Agadir", "Fes"]

places = []
for i, (name, cat, sub, niche) in enumerate(PLACES_RAW):
    for city in random.sample(CITIES, 2):
        rating = round(random.uniform(3.0, 5.0), 1)
        dist = round(random.uniform(0.2, 3.5), 1)
        is_open = random.random() > 0.2
        price = {"luxury": 3, "casual": 2, "budget": 1}[niche]
        places.append({
            "id": len(places) + 1,
            "name": f"{name} {city}",
            "category": cat,
            "subcategory": sub,
            "niche": niche,
            "city": city,
            "rating": rating,
            "distance_km": dist,
            "is_open": is_open,
            "price_level": price,
            "has_website": random.random() > 0.4,
            "has_instagram": random.random() > 0.3,
            "description": (
                f"{name} - {niche} {cat} in {city}, specializing in {sub}, "
                f"rated {rating}/5, {dist}km away, {'open' if is_open else 'closed'}"
            ),
        })

df_places = pd.DataFrame(places)
df_places.to_csv(OUT / "places.csv", index=False)
print(f"[+] places.csv: {len(df_places)} rows")


# ---------------------------------------------------------------------------
# 2. LEADS
# ---------------------------------------------------------------------------
LEAD_NAMES = {
    "restaurant": ["Al Fassia", "Le Relais", "Casa Grill", "Dar Tajine", "Bab Souika", "Riad Zitoun"],
    "cafe":       ["Cafe Art", "Coffee Story", "Terrasse", "Beans & Beyond", "Latte Land"],
    "gym":        ["ProFit", "Iron Body", "Pulse Gym", "FitZone", "MoveMore"],
    "hotel":      ["Riad Palace", "City Hotel", "Atlas Inn", "Kasbah Hotel", "Maison Bab"],
    "barbershop": ["The Cut", "Man Cave", "Blade & Blade", "Sharp Cuts", "Classic Trim"],
    "spa":        ["Serenity", "Hammam Royale", "Glow Spa", "Pure Bliss", "La Beaute"],
}

leads = []
for cat, names in LEAD_NAMES.items():
    for name in names:
        for city in ["Rabat", "Casablanca"]:
            niche = random.choice(["luxury", "casual", "budget"])
            rating = round(random.uniform(2.5, 5.0), 1)
            hw = random.random() > 0.5
            hi = random.random() > 0.4
            hv = random.random() > 0.7
            reviews = random.randint(10, 600)
            leads.append({
                "id": len(leads) + 1,
                "business_name": f"{name} {city}",
                "category": cat,
                "city": city,
                "niche": niche,
                "rating": rating,
                "has_website": hw,
                "has_instagram": hi,
                "has_video_content": hv,
                "review_count": reviews,
                "description": (
                    f"{name} - {niche} {cat} in {city}, rated {rating}/5, "
                    f"website: {'yes' if hw else 'no'}, "
                    f"instagram: {'yes' if hi else 'no'}, "
                    f"video content: {'yes' if hv else 'no'}, "
                    f"{reviews} reviews"
                ),
            })

df_leads = pd.DataFrame(leads)
df_leads.to_csv(OUT / "leads.csv", index=False)
print(f"[+] leads.csv: {len(df_leads)} rows")


# ---------------------------------------------------------------------------
# 3. TRAINING DATA (query, candidate, label, mode)
# ---------------------------------------------------------------------------
training = []


def add(query, candidate, label, mode):
    training.append({"query": query, "candidate": candidate, "label": label, "mode": mode})


# ---- PLACES HIGH (category + features match) ----
high_places = [
    ("sushi restaurant near me open now good rating",
     "Tokyo Nights - luxury sushi restaurant, 0.6km, open now, 4.7 stars"),
    ("moroccan restaurant luxury Rabat",
     "Dar Zitoun - luxury moroccan restaurant in Rabat, 4.8 stars, open"),
    ("cheap barbershop haircut open today",
     "Quick Cut - budget barbershop, affordable haircut, 0.4km, open"),
    ("luxury spa massage Casablanca",
     "Royal Spa - luxury spa, massage, Casablanca, 4.9 stars, open"),
    ("italian restaurant not expensive open",
     "Pizza Roma - budget italian restaurant, 0.8km, open, 4.1 stars"),
    ("gym yoga open morning Rabat",
     "Yoga Studio - luxury yoga gym, Rabat, open early morning, 4.6 stars"),
    ("cafe coffee cheap near me",
     "Cafe Darna - budget cafe, coffee, 0.3km, open now, 3.9 stars"),
    ("hotel luxury 5 star Marrakech",
     "Grand Palace Hotel - luxury 5-star hotel, Marrakech, 4.8 stars, open"),
    ("sushi restaurant max 1km Casablanca",
     "Sushi Garden - casual sushi restaurant, 0.9km, Casablanca, open, 4.3 stars"),
    ("french restaurant fine dining",
     "Le Grill - luxury french restaurant, 4.8 stars, open, city center"),
    ("gym fitness Casablanca open now",
     "FitLife Gym - casual gym, fitness, Casablanca, 1.2km, open now"),
    ("asian noodle restaurant budget",
     "Noodle House - casual asian restaurant, noodles, budget, 0.7km, open"),
    ("hammam spa luxury Rabat",
     "Hammam Palace - luxury spa, hammam, Rabat, 4.7 stars, open"),
    ("burger restaurant fast food near me",
     "Burger Palace - budget burger restaurant, 0.5km, open, 4.0 stars"),
    ("barber beard grooming luxury",
     "Elite Barber - luxury barbershop, beard grooming, 4.6 stars, open"),
    ("hotel budget city center open",
     "Budget Stay - budget hotel, city center, 3.5 stars, open"),
    ("coffee cafe pastry luxury morning",
     "Patisserie Royale - luxury cafe, pastry, coffee, open morning, 4.8 stars"),
    ("crossfit gym intense workout",
     "CrossFit Zone - casual gym, crossfit, 1.0km, open, 4.4 stars"),
    ("wellness spa relax near me open",
     "Zen Retreat - casual wellness spa, relaxation, 0.8km, open, 4.3 stars"),
    ("moroccan food casual lunch",
     "Cafe Atlas - casual moroccan restaurant, lunch, 0.6km, open, 4.1 stars"),
]
for q, c in high_places:
    add(q, c, "high", "places")
    # variations
    add(q + " good reviews", c + ", highly rated", "high", "places")
    add("find me " + q, "Result: " + c, "high", "places")
    add(q.replace("near me", "max 1km"), c, "high", "places")
    add(q, c + ", 5 minute walk", "high", "places")

# ---- PLACES MEDIUM (category matches, some features off) ----
medium_places = [
    ("sushi restaurant near me open now",
     "Noodle House - asian fusion restaurant, 1.8km, open, 4.0 stars (not sushi but asian)"),
    ("luxury spa massage",
     "Urban Spa - casual spa, beauty treatments, 2.5km, open, 3.8 stars"),
    ("cheap barbershop",
     "Classic Cuts - casual barbershop, standard prices, 1.5km, open"),
    ("italian restaurant luxury",
     "La Trattoria - luxury italian, 3km away, open but far, 4.5 stars"),
    ("gym yoga near me",
     "FitLife Gym - casual gym, has yoga class but not specialized, 2km, open"),
    ("moroccan restaurant cheap",
     "Dar Zitoun - luxury moroccan, 4.8 stars but expensive"),
    ("cafe coffee quick",
     "Patisserie Royale - luxury cafe, coffee but expensive and slow service"),
    ("hotel 5 star budget",
     "City Center Inn - 3-star hotel, decent quality but not 5-star"),
    ("sushi Rabat max 1km",
     "Tokyo Nights - sushi restaurant but 1.3km, slightly outside range"),
    ("french restaurant affordable",
     "Le Grill - french restaurant but luxury price, may be expensive for budget"),
    ("gym fitness open early",
     "CrossFit Zone - gym, fitness but opens at 7am, may not be early enough"),
    ("spa hammam budget",
     "Zen Retreat - wellness spa, has hammam but mid-range price"),
    ("burger fast food cheap",
     "Gourmet Burger - burger restaurant but slightly more expensive than budget"),
    ("coffee cafe near me",
     "Cafe Central - cafe, coffee, 2.2km away, a bit far"),
    ("barbershop haircut quick near me",
     "The Barber Room - barbershop, beard specialist, 1.8km, might need appointment"),
]
for q, c in medium_places:
    add(q, c, "medium", "places")
    add(q + " open today", c + ", check availability", "medium", "places")
    add("looking for " + q, c, "medium", "places")

# ---- PLACES LOW (completely wrong category) ----
low_places = [
    ("sushi restaurant near me", "Elite Barber - luxury barbershop, beard grooming, 0.4km"),
    ("luxury spa massage", "Burger Palace - budget burger restaurant, 0.5km, open"),
    ("gym yoga open morning", "Sushi Garden - casual sushi restaurant, 0.9km"),
    ("cheap barbershop", "Royal Spa - luxury spa, massage, 4.9 stars"),
    ("hotel luxury Marrakech", "Quick Cut - budget barbershop, haircut, 0.4km"),
    ("italian restaurant fine dining", "FitLife Gym - casual fitness gym, 1.2km"),
    ("coffee cafe morning", "Iron Works - weights gym, open early, 0.7km"),
    ("hammam spa", "La Trattoria - luxury italian restaurant, 3km"),
    ("french restaurant", "Yoga Studio - yoga gym, luxury, 4.6 stars"),
    ("burger fast food", "Hammam Palace - luxury hammam spa, 4.7 stars"),
    ("moroccan restaurant", "Budget Stay - budget hotel, city center"),
    ("gym crossfit", "Patisserie Royale - luxury cafe, pastry, 4.8 stars"),
    ("hotel city center", "Noodle House - casual asian restaurant, noodles"),
    ("cafe pastry", "CrossFit Zone - crossfit gym, intense workouts"),
    ("barbershop beard", "Grand Palace Hotel - luxury 5-star hotel, Marrakech"),
]
for q, c in low_places:
    add(q, c, "low", "places")
    add(q + " open now", c + ", completely different service", "low", "places")
    add("i need " + q, c, "low", "places")


# ---- LEADS HIGH ----
high_leads = [
    ("I offer video editing and reels for restaurants, target Rabat",
     "Al Fassia Rabat - luxury moroccan restaurant, Rabat, no video content, instagram yes, 4.7 stars, 245 reviews"),
    ("website creation for gyms in Casablanca no website",
     "ProFit Casablanca - casual gym, Casablanca, no website, instagram yes, 4.2 stars"),
    ("social media management for hotels no instagram Marrakech",
     "Riad Palace Casablanca - luxury hotel, no instagram, no video, website yes, 4.8 stars"),
    ("I do short form video ads for cafes",
     "Cafe Art Rabat - casual cafe, no video content, instagram yes, website no, 3.9 stars, 58 reviews"),
    ("photography for restaurants luxury niche no website",
     "Le Relais Casablanca - luxury restaurant, no website, no video, instagram yes, 4.5 stars"),
    ("I create reels for spas and wellness centers",
     "Serenity Casablanca - luxury spa, no video content, instagram no, website yes, 4.6 stars"),
    ("digital marketing for barbershops Casablanca",
     "Man Cave Casablanca - casual barbershop, no website, no instagram, 3.8 stars, 34 reviews"),
    ("brand content creation for gyms no social media",
     "Iron Body Rabat - casual gym, no instagram, no video, website no, 4.0 stars"),
    ("I build websites for restaurants that have no online presence",
     "Bab Souika Rabat - casual moroccan restaurant, no website, no instagram, 3.7 stars, 22 reviews"),
    ("video production for luxury hotels Rabat",
     "Atlas Inn Casablanca - luxury hotel, no video content, has instagram, no website, 4.4 stars"),
    ("I offer instagram content for spas with no social media",
     "Pure Bliss Casablanca - casual spa, no instagram, no video, website yes, 4.1 stars"),
    ("reels and short videos for barbershops that have no video",
     "Classic Trim Casablanca - budget barbershop, no video, instagram yes, no website, 3.6 stars"),
    ("web design for gyms Casablanca no website",
     "FitZone Casablanca - casual gym, no website, instagram no, 4.1 stars, 67 reviews"),
    ("content marketing for moroccan restaurants Rabat",
     "Dar Tajine Rabat - luxury moroccan restaurant, no video, instagram yes, website yes, 4.6 stars"),
    ("I do ads and video for cafes no social media presence",
     "Latte Land Casablanca - casual cafe, no instagram, no video, no website, 3.5 stars, 15 reviews"),
]
for q, c in high_leads:
    add(q, c, "high", "leads")
    add(q + " good potential client", c + ", clear opportunity", "high", "leads")
    add(q.replace("I offer", "looking for clients:"), c, "high", "leads")

# ---- LEADS MEDIUM ----
medium_leads = [
    ("I offer video editing for restaurants Rabat",
     "Casa Grill Casablanca - casual restaurant, has video content already, website yes, 4.3 stars"),
    ("website creation for gyms",
     "MoveMore Rabat - casual gym, has website, has instagram, 4.0 stars"),
    ("social media for hotels",
     "City Hotel Casablanca - casual hotel, has instagram, has website, some video, 3.8 stars"),
    ("I create reels for cafes",
     "Coffee Story Rabat - casual cafe, has some instagram posts, website yes, 3.9 stars"),
    ("photography for luxury restaurants",
     "Le Relais Rabat - luxury restaurant, has website, has instagram but low engagement, 4.4 stars"),
    ("digital marketing for barbershops",
     "Sharp Cuts Rabat - casual barbershop, has website but no social media, 3.7 stars"),
    ("video production for spas",
     "Glow Spa Rabat - casual spa, has website and instagram, limited video, 4.2 stars"),
    ("I build websites for restaurants",
     "Al Fassia Casablanca - luxury restaurant, already has website, instagram yes, 4.8 stars"),
    ("content creation for gyms",
     "Pulse Gym Casablanca - casual gym, has some content, partial social media, 4.0 stars"),
    ("reels for hotels",
     "Kasbah Hotel Casablanca - luxury hotel, has instagram but not active, website yes, 4.5 stars"),
]
for q, c in medium_leads:
    add(q, c, "medium", "leads")
    add(q + " looking for opportunities", c + ", some potential", "medium", "leads")
    add("potential client for " + q.split()[-1] + " services", c, "medium", "leads")

# ---- LEADS LOW ----
low_leads = [
    ("I offer video editing for restaurants",
     "FitZone Casablanca - casual gym, no website, no instagram, 4.1 stars"),
    ("website creation for gyms",
     "Hammam Royale Casablanca - luxury spa, hammam, no website, 4.7 stars"),
    ("social media for hotels",
     "ProFit Casablanca - casual gym, no instagram, no video, 4.2 stars"),
    ("reels for cafes no social media",
     "Man Cave Casablanca - casual barbershop, no website, no instagram, 3.8 stars"),
    ("photography for luxury restaurants",
     "Iron Body Rabat - casual gym, no instagram, no video, 4.0 stars"),
    ("digital marketing for barbershops",
     "Riad Palace Casablanca - luxury hotel, no instagram, 4.8 stars"),
    ("video production for spas",
     "Casa Grill Casablanca - casual restaurant, has video content, 4.3 stars"),
    ("I build websites for restaurants",
     "Serenity Casablanca - luxury spa, no video, instagram no, 4.6 stars"),
    ("content creation for gyms",
     "Blade Blade Casablanca - casual barbershop, no instagram, 3.9 stars"),
    ("reels for hotels luxury",
     "Beans Beyond Rabat - casual cafe, no video, no instagram, 3.8 stars"),
]
for q, c in low_leads:
    add(q, c, "low", "leads")
    add(q + " in any city", c + ", wrong business type", "low", "leads")
    add("looking for: " + q, c, "low", "leads")


df_train = pd.DataFrame(training)
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
df_train.to_csv(OUT / "training_data.csv", index=False)
print(f"[+] training_data.csv: {len(df_train)} rows")
print(f"    Label counts:\n{df_train['label'].value_counts().to_string()}")
