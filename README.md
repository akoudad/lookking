# Lookking

> Multi-agent AI assistant for finding **places** and **business leads** in Morocco via Telegram.
> Built for the S8 Integrated Project (AI & Big Data, UIR).

![Architecture](docs/architecture.png)

---

## What it is

Lookking is a Telegram chatbot that takes a casual natural-language query
("luxury spa Rabat", "gym in Casablanca", "I offer video editing for cafes")
and returns the **Top 3 best-matching places or leads**, each ranked by a
**fine-tuned DistilBERT classifier**, with a one-line human reason and a
clickable Google Maps + OpenStreetMap link so the user can verify the place
exists.

It is built on three pillars:

| Pillar | What it brings |
|---|---|
| **Multi-agent system** (CrewAI) | Each agent makes a distinct decision — Intent → Retrieval → Ranking. Outputs of one are the input of the next. |
| **Deep learning** (DistilBERT) | A fine-tuned 3-class match-relevance classifier (High / Medium / Low) trained on 1260 real Moroccan-place query–candidate pairs. Holdout accuracy: **80%**. |
| **Live data integration** (Nominatim) | All places are real OSM entries with verifiable coordinates. CSV fallback (361 real places we crawled) for resilience. |

---

## Quick start

```bash
git clone <repo-url> lookking
cd lookking

# Python env
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt

# Secrets
cp .env.example .env
# Edit .env: paste TELEGRAM_TOKEN + at least one LLM key
# (GROQ_API_KEY recommended — free, 70B model handles tool-use reliably)

# One-time: collect real places + build training data + train DL model
python3 data/collect_real_places.py
python3 data/build_training_v2.py
python3 model/train_v2.py

# Run the bot
python3 main.py
```

Then in Telegram, message your bot `/start` and pick a mode.

---

## Architecture

Three agents in a sequential CrewAI pipeline. Each owns a clear decision.

| # | Agent | Decision it makes | Output |
|---|---|---|---|
| 1 | **Intent Classifier** | parses casual user text into structured intent | JSON `{mode, city, category, niche, urgency, constraints}` |
| 2 | **Retrieval Strategist** | picks data source (live Nominatim vs CSV fallback), expands query, dedupes | 5 candidate places/leads with full metadata |
| 3 | **Ranking & Explanation** | calls DistilBERT scorer per candidate, ranks, writes human reason | Top 3 with match label, confidence, reason, map links |

See `docs/architecture.png` for the diagram.

---

## Deep learning model

A fine-tuned **DistilBERT** matcher.

```
[QUERY]: <user query>  [SEP]  [CANDIDATE]: <place description>
                       |
                       v
        DistilBERT (66M params, layers 0-3 frozen)
                       |
                       v   [CLS] vector
                MLP head (768 -> 256 -> 64 -> 3)
                       |
                       v   softmax
              {Low, Medium, High} match probabilities
```

**Training data:** 1260 query–candidate pairs derived from **361 real OSM places** crawled across 9 Moroccan cities and 11 categories.
- HIGH match = same category + same city
- MEDIUM match = same category, different city
- LOW match = different category entirely

**Honest metrics (real numbers, no cherry-picking):**

| Metric | Test split (1260 × 20%) | Holdout (30 hand-written unseen queries) |
|---|---|---|
| Accuracy | **88.89%** | **80.00%** |
| Macro F1 | 0.83 | 0.71 |

Plots: `model/training_curves.png`, `model/confusion_matrix_v2.png`,
`model/confusion_matrix_holdout.png`. Raw numbers: `model/metrics_v2.json`.

---

## Repository layout

```
lookking/
├── main.py                 # entry point
├── README.md               # this file
├── requirements.txt
├── .env.example
├── .gitignore
│
├── agents/
│   └── crew_setup.py       # the 3 agents + pipeline
├── bot/
│   └── telegram_bot.py     # Telegram interface, HITL state machine
├── tools/
│   ├── search_tool.py      # search_places + search_leads (CrewAI tools)
│   ├── nominatim_tool.py   # OpenStreetMap Nominatim wrapper
│   └── dl_scorer_tool.py   # DistilBERT scorer (CrewAI tool)
├── model/
│   ├── model.py            # LookkingMatcher (DistilBERT + MLP)
│   ├── train_v2.py         # training script (real data + holdout eval)
│   ├── training_curves.png
│   ├── confusion_matrix_v2.png
│   ├── confusion_matrix_holdout.png
│   └── metrics_v2.json
├── data/
│   ├── collect_real_places.py
│   ├── places_real.csv     # 361 real OSM places
│   ├── build_training_v2.py
│   ├── training_data_v2.csv
│   └── holdout_test.csv
├── scripts/
│   ├── smoke_test.py       # run pipeline on 3 sample queries
│   └── draw_architecture.py
├── utils/
│   └── logger.py           # JSON action logger
├── logs/                   # populated at runtime
└── docs/
    └── architecture.png
```

---

## Human-in-the-loop

Two HITL touch points keep the user in control:

1. **Mode selection** — `/start` → user clicks **Place** or **Leads** (no AI guessing).
2. **Refinement** — after results, user clicks **Done** (accept) or **Add Info** (refine). Refinement is merged with the prior query and re-run.

State machine per chat:

```
IDLE -> /start -> MODE_PICK -> AWAITING_QUERY -> RESULTS_SHOWN
                                                  |-> Done -> IDLE
                                                  |-> Add Info -> AWAITING_REFINEMENT
                                                                  |
                                                                  v
                                                            RESULTS_SHOWN
```

---

## LLM backends

Default = **Groq llama-3.3-70b-versatile** (verified working end-to-end with
~8k tokens per query, well below the free 12k TPM cap).

Override with env var:

```bash
LLM_PROVIDER=gemini   python3 main.py    # Gemini 2.0 Flash
LLM_PROVIDER=groq     python3 main.py    # Groq (default)
LLM_PROVIDER=cerebras python3 main.py    # Cerebras Llama-3.1-8B (last resort)
```

Routing logic lives in `agents/crew_setup.py :: _make_llm()`. Each agent has
`max_iter=3-5` to cap reasoning loops, and search tools return only 3
candidates each — both designed to keep total token usage per query well
below the strictest free-tier limit (Groq 12k TPM).

---

## Prof requirements coverage

| Requirement | Status | Evidence |
|---|---|---|
| Multi-agent system, each agent distinct role | ✅ | `agents/crew_setup.py` — 3 agents, each with unique structured output |
| Deep learning model trained from data | ✅ | `model/train_v2.py`, 1260-pair dataset, holdout-validated |
| Real-world problem | ✅ | Place + lead finder, real Moroccan cities, real OSM data |
| External API integration | ✅ | OpenStreetMap Nominatim, Google Maps link generation |
| Tool-using agents | ✅ | `search_places`, `search_leads`, `score_match` |
| Human-in-the-loop | ✅ | Mode picker + Add Info refinement |
| Logging / observability | ✅ | `logs/agent_logs.json` every action timestamped |
| Free / open stack | ✅ | All deps open-source, all APIs free tier |
| Error handling & fallback | ✅ | Nominatim → CSV fallback, missing-model startup check |
| Public GitHub repo | 🟡 | Code ready; push once branch protected |
| Architecture diagram | ✅ | `docs/architecture.png` |
| PDF report | 🟡 | Markdown ready, convert with Pandoc |
| Slides | 🟡 | Outline in `STRATEGIC_DECISIONS.md` |
| Demo video | 🟡 | To record |

---

## Limitations / future work

- **Ratings are synthetic** (computed from OSM "importance" score, not real reviews) — future: integrate Google Places / Yelp / Foursquare APIs.
- **Leads mode** still uses the synthetic CSV (no free public business-lead API exists).
- English-only — French / Arabic next.
- No persistence — approved searches aren't saved to disk yet.
- A 3B-parameter DL model could be distilled further or replaced by a smaller sentence-transformer for lower latency.
- LLM provider free tiers all have rate limits — production would need either a paid key or self-hosted inference.

---

## Author

Karim Akoudad — UIR, AI & Big Data, S8 Integrated Project (2026).
