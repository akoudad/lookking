# Strategic Decisions — 2 Days to Final Presentation

**Course:** S8 Integrated Project — UIR
**Project:** Lookking (Open Domain — option F)
**Deadline:** 2 days
**Document purpose:** Lock scope. Decide what ships, what defers, how we address prof's feedback.

---

## 1. PROF'S FEEDBACK (verbatim points)

1. **Each agent must have a clear, distinct job** — not just exist while one agent does all the work.
2. **Reconsider what the BERT model does** — is it actually good for this task?
3. **Report must be pro-grade** — clearly state: what data, what we trained, what accuracy, methodology, limits.

We address each below.

---

## 2. HONEST REVIEW OF CURRENT STATE

### 2.1 Agents — current design

| Agent | Current job | Real value-add? |
|---|---|---|
| Orchestrator | Parses query (mode, city, category) | LLM-only, no tool, output = free text. **Light.** |
| Discovery | Calls `search_places` or `search_leads` tool | Tool call wrapper. **Could be a function.** |
| Scoring | Calls `score_match` per candidate, ranks top 3 | Wraps DL tool + does ranking. **Some value.** |

**Honest verdict:** Prof is right. One LLM could do all three: parse → call tool → call scoring tool → output ranked. Three "agents" right now feel decorative.

### 2.2 DL Model — current state

| Item | Value |
|---|---|
| Base | DistilBERT (`distilbert-base-uncased`) |
| Head | MLP 768 → 256 → 64 → 3 |
| Frozen | Embeddings + transformer layers 0-3 |
| Trained | Layers 4-5 + classifier head |
| Task | Query-candidate relevance classification |
| Classes | High / Medium / Low match |
| Data | **295 synthetic pairs** (we generated them in `generate_data.py`) |
| Split | 80/20 stratified, 236 train / 59 test |
| Result | **100% test accuracy** |

**Honest verdict:** 100% accuracy is suspicious. Reasons:
1. Data is **fully synthetic** — we wrote both queries and candidates with template variations.
2. Train/test split is from same template families — easy generalization.
3. Real test should be on **unseen query patterns**, not just held-out 20%.

**Is BERT right for this?** YES — query-candidate relevance is a textbook NLP sentence-pair classification task (same family as MNLI, STS-B, MS MARCO). BERT/DistilBERT is the standard tool. But our DATA is the weak link, not the model choice.

### 2.3 Pipeline overall

Working end-to-end: Telegram → CrewAI agents → Nominatim/CSV → DL scorer → Top 3 with Google Maps links + HITL approve/refine. LLM swap-blocked by free-tier quotas (current pain point).

---

## 3. AGENT REDESIGN — make each role distinct and defensible

Replace current 3 agents with **3 agents that each make a REAL decision** with structured output.

### Agent 1: **Intent Classifier**
- **Input:** Raw natural language query.
- **Decisions it makes:**
  - Classifies *mode* (places vs leads vs out-of-scope).
  - Extracts entities: city, category, niche (luxury/budget), constraints (distance, open-now).
  - Assigns *urgency* score (e.g. "near me right now" = high).
- **Output:** Structured JSON the next agent depends on.
- **Why an agent:** uses LLM reasoning to interpret casual/messy language; rule-based parsing would miss synonyms ("traditional Moroccan" = moroccan category).

### Agent 2: **Retrieval Strategist**
- **Input:** Intent JSON from Agent 1.
- **Decisions it makes:**
  - Choose data source: Nominatim (live OSM) if specific category + city / CSV fallback otherwise.
  - Query expansion: e.g. "barber" → ["barber", "hairdresser", "salon"].
  - Filter constraints: distance, open-now, niche.
  - Deduplicate near-identical results from OSM.
  - Returns top-N candidates with raw metadata.
- **Output:** Structured candidate list.
- **Why an agent:** retrieval strategy depends on intent; LLM picks the right tool for the situation.

### Agent 3: **Ranking & Explanation**
- **Input:** Candidate list from Agent 2 + original query.
- **Decisions it makes:**
  - Calls the DL scorer (DistilBERT) for each candidate → high/med/low.
  - Re-ranks within the same DL bucket using LLM reasoning over secondary signals (rating, distance, niche fit).
  - Writes **personalized explanation** per result ("this matches because you wanted *luxury* and it's a 4.8-star spa").
- **Output:** Top 3 ranked results with DL confidence + human explanation + map links.
- **Why an agent:** DL gives numeric match; LLM turns it into a reason a user understands. **Hybrid DL+LLM is the whole point of this agent.**

### Why this is better than current

| Before | After |
|---|---|
| Orchestrator: parses, output text | Intent: outputs structured JSON, schema-validated |
| Discovery: calls one tool | Retrieval: picks strategy, expands query, dedupes |
| Scoring: ranks | Ranking: DL scoring + LLM-written natural explanation |
| Each agent could be a function | Each agent makes a real decision the next one depends on |

This addresses prof feedback point 1 directly.

---

## 4. DL MODEL — keep DistilBERT, fix presentation

### 4.1 Keep DistilBERT — defensible

- Task = sentence-pair classification → BERT family is the textbook choice.
- DistilBERT specifically: 40% smaller than BERT, 60% faster, 97% of BERT performance.
- Runs on Mac (M-series MPS) without cloud GPU.
- Frozen-base + fine-tuned-head is standard transfer learning.

### 4.2 Reframe the task crisply for the report

> **Task:** Given a free-text user query and a candidate place description, predict whether the candidate is a **Low / Medium / High** match for that user's intent.
> **Model:** Fine-tuned DistilBERT encoder + 3-layer MLP head.
> **Why this task is real:** before this model, ranking was "highest rating wins" — ignoring whether the candidate is even what the user asked for. The model learns *semantic relevance*, not just metadata sorting.

### 4.3 Honest evaluation — add a HARD test

Currently: 80/20 split on same templates → 100% accuracy (optimistic).

**Add for the report:**
- A **held-out set of 30-50 hand-written queries** in a new style (not from our templates) → realistic accuracy.
- Per-class metrics (precision, recall, F1) — already computed by `classification_report`, just need to save and print.
- Confusion matrix image — already saved at `model/confusion_matrix.png`.
- **Loss & accuracy curves over epochs** — modify train.py to save these.
- **Honest discussion** in report: "100% on synthetic split likely overfits template patterns; real-world accuracy estimated at X%."

### 4.4 What stays out of scope

- Re-training with 10K+ pairs from real users → too late, document as future work.
- Multilingual model → defer.

---

## 5. WHAT SHIPS (2-day MVP)

### MUST HAVE (deliverables for the defense)

| # | Deliverable | Effort | Owner |
|---|---|---|---|
| 1 | **Refactor agents** to the 3 distinct roles above | 2h | code |
| 2 | **Add a held-out test set** (~30 new queries) + retrain or just evaluate | 1.5h | code + data |
| 3 | **Save training curves** (loss/accuracy plots) | 30min | code |
| 4 | **Architecture diagram** (PNG, clean) | 1h | design |
| 5 | **PDF report** (8-12 pages, pro structure) | 3-4h | writing |
| 6 | **Slides** (12-15 slides) | 1.5h | design |
| 7 | **Demo video 3-5 min** (screen recording) | 1h | recording |
| 8 | **GitHub repo public + README polished** | 30min | git |
| 9 | **LLM choice locked** (whichever free tier holds) | 5min | config |
| 10 | **Defense rehearsal** (Q&A practice) | 1h | prep |

Total: ~13 hours over 2 days. Achievable.

### NICE TO HAVE (if time allows)

- Logging dashboard (just markdown summary of `logs/agent_logs.json`).
- Screenshots gallery in README.
- One worked example walked through end-to-end in the report.

---

## 6. WHAT DEFERS — "Next Update" / Future Work section

These go in the report as **planned roadmap** so the prof sees we know what's missing:

1. **Real review data** — integrate Google Places API or Yelp API for genuine ratings and price levels.
2. **Multilingual support** — French + Arabic query understanding (currently English only).
3. **Real lead discovery** — LinkedIn / Apollo / Hunter.io integration for business prospect data.
4. **User accounts + persistence** — SQLite/Postgres to save approved searches per user.
5. **Web app version** — Chainlit or Streamlit interface in addition to Telegram.
6. **Caching layer** — Redis to memoize identical queries (cuts LLM calls + latency).
7. **A/B test the DL model** vs a naive rule-based scorer to quantify uplift.
8. **Active learning loop** — when user clicks "Done" → log as positive; "Add Info" → re-train.
9. **Larger labeled dataset** — recruit 20+ users for 10K real query-result-label triples.
10. **Production deployment** — Docker + cloud host (Fly.io, Render free tier).

---

## 7. CONCRETE 2-DAY PLAN

### Day 1 (today)

**Morning (3h)**
- [ ] Refactor agents into Intent Classifier / Retrieval Strategist / Ranking & Explanation
- [ ] Update agent task descriptions in `crew_setup.py`
- [ ] Verify Telegram pipeline still works end-to-end

**Afternoon (4h)**
- [ ] Write 30 hand-crafted unseen queries → save as `data/holdout_test.csv`
- [ ] Add evaluation script `model/evaluate_holdout.py` → outputs real-world accuracy
- [ ] Update `train.py` to save loss/accuracy curves PNG
- [ ] Re-run evaluation, capture numbers

**Evening (1h)**
- [ ] Draw architecture diagram (excalidraw.com, free, no signup) → export PNG
- [ ] Pick final LLM (whichever has quota left), lock in `crew_setup.py`

### Day 2 (tomorrow)

**Morning (4h)**
- [ ] Write PDF report (8-12 pages) — use template in section 8 below
- [ ] Convert MD → PDF via Pandoc

**Afternoon (3h)**
- [ ] Build slides (12-15 slides) — template in section 9 below
- [ ] Record demo video (QuickTime → screen recording, ~5 min script)
- [ ] Push everything to GitHub

**Evening (1h)**
- [ ] Defense rehearsal (Q&A list in section 10)
- [ ] Final sanity check

---

## 8. REPORT STRUCTURE (PDF, 8-12 pages)

1. **Cover page** — title, author, course, supervisor, date
2. **Abstract** (½ page) — what + why + result in 100 words
3. **Introduction** (1 page) — problem statement, motivation, contribution
4. **Related work** (½ page) — CrewAI, RAG, BERT for relevance classification
5. **System architecture** (1.5 pages) — diagram + walk-through of each component
6. **Multi-agent design** (1.5 pages) — the 3 agents, their decisions, why each is irreplaceable
7. **Deep learning model** (2 pages) — base model, head, training, data, metrics, confusion matrix, curves
8. **Data integration** (1 page) — Nominatim, CSV fallback, geographic parsing
9. **HITL design** (½ page) — state machine, mode picker, refinement
10. **Results** (1 page) — example queries, screenshots, latency, accuracy
11. **Limitations & future work** (½ page) — synthetic data, latency, defer list
12. **Conclusion** (¼ page)
13. **References** (¼ page)

---

## 9. SLIDE DECK (12-15 slides)

1. Title
2. Problem statement
3. Use case examples (visual)
4. System architecture (diagram)
5. Three agents — each with one slide showing decision + output schema
6. (continued — agent 2)
7. (continued — agent 3)
8. DL model — diagram of DistilBERT + head
9. DL model — training data + metrics + confusion matrix
10. Live OSM integration — map screenshot with pin
11. HITL flow — state machine diagram
12. Tech stack
13. Demo video embed
14. Limitations + future work
15. Q&A

---

## 10. EXPECTED QUESTIONS — defense prep

| Q | Strong answer |
|---|---|
| Why CrewAI, not just one LLM call? | Each agent makes a different *decision* with structured output. Schema-validated handoffs make pipeline debuggable and extensible. |
| Why BERT for this task? | Sentence-pair relevance classification is BERT's home territory (MNLI, STS-B). DistilBERT is the lightweight standard. |
| Why is your test acc 100%? Isn't that overfitting? | On the synthetic split — yes, it's too easy. On the held-out hand-written test set, accuracy drops to ~X%, which is the realistic number we report. |
| Why synthetic data? | Real labeled query-result pairs would require user studies (not feasible in 8 weeks). Synthetic data is a stand-in to prove the pipeline; future work is real data. |
| What if Nominatim is down? | CSV fallback in `search_tool.py` catches the exception, logs it, returns synthetic results. Demonstrated in code. |
| How does the human-in-the-loop work? | Mode picker (Place vs Leads) eliminates ambiguity up front. After results, user clicks Done (confirms) or Add Info (triggers refinement merge). |
| How would you scale to 100k users? | Cache LLM calls (same query → same result for N min). Move DL model to dedicated inference server. Replace CrewAI sequential with parallel where possible. |
| Why didn't you use Google Maps API? | Costs money + needs a billing card. Nominatim is the free open equivalent. Listed in future work. |

---

## 11. FINAL CHECKLIST (run through before defense)

- [ ] Code on GitHub, public, README runnable in <5min
- [ ] `requirements.txt` pinned
- [ ] `.env.example` with placeholder keys (real `.env` gitignored)
- [ ] Architecture diagram PNG in repo
- [ ] Confusion matrix PNG in repo
- [ ] Training curves PNG in repo
- [ ] Report PDF in repo
- [ ] Slides PDF in repo
- [ ] Demo video uploaded (YouTube unlisted or Drive link in README)
- [ ] Live bot running during defense
- [ ] Backup screenshots in case bot fails mid-demo

---

*End of strategic doc. Lock this. Stop scope creep. Ship.*
