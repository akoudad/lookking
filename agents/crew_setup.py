"""
Lookking — Multi-Agent Crew (V2).

3 agents, each making a distinct decision the next depends on:
  1. Intent Classifier      → structured JSON intent
  2. Retrieval Strategist   → 3 candidates from Nominatim or CSV
  3. Ranking & Explanation  → top 3 with DL match + LLM reason

Prompts are short on purpose: every token costs free-tier budget.
Provider fallback: groq → gemini → cerebras (auto on rate limit).
"""
import os
import sys
from pathlib import Path

from crewai import Agent, Task, Crew, Process, LLM
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from tools.search_tool import search_places, search_leads
from tools.dl_scorer_tool import score_match

load_dotenv(Path(__file__).parent.parent / ".env")

_RATE_KEYWORDS = ("rate", "limit", "quota", "429", "503", "overloaded", "capacity")


def _make_llm(provider: str) -> LLM:
    """Build LLM for the given provider. Raises RuntimeError if key missing."""
    if provider == "groq" and os.getenv("GROQ_API_KEY"):
        return LLM(
            model="groq/llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.2,
            max_tokens=512,   # tight cap — keeps total TPM well under 12k
        )
    if provider == "gemini" and os.getenv("GEMINI_API_KEY"):
        return LLM(
            model="gemini/gemini-2.0-flash",
            api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.2,
        )
    if provider == "cerebras" and os.getenv("CEREBRAS_API_KEY"):
        return LLM(
            model="cerebras/llama3.1-8b",
            api_key=os.getenv("CEREBRAS_API_KEY"),
            temperature=0.2,
        )
    raise RuntimeError(f"Provider {provider!r}: key missing or unknown.")


def _provider_order() -> list[str]:
    """Return providers to try, respecting LLM_PROVIDER env override."""
    preferred = os.getenv("LLM_PROVIDER", "groq").lower()
    order = ["groq", "gemini", "cerebras"]
    if preferred in order:
        order.remove(preferred)
        order.insert(0, preferred)
    return order


def build_crew(user_query: str, llm: LLM) -> Crew:
    """Build fresh agents + crew for one query with the given LLM."""

    intent_classifier = Agent(
        role="Intent Classifier",
        goal=(
            "Parse user text into JSON: "
            "{mode, city, category, niche, urgency, constraints}. "
            "Output valid JSON only."
        ),
        backstory=(
            "Normalize local-search queries into structured intent. "
            "If query starts with [MODE: x] use that mode. "
            "If 'refinement:' appears, refinement overrides conflicting fields."
        ),
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=3,
    )

    retrieval_strategist = Agent(
        role="Retrieval Strategist",
        goal=(
            "Fetch 3 candidates. "
            "Use Search Places for mode=places. "
            "Use Search Leads for mode=leads."
        ),
        backstory=(
            "Pick right data source. Pass short query like 'spa Rabat' to tool. "
            "Return candidates verbatim with any URLs."
        ),
        tools=[search_places, search_leads],
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=3,
    )

    ranking_explainer = Agent(
        role="Ranking and Explanation",
        goal=(
            "Score each candidate with Score Match. "
            "Rank High>Medium>Low by confidence. "
            "Return TOP 3: name, details, match label, confidence, 1-sentence reason, Map/OSM URLs."
        ),
        backstory=(
            "Combine DistilBERT score with short LLM reason. Keep URLs unchanged."
        ),
        tools=[score_match],
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=5,
    )

    intent_task = Task(
        description=(
            f"Query: {user_query!r}\n"
            "Output ONLY JSON: {mode, city, category, niche, urgency, constraints}."
        ),
        expected_output="JSON intent.",
        agent=intent_classifier,
    )

    retrieval_task = Task(
        description=(
            f"Query: {user_query!r}\n"
            "Use intent. Call Search Places (places) or Search Leads (leads). "
            "Return candidates verbatim."
        ),
        expected_output="3 candidates with URLs.",
        agent=retrieval_strategist,
        context=[intent_task],
    )

    ranking_task = Task(
        description=(
            f"Query: {user_query!r}\n"
            "Call Score Match for EACH candidate: "
            "'QUERY: <q> | CANDIDATE: <text>'. "
            "Rank by label (High>Med>Low) then confidence. Output:\n"
            "Rank N. <Name>\nDetails: <cat, city, rating>\n"
            "Match: <label> (<conf%>)\nWhy: <1 sentence>\n"
            "Map: <url>\nOSM: <url>"
        ),
        expected_output="Top 3 ranked with reason and URLs.",
        agent=ranking_explainer,
        context=[intent_task, retrieval_task],
    )

    return Crew(
        agents=[intent_classifier, retrieval_strategist, ranking_explainer],
        tasks=[intent_task, retrieval_task, ranking_task],
        process=Process.sequential,
        verbose=False,
    )


def run_lookking(user_query: str) -> str:
    """Run the 3-agent pipeline with automatic provider fallback on rate limit."""
    last_err = None
    for provider in _provider_order():
        try:
            llm = _make_llm(provider)
        except RuntimeError:
            continue

        try:
            crew = build_crew(user_query, llm)
            result = crew.kickoff(inputs={"query": user_query})
            return str(result)
        except Exception as e:
            last_err = e
            err_str = str(e).lower()
            if any(kw in err_str for kw in _RATE_KEYWORDS):
                # Rate-limited — silently try next provider
                continue
            # Non-rate error — don't retry
            return f"System error: {type(e).__name__}: {e}"

    return (
        "⚠️ All AI providers are rate-limited right now. "
        "Please wait 60 seconds and try again.\n"
        f"(Last error: {type(last_err).__name__})"
    )
