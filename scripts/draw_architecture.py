"""
Generate architecture.png — visual diagram of the Lookking pipeline.

Run from project root:
    python3 scripts/draw_architecture.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent.parent / "docs" / "architecture.png"


def box(ax, x, y, w, h, text, color="#cfe2ff", edge="#1a4a8a"):
    rect = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.8, edgecolor=edge, facecolor=color,
    )
    ax.add_patch(rect)
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center", fontsize=10, fontweight="bold", wrap=True)


def arrow(ax, x1, y1, x2, y2, label=""):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=18,
        linewidth=1.6, color="#444",
    ))
    if label:
        ax.text((x1 + x2) / 2 + 0.05, (y1 + y2) / 2, label,
                fontsize=8, color="#444", style="italic")


def main():
    OUT.parent.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(15, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.set_aspect("equal")
    ax.axis("off")

    # Title
    ax.text(7, 9.6, "Lookking — System Architecture",
            ha="center", fontsize=18, fontweight="bold")
    ax.text(7, 9.2,
            "Multi-agent AI pipeline · CrewAI + DistilBERT + OpenStreetMap · "
            "Telegram HITL UX",
            ha="center", fontsize=10, color="#444", style="italic")

    # User
    box(ax, 0.5, 7.5, 2.4, 1.0, "USER\n(Telegram)", "#fff3cd", "#7a5c00")

    # Bot layer
    box(ax, 4.0, 7.5, 3.0, 1.0,
        "Telegram Bot\nState machine\nHITL: Mode + Refine",
        "#d1ecf1", "#0c5460")

    # Pipeline title
    ax.text(7, 6.7, "Multi-Agent Pipeline (CrewAI)",
            ha="center", fontsize=12, fontweight="bold", color="#1a4a8a")

    # 3 agents
    box(ax, 0.5, 4.6, 4.0, 1.6,
        "AGENT 1\nIntent Classifier\n"
        "(LLM)\n"
        "-> JSON: mode, city,\ncategory, niche, urgency",
        "#e3f2ff", "#1a4a8a")

    box(ax, 5.0, 4.6, 4.0, 1.6,
        "AGENT 2\nRetrieval Strategist\n"
        "(LLM + Search tools)\n"
        "-> 5-8 candidates from\nNominatim OSM / CSV",
        "#e3f2ff", "#1a4a8a")

    box(ax, 9.5, 4.6, 4.0, 1.6,
        "AGENT 3\nRanking + Explanation\n"
        "(LLM + DL Tool)\n"
        "-> Top 3 ranked +\nhuman-readable reason",
        "#e3f2ff", "#1a4a8a")

    # Tools layer
    ax.text(7, 4.2, "Tools layer", ha="center",
            fontsize=11, fontweight="bold", color="#444")

    box(ax, 5.0, 2.6, 4.0, 1.2,
        "Search Tool\nNominatim API (live OSM)\n"
        "-> places_real.csv fallback",
        "#d4edda", "#155724")

    box(ax, 9.5, 2.6, 4.0, 1.2,
        "DL Scorer Tool\nDistilBERT + MLP head\n"
        "-> High/Med/Low match",
        "#d4edda", "#155724")

    # Data / model layer
    box(ax, 0.5, 0.5, 4.0, 1.2,
        "Data\nplaces_real.csv (361 places)\ntraining_data_v2 (1260 pairs)\nholdout (30 queries)",
        "#f8d7da", "#721c24")

    box(ax, 5.0, 0.5, 4.0, 1.2,
        "OpenStreetMap\nNominatim API\nfree, no key, live data",
        "#f8d7da", "#721c24")

    box(ax, 9.5, 0.5, 4.0, 1.2,
        "Trained Model\nlookking_model.pt\nholdout acc: 80%",
        "#f8d7da", "#721c24")

    # Arrows
    arrow(ax, 2.9, 8.0, 4.0, 8.0)              # user → bot
    arrow(ax, 7.0, 8.0, 2.9, 8.05, "")         # bot → user (reply)
    arrow(ax, 5.5, 7.5, 2.5, 6.3, "query")     # bot → agent1
    arrow(ax, 4.5, 5.4, 5.0, 5.4, "intent")    # a1 → a2
    arrow(ax, 9.0, 5.4, 9.5, 5.4, "candidates")# a2 → a3
    arrow(ax, 7.0, 4.6, 7.0, 3.8, "tool call") # a2 → search tool
    arrow(ax, 11.5, 4.6, 11.5, 3.8, "tool call")# a3 → dl tool
    arrow(ax, 7.0, 2.6, 7.0, 1.7, "")          # search → osm
    arrow(ax, 7.0, 2.65, 4.5, 1.7, "")         # search → data csv
    arrow(ax, 11.5, 2.6, 11.5, 1.7, "")        # dl → model
    arrow(ax, 11.5, 6.2, 7.0, 7.5, "top 3 reply")# a3 → bot

    # Legend
    legend_handles = [
        mpatches.Patch(color="#cfe2ff", label="Agents (LLM-driven)"),
        mpatches.Patch(color="#fff3cd", label="User"),
        mpatches.Patch(color="#d1ecf1", label="Interface"),
        mpatches.Patch(color="#d4edda", label="Tools"),
        mpatches.Patch(color="#f8d7da", label="Data / Models / APIs"),
    ]
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.05), ncol=5, frameon=False, fontsize=9)

    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    print(f"[+] Architecture diagram → {OUT}")


if __name__ == "__main__":
    main()
