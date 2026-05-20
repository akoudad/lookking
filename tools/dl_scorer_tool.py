"""
DL Scorer Tool: wraps the trained LookkingMatcher to score query-candidate pairs.
This is the deep learning model used as a CrewAI tool.
"""
import sys
from pathlib import Path
from typing import Optional

import torch
from crewai.tools import tool

sys.path.insert(0, str(Path(__file__).parent.parent))
from model.model import LookkingMatcher, get_tokenizer, get_device, LABEL_NAMES
from utils.logger import log_action

MODEL_PATH = Path(__file__).parent.parent / "model" / "lookking_model.pt"
MAX_LEN = 128

_model: Optional[LookkingMatcher] = None
_tokenizer = None
_device = None


def _load_model():
    global _model, _tokenizer, _device
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Run: python3 model/train.py"
            )
        # Force CPU — MPS is not thread-safe inside ThreadPoolExecutor
        _device = torch.device("cpu")
        _tokenizer = get_tokenizer()
        _model = LookkingMatcher(num_classes=3)
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=_device, weights_only=True))
        _model.to(_device)
        _model.eval()


def score_pair(query: str, candidate: str) -> dict:
    """Score one query-candidate pair. Returns label, confidence, and all scores."""
    _load_model()
    text = f"[QUERY]: {query} [SEP] [CANDIDATE]: {candidate}"
    enc = _tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = _model(
            enc["input_ids"].to(_device),
            enc["attention_mask"].to(_device),
        )
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    label_idx = int(probs.argmax())
    return {
        "label": LABEL_NAMES[label_idx],
        "confidence": float(round(probs[label_idx], 3)),
        "scores": {
            "Low Match": float(round(probs[0], 3)),
            "Medium Match": float(round(probs[1], 3)),
            "High Match": float(round(probs[2], 3)),
        },
    }


@tool("Score Match")
def score_match(query_and_candidate: str) -> str:
    """
    Use the trained deep learning model to score how well a candidate matches a user query.
    Input format: "QUERY: <user query> | CANDIDATE: <candidate description>"
    Returns: match label (High/Medium/Low Match), confidence score, and all class probabilities.
    This is the core AI scoring tool that powers Lookking's ranking.
    """
    try:
        if "QUERY:" in query_and_candidate and "CANDIDATE:" in query_and_candidate:
            parts = query_and_candidate.split("CANDIDATE:")
            query = parts[0].replace("QUERY:", "").strip().strip("|").strip()
            candidate = parts[1].strip()
        else:
            query = query_and_candidate
            candidate = query_and_candidate

        result = score_pair(query, candidate)

        output = (
            f"Match Label: {result['label']}\n"
            f"Confidence: {result['confidence']*100:.1f}%\n"
            f"Scores — High: {result['scores']['High Match']*100:.1f}% | "
            f"Medium: {result['scores']['Medium Match']*100:.1f}% | "
            f"Low: {result['scores']['Low Match']*100:.1f}%"
        )
        log_action("DLScorerTool", "score_match", {"input": query_and_candidate[:200]}, output)
        return output

    except FileNotFoundError as e:
        return f"Error: {e}"
    except Exception as e:
        return f"Scoring error: {type(e).__name__}: {e}"
