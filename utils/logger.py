import json
import os
from datetime import datetime
from pathlib import Path

LOG_FILE = Path(__file__).parent.parent / "logs" / "agent_logs.json"


def log_action(agent: str, action: str, input_data: dict, output: str) -> None:
    LOG_FILE.parent.mkdir(exist_ok=True)

    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "agent": agent,
        "action": action,
        "input": input_data,
        "output": output[:800] if len(output) > 800 else output,
    }

    logs = []
    if LOG_FILE.exists():
        try:
            with open(LOG_FILE) as f:
                logs = json.load(f)
        except json.JSONDecodeError:
            logs = []

    logs.append(entry)

    with open(LOG_FILE, "w") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)
