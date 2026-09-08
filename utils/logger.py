"""Structured JSON experiment logger."""

import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class ExperimentLogger:
    """Accumulates results and writes them as a JSON file to outputs/logs/."""

    def __init__(self, experiment_name: str):
        self.name = experiment_name
        self.start_time = time.time()
        self.data = {
            "experiment": experiment_name,
            "started_at": datetime.now().isoformat(),
            "config": {
                "seed": config.SEED, "epochs": config.EPOCHS,
                "batch_size": config.BATCH_SIZE, "learning_rate": config.LEARNING_RATE,
                "poison_rate": config.POISON_RATE, "device": str(config.DEVICE),
            },
            "results": {},
        }

    def record(self, key: str, value: Any):
        self.data["results"][key] = value

    def record_nested(self, section: str, key: str, value: Any):
        if section not in self.data["results"]:
            self.data["results"][section] = {}
        self.data["results"][section][key] = value

    def save(self) -> str:
        self.data["elapsed_seconds"] = round(time.time() - self.start_time, 2)
        self.data["finished_at"] = datetime.now().isoformat()
        path = os.path.join(config.LOG_DIR, f"{self.name}.json")
        with open(path, "w") as f:
            json.dump(self.data, f, indent=2, default=str)
        print(f"  [Log] Saved → {path}")
        return path


def load_log(experiment_name: str) -> Optional[Dict]:
    path = os.path.join(config.LOG_DIR, f"{experiment_name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def load_all_logs() -> Dict[str, Dict]:
    logs = {}
    for fname in sorted(os.listdir(config.LOG_DIR)):
        if fname.endswith(".json"):
            with open(os.path.join(config.LOG_DIR, fname)) as f:
                data = json.load(f)
                logs[data["experiment"]] = data
    return logs
