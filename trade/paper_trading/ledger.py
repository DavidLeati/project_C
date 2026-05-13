from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class LedgerEvent:
    timestamp: str
    symbol: str
    close_time: str
    model_position: float
    action: str
    close_price: float
    target_notional: str
    current_notional: str
    delta_notional: str
    order_side: str | None
    order_quantity: str
    reduce_only: bool
    local_equity: float
    gross_pnl: float
    estimated_cost: float
    order_response: dict[str, Any] | None
    error: str | None = None


class PaperLedger:
    def __init__(self, artifact_dir: Path):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.artifact_dir / "events.jsonl"
        self.csv_path = self.artifact_dir / "events.csv"
        self._csv_header_written = self.csv_path.exists() and self.csv_path.stat().st_size > 0

    def append(self, event: LedgerEvent) -> None:
        row = asdict(event)
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True, default=str) + "\n")

        flat = dict(row)
        flat["order_response"] = json.dumps(flat["order_response"], ensure_ascii=True, default=str)
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(flat.keys()))
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerow(flat)
