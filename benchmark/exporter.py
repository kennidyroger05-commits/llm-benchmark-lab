"""Standard JSON and CSV export for benchmark result collections."""

import csv
import json
import re
import uuid
from dataclasses import fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

from benchmark.results import BenchmarkResult


def _new_run_id() -> str:
    """Create a readable UTC run ID with a random suffix to prevent collisions."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    return f"{timestamp}_{uuid.uuid4().hex[:8]}"


def _model_slug(model_id: str) -> str:
    """Convert a model identifier to a portable directory name."""
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", model_id).strip("-._").lower()
    if not slug:
        raise ValueError("model_id must contain at least one alphanumeric character")
    return slug


def export_results(
    results: Iterable[BenchmarkResult],
    model_id: str,
    phase: str = "phase1",
    condition: str = "raw",
    results_root: Path = Path("results"),
    run_id: Optional[str] = None,
) -> tuple[Path, Path]:
    """Write one JSON and one CSV file without overwriting a previous run."""
    run_id = run_id or _new_run_id()
    if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("run_id may contain only letters, numbers, underscores, and hyphens")

    output_dir = Path(results_root) / condition / phase / _model_slug(model_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"benchmark_{run_id}.json"
    csv_path = output_dir / f"benchmark_{run_id}.csv"
    rows = [result.to_dict() for result in results]
    fieldnames = [field.name for field in fields(BenchmarkResult)]

    # Exclusive creation fails if either filename already exists instead of overwriting.
    with json_path.open("x", encoding="utf-8") as json_file:
        json.dump(rows, json_file, ensure_ascii=False, indent=2)
        json_file.write("\n")
    with csv_path.open("x", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return json_path, csv_path
