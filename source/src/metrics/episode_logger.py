"""
episode_logger.py — CSV logging for simulation episodes.

Saves episode results to CSV files for later analysis.
Improved version of the original swarmLogger.py.
"""

import csv
import os
from datetime import datetime
from typing import Dict, Any, List


class EpisodeLogger:
    """
    Logs simulation results to CSV files.

    Creates a timestamped CSV file and writes episode summaries,
    per-predator statistics, and other metrics.
    """

    def __init__(self, output_dir: str = "logs", filename_prefix: str = "episode"):
        """
        Initialize logger.

        Args:
            output_dir: Directory to save log files
            filename_prefix: Prefix for log filenames
        """
        os.makedirs(output_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = os.path.join(output_dir, f"{filename_prefix}_{timestamp}.csv")
        self.episode_id = timestamp

        self.rows: List[Dict[str, Any]] = []

    def log_episode_summary(self, **fields):
        """
        Log overall episode statistics.

        Args:
            **fields: Arbitrary key-value pairs to log
        """
        row = {"type": "episode_summary", "episode_id": self.episode_id}
        row.update(fields)
        self.rows.append(row)

    def log_predator_summary(self, predator_id: int, **fields):
        """
        Log per-predator statistics.

        Args:
            predator_id: Predator agent ID
            **fields: Arbitrary key-value pairs to log
        """
        row = {
            "type": "predator_summary",
            "episode_id": self.episode_id,
            "predator_id": predator_id,
        }
        row.update(fields)
        self.rows.append(row)

    def flush(self):
        """Write all logged rows to CSV file."""
        if not self.rows:
            return

        # Collect all field names
        fieldnames = set()
        for row in self.rows:
            fieldnames.update(row.keys())
        fieldnames = sorted(fieldnames)

        # Write to CSV
        with open(self.filepath, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)

        print(f"Logged {len(self.rows)} rows to {self.filepath}")
