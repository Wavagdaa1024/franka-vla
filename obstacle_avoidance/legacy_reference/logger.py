"""
Logging and manifest manager for AG-S4-002 / S4 End-to-End.
Ensures:
1. JSONL step-by-step logging.
2. Per-episode JSON summaries.
3. run_manifest.json generation.
4. Source tag explicitly set to 'synthetic_unit_test'.
5. Failed/timeout episodes are fully preserved with explicit termination reasons.
6. Refuses in-place overwriting of existing episode evidence; preserves separate attempts.
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from src.aggregator import AggregationSummary, AggregatorConfig, EventAggregator
from src.types import StepRecord, TerminationReason, ValidationError


class EpisodeLogger:
    """
    Logs an individual episode to JSONL and produces a finalized JSON summary.
    Preserves prior runs and failure evidence by auto-versioning attempts if file exists.
    """

    def __init__(
        self,
        output_dir: Path,
        episode_id: str,
        scene_seed: int,
        attempt_id: Optional[int] = None,
        aggregator_config: Optional[AggregatorConfig] = None,
        source: str = "synthetic_unit_test",
        refuse_overwrite: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.episode_id = str(episode_id)
        self.scene_seed = int(scene_seed)
        self.source = str(source)

        self.step_logs_dir = self.output_dir / "step_logs"
        self.summaries_dir = self.output_dir / "summaries"

        self.step_logs_dir.mkdir(parents=True, exist_ok=True)
        self.summaries_dir.mkdir(parents=True, exist_ok=True)

        base_jsonl = self.step_logs_dir / f"episode_{self.episode_id}.jsonl"
        base_summary = self.summaries_dir / f"episode_{self.episode_id}.json"

        if attempt_id is not None:
            self.attempt_id = attempt_id
            self.jsonl_path = self.step_logs_dir / f"episode_{self.episode_id}_attempt_{attempt_id}.jsonl"
            self.summary_path = self.summaries_dir / f"episode_{self.episode_id}_attempt_{attempt_id}.json"
        else:
            if base_jsonl.exists() or base_summary.exists():
                if refuse_overwrite:
                    raise FileExistsError(
                        f"Episode log '{base_jsonl}' already exists. Overwriting evidence is forbidden. "
                        "Specify an attempt_id to record a retry."
                    )
                # Auto-version to attempt 2, 3, ... to preserve existing evidence
                k = 2
                while True:
                    candidate_jsonl = self.step_logs_dir / f"episode_{self.episode_id}_attempt_{k}.jsonl"
                    candidate_summary = self.summaries_dir / f"episode_{self.episode_id}_attempt_{k}.json"
                    if not candidate_jsonl.exists() and not candidate_summary.exists():
                        break
                    k += 1
                self.attempt_id = k
                self.jsonl_path = candidate_jsonl
                self.summary_path = candidate_summary
            else:
                self.attempt_id = 1
                self.jsonl_path = base_jsonl
                self.summary_path = base_summary

        self.aggregator = EventAggregator(config=aggregator_config)
        self._step_records: List[StepRecord] = []
        self._file_handle = open(self.jsonl_path, "w", encoding="utf-8")
        self._is_finalized = False

    def log_step(self, step: StepRecord) -> None:
        if self._is_finalized:
            raise ValidationError(f"Cannot log step to finalized episode {self.episode_id}")

        if step.episode_id != self.episode_id:
            raise ValidationError(
                f"Episode ID mismatch: step has {step.episode_id}, logger has {self.episode_id}"
            )
        if step.source != self.source:
            raise ValidationError(
                f"Source mismatch: step has {step.source}, logger has {self.source}"
            )

        # Let step validate itself and let aggregator validate ordering & types
        self.aggregator.process_step(step)
        self._step_records.append(step)

        line = json.dumps(step.to_dict(), ensure_ascii=False)
        self._file_handle.write(line + "\n")
        self._file_handle.flush()

    def finalize(self, termination_reason: str) -> Dict[str, Any]:
        if self._is_finalized:
            raise ValidationError(f"Episode {self.episode_id} is already finalized.")

        self._is_finalized = True
        self._file_handle.close()

        summary: AggregationSummary = self.aggregator.finalize_episode(termination_reason)

        episode_summary = {
            "episode_id": self.episode_id,
            "attempt_id": self.attempt_id,
            "scene_seed": self.scene_seed,
            "source": self.source,
            "termination_reason": str(termination_reason),
            "step_count": len(self._step_records),
            "aggregation": summary.to_dict(),
            "windows": [w.to_dict() for w in self.aggregator.get_windows()],
            "jsonl_path": str(self.jsonl_path.relative_to(self.output_dir)),
        }

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(episode_summary, f, indent=2, ensure_ascii=False)

        return episode_summary


class ManifestManager:
    """
    Manages run_manifest.json aggregating all episodes in a run.
    Prevents silent dropping or loss of failure runs.
    """

    def __init__(self, output_dir: Path, run_id: str = "AG-S4-002-SYNTHETIC") -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.manifest_path = self.output_dir / "run_manifest.json"
        self.episodes: List[Dict[str, Any]] = []

    def add_episode_summary(self, episode_summary: Dict[str, Any]) -> None:
        self.episodes.append(episode_summary)

    def write_manifest(self) -> Dict[str, Any]:
        total_evaluable = sum(e["aggregation"]["evaluable_windows"] for e in self.episodes)
        total_censored = sum(e["aggregation"]["censored_windows"] for e in self.episodes)
        total_repeated = sum(e["aggregation"]["repeated_conflicts"] for e in self.episodes)
        total_steps = sum(e["step_count"] for e in self.episodes)

        reasons = {}
        for e in self.episodes:
            r = e["termination_reason"]
            reasons[r] = reasons.get(r, 0) + 1

        overall_repeat_rate = (total_repeated / total_evaluable) if total_evaluable > 0 else None

        manifest_data = {
            "manifest_version": "1.1",
            "run_id": self.run_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "source": "synthetic_unit_test",
            "total_episodes": len(self.episodes),
            "termination_breakdown": reasons,
            "aggregate_statistics": {
                "total_steps": total_steps,
                "total_evaluable_windows": total_evaluable,
                "total_censored_windows": total_censored,
                "total_repeated_conflicts": total_repeated,
                "overall_repeat_rate": overall_repeat_rate,
                "overall_denominator": total_evaluable,
            },
            "episodes": self.episodes,
        }

        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2, ensure_ascii=False)

        return manifest_data
