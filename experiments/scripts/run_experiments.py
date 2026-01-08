#!/usr/bin/env python3
"""
Experiment runner for Docker Neural Memory.

Runs standardized experiments and logs results to Langfuse.

Usage:
    python run_experiments.py --suite learning
    python run_experiments.py --suite all --dim 256 --lr 0.01
    python run_experiments.py --suite capacity --output results/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import torch

from src.config import MemoryConfig
from src.memory.neural_memory import NeuralMemory

# Optional Langfuse import
try:
    from langfuse import Langfuse
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False
    Langfuse = None


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    suite: str = "all"
    dim: int = 256
    learning_rate: float = 0.01
    device: str = "cpu"
    output_dir: Path = field(default_factory=lambda: Path("experiments/results"))
    use_langfuse: bool = True
    experiment_name: str = ""

    def __post_init__(self) -> None:
        if not self.experiment_name:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"exp_{self.suite}_{timestamp}"


@dataclass
class ExperimentResult:
    """Result of a single test."""
    test_id: str
    passed: bool
    metrics: dict[str, Any]
    duration_ms: float
    error: str | None = None


class ExperimentRunner:
    """Runs experiments and tracks results."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.memory_config = MemoryConfig(
            dim=config.dim,
            learning_rate=config.learning_rate,
            device=config.device,
        )
        self.memory: NeuralMemory | None = None
        self.langfuse: Langfuse | None = None
        self.results: list[ExperimentResult] = []

        # Initialize Langfuse if available and requested
        if config.use_langfuse and LANGFUSE_AVAILABLE:
            try:
                self.langfuse = Langfuse()
                print(f"Langfuse connected: tracking to {config.experiment_name}")
            except Exception as e:
                print(f"Langfuse init failed: {e}")
                self.langfuse = None

    def reset_memory(self) -> None:
        """Create a fresh memory instance."""
        self.memory = NeuralMemory(self.memory_config)

    def _create_trace(self, name: str, metadata: dict | None = None) -> Any:
        """Create a Langfuse trace if available."""
        if self.langfuse:
            return self.langfuse.trace(
                name=name,
                session_id=self.config.experiment_name,
                metadata=metadata or {},
            )
        return None

    def _log_score(self, trace: Any, name: str, value: float, comment: str = "") -> None:
        """Log a score to Langfuse trace."""
        if trace and self.langfuse:
            trace.score(name=name, value=value, comment=comment)

    def run_learning_suite(self) -> list[ExperimentResult]:
        """Run learning verification tests."""
        dataset_path = Path("experiments/datasets/learning.json")
        with open(dataset_path) as f:
            dataset = json.load(f)

        results = []

        for test in dataset["tests"]:
            self.reset_memory()
            result = self._run_learning_test(test)
            results.append(result)
            self._print_result(result)

        return results

    def _run_learning_test(self, test: dict) -> ExperimentResult:
        """Run a single learning test."""
        test_id = test["id"]
        trace = self._create_trace(f"learning/{test_id}", {"test": test})

        start_time = time.perf_counter()
        metrics: dict[str, Any] = {"surprises": [], "weight_deltas": []}
        passed = True
        error = None

        try:
            if "inputs" in test:
                # Simple input sequence test
                for i, inp in enumerate(test["inputs"]):
                    result = self.memory.observe(inp)
                    metrics["surprises"].append(result["surprise"])
                    metrics["weight_deltas"].append(result["weight_delta"])

                    if trace:
                        span = trace.span(name=f"observe_{i}")
                        span.update(
                            input={"content": inp[:100]},
                            output={"surprise": result["surprise"]},
                        )
                        span.end()

                # Check expected outcomes
                expected = test.get("expected", {})

                if "surprise_trend" in expected:
                    if expected["surprise_trend"] == "decreasing":
                        # Check if trend is generally decreasing
                        first_half_avg = sum(metrics["surprises"][:len(metrics["surprises"])//2]) / max(1, len(metrics["surprises"])//2)
                        second_half_avg = sum(metrics["surprises"][len(metrics["surprises"])//2:]) / max(1, len(metrics["surprises"]) - len(metrics["surprises"])//2)
                        passed = passed and (second_half_avg < first_half_avg)
                        metrics["trend_check"] = {
                            "first_half_avg": first_half_avg,
                            "second_half_avg": second_half_avg,
                            "is_decreasing": second_half_avg < first_half_avg,
                        }

                if "final_surprise_max" in expected:
                    final = metrics["surprises"][-1]
                    passed = passed and (final <= expected["final_surprise_max"])
                    metrics["final_surprise"] = final

            elif "sequence" in test:
                # Sequence with checkpoints/recalls
                checkpoints: dict[str, str] = {}

                for step in test["sequence"]:
                    if "input" in step:
                        result = self.memory.observe(step["input"])
                        metrics["surprises"].append(result["surprise"])
                        if "checkpoint" in step:
                            checkpoints[step["checkpoint"]] = self.memory.get_weight_hash()

                    elif "recall" in step:
                        surprise = self.memory.surprise(step["recall"])
                        metrics["recall_surprise"] = surprise
                        if "expect_surprise_max" in step:
                            passed = passed and (surprise <= step["expect_surprise_max"])

            # Log scores to Langfuse
            if trace:
                self._log_score(trace, "passed", 1.0 if passed else 0.0)
                if metrics["surprises"]:
                    self._log_score(trace, "final_surprise", metrics["surprises"][-1])
                    self._log_score(trace, "surprise_reduction",
                                  metrics["surprises"][0] - metrics["surprises"][-1])

        except Exception as e:
            passed = False
            error = str(e)

        duration_ms = (time.perf_counter() - start_time) * 1000

        return ExperimentResult(
            test_id=test_id,
            passed=passed,
            metrics=metrics,
            duration_ms=duration_ms,
            error=error,
        )

    def run_retention_suite(self) -> list[ExperimentResult]:
        """Run retention verification tests."""
        dataset_path = Path("experiments/datasets/retention.json")
        with open(dataset_path) as f:
            dataset = json.load(f)

        results = []

        for test in dataset["tests"]:
            self.reset_memory()
            result = self._run_retention_test(test)
            results.append(result)
            self._print_result(result)

        return results

    def _run_retention_test(self, test: dict) -> ExperimentResult:
        """Run a single retention test."""
        test_id = test["id"]
        trace = self._create_trace(f"retention/{test_id}", {"test": test})

        start_time = time.perf_counter()
        metrics: dict[str, Any] = {}
        passed = True
        error = None

        try:
            if "learn" in test and "recall" in test:
                # Simple learn-recall test
                learns = test["learn"] if isinstance(test["learn"], list) else [test["learn"]]
                for content in learns:
                    self.memory.observe(content)

                recall_surprise = self.memory.surprise(test["recall"])
                metrics["recall_surprise"] = recall_surprise

                expected = test.get("expected", {})
                if "surprise_max" in expected:
                    passed = recall_surprise <= expected["surprise_max"]

            elif "sequence" in test:
                # Sequence with interleaved operations
                for step in test["sequence"]:
                    if "learn" in step:
                        result = self.memory.observe(step["learn"])
                        metrics.setdefault("learn_surprises", []).append(result["surprise"])

                    elif "recall" in step:
                        surprise = self.memory.surprise(step["recall"])
                        metrics.setdefault("recall_surprises", []).append(surprise)
                        if "expect_surprise_max" in step:
                            passed = passed and (surprise <= step["expect_surprise_max"])

                    elif "consolidate" in step:
                        # TODO: Implement consolidation in memory
                        metrics["consolidated"] = True

            if trace:
                self._log_score(trace, "passed", 1.0 if passed else 0.0)
                if "recall_surprise" in metrics:
                    self._log_score(trace, "recall_surprise", metrics["recall_surprise"])

        except Exception as e:
            passed = False
            error = str(e)

        duration_ms = (time.perf_counter() - start_time) * 1000

        return ExperimentResult(
            test_id=test_id,
            passed=passed,
            metrics=metrics,
            duration_ms=duration_ms,
            error=error,
        )

    def run_generalization_suite(self) -> list[ExperimentResult]:
        """Run generalization verification tests."""
        dataset_path = Path("experiments/datasets/generalization.json")
        with open(dataset_path) as f:
            dataset = json.load(f)

        results = []

        for test in dataset["tests"]:
            self.reset_memory()
            result = self._run_generalization_test(test)
            results.append(result)
            self._print_result(result)

        return results

    def _run_generalization_test(self, test: dict) -> ExperimentResult:
        """Run a single generalization test."""
        test_id = test["id"]
        trace = self._create_trace(f"generalization/{test_id}", {"test": test})

        start_time = time.perf_counter()
        metrics: dict[str, Any] = {}
        passed = True
        error = None

        try:
            # Learn content
            learns = test.get("learn", [])
            if isinstance(learns, str):
                learns = [learns]

            for content in learns:
                self.memory.observe(content)

            # Test paraphrases
            if "test_paraphrases" in test:
                paraphrase_surprises = []
                for para in test["test_paraphrases"]:
                    surprise = self.memory.surprise(para)
                    paraphrase_surprises.append(surprise)

                avg_surprise = sum(paraphrase_surprises) / len(paraphrase_surprises)
                metrics["paraphrase_surprises"] = paraphrase_surprises
                metrics["avg_paraphrase_surprise"] = avg_surprise

                expected = test.get("expected", {})
                if "avg_surprise_max" in expected:
                    passed = passed and (avg_surprise <= expected["avg_surprise_max"])

            # Test novel related content
            if "test_novel" in test:
                novel_surprises = []
                for novel in test["test_novel"]:
                    surprise = self.memory.surprise(novel)
                    novel_surprises.append(surprise)

                metrics["novel_surprises"] = novel_surprises
                metrics["avg_novel_surprise"] = sum(novel_surprises) / len(novel_surprises)

            # Test unrelated content (should stay high surprise)
            if "test_unrelated" in test:
                unrelated_surprises = []
                for unrelated in test["test_unrelated"]:
                    surprise = self.memory.surprise(unrelated)
                    unrelated_surprises.append(surprise)

                avg_unrelated = sum(unrelated_surprises) / len(unrelated_surprises)
                metrics["unrelated_surprises"] = unrelated_surprises
                metrics["avg_unrelated_surprise"] = avg_unrelated

                expected = test.get("expected", {})
                if "unrelated_surprise_min" in expected:
                    passed = passed and (avg_unrelated >= expected["unrelated_surprise_min"])

            if trace:
                self._log_score(trace, "passed", 1.0 if passed else 0.0)

        except Exception as e:
            passed = False
            error = str(e)

        duration_ms = (time.perf_counter() - start_time) * 1000

        return ExperimentResult(
            test_id=test_id,
            passed=passed,
            metrics=metrics,
            duration_ms=duration_ms,
            error=error,
        )

    def run_capacity_suite(self) -> list[ExperimentResult]:
        """Run capacity stress tests."""
        dataset_path = Path("experiments/datasets/capacity.json")
        with open(dataset_path) as f:
            dataset = json.load(f)

        results = []

        # Run scaling test
        scaling_test = next((t for t in dataset["tests"] if t["id"] == "scaling_test"), None)
        if scaling_test:
            result = self._run_scaling_test(scaling_test, dataset.get("generators", {}))
            results.append(result)
            self._print_result(result)

        return results

    def _run_scaling_test(self, test: dict, generators: dict) -> ExperimentResult:
        """Run scaling/capacity test."""
        test_id = test["id"]
        trace = self._create_trace(f"capacity/{test_id}", {"test": test})

        start_time = time.perf_counter()
        metrics: dict[str, Any] = {"scaling_results": []}
        passed = True
        error = None

        try:
            gen_config = generators.get("unique_observations", {})
            domains = gen_config.get("domains", ["Docker", "Python", "Go"])
            predicates = gen_config.get("predicates", ["uses", "provides"])
            objects = gen_config.get("objects", ["containers", "modules"])

            for obs_count in test["observation_counts"]:
                self.reset_memory()

                surprises = []
                latencies = []

                for i in range(obs_count):
                    # Generate unique observation
                    domain = domains[i % len(domains)]
                    pred = predicates[i % len(predicates)]
                    obj = objects[i % len(objects)]
                    content = f"Fact {i}: {domain} {pred} {obj}"

                    obs_start = time.perf_counter()
                    result = self.memory.observe(content)
                    obs_time = (time.perf_counter() - obs_start) * 1000

                    surprises.append(result["surprise"])
                    latencies.append(obs_time)

                # Record metrics for this observation count
                result_entry = {
                    "observation_count": obs_count,
                    "avg_surprise": sum(surprises) / len(surprises),
                    "final_surprise": surprises[-1],
                    "avg_latency_ms": sum(latencies) / len(latencies),
                    "p99_latency_ms": sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0,
                }
                metrics["scaling_results"].append(result_entry)

                print(f"  {obs_count} obs: avg_surprise={result_entry['avg_surprise']:.3f}, "
                      f"p99_latency={result_entry['p99_latency_ms']:.1f}ms")

            # Check expectations
            expected = test.get("expected", {})
            if "latency_p99_max_ms" in expected:
                for result_entry in metrics["scaling_results"]:
                    if result_entry["p99_latency_ms"] > expected["latency_p99_max_ms"]:
                        passed = False
                        break

            if trace:
                self._log_score(trace, "passed", 1.0 if passed else 0.0)

        except Exception as e:
            passed = False
            error = str(e)

        duration_ms = (time.perf_counter() - start_time) * 1000

        return ExperimentResult(
            test_id=test_id,
            passed=passed,
            metrics=metrics,
            duration_ms=duration_ms,
            error=error,
        )

    def _print_result(self, result: ExperimentResult) -> None:
        """Print a test result."""
        status = "PASS" if result.passed else "FAIL"
        print(f"  [{status}] {result.test_id} ({result.duration_ms:.1f}ms)")
        if result.error:
            print(f"       Error: {result.error}")

    def run(self) -> dict[str, Any]:
        """Run all requested experiment suites."""
        all_results: dict[str, list[ExperimentResult]] = {}

        suites = {
            "learning": self.run_learning_suite,
            "retention": self.run_retention_suite,
            "generalization": self.run_generalization_suite,
            "capacity": self.run_capacity_suite,
        }

        if self.config.suite == "all":
            to_run = list(suites.keys())
        else:
            to_run = [self.config.suite]

        print(f"\n{'='*60}")
        print(f"Experiment: {self.config.experiment_name}")
        print(f"Config: dim={self.config.dim}, lr={self.config.learning_rate}")
        print(f"{'='*60}\n")

        for suite_name in to_run:
            if suite_name in suites:
                print(f"\n--- {suite_name.upper()} SUITE ---")
                results = suites[suite_name]()
                all_results[suite_name] = results

        # Summary
        total = sum(len(r) for r in all_results.values())
        passed = sum(sum(1 for r in results if r.passed) for results in all_results.values())

        print(f"\n{'='*60}")
        print(f"SUMMARY: {passed}/{total} tests passed")
        print(f"{'='*60}\n")

        # Save results
        self._save_results(all_results)

        # Flush Langfuse
        if self.langfuse:
            self.langfuse.flush()

        return {
            "experiment_name": self.config.experiment_name,
            "total_tests": total,
            "passed": passed,
            "suites": {
                name: [{"test_id": r.test_id, "passed": r.passed, "metrics": r.metrics}
                       for r in results]
                for name, results in all_results.items()
            },
        }

    def _save_results(self, all_results: dict[str, list[ExperimentResult]]) -> None:
        """Save results to JSON file."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        output_file = self.config.output_dir / f"{self.config.experiment_name}.json"

        data = {
            "experiment_name": self.config.experiment_name,
            "config": {
                "dim": self.config.dim,
                "learning_rate": self.config.learning_rate,
                "device": self.config.device,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "suites": {
                name: [
                    {
                        "test_id": r.test_id,
                        "passed": r.passed,
                        "metrics": r.metrics,
                        "duration_ms": r.duration_ms,
                        "error": r.error,
                    }
                    for r in results
                ]
                for name, results in all_results.items()
            },
        }

        with open(output_file, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Results saved to: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run neural memory experiments")
    parser.add_argument("--suite", default="all",
                       choices=["all", "learning", "retention", "generalization", "capacity"],
                       help="Which test suite to run")
    parser.add_argument("--dim", type=int, default=256, help="Memory dimension")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--device", default="cpu", help="Device (cpu, cuda, mps)")
    parser.add_argument("--output", type=Path, default=Path("experiments/results"),
                       help="Output directory for results")
    parser.add_argument("--no-langfuse", action="store_true", help="Disable Langfuse tracking")
    parser.add_argument("--name", default="", help="Experiment name (auto-generated if empty)")

    args = parser.parse_args()

    config = ExperimentConfig(
        suite=args.suite,
        dim=args.dim,
        learning_rate=args.lr,
        device=args.device,
        output_dir=args.output,
        use_langfuse=not args.no_langfuse,
        experiment_name=args.name,
    )

    runner = ExperimentRunner(config)
    runner.run()


if __name__ == "__main__":
    main()
