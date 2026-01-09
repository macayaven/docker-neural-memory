"""
Langfuse-integrated observability for neural memory.

Tracks both training evolution and inference-time traces.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from collections.abc import Generator

    from langfuse import Langfuse
    from langfuse.client import StatefulSpanClient, StatefulTraceClient

    from ..memory.neural_memory import NeuralMemory


@dataclass
class MetricsSnapshot:
    """Point-in-time snapshot of memory metrics."""

    timestamp: str
    observation_count: int
    surprise: float
    weight_delta: float
    weight_hash: str
    latency_ms: float
    patterns_activated: List[str] = field(default_factory=list)
    learned: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for Langfuse."""
        return {
            "timestamp": self.timestamp,
            "observation_count": self.observation_count,
            "surprise": self.surprise,
            "weight_delta": self.weight_delta,
            "weight_hash": self.weight_hash,
            "latency_ms": self.latency_ms,
            "patterns_activated": self.patterns_activated,
            "learned": self.learned,
        }


class MemoryObserver:
    """
    Observability wrapper for NeuralMemory using Langfuse.

    Tracks:
    - Training: surprise evolution, weight deltas, learning rate
    - Inference: query latency, confidence, patterns activated

    Usage:
        from langfuse import Langfuse
        langfuse = Langfuse()
        observer = MemoryObserver(memory, langfuse)

        # Training with tracing
        result = observer.observe("Python uses indentation")

        # Inference with tracing
        result = observer.infer("What does Python use?")

        # Get metrics summary
        summary = observer.get_summary()
    """

    def __init__(
        self,
        memory: NeuralMemory,
        langfuse: Langfuse | None = None,
        session_id: str | None = None,
    ) -> None:
        """
        Initialize observer.

        Args:
            memory: NeuralMemory instance to observe
            langfuse: Langfuse client (optional, metrics still collected locally)
            session_id: Session ID for grouping traces
        """
        self.memory = memory
        self.langfuse = langfuse
        self.session_id = session_id or f"session_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

        # Local metrics storage
        self._observations: list[MetricsSnapshot] = []
        self._inferences: list[MetricsSnapshot] = []
        self._current_trace: StatefulTraceClient | None = None

    def _get_timestamp(self) -> str:
        """Get ISO timestamp."""
        return datetime.now(timezone.utc).isoformat()

    @contextmanager
    def _trace(
        self, name: str, **metadata: Any
    ) -> Generator[StatefulSpanClient | None, None, None]:
        """Context manager for Langfuse tracing."""
        if self.langfuse is None:
            yield None
            return

        trace = self.langfuse.trace(
            name=name,
            session_id=self.session_id,
            metadata=metadata,
        )
        self._current_trace = trace

        span = trace.span(name=name)
        try:
            yield span
        finally:
            span.end()
            self._current_trace = None

    def observe(
        self,
        content: str,
        learning_rate: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Observe content with full tracing.

        Args:
            content: Text to learn from
            learning_rate: Optional learning rate override
            metadata: Additional metadata for trace

        Returns:
            Observation result with metrics
        """
        start_time = time.perf_counter()

        with self._trace("observe", content_length=len(content), **(metadata or {})) as span:
            # Execute observation
            result = self.memory.observe(content, learning_rate=learning_rate)

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Create snapshot
            snapshot = MetricsSnapshot(
                timestamp=self._get_timestamp(),
                observation_count=self.memory._observation_count,
                surprise=result["surprise"],
                weight_delta=result["weight_delta"],
                weight_hash=self.memory.get_weight_hash(),
                latency_ms=latency_ms,
                patterns_activated=result.get("patterns_activated", []),
                learned=result.get("learned", False),
            )
            self._observations.append(snapshot)

            # Log to Langfuse
            if span is not None:
                span.update(
                    input={"content": content[:500]},  # Truncate for storage
                    output=snapshot.to_dict(),
                )

                # Score the observation
                if self._current_trace:
                    self._current_trace.score(
                        name="surprise",
                        value=result["surprise"],
                        comment="Lower is better (more familiar)",
                    )
                    self._current_trace.score(
                        name="weight_delta",
                        value=min(result["weight_delta"], 1.0),  # Normalize
                        comment="Learning magnitude",
                    )

            return {**result, "latency_ms": latency_ms, "snapshot": snapshot}

    def infer(
        self,
        query: str,
        temperature: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Query memory with full tracing.

        Args:
            query: Text to query
            temperature: Temperature parameter
            metadata: Additional metadata for trace

        Returns:
            Inference result with metrics
        """
        start_time = time.perf_counter()

        with self._trace("infer", query_length=len(query), **(metadata or {})) as span:
            # Execute inference
            result = self.memory.infer(query, temperature=temperature)

            latency_ms = (time.perf_counter() - start_time) * 1000

            # Create snapshot
            snapshot = MetricsSnapshot(
                timestamp=self._get_timestamp(),
                observation_count=self.memory._observation_count,
                surprise=1.0 - result["confidence"],  # Inverse of confidence
                weight_delta=0.0,  # No learning during inference
                weight_hash=self.memory.get_weight_hash(),
                latency_ms=latency_ms,
            )
            self._inferences.append(snapshot)

            # Log to Langfuse
            if span is not None:
                span.update(
                    input={"query": query[:500]},
                    output={
                        "confidence": result["confidence"],
                        "latency_ms": latency_ms,
                    },
                )

                if self._current_trace:
                    self._current_trace.score(
                        name="confidence",
                        value=result["confidence"],
                        comment="Higher is better",
                    )
                    self._current_trace.score(
                        name="latency_ms",
                        value=min(latency_ms / 100, 1.0),  # Normalize to 0-1 (100ms = 1.0)
                        comment="Lower is better",
                    )

            return {**result, "latency_ms": latency_ms, "snapshot": snapshot}

    def surprise(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> float:
        """
        Check surprise without learning, with tracing.

        Args:
            content: Text to check
            metadata: Additional metadata

        Returns:
            Surprise score (0-1)
        """
        start_time = time.perf_counter()

        with self._trace("surprise", content_length=len(content), **(metadata or {})) as span:
            score = self.memory.surprise(content)
            latency_ms = (time.perf_counter() - start_time) * 1000

            if span is not None:
                span.update(
                    input={"content": content[:500]},
                    output={"surprise": score, "latency_ms": latency_ms},
                )

            return score

    def get_summary(self) -> dict[str, Any]:
        """
        Get summary of all collected metrics.

        Returns:
            Dictionary with training and inference statistics
        """
        obs_surprises = [o.surprise for o in self._observations]
        obs_deltas = [o.weight_delta for o in self._observations]
        obs_latencies = [o.latency_ms for o in self._observations]
        inf_latencies = [i.latency_ms for i in self._inferences]

        def safe_avg(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0

        return {
            "session_id": self.session_id,
            "training": {
                "total_observations": len(self._observations),
                "avg_surprise": safe_avg(obs_surprises),
                "surprise_trend": obs_surprises[-10:] if obs_surprises else [],
                "avg_weight_delta": safe_avg(obs_deltas),
                "avg_latency_ms": safe_avg(obs_latencies),
                "learned_count": sum(1 for o in self._observations if o.learned),
            },
            "inference": {
                "total_queries": len(self._inferences),
                "avg_latency_ms": safe_avg(inf_latencies),
                "p99_latency_ms": sorted(inf_latencies)[int(len(inf_latencies) * 0.99)]
                if inf_latencies
                else 0,
            },
            "memory": self.memory.get_stats(),
        }

    def flush(self) -> None:
        """Flush pending traces to Langfuse."""
        if self.langfuse:
            self.langfuse.flush()
