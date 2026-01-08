#!/usr/bin/env python3
"""
Gordon + Neural Memory Comparison Test

Tests Docker's Gordon AI with and without Neural Memory context injection.
Run this on a machine with Docker Desktop installed.

Usage:
    python test_gordon.py
"""

import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

# Add parent to path for neural memory
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import MemoryConfig
from src.memory.neural_memory import NeuralMemory


@dataclass
class TestResult:
    """Result of a Gordon test."""
    query: str
    response_without_memory: str
    response_with_memory: str
    time_without: float
    time_with: float
    asked_clarification_without: bool
    asked_clarification_with: bool
    surprise_before: float
    surprise_after: float


@dataclass
class TestSession:
    """Track test session metrics."""
    results: list[TestResult] = field(default_factory=list)
    preferences_learned: dict = field(default_factory=dict)


def call_gordon(query: str, context: str = "") -> tuple[str, float]:
    """
    Call Gordon via docker ai CLI.

    Args:
        query: The question to ask Gordon
        context: Optional context to prepend

    Returns:
        (response, elapsed_time)
    """
    full_query = query
    if context:
        full_query = f"Context from previous interactions:\n{context}\n\nUser question: {query}"

    try:
        start = time.time()
        result = subprocess.run(
            ["docker", "ai", full_query],
            capture_output=True,
            text=True,
            timeout=60,
        )
        elapsed = time.time() - start

        response = result.stdout.strip()
        if result.returncode != 0:
            response = f"Error: {result.stderr}"

        return response, elapsed

    except subprocess.TimeoutExpired:
        return "Timeout", 60.0
    except FileNotFoundError:
        return "Docker CLI not found. Install Docker Desktop.", 0.0


def asked_clarification(response: str) -> bool:
    """Check if Gordon asked for clarification."""
    clarification_phrases = [
        "which",
        "what",
        "could you specify",
        "please provide",
        "can you tell me",
        "do you want",
        "would you like",
        "?",  # Questions often indicate clarification needed
    ]
    response_lower = response.lower()

    # Count question indicators
    question_count = sum(1 for phrase in clarification_phrases if phrase in response_lower)

    # If multiple question indicators, likely asking for clarification
    return question_count >= 2


def extract_preferences(query: str, response: str) -> dict:
    """Extract Docker preferences from interaction."""
    preferences = {}

    # Python version preferences
    for version in ["3.11", "3.12", "3.10", "3.9"]:
        if version in query.lower() or version in response.lower():
            preferences["python_version"] = version
            break

    # Base image preferences
    if "slim" in query.lower() or "slim" in response.lower():
        preferences["base_image"] = "slim"
    elif "alpine" in query.lower() or "alpine" in response.lower():
        preferences["base_image"] = "alpine"
    elif "bookworm" in query.lower() or "bookworm" in response.lower():
        preferences["base_image"] = "bookworm"

    # Compose preferences
    if "compose" in query.lower():
        if "v2" in query.lower() or "version 2" in query.lower():
            preferences["compose_version"] = "v2"
        else:
            preferences["compose_version"] = "v3"

    return preferences


def build_memory_context(memory: NeuralMemory, preferences: dict) -> str:
    """Build context string from neural memory and preferences."""
    if not preferences:
        return ""

    context_parts = ["User's known preferences:"]

    if "python_version" in preferences:
        context_parts.append(f"- Prefers Python {preferences['python_version']}")
    if "base_image" in preferences:
        context_parts.append(f"- Prefers {preferences['base_image']} base images")
    if "compose_version" in preferences:
        context_parts.append(f"- Uses Docker Compose {preferences['compose_version']}")

    # Add memory stats
    stats = memory.get_stats()
    context_parts.append(f"\nMemory has learned from {stats['total_observations']} interactions.")

    return "\n".join(context_parts)


def run_test_scenario(
    memory: NeuralMemory,
    session: TestSession,
    query: str,
) -> TestResult:
    """Run a single test scenario."""

    # Build context from learned preferences
    context = build_memory_context(memory, session.preferences_learned)

    # Check surprise BEFORE learning
    surprise_before = memory.surprise(query)

    # Test WITHOUT memory context
    print(f"\n{'='*60}")
    print(f"Query: {query}")
    print(f"{'='*60}")

    print("\n[Without Neural Memory]")
    response_without, time_without = call_gordon(query)
    print(f"Response: {response_without[:200]}...")
    print(f"Time: {time_without:.2f}s")
    clarified_without = asked_clarification(response_without)
    print(f"Asked clarification: {clarified_without}")

    # Test WITH memory context
    print("\n[With Neural Memory]")
    response_with, time_with = call_gordon(query, context)
    print(f"Response: {response_with[:200]}...")
    print(f"Time: {time_with:.2f}s")
    clarified_with = asked_clarification(response_with)
    print(f"Asked clarification: {clarified_with}")

    # Learn from this interaction
    memory.observe(f"Query: {query}\nResponse: {response_with}")
    surprise_after = memory.surprise(query)

    # Extract and store preferences
    new_prefs = extract_preferences(query, response_with)
    session.preferences_learned.update(new_prefs)

    print(f"\nSurprise: {surprise_before:.3f} → {surprise_after:.3f}")
    print(f"Preferences learned: {session.preferences_learned}")

    result = TestResult(
        query=query,
        response_without_memory=response_without,
        response_with_memory=response_with,
        time_without=time_without,
        time_with=time_with,
        asked_clarification_without=clarified_without,
        asked_clarification_with=clarified_with,
        surprise_before=surprise_before,
        surprise_after=surprise_after,
    )

    session.results.append(result)
    return result


def print_summary(session: TestSession) -> None:
    """Print test session summary."""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    total = len(session.results)
    if total == 0:
        print("No tests run.")
        return

    clarifications_without = sum(1 for r in session.results if r.asked_clarification_without)
    clarifications_with = sum(1 for r in session.results if r.asked_clarification_with)

    avg_surprise_before = sum(r.surprise_before for r in session.results) / total
    avg_surprise_after = sum(r.surprise_after for r in session.results) / total

    print(f"""
| Metric                    | Without Memory | With Memory |
|---------------------------|----------------|-------------|
| Clarifying Questions      | {clarifications_without}/{total}            | {clarifications_with}/{total}           |
| Avg Response Time         | {sum(r.time_without for r in session.results)/total:.2f}s          | {sum(r.time_with for r in session.results)/total:.2f}s         |
| Surprise (start → end)    | N/A            | {avg_surprise_before:.2f} → {avg_surprise_after:.2f}   |

Preferences Learned: {json.dumps(session.preferences_learned, indent=2)}

Key Insight:
- Without memory: Gordon asks {clarifications_without} clarifying questions
- With memory: Gordon asks {clarifications_with} clarifying questions
- Surprise dropped from {avg_surprise_before:.2f} to {avg_surprise_after:.2f} (learning!)
""")


# Test scenarios - Docker-focused queries
TEST_QUERIES = [
    # First interaction - establishes preferences
    "Create a Dockerfile for a Python web application",

    # Second - should remember Python preference
    "Add a multi-stage build to optimize the image size",

    # Third - should remember base image preference
    "Create a docker-compose.yml for this app with Redis",

    # Fourth - test generalization
    "What's the best way to handle secrets in this setup?",

    # Fifth - should know the full context now
    "Optimize this Dockerfile for production",
]


def main():
    """Run the Gordon comparison test."""
    print("Gordon + Neural Memory Comparison Test")
    print("="*60)

    # Check Docker availability
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        print(f"Docker: {result.stdout.strip()}")
    except FileNotFoundError:
        print("ERROR: Docker CLI not found. Please install Docker Desktop.")
        return

    # Check Gordon availability
    try:
        result = subprocess.run(["docker", "ai", "--help"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print("ERROR: docker ai not available. Enable Gordon in Docker Desktop settings.")
            return
        print("Gordon: Available")
    except Exception as e:
        print(f"ERROR: Could not access Gordon: {e}")
        return

    # Initialize neural memory
    print("\nInitializing Neural Memory...")
    memory = NeuralMemory(MemoryConfig(dim=256, learning_rate=0.02))
    session = TestSession()

    print(f"Memory initialized: {sum(p.numel() for p in memory.parameters()):,} parameters")

    # Run test scenarios
    print("\nRunning test scenarios...")

    for query in TEST_QUERIES:
        run_test_scenario(memory, session, query)
        time.sleep(1)  # Rate limiting

    # Print summary
    print_summary(session)

    # Save results
    output_path = Path(__file__).parent / "gordon_test_results.json"
    with open(output_path, "w") as f:
        json.dump({
            "preferences_learned": session.preferences_learned,
            "results": [
                {
                    "query": r.query,
                    "clarification_without": r.asked_clarification_without,
                    "clarification_with": r.asked_clarification_with,
                    "surprise_before": r.surprise_before,
                    "surprise_after": r.surprise_after,
                }
                for r in session.results
            ]
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
