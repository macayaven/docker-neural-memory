#!/usr/bin/env python3
"""
Interactive Gordon + Neural Memory Test

Run this to interactively test Gordon with neural memory.
Each query builds on the previous, showing learning in action.

Usage:
    python interactive_test.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import MemoryConfig
from src.memory.neural_memory import NeuralMemory


def main():
    print("="*60)
    print("Gordon + Neural Memory Interactive Test")
    print("="*60)

    # Initialize
    memory = NeuralMemory(MemoryConfig(dim=256, learning_rate=0.02))
    preferences = {}

    print(f"""
Neural Memory initialized: {sum(p.numel() for p in memory.parameters()):,} parameters

Instructions:
1. I'll suggest queries to test with Gordon
2. Run them in your terminal: docker ai "<query>"
3. Tell me what Gordon responded
4. I'll learn from the interaction and build context

Type 'quit' to exit, 'stats' for memory stats, 'context' to see learned context.
""")

    suggested_queries = [
        "Create a Dockerfile for a Python web application",
        "I want to use Python 3.11 with a slim base image",
        "Add a multi-stage build to this Dockerfile",
        "Create a docker-compose.yml with Redis for caching",
        "How should I handle environment variables securely?",
    ]

    query_idx = 0

    while True:
        print("\n" + "-"*40)

        if query_idx < len(suggested_queries):
            suggested = suggested_queries[query_idx]
            print(f"Suggested query #{query_idx + 1}:")
            print(f"  docker ai \"{suggested}\"")
            print()

        user_input = input("Your input (or paste Gordon's response): ").strip()

        if user_input.lower() == 'quit':
            break

        if user_input.lower() == 'stats':
            stats = memory.get_stats()
            print(f"""
Memory Stats:
- Observations: {stats['total_observations']}
- Avg Surprise: {stats['avg_surprise']:.3f}
- Parameters: {stats['weight_parameters']:,}
- Current hash: {memory.get_weight_hash()}
""")
            continue

        if user_input.lower() == 'context':
            print("\nLearned preferences:")
            for k, v in preferences.items():
                print(f"  - {k}: {v}")
            print("\nContext for Gordon:")
            print(build_context(preferences))
            continue

        if not user_input:
            continue

        # Check surprise before learning
        surprise_before = memory.surprise(user_input)

        # Learn from the input
        result = memory.observe(user_input)

        # Extract preferences
        new_prefs = extract_preferences(user_input)
        preferences.update(new_prefs)

        print(f"""
Learned!
- Surprise: {surprise_before:.3f} → {result['surprise']:.3f}
- Weight delta: {result['weight_delta']:.6f}
- New preferences: {new_prefs if new_prefs else 'None detected'}
""")

        if new_prefs:
            print("Updated context for next Gordon query:")
            print(build_context(preferences))

        query_idx += 1

    # Final stats
    print("\n" + "="*60)
    print("Session Summary")
    print("="*60)
    stats = memory.get_stats()
    print(f"""
Total observations: {stats['total_observations']}
Final avg surprise: {stats['avg_surprise']:.3f}
Preferences learned: {preferences}

The more you use it, the lower the surprise → Gordon needs less clarification!
""")


def extract_preferences(text: str) -> dict:
    """Extract Docker preferences from text."""
    prefs = {}
    text_lower = text.lower()

    # Python versions
    for v in ["3.12", "3.11", "3.10", "3.9"]:
        if v in text_lower:
            prefs["python_version"] = v
            break

    # Base images
    if "slim" in text_lower:
        prefs["base_image"] = "slim"
    elif "alpine" in text_lower:
        prefs["base_image"] = "alpine"
    elif "bookworm" in text_lower:
        prefs["base_image"] = "bookworm"

    # Other patterns
    if "multi-stage" in text_lower or "multistage" in text_lower:
        prefs["uses_multistage"] = True
    if "redis" in text_lower:
        prefs["uses_redis"] = True
    if "compose" in text_lower:
        prefs["uses_compose"] = True

    return prefs


def build_context(preferences: dict) -> str:
    """Build context string for Gordon."""
    if not preferences:
        return "(No preferences learned yet)"

    parts = ["User's known Docker preferences:"]

    if "python_version" in preferences:
        parts.append(f"- Uses Python {preferences['python_version']}")
    if "base_image" in preferences:
        parts.append(f"- Prefers {preferences['base_image']} base images")
    if preferences.get("uses_multistage"):
        parts.append("- Uses multi-stage builds")
    if preferences.get("uses_redis"):
        parts.append("- Uses Redis for caching")
    if preferences.get("uses_compose"):
        parts.append("- Uses Docker Compose")

    parts.append("\nPlease apply these preferences to your response.")

    return "\n".join(parts)


if __name__ == "__main__":
    main()
