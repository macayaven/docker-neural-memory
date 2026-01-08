#!/usr/bin/env python3
"""
Docker Neural Memory - Killer Demo

The goal: Make viewers IMMEDIATELY understand why neural memory
differs from static memory (RAG, vector DBs).

Run: docker compose -f docker-compose.dev.yml run --rm demo

Four demo points:
1. Weights Actually Change - proof that learning happened
2. Surprise Decreases - pattern recognition in action
3. Bounded Capacity - unlike vector DBs that grow forever
4. Persistence - knowledge survives restart
"""

import sys
import time
from pathlib import Path

# Rich for beautiful terminal output
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for prettier output")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory.neural_memory import NeuralMemory
from src.state.checkpoint import CheckpointManager
from src.config import MemoryConfig


def create_console():
    """Create console for output."""
    if RICH_AVAILABLE:
        return Console()
    return None


def print_header(console, title: str):
    """Print section header."""
    if console:
        console.print()
        console.print(Panel(title, style="bold cyan", box=box.DOUBLE))
    else:
        print(f"\n{'='*60}")
        print(f"  {title}")
        print('='*60)


def print_result(console, label: str, before: str, after: str, highlight: bool = False):
    """Print before/after comparison."""
    if console:
        style = "bold green" if highlight else ""
        console.print(f"  {label}")
        console.print(f"    Before: [dim]{before}[/dim]")
        console.print(f"    After:  [{style}]{after}[/{style}]")
    else:
        print(f"  {label}")
        print(f"    Before: {before}")
        print(f"    After:  {after}")


def demo_1_weights_change(console, memory: NeuralMemory):
    """
    DEMO POINT 1: Weights Actually Change

    This proves the system LEARNS, not just stores.
    Static memory: no weight change.
    Neural memory: weights update via gradient descent.
    """
    print_header(console, "1. WEIGHTS ACTUALLY CHANGE")

    if console:
        console.print("  [dim]Proving that learning happens at the weight level...[/dim]")
    else:
        print("  Proving that learning happens at the weight level...")

    before = memory.get_weight_hash()

    # Feed content
    memory.observe("Python uses indentation for code blocks")

    after = memory.get_weight_hash()

    print_result(console, "Weight Hash:", before, after, highlight=True)

    if console:
        if before != after:
            console.print("\n  [bold green]WEIGHTS CHANGED![/bold green] This is real learning.")
            console.print("  [dim]Static memory (RAG): weights never change.[/dim]")
        else:
            console.print("\n  [bold red]ERROR: Weights did not change![/bold red]")
    else:
        if before != after:
            print("\n  WEIGHTS CHANGED! This is real learning.")
            print("  Static memory (RAG): weights never change.")
        else:
            print("\n  ERROR: Weights did not change!")

    return before != after


def demo_2_surprise_decreases(console, memory: NeuralMemory):
    """
    DEMO POINT 2: Surprise Decreases on Repeated Patterns

    THE KILLER FEATURE: The system recognizes patterns it has seen.
    Feed similar content 3 times, surprise decreases.
    """
    print_header(console, "2. PATTERN RECOGNITION (Surprise Decreases)")

    if console:
        console.print("  [dim]Feeding similar patterns, watching surprise decrease...[/dim]\n")
    else:
        print("  Feeding similar patterns, watching surprise decrease...\n")

    patterns = [
        "Python uses whitespace for structure",
        "In Python, indentation defines code blocks",
        "Python relies on indentation instead of braces",
    ]

    surprises = []

    if RICH_AVAILABLE and console:
        table = Table(box=box.ROUNDED)
        table.add_column("#", style="dim")
        table.add_column("Content", style="cyan")
        table.add_column("Surprise", justify="right")
        table.add_column("Trend", justify="center")

    for i, pattern in enumerate(patterns):
        result = memory.observe(pattern)
        surprises.append(result["surprise"])

        trend = ""
        if i > 0:
            if surprises[i] < surprises[i-1]:
                trend = "decreasing"
            else:
                trend = "increasing"

        if RICH_AVAILABLE and console:
            trend_icon = "" if i == 0 else ("" if trend == "decreasing" else "")
            color = "green" if trend == "decreasing" else ("red" if trend == "increasing" else "white")
            table.add_row(
                str(i+1),
                pattern[:40] + "...",
                f"[{color}]{result['surprise']:.3f}[/{color}]",
                trend_icon
            )
        else:
            print(f"  {i+1}. Surprise: {result['surprise']:.3f} - {pattern[:40]}...")

    if RICH_AVAILABLE and console:
        console.print(table)

    # Summary
    decrease = (surprises[0] - surprises[-1]) / surprises[0] * 100

    if console:
        console.print()
        if decrease > 20:
            console.print(f"  [bold green]Surprise decreased {decrease:.0f}%![/bold green]")
            console.print("  [dim]The system LEARNED the pattern. RAG would show 3 identical results.[/dim]")
        else:
            console.print(f"  [yellow]Surprise only decreased {decrease:.0f}%[/yellow]")
    else:
        print(f"\n  Surprise decreased {decrease:.0f}%!")
        if decrease > 20:
            print("  The system LEARNED the pattern. RAG would show 3 identical results.")

    return decrease > 20


def demo_3_bounded_capacity(console, memory: NeuralMemory):
    """
    DEMO POINT 3: Bounded Capacity

    Unlike vector DBs that grow with every insert,
    neural memory has FIXED capacity. It compresses, not stores.
    """
    print_header(console, "3. BOUNDED CAPACITY (Memory Doesn't Grow)")

    if console:
        console.print("  [dim]Feeding 500 facts, checking parameter count...[/dim]\n")
    else:
        print("  Feeding 500 facts, checking parameter count...\n")

    params_before = sum(p.numel() for p in memory.parameters())

    # Feed many observations
    num_observations = 500

    if RICH_AVAILABLE and console:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Learning facts...", total=num_observations)
            for i in range(num_observations):
                memory.observe(f"Fact {i}: The number {i} is {'even' if i % 2 == 0 else 'odd'}")
                progress.advance(task)
    else:
        for i in range(num_observations):
            memory.observe(f"Fact {i}: The number {i} is {'even' if i % 2 == 0 else 'odd'}")
            if i % 100 == 0:
                print(f"  ... fed {i} facts")

    params_after = sum(p.numel() for p in memory.parameters())

    if console:
        console.print()
        console.print(f"  Parameters before: [dim]{params_before:,}[/dim]")
        console.print(f"  Parameters after:  [dim]{params_after:,}[/dim]")
        console.print()

        if params_before == params_after:
            console.print(f"  [bold green]FIXED CAPACITY![/bold green] {num_observations} observations, same size.")
            console.print("  [dim]Vector DB: would have grown by {num_observations} rows.[/dim]")
        else:
            console.print("  [bold red]ERROR: Parameters changed![/bold red]")
    else:
        print(f"\n  Parameters before: {params_before:,}")
        print(f"  Parameters after:  {params_after:,}")
        if params_before == params_after:
            print(f"\n  FIXED CAPACITY! {num_observations} observations, same size.")
            print(f"  Vector DB: would have grown by {num_observations} rows.")

    return params_before == params_after


def demo_4_persistence(console, checkpoint_dir: Path):
    """
    DEMO POINT 4: Persistence Across Restart

    Learned state survives process restart.
    Just like Docker volumes preserve container state.
    """
    print_header(console, "4. PERSISTENCE (Survives Restart)")

    if console:
        console.print("  [dim]Simulating container restart...[/dim]\n")
    else:
        print("  Simulating container restart...\n")

    # First "container" run
    memory1 = NeuralMemory(MemoryConfig(dim=256))
    manager1 = CheckpointManager(checkpoint_dir=checkpoint_dir)

    # Learn secret knowledge
    secret = "The secret code is ALPHA-BRAVO-CHARLIE-42"
    memory1.observe(secret)
    surprise_before = memory1.surprise(secret)

    manager1.checkpoint(memory1, tag="demo-state", description="Demo checkpoint")

    if console:
        console.print("  [dim]First container:[/dim]")
        console.print(f"    Learned: \"{secret[:30]}...\"")
        console.print(f"    Surprise: {surprise_before:.3f}")
        console.print("    Checkpointed to volume")
        console.print()
        console.print("  [yellow]--- CONTAINER RESTART ---[/yellow]")
        console.print()
    else:
        print(f"  First container:")
        print(f"    Learned: \"{secret[:30]}...\"")
        print(f"    Surprise: {surprise_before:.3f}")
        print("    Checkpointed to volume")
        print("\n  --- CONTAINER RESTART ---\n")

    # Simulate restart
    del memory1
    del manager1
    time.sleep(0.5)

    # Second "container" run
    memory2 = NeuralMemory(MemoryConfig(dim=256))
    manager2 = CheckpointManager(checkpoint_dir=checkpoint_dir)

    manager2.restore(memory2, tag="demo-state")
    surprise_after = memory2.surprise(secret)

    if console:
        console.print("  [dim]Second container (after restart):[/dim]")
        console.print(f"    Restored from checkpoint")
        console.print(f"    Surprise for same content: {surprise_after:.3f}")
        console.print()

        if abs(surprise_before - surprise_after) < 0.05:
            console.print("  [bold green]KNOWLEDGE PERSISTED![/bold green]")
            console.print("  [dim]Learned state survived restart via Docker volume.[/dim]")
        else:
            console.print("  [bold red]ERROR: Knowledge was lost![/bold red]")
    else:
        print("  Second container (after restart):")
        print(f"    Restored from checkpoint")
        print(f"    Surprise for same content: {surprise_after:.3f}")
        if abs(surprise_before - surprise_after) < 0.05:
            print("\n  KNOWLEDGE PERSISTED!")
            print("  Learned state survived restart via Docker volume.")

    return abs(surprise_before - surprise_after) < 0.05


def main():
    """Run the killer demo."""
    console = create_console()

    # Header
    if console:
        console.print()
        console.print(Panel.fit(
            "[bold cyan]DOCKER NEURAL MEMORY[/bold cyan]\n"
            "[dim]Memory that LEARNS, not just stores[/dim]",
            box=box.DOUBLE,
            padding=(1, 4)
        ))
    else:
        print("\n" + "="*60)
        print("  DOCKER NEURAL MEMORY")
        print("  Memory that LEARNS, not just stores")
        print("="*60)

    # Setup
    config = MemoryConfig(dim=256, learning_rate=0.02)
    memory = NeuralMemory(config)
    checkpoint_dir = Path("/app/weights")
    checkpoint_dir.mkdir(exist_ok=True)

    results = []

    # Run demos
    results.append(("Weights Change", demo_1_weights_change(console, memory)))
    results.append(("Pattern Recognition", demo_2_surprise_decreases(console, memory)))
    results.append(("Bounded Capacity", demo_3_bounded_capacity(console, memory)))
    results.append(("Persistence", demo_4_persistence(console, checkpoint_dir)))

    # Summary
    if console:
        console.print()
        console.print(Panel.fit(
            "[bold]SUMMARY[/bold]",
            box=box.DOUBLE
        ))

        for name, passed in results:
            icon = "" if passed else ""
            color = "green" if passed else "red"
            console.print(f"  [{color}]{icon}[/{color}] {name}")

        console.print()
        all_passed = all(p for _, p in results)
        if all_passed:
            console.print(Panel(
                "[bold green]This is REAL AI memory.[/bold green]\n"
                "[dim]It learns. It persists. It doesn't grow forever.\n"
                "RAG/Vector DBs can't do this.[/dim]",
                box=box.ROUNDED
            ))
        else:
            console.print("[bold red]Some demos failed - check implementation[/bold red]")
    else:
        print("\n" + "="*60)
        print("  SUMMARY")
        print("="*60)
        for name, passed in results:
            icon = "PASS" if passed else "FAIL"
            print(f"  [{icon}] {name}")
        print()
        if all(p for _, p in results):
            print("  This is REAL AI memory.")
            print("  It learns. It persists. It doesn't grow forever.")
            print("  RAG/Vector DBs can't do this.")

    return 0 if all(p for _, p in results) else 1


if __name__ == "__main__":
    sys.exit(main())
