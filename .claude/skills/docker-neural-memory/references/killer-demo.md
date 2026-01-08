# The Killer Demo

## Purpose

Make viewers **immediately understand** why neural memory differs from static memory. Goal: visceral "aha" moment.

## Key Demonstrations

### 1. Weights Actually Change
```python
before = memory.get_weight_hash()
memory.observe("Python uses indentation")
after = memory.get_weight_hash()
assert before != after  # Static memory: no change!
```

### 2. Pattern Recognition (Surprise Decreases)
```python
r1 = memory.observe("Python uses whitespace for structure")
r2 = memory.observe("In Python, indentation defines blocks")
r3 = memory.observe("Python relies on indentation")

# Surprise decreases: 0.87 → 0.65 → 0.42
# System learned the pattern!

# Novel question it NEVER saw:
surprise = memory.surprise("Does Python use braces?")
# Low surprise = recognizes related concept!
```

### 3. Bounded Capacity
```python
params_before = count_parameters(memory)
for i in range(1000):
    memory.observe(f"Fact {i}")
params_after = count_parameters(memory)

assert params_before == params_after  # Fixed capacity!
# Static memory: grew by 1000 rows
```

### 4. Persistence
```bash
docker run -v weights:/app/weights neural-memory
# teach it
docker stop neural-memory
docker start neural-memory
# still knows! (from weights, not database)
```

## Demo Script

```python
#!/usr/bin/env python3
"""Run: python demo/killer_demo.py"""

from neural_memory import NeuralMemory, MemoryConfig

def main():
    memory = NeuralMemory(MemoryConfig(dim=256))
    
    print("🧠 NEURAL MEMORY DEMO\n")
    
    # Demo 1: Learning
    print("1️⃣ Watch weights change:")
    before = memory.get_weight_hash()
    memory.observe("Test content")
    after = memory.get_weight_hash()
    print(f"   Before: {before} → After: {after}\n")
    
    # Demo 2: Pattern recognition
    print("2️⃣ Pattern recognition:")
    for text in ["A", "A again", "A variant"]:
        r = memory.observe(text)
        print(f"   Surprise: {r['surprise']:.3f} - '{text}'")
    
    print("\n✅ Neural memory LEARNS. Static memory just stores.")

if __name__ == "__main__":
    main()
```

## Video Script (2 min)

**[0:00]** "Every AI memory today is fake. They call it memory, but it's search."

**[0:30]** Show weights changing. "See that? It physically learned."

**[1:00]** Feed 3 examples, ask novel question. "I never taught it this. Watch."

**[1:30]** Docker restart. "Learned state survives. Docker volumes."

**[1:50]** "This is real AI memory. Docker can own this."
