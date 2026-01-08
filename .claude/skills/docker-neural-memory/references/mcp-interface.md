# MCP Interface Specification

Complete specification for Docker Neural Memory MCP tools.

## Design Philosophy

Traditional memory MCP tools map to CRUD operations:
```
store()  → CREATE
query()  → READ
update() → UPDATE
delete() → DELETE
```

Neural memory tools map to LEARNING operations:
```
observe()     → LEARN (weights update automatically)
infer()       → GENERATE (from learned model)
surprise()    → MEASURE (novelty detection)
consolidate() → COMPRESS (pattern stabilization)
```

**No explicit storage.** The system learns what matters.

## Tool Definitions

### Learning Operations

#### observe

Feed context to the memory. Weights update automatically.

```typescript
interface ObserveInput {
  // Content to learn from
  context: string;
  
  // Optional: Override default learning rate
  learning_rate?: number;
  
  // Optional: Domain tag for routing (in multi-memory setups)
  domain?: string;
  
  // Optional: Encoding to use for text
  encoding?: "utf-8" | "base64";
}

interface ObserveOutput {
  // How novel was this input? (0-1, higher = more surprising)
  surprise: number;
  
  // Magnitude of weight change
  weight_delta: number;
  
  // Patterns that were activated/strengthened
  patterns_activated: string[];
  
  // Whether learning occurred (might skip if surprise too low)
  learned: boolean;
  
  // Current memory utilization estimate
  capacity_used: number;
}
```

**Example usage:**
```json
{
  "tool": "observe",
  "input": {
    "context": "In Python, always use context managers (with statement) for resource handling. This ensures proper cleanup even if exceptions occur.",
    "domain": "python"
  }
}
```

**Response:**
```json
{
  "surprise": 0.73,
  "weight_delta": 0.0042,
  "patterns_activated": ["resource_management", "exception_handling"],
  "learned": true,
  "capacity_used": 0.23
}
```

#### infer

Query the memory using learned representations. Unlike traditional query, this GENERATES from the learned model rather than retrieving stored content.

```typescript
interface InferInput {
  // What to infer about
  prompt: string;
  
  // Optional: Generation temperature (0-1)
  temperature?: number;
  
  // Optional: Maximum response tokens
  max_tokens?: number;
  
  // Optional: Domain to query
  domain?: string;
  
  // Optional: Return attention weights for interpretability
  return_attention?: boolean;
}

interface InferOutput {
  // Generated response from learned patterns
  response: string;
  
  // Model's confidence in the response (0-1)
  confidence: number;
  
  // Optional: What memory attended to
  attention_weights?: Array<{
    pattern: string;
    weight: number;
  }>;
  
  // How many patterns contributed
  patterns_used: number;
}
```

**Example:**
```json
{
  "tool": "infer",
  "input": {
    "prompt": "How should I handle database connections in Python?",
    "return_attention": true
  }
}
```

**Response:**
```json
{
  "response": "Use context managers for database connections to ensure proper cleanup...",
  "confidence": 0.82,
  "attention_weights": [
    {"pattern": "resource_management", "weight": 0.67},
    {"pattern": "exception_handling", "weight": 0.21}
  ],
  "patterns_used": 3
}
```

#### surprise

Measure how novel/surprising an input is without learning from it.

```typescript
interface SurpriseInput {
  // Content to evaluate
  input: string;
  
  // Optional: Domain to check against
  domain?: string;
}

interface SurpriseOutput {
  // Surprise score (0-1)
  score: number;
  
  // Most similar learned pattern
  nearest_pattern: string;
  
  // Recommended action
  recommendation: "learn" | "skip" | "consolidate";
  
  // Why this recommendation
  reason: string;
}
```

**Recommendations:**
- `learn`: High surprise, worth updating weights
- `skip`: Low surprise, already known
- `consolidate`: Memory near capacity or unstable

#### consolidate

Trigger memory consolidation (like sleep). Compresses recent learning into stable patterns.

```typescript
interface ConsolidateInput {
  // Optional: Number of consolidation steps
  steps?: number;
  
  // Optional: EWC regularization strength
  ewc_lambda?: number;
}

interface ConsolidateOutput {
  // Number of patterns merged/compressed
  patterns_merged: number;
  
  // Memory size reduction percentage
  memory_compressed_by: number;
  
  // Stability score after consolidation (0-1)
  stability_score: number;
  
  // Duration of consolidation
  duration_ms: number;
}
```

### State Management

#### checkpoint

Save current learned state as a named checkpoint.

```typescript
interface CheckpointInput {
  // Tag name (e.g., "v1.0", "pre-experiment")
  tag: string;
  
  // Optional: Description
  description?: string;
  
  // Optional: Include momentum buffers
  include_momentum?: boolean;
}

interface CheckpointOutput {
  // Unique checkpoint ID
  checkpoint_id: string;
  
  // Size on disk
  size_mb: number;
  
  // Hash for integrity verification
  weight_hash: string;
  
  // Full path in Docker volume
  path: string;
}
```

#### restore

Restore memory to a previous checkpoint.

```typescript
interface RestoreInput {
  // Tag to restore
  tag: string;
  
  // Optional: Also restore momentum buffers
  restore_momentum?: boolean;
}

interface RestoreOutput {
  // Success status
  restored: boolean;
  
  // Hash of restored weights
  weight_hash: string;
  
  // Learning steps since this checkpoint was created
  learning_since_checkpoint: number;
  
  // Warning if significant learning will be lost
  warning?: string;
}
```

#### fork

Fork memory state into a new branch.

```typescript
interface ForkInput {
  // Source checkpoint tag
  source_tag: string;
  
  // New tag for the fork
  new_tag: string;
  
  // Optional: Description for the fork
  description?: string;
}

interface ForkOutput {
  // Success status
  forked: boolean;
  
  // Source checkpoint hash
  source_hash: string;
  
  // New checkpoint hash (same as source initially)
  new_hash: string;
  
  // Path to new checkpoint
  path: string;
}
```

#### list_checkpoints

List all available checkpoints.

```typescript
interface ListCheckpointsInput {
  // Optional: Filter by tag prefix
  prefix?: string;
  
  // Optional: Limit results
  limit?: number;
}

interface ListCheckpointsOutput {
  checkpoints: Array<{
    tag: string;
    checkpoint_id: string;
    created_at: string;  // ISO datetime
    size_mb: number;
    description?: string;
    weight_hash: string;
  }>;
  
  total_count: number;
  total_size_mb: number;
}
```

### Introspection

#### stats

Get memory statistics.

```typescript
interface StatsInput {
  // Optional: Include detailed breakdown
  detailed?: boolean;
}

interface StatsOutput {
  // Total observations processed
  total_observations: number;
  
  // Number of weight parameters
  weight_parameters: number;
  
  // Estimated capacity utilization (0-1)
  capacity_used: number;
  
  // Average recent surprise
  avg_surprise: number;
  
  // Known domains
  domains: string[];
  
  // Optional detailed stats
  details?: {
    learning_rate: number;
    momentum: number;
    last_consolidation: string;
    patterns_learned: number;
  };
}
```

#### attention_map

Visualize what memory attends to for a query.

```typescript
interface AttentionMapInput {
  // Query to analyze
  query: string;
  
  // Optional: Number of top patterns
  top_k?: number;
}

interface AttentionMapOutput {
  // Attention distribution
  attention_weights: Array<{
    pattern: string;
    weight: number;
    examples: string[];  // What triggered this pattern
  }>;
  
  // Optional: URL to rendered visualization
  visualization_url?: string;
}
```

#### explain

Export learned patterns as interpretable summaries.

```typescript
interface ExplainInput {
  // Number of top patterns to explain
  top_k?: number;
  
  // Optional: Explain patterns in a domain
  domain?: string;
}

interface ExplainOutput {
  patterns: Array<{
    // Human-readable description
    description: string;
    
    // Pattern strength (how strongly learned)
    strength: number;
    
    // Examples that triggered this pattern
    examples: string[];
    
    // When pattern was learned
    learned_at: string;
    
    // Times pattern was activated
    activation_count: number;
  }>;
}
```

## MCP Server Implementation

```python
from mcp.server import Server
from mcp.types import Tool, TextContent
import json

app = Server("neural-memory")

# Initialize memory module
memory = NeuralMemory(dim=512)
state_manager = StateManager("/app/weights", "/app/checkpoints")


@app.tool()
async def observe(context: str, learning_rate: float = None, domain: str = None) -> dict:
    """Feed context to neural memory. Weights update automatically."""
    
    # Encode text to tensor
    x = encode_text(context)
    
    # Override learning rate if specified
    if learning_rate:
        original_lr = memory.lr.data.clone()
        memory.lr.data = torch.tensor(learning_rate)
    
    # Forward pass with learning
    output, metrics = memory(x, learn=True, return_metrics=True)
    
    # Restore learning rate
    if learning_rate:
        memory.lr.data = original_lr
    
    return {
        "surprise": metrics["surprise"],
        "weight_delta": metrics["weight_delta"],
        "patterns_activated": extract_patterns(output),
        "learned": metrics["surprise"] > 0.1,
        "capacity_used": estimate_capacity(memory)
    }


@app.tool()
async def infer(prompt: str, temperature: float = 0.7, return_attention: bool = False) -> dict:
    """Generate response from learned patterns."""
    
    x = encode_text(prompt)
    
    # Inference without learning
    with torch.no_grad():
        output = memory.memory_net(x)
        confidence = compute_confidence(output)
    
    # Decode to text
    response = decode_output(output, temperature)
    
    result = {
        "response": response,
        "confidence": confidence,
        "patterns_used": count_active_patterns(output)
    }
    
    if return_attention:
        result["attention_weights"] = compute_attention_weights(x, output)
    
    return result


@app.tool()
async def surprise(input: str) -> dict:
    """Measure novelty without learning."""
    
    x = encode_text(input)
    score = memory.compute_surprise(x)
    
    # Determine recommendation
    if score > 0.7:
        recommendation = "learn"
        reason = "High novelty, worth learning"
    elif score < 0.2:
        recommendation = "skip"
        reason = "Already well-known pattern"
    elif estimate_capacity(memory) > 0.9:
        recommendation = "consolidate"
        reason = "Memory near capacity"
    else:
        recommendation = "learn"
        reason = "Moderate novelty"
    
    return {
        "score": score,
        "nearest_pattern": find_nearest_pattern(x),
        "recommendation": recommendation,
        "reason": reason
    }


@app.tool()
async def consolidate(steps: int = 100, ewc_lambda: float = 1000.0) -> dict:
    """Compress recent learning into stable patterns."""
    
    consolidator = MemoryConsolidator(memory, ewc_lambda)
    
    # Get recent observations for replay
    replay_data = state_manager.get_recent_observations()
    
    import time
    start = time.time()
    metrics = consolidator.consolidate(replay_data, steps)
    duration = (time.time() - start) * 1000
    
    return {
        "patterns_merged": estimate_patterns_merged(metrics),
        "memory_compressed_by": 1 - (metrics["final_surprise"] / metrics["initial_surprise"]),
        "stability_score": compute_stability(memory),
        "duration_ms": duration
    }


@app.tool()
async def checkpoint(tag: str, description: str = None) -> dict:
    """Save current learned state."""
    
    return state_manager.save_checkpoint(
        memory=memory,
        tag=tag,
        description=description
    )


@app.tool()
async def restore(tag: str) -> dict:
    """Restore memory to a previous checkpoint."""
    
    return state_manager.restore_checkpoint(
        memory=memory,
        tag=tag
    )


@app.tool()
async def fork(source_tag: str, new_tag: str, description: str = None) -> dict:
    """Fork memory state into a new branch."""
    
    return state_manager.fork_checkpoint(
        source_tag=source_tag,
        new_tag=new_tag,
        description=description
    )


@app.tool()
async def list_checkpoints(prefix: str = None, limit: int = 100) -> dict:
    """List available checkpoints."""
    
    return state_manager.list_checkpoints(prefix=prefix, limit=limit)


@app.tool()
async def stats(detailed: bool = False) -> dict:
    """Get memory statistics."""
    
    return {
        "total_observations": state_manager.observation_count,
        "weight_parameters": sum(p.numel() for p in memory.memory_net.parameters()),
        "capacity_used": estimate_capacity(memory),
        "avg_surprise": state_manager.recent_surprise_avg(),
        "domains": state_manager.known_domains(),
        **({"details": get_detailed_stats(memory)} if detailed else {})
    }


if __name__ == "__main__":
    app.run()
```

## Claude Desktop Configuration

```json
{
  "mcpServers": {
    "neural-memory": {
      "type": "http",
      "url": "http://localhost:8765/mcp"
    }
  }
}
```

Or for stdio mode:
```json
{
  "mcpServers": {
    "neural-memory": {
      "command": "docker",
      "args": ["exec", "-i", "neural-memory", "python", "-m", "mcp_server"]
    }
  }
}
```

## Error Handling

All tools should handle these error cases:

```typescript
interface MCPError {
  code: string;
  message: string;
  details?: any;
}

// Common error codes
const ERRORS = {
  MEMORY_SATURATED: "Memory at capacity, consolidate before learning",
  CHECKPOINT_NOT_FOUND: "Checkpoint tag does not exist",
  INVALID_INPUT: "Input could not be encoded",
  LEARNING_FAILED: "Weight update failed (gradient issue)",
  STATE_CORRUPTED: "Checkpoint file corrupted"
};
```
