# Neural Memory Experiments

Standardized experiment framework for validating and benchmarking the neural memory system.

## Quick Start

```bash
# Run all experiments (in Docker)
docker compose -f docker-compose.dev.yml run --rm dev \
    python experiments/scripts/run_experiments.py --suite all

# Run specific suite
docker compose -f docker-compose.dev.yml run --rm dev \
    python experiments/scripts/run_experiments.py --suite learning

# With custom hyperparameters
docker compose -f docker-compose.dev.yml run --rm dev \
    python experiments/scripts/run_experiments.py \
        --suite all --dim 512 --lr 0.005
```

## Experiment Suites

### 1. Learning (`learning.json`)
Verifies that the memory actually learns:
- **repeat_exact**: Surprise decreases on repeated inputs
- **semantic_variation**: Learning transfers to similar content
- **domain_separation**: Unrelated domains stay high surprise
- **incremental_learning**: New learning doesn't destroy old

### 2. Retention (`retention.json`)
Verifies that memory retains learned content:
- **immediate_recall**: Low surprise immediately after learning
- **interference_resistance**: Original learning survives new learning
- **interleaved_domains**: Multiple domains retained simultaneously
- **consolidation_benefit**: Consolidation improves retention

### 3. Generalization (`generalization.json`)
Verifies that memory generalizes beyond exact matches:
- **paraphrase_recognition**: Recognizes rephrased content
- **concept_transfer**: Learning transfers to related concepts
- **abstraction_learning**: Learns abstract patterns from examples
- **negative_transfer**: Unrelated content stays high surprise

### 4. Capacity (`capacity.json`)
Stress tests memory limits:
- **scaling_test**: Performance vs observation count
- **saturation_detection**: When is memory "full"?
- **recovery_after_consolidation**: Consolidation frees capacity
- **dimension_vs_capacity**: How dim affects capacity

## Langfuse Integration

Results are automatically tracked to Langfuse if configured:

```bash
export LANGFUSE_PUBLIC_KEY="pk-..."
export LANGFUSE_SECRET_KEY="sk-..."
export LANGFUSE_HOST="https://cloud.langfuse.com"  # or your self-hosted URL
```

Disable with `--no-langfuse` flag.

## Output

Results are saved to `experiments/results/` as JSON:

```json
{
  "experiment_name": "exp_all_20240115_143052",
  "config": {"dim": 256, "learning_rate": 0.01},
  "suites": {
    "learning": [
      {"test_id": "repeat_exact", "passed": true, "metrics": {...}}
    ]
  }
}
```

## Metrics Tracked

| Metric | Description | Goal |
|--------|-------------|------|
| `surprise` | Prediction error (0-1) | Lower = more familiar |
| `weight_delta` | Learning magnitude | Higher = more learning |
| `latency_ms` | Observation time | < 100ms p99 |
| `recall_surprise` | Surprise on previously seen | < 0.3 |
| `transfer_ratio` | Learning transfer rate | > 0.5 |

## Adding New Tests

1. Add test definition to appropriate `datasets/*.json`
2. Implement handler in `scripts/run_experiments.py`
3. Run and validate results

## Hyperparameter Sweep

```bash
# Example sweep script
for dim in 128 256 512; do
    for lr in 0.001 0.01 0.1; do
        python experiments/scripts/run_experiments.py \
            --suite learning --dim $dim --lr $lr \
            --name "sweep_dim${dim}_lr${lr}"
    done
done
```

Results can be compared in Langfuse dashboard.
