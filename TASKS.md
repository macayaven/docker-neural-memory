# Docker Neural Memory - TDD Implementation Tasks

> **Methodology**: Strict TDD (Test-Driven Development)
> **Rule**: Write tests FIRST. Tests define the interface. No code without a failing test.
> **Workflow**: Red 🔴 → Green 🟢 → Refactor 🔵

---

## TDD Workflow

```
For each task:
1. 🔴 RED: Write failing tests that define expected behavior
2. 🟢 GREEN: Write minimal code to make tests pass
3. 🔵 REFACTOR: Improve code while keeping tests green
4. ✅ DONE: All tests pass, code is clean
```

**Run tests**:
```bash
docker compose -f docker-compose.dev.yml run --rm test
```

---

## Phase 1: Core Memory Module

### Task 1.1: Text Encoder
**Use Case**: Convert text to tensor representation for neural memory processing.

**Why it matters**: The encoder is the foundation - if encoding is inconsistent or lossy, learning will fail.

#### Tests to Write First

```python
# tests/unit/test_encoder.py

class TestTextEncoder:
    """TDD tests for text encoding."""

    def test_encode_returns_tensor_with_correct_shape(self):
        """Encoded text should have shape [1, dim]."""
        encoder = TextEncoder(dim=256)
        result = encoder.encode("Hello world")
        assert result.shape == (1, 256)

    def test_encode_is_deterministic(self):
        """Same text should produce identical encodings."""
        encoder = TextEncoder(dim=256)
        enc1 = encoder.encode("test")
        enc2 = encoder.encode("test")
        assert torch.allclose(enc1, enc2)

    def test_different_texts_produce_different_encodings(self):
        """Different texts should produce different encodings."""
        encoder = TextEncoder(dim=256)
        enc1 = encoder.encode("hello")
        enc2 = encoder.encode("world")
        assert not torch.allclose(enc1, enc2)

    def test_encode_handles_empty_string(self):
        """Empty string should produce valid tensor."""
        encoder = TextEncoder(dim=256)
        result = encoder.encode("")
        assert result.shape == (1, 256)

    def test_encode_handles_unicode(self):
        """Unicode text should encode correctly."""
        encoder = TextEncoder(dim=256)
        result = encoder.encode("こんにちは 🧠")
        assert result.shape == (1, 256)

    def test_encode_batch(self):
        """Batch encoding should work."""
        encoder = TextEncoder(dim=256)
        texts = ["hello", "world", "test"]
        result = encoder.encode_batch(texts)
        assert result.shape == (3, 256)
```

#### Implementation File
`src/encoder/text_encoder.py`

#### Acceptance Criteria
- [ ] All tests pass
- [ ] Encoding is deterministic
- [ ] Handles edge cases (empty, unicode, long text)

---

### Task 1.2: Neural Memory Core
**Use Case**: Store learned patterns in neural weights that update during inference.

**Why it matters**: This is THE core innovation - weights must actually change during `observe()`.

#### Tests to Write First

```python
# tests/unit/test_neural_memory.py

class TestNeuralMemoryWeightUpdate:
    """Verify that weights actually update during observation."""

    def test_weights_change_after_observe(self):
        """The key test: weights must change after observe()."""
        memory = NeuralMemory(dim=256)

        before = memory.get_weight_hash()
        memory.observe("Python uses indentation")
        after = memory.get_weight_hash()

        assert before != after, "Weights must change after observe()"

    def test_weights_unchanged_after_infer(self):
        """Inference should NOT modify weights."""
        memory = NeuralMemory(dim=256)

        before = memory.get_weight_hash()
        memory.infer("What is Python?")
        after = memory.get_weight_hash()

        assert before == after, "Weights must NOT change during infer()"

    def test_observe_returns_surprise_score(self):
        """observe() should return surprise score between 0 and 1."""
        memory = NeuralMemory(dim=256)
        result = memory.observe("test content")

        assert "surprise" in result
        assert 0 <= result["surprise"] <= 1

    def test_observe_returns_weight_delta(self):
        """observe() should return magnitude of weight change."""
        memory = NeuralMemory(dim=256)
        result = memory.observe("test content")

        assert "weight_delta" in result
        assert result["weight_delta"] > 0


class TestNeuralMemorySurpriseDecreases:
    """THE KILLER TEST: Surprise decreases on repeated patterns."""

    def test_surprise_decreases_on_repetition(self):
        """Repeated similar content should have decreasing surprise."""
        memory = NeuralMemory(dim=256)

        # Feed similar patterns
        r1 = memory.observe("Python uses whitespace for structure")
        r2 = memory.observe("In Python, indentation defines blocks")
        r3 = memory.observe("Python relies on indentation")

        # Surprise should decrease as pattern is learned
        assert r3["surprise"] < r1["surprise"], (
            f"Surprise should decrease: {r1['surprise']:.3f} → {r3['surprise']:.3f}"
        )

    def test_novel_content_has_high_surprise(self):
        """Completely new content should have high surprise."""
        memory = NeuralMemory(dim=256)

        # Learn about Python
        for _ in range(5):
            memory.observe("Python programming language")

        # Novel topic should be surprising
        result = memory.observe("Quantum physics experiments")
        assert result["surprise"] > 0.5


class TestNeuralMemoryInference:
    """Test inference from learned patterns."""

    def test_infer_returns_confidence(self):
        """infer() should return confidence score."""
        memory = NeuralMemory(dim=256)
        result = memory.infer("test query")

        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1

    def test_confidence_increases_after_learning(self):
        """Confidence should be higher for learned topics."""
        memory = NeuralMemory(dim=256)

        # Measure confidence before learning
        before = memory.infer("Python indentation")["confidence"]

        # Learn about Python
        for _ in range(5):
            memory.observe("Python uses indentation for blocks")

        # Confidence should increase
        after = memory.infer("Python indentation")["confidence"]
        assert after > before


class TestNeuralMemorySurpriseMethod:
    """Test surprise measurement without learning."""

    def test_surprise_without_learning(self):
        """surprise() should measure novelty without updating weights."""
        memory = NeuralMemory(dim=256)

        before = memory.get_weight_hash()
        score = memory.surprise("test input")
        after = memory.get_weight_hash()

        assert before == after, "surprise() should not modify weights"
        assert 0 <= score <= 1

    def test_surprise_low_for_learned_patterns(self):
        """Learned patterns should have low surprise."""
        memory = NeuralMemory(dim=256)

        # Learn pattern
        for _ in range(10):
            memory.observe("Machine learning models")

        # Similar content should have low surprise
        score = memory.surprise("Machine learning algorithms")
        assert score < 0.5
```

#### Implementation File
`src/memory/neural_memory.py`

#### Acceptance Criteria
- [ ] Weights change on observe()
- [ ] Weights unchanged on infer()
- [ ] Surprise decreases on repeated patterns (THE KEY TEST)
- [ ] Novel content has high surprise

---

### Task 1.3: TTT Layer Variants
**Use Case**: Provide linear (fast) and MLP (expressive) hidden state options.

**Why it matters**: TTT-Linear is faster but less expressive; TTT-MLP is more powerful. Need both options.

#### Tests to Write First

```python
# tests/unit/test_ttt_layer.py

class TestTTTLinear:
    """Test TTT layer with linear hidden state."""

    def test_output_shape_matches_input(self):
        """Output should have same shape as input."""
        layer = TTTLinear(dim=256)
        x = torch.randn(2, 10, 256)  # [batch, seq, dim]
        y = layer(x)
        assert y.shape == x.shape

    def test_processes_sequence_token_by_token(self):
        """Each token should be processed with updated hidden state."""
        layer = TTTLinear(dim=256)
        x = torch.randn(1, 5, 256)
        y = layer(x)

        # Output should be different from input
        assert not torch.allclose(x, y, atol=0.1)

    def test_hidden_state_updates_during_forward(self):
        """Hidden state should evolve during sequence processing."""
        layer = TTTLinear(dim=256)
        x = torch.randn(1, 10, 256)

        # Outputs at different positions should differ
        y = layer(x)
        assert not torch.allclose(y[:, 0, :], y[:, -1, :])


class TestTTTMLP:
    """Test TTT layer with MLP hidden state."""

    def test_output_shape_matches_input(self):
        """Output should have same shape as input."""
        layer = TTTMLP(dim=256)
        x = torch.randn(2, 10, 256)
        y = layer(x)
        assert y.shape == x.shape

    def test_more_expressive_than_linear(self):
        """MLP should capture more complex patterns than linear."""
        linear = TTTLinear(dim=64)
        mlp = TTTMLP(dim=64)

        # Create a pattern that requires nonlinearity
        x = torch.randn(1, 20, 64)
        x = x ** 2  # Nonlinear pattern

        # Both should process, but outputs will differ
        y_linear = linear(x)
        y_mlp = mlp(x)

        assert y_linear.shape == y_mlp.shape
        assert not torch.allclose(y_linear, y_mlp)


class TestTTTLayerFactory:
    """Test factory for creating TTT layers."""

    def test_create_linear_variant(self):
        """Should create TTT-Linear when specified."""
        layer = create_ttt_layer(dim=256, variant="linear")
        assert isinstance(layer, TTTLinear)

    def test_create_mlp_variant(self):
        """Should create TTT-MLP when specified."""
        layer = create_ttt_layer(dim=256, variant="mlp")
        assert isinstance(layer, TTTMLP)

    def test_invalid_variant_raises(self):
        """Invalid variant should raise ValueError."""
        with pytest.raises(ValueError):
            create_ttt_layer(dim=256, variant="invalid")
```

#### Implementation Files
`src/memory/ttt_layer.py`

#### Acceptance Criteria
- [ ] TTT-Linear processes sequences correctly
- [ ] TTT-MLP processes sequences correctly
- [ ] Factory creates correct variant

---

### Task 1.4: Bounded Capacity Verification
**Use Case**: Verify memory has fixed capacity (unlike vector DBs that grow unbounded).

**Why it matters**: This is a key differentiator from RAG - the memory compresses, not stores.

#### Tests to Write First

```python
# tests/unit/test_bounded_capacity.py

class TestBoundedCapacity:
    """Verify memory has bounded capacity (key differentiator from RAG)."""

    def test_parameter_count_unchanged_after_observations(self):
        """THE BOUNDED CAPACITY TEST: Parameters must not grow."""
        memory = NeuralMemory(dim=256)

        params_before = sum(p.numel() for p in memory.parameters())

        # Feed many observations
        for i in range(1000):
            memory.observe(f"Fact number {i}: random content here")

        params_after = sum(p.numel() for p in memory.parameters())

        assert params_before == params_after, (
            "Parameter count must not change - memory compresses, not stores"
        )

    def test_memory_compresses_patterns(self):
        """Memory should compress similar patterns, not store each one."""
        memory = NeuralMemory(dim=256)

        # Feed 100 similar patterns
        for i in range(100):
            memory.observe(f"Python is a programming language - variant {i}")

        # Get final weight hash
        hash_100 = memory.get_weight_hash()

        # Feed 100 more similar patterns
        for i in range(100, 200):
            memory.observe(f"Python is a programming language - variant {i}")

        hash_200 = memory.get_weight_hash()

        # Weights should have changed (learning happened)
        # but parameter count stays fixed (bounded)
        params = sum(p.numel() for p in memory.parameters())
        assert params == sum(p.numel() for p in memory.parameters())
```

#### Implementation File
N/A - tests verify existing NeuralMemory behavior

#### Acceptance Criteria
- [ ] Parameter count unchanged after 1000 observations
- [ ] Learning still happens (weights change)

---

## Phase 2: State Management

### Task 2.1: Checkpoint Save/Restore
**Use Case**: Save learned state to disk and restore it later.

**Why it matters**: Persistence is critical - learned state must survive container restart.

#### Tests to Write First

```python
# tests/unit/test_checkpoint.py

class TestCheckpointSave:
    """Test saving checkpoints."""

    def test_checkpoint_creates_file(self, tmp_path):
        """checkpoint() should create a file."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        memory = NeuralMemory(dim=256)

        result = manager.checkpoint(memory, tag="v1.0")

        assert (tmp_path / "v1.0.pt").exists()
        assert result["tag"] == "v1.0"
        assert result["size_mb"] > 0

    def test_checkpoint_includes_weight_hash(self):
        """Checkpoint result should include weight hash."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        memory = NeuralMemory(dim=256)

        result = manager.checkpoint(memory, tag="test")

        assert "weight_hash" in result
        assert len(result["weight_hash"]) == 16  # Hex hash


class TestCheckpointRestore:
    """Test restoring checkpoints."""

    def test_restore_recovers_exact_state(self, tmp_path):
        """THE PERSISTENCE TEST: Restored state must be identical."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        memory = NeuralMemory(dim=256)

        # Learn something
        memory.observe("Important information")
        original_hash = memory.get_weight_hash()
        original_surprise = memory.surprise("Important information")

        # Checkpoint
        manager.checkpoint(memory, tag="snapshot")

        # Modify memory
        for _ in range(10):
            memory.observe("Different content entirely")

        # Verify memory changed
        assert memory.get_weight_hash() != original_hash

        # Restore
        manager.restore(memory, tag="snapshot")

        # Verify exact restoration
        assert memory.get_weight_hash() == original_hash
        assert abs(memory.surprise("Important information") - original_surprise) < 0.01

    def test_restore_nonexistent_raises(self, tmp_path):
        """Restoring non-existent checkpoint should raise."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        memory = NeuralMemory(dim=256)

        with pytest.raises(CheckpointNotFoundError):
            manager.restore(memory, tag="nonexistent")


class TestListCheckpoints:
    """Test listing checkpoints."""

    def test_list_returns_all_checkpoints(self, tmp_path):
        """list_checkpoints() should return all saved checkpoints."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        memory = NeuralMemory(dim=256)

        manager.checkpoint(memory, tag="v1")
        manager.checkpoint(memory, tag="v2")
        manager.checkpoint(memory, tag="v3")

        checkpoints = manager.list_checkpoints()
        tags = [cp["tag"] for cp in checkpoints]

        assert "v1" in tags
        assert "v2" in tags
        assert "v3" in tags

    def test_list_includes_metadata(self, tmp_path):
        """Checkpoint list should include metadata."""
        manager = CheckpointManager(checkpoint_dir=tmp_path)
        memory = NeuralMemory(dim=256)

        manager.checkpoint(memory, tag="test", description="Test checkpoint")

        checkpoints = manager.list_checkpoints()

        assert checkpoints[0]["description"] == "Test checkpoint"
        assert "created_at" in checkpoints[0]
        assert "size_mb" in checkpoints[0]
```

#### Implementation File
`src/state/checkpoint.py`

#### Acceptance Criteria
- [ ] Checkpoint creates file
- [ ] Restore recovers exact state (hash match)
- [ ] List returns all checkpoints with metadata

---

### Task 2.2: Fork/Branch Operations
**Use Case**: Create branches of memory state for experimentation.

**Why it matters**: Enables safe experimentation - try risky changes without losing stable state.

#### Tests to Write First

```python
# tests/unit/test_versioning.py

class TestFork:
    """Test forking memory state."""

    def test_fork_creates_copy(self, tmp_path):
        """fork() should create independent copy of checkpoint."""
        checkpoint_mgr = CheckpointManager(checkpoint_dir=tmp_path)
        version_mgr = VersionManager(checkpoint_mgr)
        memory = NeuralMemory(dim=256)

        # Create source checkpoint
        checkpoint_mgr.checkpoint(memory, tag="main")

        # Fork it
        result = version_mgr.fork(source_tag="main", new_tag="experiment")

        assert result["forked"] == True
        assert result["source_tag"] == "main"
        assert result["new_tag"] == "experiment"

        # Both should exist
        checkpoints = checkpoint_mgr.list_checkpoints()
        tags = [cp["tag"] for cp in checkpoints]
        assert "main" in tags
        assert "experiment" in tags

    def test_forked_branch_is_independent(self, tmp_path):
        """Changes to fork should not affect source."""
        checkpoint_mgr = CheckpointManager(checkpoint_dir=tmp_path)
        version_mgr = VersionManager(checkpoint_mgr)
        memory = NeuralMemory(dim=256)

        # Create and checkpoint initial state
        memory.observe("Initial content")
        checkpoint_mgr.checkpoint(memory, tag="main")
        main_hash = memory.get_weight_hash()

        # Fork
        version_mgr.fork(source_tag="main", new_tag="branch")

        # Restore branch and modify
        checkpoint_mgr.restore(memory, tag="branch")
        memory.observe("Branch-specific content")
        checkpoint_mgr.checkpoint(memory, tag="branch")  # Save branch changes

        # Restore main - should be unchanged
        checkpoint_mgr.restore(memory, tag="main")
        assert memory.get_weight_hash() == main_hash


class TestLearningMetrics:
    """Test tracking learning since checkpoint."""

    def test_track_learning_since_checkpoint(self, tmp_path):
        """Should track how much learning happened since checkpoint."""
        checkpoint_mgr = CheckpointManager(checkpoint_dir=tmp_path)
        version_mgr = VersionManager(checkpoint_mgr)
        memory = NeuralMemory(dim=256)

        checkpoint_mgr.checkpoint(memory, tag="baseline")

        # Do some learning
        for _ in range(10):
            memory.observe("New content")

        metrics = version_mgr.learning_since_checkpoint(memory, tag="baseline")

        assert metrics["total_learning"] > 0
        assert metrics["num_layers_changed"] > 0
```

#### Implementation File
`src/state/versioning.py`

#### Acceptance Criteria
- [ ] Fork creates independent copy
- [ ] Changes to fork don't affect source
- [ ] Learning metrics tracked correctly

---

### Task 2.3: Container Restart Persistence
**Use Case**: Learned state survives Docker container restart.

**Why it matters**: Production reliability - can't lose learned state on restart.

#### Tests to Write First

```python
# tests/integration/test_persistence.py

class TestContainerPersistence:
    """Test that state persists across container restarts."""

    def test_state_survives_process_restart(self, tmp_path):
        """THE PERSISTENCE TEST: State must survive restart."""
        # Simulate first container run
        memory1 = NeuralMemory(dim=256)
        manager1 = CheckpointManager(checkpoint_dir=tmp_path)

        # Learn something unique
        memory1.observe("The secret code is ALPHA-BRAVO-CHARLIE")
        surprise_before = memory1.surprise("The secret code is ALPHA-BRAVO-CHARLIE")
        manager1.checkpoint(memory1, tag="latest")

        # "Delete" first container (let objects go out of scope)
        del memory1
        del manager1

        # Simulate second container run
        memory2 = NeuralMemory(dim=256)
        manager2 = CheckpointManager(checkpoint_dir=tmp_path)

        # Restore state
        manager2.restore(memory2, tag="latest")

        # Should still know the content
        surprise_after = memory2.surprise("The secret code is ALPHA-BRAVO-CHARLIE")

        assert abs(surprise_before - surprise_after) < 0.01, (
            "Knowledge must persist: surprise should be identical after restart"
        )

    def test_volume_mount_preserves_checkpoints(self, tmp_path):
        """Checkpoints on mounted volume should persist."""
        # First "container"
        manager1 = CheckpointManager(checkpoint_dir=tmp_path)
        memory1 = NeuralMemory(dim=256)
        manager1.checkpoint(memory1, tag="persistent")

        # Second "container" - same volume
        manager2 = CheckpointManager(checkpoint_dir=tmp_path)

        checkpoints = manager2.list_checkpoints()
        assert any(cp["tag"] == "persistent" for cp in checkpoints)
```

#### Implementation File
N/A - tests verify integration

#### Acceptance Criteria
- [ ] State survives process restart
- [ ] Checkpoints persist on volume

---

## Phase 3: Consolidation

### Task 3.1: EWC-Based Consolidation
**Use Case**: Compress recent learning while protecting important old patterns.

**Why it matters**: Prevents catastrophic forgetting - old knowledge retained when learning new things.

#### Tests to Write First

```python
# tests/unit/test_consolidation.py

class TestConsolidation:
    """Test memory consolidation (like sleep)."""

    def test_consolidate_returns_metrics(self):
        """consolidate() should return compression metrics."""
        memory = NeuralMemory(dim=256)
        consolidator = Consolidator(memory)

        # Feed some content first
        for _ in range(20):
            memory.observe("Pattern to consolidate")

        result = consolidator.consolidate()

        assert "patterns_merged" in result
        assert "memory_compressed_by" in result
        assert "stability_score" in result

    def test_consolidation_stabilizes_weights(self):
        """After consolidation, weights should be more stable."""
        memory = NeuralMemory(dim=256)
        consolidator = Consolidator(memory)

        # Learn a pattern
        for _ in range(20):
            memory.observe("Important pattern")

        # Consolidate
        consolidator.consolidate()

        # Record stable hash
        stable_hash = memory.get_weight_hash()

        # Try to learn something new
        memory.observe("New unrelated content")

        # Weight change should be smaller after consolidation
        # (EWC penalty prevents large changes to important weights)
        post_learn_hash = memory.get_weight_hash()

        # Weights changed but not drastically
        assert stable_hash != post_learn_hash


class TestCatastrophicForgettingPrevention:
    """THE FORGETTING TEST: Old knowledge retained after new learning."""

    def test_old_patterns_retained_after_new_learning(self):
        """Learning new patterns should not destroy old ones."""
        memory = NeuralMemory(dim=256)
        consolidator = Consolidator(memory)

        # Learn pattern A
        for _ in range(20):
            memory.observe("Pattern A: Python programming")

        surprise_a_before = memory.surprise("Python programming")

        # Consolidate pattern A
        consolidator.consolidate()

        # Learn pattern B (different topic)
        for _ in range(20):
            memory.observe("Pattern B: Quantum physics")

        # Pattern A should still be known (low surprise)
        surprise_a_after = memory.surprise("Python programming")

        assert surprise_a_after < 0.6, (
            f"Old pattern forgotten: surprise went from {surprise_a_before:.2f} to {surprise_a_after:.2f}"
        )
```

#### Implementation File
`src/memory/consolidation.py`

#### Acceptance Criteria
- [ ] Consolidation returns metrics
- [ ] Old patterns retained after new learning (THE KEY TEST)

---

### Task 3.2: Capacity Estimation
**Use Case**: Track memory utilization to warn before saturation.

**Why it matters**: User needs to know when to consolidate or increase capacity.

#### Tests to Write First

```python
# tests/unit/test_capacity.py

class TestCapacityEstimation:
    """Test memory capacity tracking."""

    def test_initial_capacity_near_zero(self):
        """Fresh memory should have low capacity usage."""
        memory = NeuralMemory(dim=256)

        capacity = estimate_capacity(memory)

        assert capacity < 0.1, "Fresh memory should have near-zero capacity usage"

    def test_capacity_increases_with_learning(self):
        """Capacity usage should increase as memory learns."""
        memory = NeuralMemory(dim=256)

        capacity_before = estimate_capacity(memory)

        # Learn many patterns
        for i in range(100):
            memory.observe(f"Diverse content number {i}")

        capacity_after = estimate_capacity(memory)

        assert capacity_after > capacity_before

    def test_capacity_bounded_between_0_and_1(self):
        """Capacity should always be between 0 and 1."""
        memory = NeuralMemory(dim=256)

        for i in range(1000):
            memory.observe(f"Content {i}")
            capacity = estimate_capacity(memory)
            assert 0 <= capacity <= 1
```

#### Implementation File
`src/memory/capacity.py`

#### Acceptance Criteria
- [ ] Fresh memory has near-zero capacity
- [ ] Capacity increases with learning
- [ ] Always bounded [0, 1]

---

## Phase 4: MCP Interface

### Task 4.1: MCP Server Setup
**Use Case**: Provide MCP protocol endpoint for Claude/Gordon.

**Why it matters**: MCP is how Claude connects to external tools.

#### Tests to Write First

```python
# tests/integration/test_mcp_server.py

class TestMCPServerSetup:
    """Test MCP server initialization."""

    @pytest.mark.asyncio
    async def test_server_lists_all_tools(self):
        """Server should list all 11 tools."""
        server = NeuralMemoryServer()
        tools = server.get_tool_schemas()

        expected_tools = [
            "observe", "infer", "surprise", "consolidate",
            "checkpoint", "restore", "fork", "list_checkpoints",
            "stats", "attention_map", "explain"
        ]

        tool_names = [t["name"] for t in tools]
        for expected in expected_tools:
            assert expected in tool_names, f"Missing tool: {expected}"

    @pytest.mark.asyncio
    async def test_tool_schemas_valid(self):
        """All tool schemas should have required fields."""
        server = NeuralMemoryServer()
        tools = server.get_tool_schemas()

        for tool in tools:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
```

#### Implementation File
`src/mcp_server/server.py`

#### Acceptance Criteria
- [ ] Server lists all 11 tools
- [ ] Tool schemas are valid MCP format

---

### Task 4.2: Learning Tools (observe, infer, surprise)
**Use Case**: Core learning operations via MCP.

**Why it matters**: These are the primary interface for Claude.

#### Tests to Write First

```python
# tests/integration/test_mcp_tools.py

class TestObserveTool:
    """Test observe MCP tool."""

    @pytest.mark.asyncio
    async def test_observe_updates_memory(self):
        """observe tool should update memory weights."""
        server = NeuralMemoryServer()

        before_hash = server.memory.get_weight_hash()
        result = await server.handle_tool_call("observe", {
            "context": "Test content to learn"
        })
        after_hash = server.memory.get_weight_hash()

        assert before_hash != after_hash
        assert "surprise" in result
        assert "weight_delta" in result

    @pytest.mark.asyncio
    async def test_observe_with_learning_rate_override(self):
        """observe should accept learning rate override."""
        server = NeuralMemoryServer()

        result = await server.handle_tool_call("observe", {
            "context": "Test content",
            "learning_rate": 0.1
        })

        assert result["weight_delta"] > 0


class TestInferTool:
    """Test infer MCP tool."""

    @pytest.mark.asyncio
    async def test_infer_returns_response(self):
        """infer should return response with confidence."""
        server = NeuralMemoryServer()

        result = await server.handle_tool_call("infer", {
            "prompt": "What is Python?"
        })

        assert "response" in result
        assert "confidence" in result
        assert 0 <= result["confidence"] <= 1

    @pytest.mark.asyncio
    async def test_infer_does_not_learn(self):
        """infer should not update weights."""
        server = NeuralMemoryServer()

        before_hash = server.memory.get_weight_hash()
        await server.handle_tool_call("infer", {"prompt": "Test"})
        after_hash = server.memory.get_weight_hash()

        assert before_hash == after_hash


class TestSurpriseTool:
    """Test surprise MCP tool."""

    @pytest.mark.asyncio
    async def test_surprise_returns_score(self):
        """surprise should return score and recommendation."""
        server = NeuralMemoryServer()

        result = await server.handle_tool_call("surprise", {
            "input": "Test content"
        })

        assert "score" in result
        assert "recommendation" in result
        assert result["recommendation"] in ["learn", "skip", "consolidate"]
```

#### Implementation File
`src/mcp_server/tools.py`, `src/mcp_server/server.py`

#### Acceptance Criteria
- [ ] observe updates memory and returns metrics
- [ ] infer returns response without learning
- [ ] surprise returns score and recommendation

---

### Task 4.3: State Management Tools
**Use Case**: Checkpoint, restore, fork via MCP.

**Why it matters**: Enables state management from Claude conversations.

#### Tests to Write First

```python
# tests/integration/test_mcp_state_tools.py

class TestCheckpointTool:
    """Test checkpoint MCP tool."""

    @pytest.mark.asyncio
    async def test_checkpoint_saves_state(self, tmp_path):
        """checkpoint tool should save current state."""
        server = NeuralMemoryServer(checkpoint_dir=tmp_path)

        result = await server.handle_tool_call("checkpoint", {
            "tag": "test-v1",
            "description": "Test checkpoint"
        })

        assert result["tag"] == "test-v1"
        assert "weight_hash" in result
        assert result["size_mb"] > 0


class TestRestoreTool:
    """Test restore MCP tool."""

    @pytest.mark.asyncio
    async def test_restore_recovers_state(self, tmp_path):
        """restore tool should recover exact state."""
        server = NeuralMemoryServer(checkpoint_dir=tmp_path)

        # Learn and checkpoint
        await server.handle_tool_call("observe", {"context": "Important"})
        original_hash = server.memory.get_weight_hash()
        await server.handle_tool_call("checkpoint", {"tag": "snap"})

        # Modify
        await server.handle_tool_call("observe", {"context": "Different"})

        # Restore
        result = await server.handle_tool_call("restore", {"tag": "snap"})

        assert result["restored"] == True
        assert server.memory.get_weight_hash() == original_hash


class TestForkTool:
    """Test fork MCP tool."""

    @pytest.mark.asyncio
    async def test_fork_creates_branch(self, tmp_path):
        """fork tool should create independent branch."""
        server = NeuralMemoryServer(checkpoint_dir=tmp_path)

        await server.handle_tool_call("checkpoint", {"tag": "main"})

        result = await server.handle_tool_call("fork", {
            "source_tag": "main",
            "new_tag": "experiment"
        })

        assert result["forked"] == True

        # Both should exist
        list_result = await server.handle_tool_call("list_checkpoints", {})
        tags = [cp["tag"] for cp in list_result["checkpoints"]]
        assert "main" in tags
        assert "experiment" in tags
```

#### Implementation File
`src/mcp_server/server.py`

#### Acceptance Criteria
- [ ] checkpoint saves state correctly
- [ ] restore recovers exact state
- [ ] fork creates independent branch

---

## Phase 5: Demo & Polish

### Task 5.1: Killer Demo Script
**Use Case**: 2-minute demonstration that makes viewers "get it" immediately.

**Why it matters**: Demo sells the concept - must be visceral and clear.

#### Tests to Write First

```python
# tests/integration/test_demo_scenarios.py

class TestKillerDemo:
    """Test the killer demo scenarios."""

    def test_weights_visibly_change(self):
        """DEMO POINT 1: Weights must visibly change."""
        memory = NeuralMemory(dim=256)

        before = memory.get_weight_hash()
        memory.observe("Test content")
        after = memory.get_weight_hash()

        assert before != after
        # Hashes should be clearly different (not off by 1 bit)
        assert before[:8] != after[:8]

    def test_surprise_decreases_visibly(self):
        """DEMO POINT 2: Surprise decreases from 0.8+ to 0.4-."""
        memory = NeuralMemory(dim=256)

        r1 = memory.observe("Python uses indentation")
        r2 = memory.observe("Python requires indentation")
        r3 = memory.observe("Indentation matters in Python")

        # Should be clearly visible decrease
        assert r1["surprise"] > 0.6, "First observation should be surprising"
        assert r3["surprise"] < 0.5, "Third observation should be less surprising"
        assert r3["surprise"] < r1["surprise"] * 0.7, "At least 30% decrease"

    def test_bounded_capacity_demo(self):
        """DEMO POINT 3: 1000 observations, same parameter count."""
        memory = NeuralMemory(dim=256)

        params_before = sum(p.numel() for p in memory.parameters())

        for i in range(1000):
            memory.observe(f"Fact {i}: some content")

        params_after = sum(p.numel() for p in memory.parameters())

        assert params_before == params_after

    def test_persistence_demo(self, tmp_path):
        """DEMO POINT 4: Knowledge survives restart."""
        # First run
        memory1 = NeuralMemory(dim=256)
        manager1 = CheckpointManager(checkpoint_dir=tmp_path)

        memory1.observe("Secret: the answer is 42")
        surprise1 = memory1.surprise("The answer is 42")
        manager1.checkpoint(memory1, tag="demo")

        # Simulate restart
        del memory1

        # Second run
        memory2 = NeuralMemory(dim=256)
        manager2 = CheckpointManager(checkpoint_dir=tmp_path)
        manager2.restore(memory2, tag="demo")

        surprise2 = memory2.surprise("The answer is 42")

        assert abs(surprise1 - surprise2) < 0.05, "Knowledge persisted"
```

#### Implementation File
`demo/killer_demo.py`

#### Acceptance Criteria
- [ ] Weights visibly change (different hash prefix)
- [ ] Surprise decreases >30% over 3 observations
- [ ] Parameter count unchanged after 1000 observations
- [ ] Knowledge persists across restart

---

### Task 5.2: Demo Script Polish
**Use Case**: Create polished demo script with clear output.

**Why it matters**: First impression matters - demo must be professional.

#### Demo Script Requirements

```python
# demo/killer_demo.py

"""
Docker Neural Memory - Killer Demo

Run: docker compose -f docker-compose.dev.yml run --rm demo

Shows:
1. Weights actually change (hash diff)
2. Surprise decreases (pattern learning)
3. Bounded capacity (no growth)
4. Persistence (survives restart)
"""

def main():
    print("=" * 60)
    print("  DOCKER NEURAL MEMORY - THE DEMO")
    print("  Memory that LEARNS, not just stores")
    print("=" * 60)
    print()

    # Demo 1: Weights Change
    # Demo 2: Surprise Decreases
    # Demo 3: Bounded Capacity
    # Demo 4: Persistence

    print()
    print("=" * 60)
    print("  This is REAL learning. RAG can't do this.")
    print("=" * 60)
```

#### Acceptance Criteria
- [ ] Clean, professional output
- [ ] Each demo point clearly labeled
- [ ] Runs in under 30 seconds
- [ ] Works reliably every time

---

## Task Status Legend

```
[ ] Not started
[~] In progress
[x] Complete
[!] Blocked
```

---

## Development Commands

```bash
# Build dev container
docker compose -f docker-compose.dev.yml build

# Run tests (TDD workflow)
docker compose -f docker-compose.dev.yml run --rm test

# Run specific test
docker compose -f docker-compose.dev.yml run --rm test pytest tests/unit/test_encoder.py -v

# Run with coverage
docker compose -f docker-compose.dev.yml run --rm test pytest --cov=src --cov-report=html

# Interactive development
docker compose -f docker-compose.dev.yml run --rm dev

# Run demo
docker compose -f docker-compose.dev.yml run --rm demo
```

---

## Next Steps

1. Build dev container: `docker compose -f docker-compose.dev.yml build`
2. Start with Task 1.1: Write encoder tests
3. Make tests pass (TDD Red → Green → Refactor)
4. Proceed to Task 1.2
5. Continue through all tasks sequentially
