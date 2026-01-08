# Titans Architecture Reference

Detailed implementation guide for Titans neural memory and TTT layers.

## Titans Memory Module

From "Titans: Learning to Memorize at Test Time" (Google Research, Dec 2024).

### Core Architecture

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import copy


class NeuralMemory(nn.Module):
    """
    Titans-style neural long-term memory module.
    
    Key innovation: The hidden state IS a neural network that updates
    its weights during inference via self-supervised learning.
    
    Args:
        dim: Input/output dimension
        memory_dim: Internal memory dimension (default: dim * 4)
        num_layers: Depth of memory network
        learning_rate: Initial learning rate for test-time updates
        momentum: Momentum for weight updates (smooths learning)
    """
    
    def __init__(
        self,
        dim: int,
        memory_dim: Optional[int] = None,
        num_layers: int = 2,
        learning_rate: float = 0.01,
        momentum: float = 0.9
    ):
        super().__init__()
        
        memory_dim = memory_dim or dim * 4
        
        # The memory is a small neural network
        layers = []
        in_dim = dim
        for i in range(num_layers - 1):
            layers.extend([
                nn.Linear(in_dim, memory_dim),
                nn.GELU(),
                nn.LayerNorm(memory_dim)
            ])
            in_dim = memory_dim
        layers.append(nn.Linear(in_dim, dim))
        
        self.memory_net = nn.Sequential(*layers)
        
        # Learnable learning rate (meta-learned)
        self.lr = nn.Parameter(torch.tensor(learning_rate))
        self.momentum = momentum
        
        # Momentum buffers for smoother updates
        self._momentum_buffers = {}
        
        # Input projection for self-supervised target
        self.target_proj = nn.Linear(dim, dim)
        
    def forward(
        self,
        x: Tensor,
        learn: bool = True,
        return_metrics: bool = False
    ) -> Tuple[Tensor, Optional[dict]]:
        """
        Process input and optionally update memory weights.
        
        Args:
            x: Input tensor [batch, seq, dim]
            learn: Whether to perform test-time training
            return_metrics: Whether to return learning metrics
            
        Returns:
            output: Memory-augmented representation [batch, seq, dim]
            metrics: Optional dict with surprise, weight_delta, etc.
        """
        batch, seq_len, dim = x.shape
        
        # Query the memory
        memory_output = self.memory_net(x)
        
        metrics = None
        
        if learn and seq_len > 1:
            # Self-supervised learning signal
            with torch.enable_grad():
                # Target: shifted input (predict next token representation)
                target = self.target_proj(x[:, 1:, :])
                pred = memory_output[:, :-1, :]
                
                # Surprise = prediction error
                surprise = F.mse_loss(pred, target)
                
                # Compute gradients for memory network only
                grads = torch.autograd.grad(
                    surprise,
                    self.memory_net.parameters(),
                    create_graph=False
                )
                
                # Update weights with momentum
                weight_delta = 0.0
                with torch.no_grad():
                    for (name, param), grad in zip(
                        self.memory_net.named_parameters(),
                        grads
                    ):
                        # Initialize momentum buffer
                        if name not in self._momentum_buffers:
                            self._momentum_buffers[name] = torch.zeros_like(grad)
                        
                        # Update momentum
                        self._momentum_buffers[name] = (
                            self.momentum * self._momentum_buffers[name] +
                            (1 - self.momentum) * grad
                        )
                        
                        # Apply update
                        update = self.lr * self._momentum_buffers[name]
                        param -= update
                        weight_delta += update.abs().mean().item()
                
                if return_metrics:
                    metrics = {
                        'surprise': surprise.item(),
                        'weight_delta': weight_delta,
                        'learning_rate': self.lr.item()
                    }
        
        if return_metrics:
            return memory_output, metrics
        return memory_output, None

    def compute_surprise(self, x: Tensor) -> float:
        """Compute surprise without updating weights."""
        with torch.no_grad():
            memory_output = self.memory_net(x)
            if x.shape[1] > 1:
                target = self.target_proj(x[:, 1:, :])
                pred = memory_output[:, :-1, :]
                return F.mse_loss(pred, target).item()
            return 0.0
    
    def reset_momentum(self):
        """Reset momentum buffers (e.g., when switching domains)."""
        self._momentum_buffers.clear()


class NeuralMemoryWithGating(NeuralMemory):
    """
    Extended Titans memory with gating mechanism.
    
    Adds learned gates to control:
    - How much to use memory vs pass-through
    - How much to learn from each input
    """
    
    def __init__(self, dim: int, **kwargs):
        super().__init__(dim, **kwargs)
        
        # Gate for memory usage
        self.use_gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid()
        )
        
        # Gate for learning intensity
        self.learn_gate = nn.Sequential(
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        x: Tensor,
        learn: bool = True,
        return_metrics: bool = False
    ) -> Tuple[Tensor, Optional[dict]]:
        # Get memory output
        memory_out, metrics = super().forward(x, learn=False, return_metrics=return_metrics)
        
        # Compute gates
        combined = torch.cat([x, memory_out], dim=-1)
        use_weight = self.use_gate(combined)
        
        # Gated output
        output = use_weight * memory_out + (1 - use_weight) * x
        
        # Gated learning
        if learn:
            learn_weight = self.learn_gate(x).mean()
            if learn_weight > 0.1:  # Threshold for learning
                # Scale learning rate by gate
                original_lr = self.lr.data.clone()
                self.lr.data *= learn_weight
                _, update_metrics = super().forward(x, learn=True, return_metrics=True)
                self.lr.data = original_lr
                
                if metrics and update_metrics:
                    metrics.update(update_metrics)
                    metrics['learn_gate'] = learn_weight.item()
        
        return output, metrics
```

## TTT Layers

From "Learning to (Learn at Test Time)" (Stanford/Meta, July 2024).

### TTT-Linear

```python
class TTTLinear(nn.Module):
    """
    Test-Time Training layer with linear hidden state.
    
    The hidden state is a linear model that updates via
    self-supervised learning during the forward pass.
    
    Faster than TTT-MLP but less expressive.
    """
    
    def __init__(
        self,
        dim: int,
        learning_rate: float = 0.1,
        num_steps: int = 1
    ):
        super().__init__()
        
        # Hidden state: a linear model
        self.W = nn.Parameter(torch.eye(dim) * 0.01)
        
        # Project to key/value for self-supervised objective
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
        # Learnable learning rate
        self.eta = nn.Parameter(torch.tensor(learning_rate))
        self.num_steps = num_steps
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Process sequence with test-time training.
        
        Args:
            x: Input [batch, seq, dim]
            
        Returns:
            Output [batch, seq, dim]
        """
        batch, seq_len, dim = x.shape
        
        # Clone W for this batch (don't modify original)
        W = self.W.clone()
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t, :]  # [batch, dim]
            
            # Self-supervised targets
            k_t = self.to_k(x_t)  # [batch, dim]
            v_t = self.to_v(x_t)  # [batch, dim]
            
            # Forward through hidden state
            y_t = x_t @ W  # [batch, dim]
            
            # Self-supervised loss: predict value from key
            for _ in range(self.num_steps):
                pred = k_t @ W
                loss = ((pred - v_t) ** 2).sum()
                
                # Gradient w.r.t. W
                grad = 2 * k_t.T @ (pred - v_t) / batch
                
                # Update W
                W = W - self.eta * grad
            
            outputs.append(y_t)
        
        return torch.stack(outputs, dim=1)


class TTTMLP(nn.Module):
    """
    Test-Time Training layer with MLP hidden state.
    
    More expressive than TTT-Linear but slower.
    The hidden state is a two-layer MLP.
    """
    
    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        learning_rate: float = 0.05,
        num_steps: int = 1
    ):
        super().__init__()
        
        hidden_dim = hidden_dim or dim * 2
        
        # Hidden state: a 2-layer MLP
        self.hidden_net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dim)
        )
        
        # Project to key/value
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        
        self.eta = nn.Parameter(torch.tensor(learning_rate))
        self.num_steps = num_steps
    
    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, dim = x.shape
        
        # Clone hidden net for this sequence
        hidden_net = copy.deepcopy(self.hidden_net)
        
        # Enable gradients for the copy
        for param in hidden_net.parameters():
            param.requires_grad = True
        
        outputs = []
        
        for t in range(seq_len):
            x_t = x[:, t:t+1, :]  # [batch, 1, dim]
            
            # Self-supervised targets
            k_t = self.to_k(x_t)
            v_t = self.to_v(x_t)
            
            # Forward through hidden state
            y_t = hidden_net(x_t)
            
            # Self-supervised update
            for _ in range(self.num_steps):
                pred = hidden_net(k_t)
                loss = F.mse_loss(pred, v_t)
                
                # Compute and apply gradients
                grads = torch.autograd.grad(
                    loss,
                    hidden_net.parameters(),
                    create_graph=False
                )
                
                with torch.no_grad():
                    for param, grad in zip(hidden_net.parameters(), grads):
                        param -= self.eta * grad
            
            outputs.append(y_t.squeeze(1))
        
        return torch.stack(outputs, dim=1)
```

## Memory with Attention Integration

Titans combines neural memory with attention (memory as long-term, attention as short-term):

```python
class TitansBlock(nn.Module):
    """
    Full Titans block: Neural Memory + Attention.
    
    Memory handles long-term patterns.
    Attention handles local dependencies.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        memory_config: Optional[dict] = None
    ):
        super().__init__()
        
        # Neural long-term memory
        memory_config = memory_config or {}
        self.memory = NeuralMemory(dim, **memory_config)
        
        # Standard multi-head attention (short-term)
        self.attention = nn.MultiheadAttention(
            dim,
            num_heads,
            batch_first=True
        )
        
        # Combine memory and attention outputs
        self.combine = nn.Linear(dim * 2, dim)
        
        # Layer norms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(
        self,
        x: Tensor,
        learn: bool = True,
        attention_mask: Optional[Tensor] = None
    ) -> Tensor:
        # Memory pathway (long-term)
        memory_out, _ = self.memory(self.norm1(x), learn=learn)
        
        # Attention pathway (short-term)
        attn_out, _ = self.attention(
            self.norm1(x),
            self.norm1(x),
            self.norm1(x),
            attn_mask=attention_mask
        )
        
        # Combine pathways
        combined = torch.cat([memory_out, attn_out], dim=-1)
        x = x + self.combine(combined)
        
        # FFN
        x = x + self.ffn(self.norm2(x))
        
        return x
```

## Consolidation (Memory Compression)

Compress recent learning into stable long-term patterns:

```python
class MemoryConsolidator:
    """
    Consolidate memory patterns (like sleep for the brain).
    
    Compresses recent learning into stable representations
    while protecting important existing patterns.
    """
    
    def __init__(
        self,
        memory: NeuralMemory,
        ewc_lambda: float = 1000.0
    ):
        self.memory = memory
        self.ewc_lambda = ewc_lambda
        
        # Fisher information (importance of each weight)
        self.fisher = {}
        # Optimal weights (snapshot after consolidation)
        self.optimal = {}
    
    def consolidate(
        self,
        replay_data: Tensor,
        num_steps: int = 100
    ) -> dict:
        """
        Consolidate memory using replay.
        
        Args:
            replay_data: Representative samples to replay
            num_steps: Number of consolidation steps
            
        Returns:
            Metrics about consolidation
        """
        # Compute Fisher information (which weights matter)
        self._compute_fisher(replay_data)
        
        # Store current optimal weights
        for name, param in self.memory.memory_net.named_parameters():
            self.optimal[name] = param.clone().detach()
        
        # Fine-tune with EWC regularization
        optimizer = torch.optim.Adam(
            self.memory.memory_net.parameters(),
            lr=0.001
        )
        
        initial_surprise = self.memory.compute_surprise(replay_data)
        
        for step in range(num_steps):
            optimizer.zero_grad()
            
            # Standard learning objective
            output, metrics = self.memory(replay_data, learn=False, return_metrics=True)
            target = self.memory.target_proj(replay_data[:, 1:, :])
            pred = output[:, :-1, :]
            task_loss = F.mse_loss(pred, target)
            
            # EWC regularization
            ewc_loss = 0
            for name, param in self.memory.memory_net.named_parameters():
                if name in self.fisher:
                    ewc_loss += (
                        self.fisher[name] *
                        (param - self.optimal[name]) ** 2
                    ).sum()
            
            # Combined loss
            loss = task_loss + self.ewc_lambda * ewc_loss
            loss.backward()
            optimizer.step()
        
        final_surprise = self.memory.compute_surprise(replay_data)
        
        return {
            'initial_surprise': initial_surprise,
            'final_surprise': final_surprise,
            'compression_ratio': initial_surprise / (final_surprise + 1e-8),
            'steps': num_steps
        }
    
    def _compute_fisher(self, data: Tensor):
        """Compute Fisher information for EWC."""
        self.memory.memory_net.zero_grad()
        
        output = self.memory.memory_net(data)
        target = self.memory.target_proj(data[:, 1:, :])
        pred = output[:, :-1, :]
        loss = F.mse_loss(pred, target)
        loss.backward()
        
        for name, param in self.memory.memory_net.named_parameters():
            if param.grad is not None:
                self.fisher[name] = param.grad.clone().detach() ** 2
```

## Practical Notes

### Learning Rate Selection

- **High LR (0.05-0.1)**: Fast adaptation, risk of forgetting
- **Low LR (0.001-0.01)**: Stable, slower learning
- **Meta-learned LR**: Let the model learn optimal rate

### When to Consolidate

- After learning burst (many high-surprise inputs)
- Before checkpoint (stabilize weights)
- Periodically (e.g., every N observations)

### Memory Capacity

Unlike vector DBs, neural memory has bounded capacity. Signs of saturation:
- Surprise stays high even for repeated patterns
- Weight deltas become large
- Performance degrades

Solution: Consolidate or increase memory dimension.
