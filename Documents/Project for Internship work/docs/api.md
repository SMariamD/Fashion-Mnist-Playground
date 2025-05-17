# S3CMN API Documentation

## Memory System

### Class: Memory
A memory system that integrates attention-based retrieval, compression, and optimization.

#### Initialization
```python
class Memory(nn.Module):
    def __init__(
        self,
        initial_size: int,
        embedding_dim: int,
        compressed_dim: int = 128,
        max_size: Optional[int] = None,
        growth_factor: float = 1.5,
        prune_threshold: float = 0.1,
        compression_interval: int = 10000,
        num_streams: int = 4,
        device: str = 'cpu',
        distributed: bool = False,
        world_size: int = 1,
        rank: int = 0,
        optimize_memory: bool = True,
        quantize: bool = True,
        quantization_bits: int = 8
    )
```

#### Parameters
- `initial_size`: Initial memory capacity
- `embedding_dim`: Dimension of input embeddings
- `compressed_dim`: Dimension after compression
- `max_size`: Maximum memory size
- `growth_factor`: Factor for memory expansion
- `prune_threshold`: Threshold for memory pruning
- `compression_interval`: Interval for compression
- `num_streams`: Number of CUDA streams
- `device`: Device for computation
- `distributed`: Enable distributed training
- `world_size`: Number of distributed nodes
- `rank`: Node rank in distributed training
- `optimize_memory`: Enable memory optimization
- `quantize`: Enable quantization
- `quantization_bits`: Number of bits for quantization

#### Methods

```python
# Add new entry to memory
def add_to_memory(self, x: torch.Tensor, y: torch.Tensor) -> None
```

```python
# Get top-k similar memories
def get_memory(self, query: torch.Tensor, top_k: int = 5) -> torch.Tensor
```

```python
# Get memory statistics
def get_memory_stats(self) -> dict
```

### Usage Examples

#### Basic Usage
```python
# Initialize memory
memory = Memory(
    initial_size=1000,
    embedding_dim=768,
    compressed_dim=128,
    device='cuda'
)

# Add memory entry
memory.add_to_memory(input_tensor, target_tensor)

# Retrieve memories
similar_memories = memory.get_memory(query_tensor, top_k=5)

# Get statistics
stats = memory.get_memory_stats()
```

#### Distributed Usage
```python
# Initialize distributed memory
memory = Memory(
    initial_size=1000,
    embedding_dim=768,
    compressed_dim=128,
    distributed=True,
    world_size=4,
    rank=0,
    device='cuda'
)
```

#### Quantized Usage
```python
# Initialize quantized memory
memory = Memory(
    initial_size=1000,
    embedding_dim=768,
    compressed_dim=128,
    quantize=True,
    quantization_bits=8,
    device='cuda'
)
```

## Spiking Neural Network Layer

### Class: SpikingLayer
A spiking neural network layer that integrates memory operations.

#### Initialization
```python
class SpikingLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        memory: Memory,
        beta: float = 0.95,
        threshold: float = 1.0,
        spike_grad: str = 'fast_sigmoid'
    )
```

#### Parameters
- `in_features`: Number of input features
- `out_features`: Number of output features
- `memory`: Memory system instance
- `beta`: Membrane potential decay factor
- `threshold`: Spiking threshold
- `spike_grad`: Spike gradient function

#### Methods

```python
# Forward pass
def forward(self, x: torch.Tensor) -> torch.Tensor
```

```python
# Reset state
def reset_state(self) -> None
```

```python
# Get statistics
def get_stats(self) -> dict
```

### Usage Examples

#### Basic Usage
```python
# Initialize spiking layer
layer = SpikingLayer(
    in_features=768,
    out_features=512,
    memory=memory,
    beta=0.95,
    threshold=1.0
)

# Forward pass
output = layer(input_tensor)

# Reset state
layer.reset_state()
```

## Memory System Internals

### Memory Operations
- Attention-based retrieval
- Adaptive compression
- Memory pruning
- Memory expansion
- Parallel processing
- Distributed synchronization
- Tensor optimization
- Quantization

### Spiking Dynamics
- Membrane potential
- Leaky integration
- Spike generation
- Spike timing
- Spike gradients

## Performance Metrics

### Memory Metrics
- Hit rate
- Similarity
- Usage
- Compression ratio
- Memory usage
- Expansion ratio
- Prune ratio
- Quantization metrics

### Spiking Metrics
- Spike rate
- Spike timing
- Spike distribution
- Spike efficiency
- Spike accuracy

## Best Practices

### Memory Management
1. Start with moderate initial size
2. Set appropriate growth factor
3. Monitor memory usage
4. Adjust pruning threshold
5. Enable optimization
6. Use appropriate quantization

### Spiking Configuration
1. Set appropriate beta
2. Adjust threshold
3. Choose spike gradient
4. Monitor spike rate
5. Adjust learning rate
6. Enable memory integration
