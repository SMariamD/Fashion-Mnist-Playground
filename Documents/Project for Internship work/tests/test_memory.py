import pytest
import torch
from s3cmn.model.memory_v2 import Memory

def test_memory_initialization():
    """Test memory initialization."""
    memory = Memory(
        initial_size=100,
        embedding_dim=10,
        compressed_dim=5,
        device='cpu'
    )
    
    assert memory.memory_size == 100
    assert memory.embedding_dim == 10
    assert memory.compressed_dim == 5
    assert memory.device == 'cpu'
    assert memory.current_size == 0
    assert memory.memory.shape == (100, 5)

def test_memory_addition():
    """Test adding entries to memory."""
    memory = Memory(
        initial_size=100,
        embedding_dim=10,
        compressed_dim=5,
        device='cpu'
    )
    
    x = torch.randn(10)
    y = torch.randn(5)
    
    memory.add_to_memory(x, y)
    assert memory.current_size == 1
    assert memory.memory[0].sum() != 0
    
    # Test memory expansion
    for _ in range(100):
        memory.add_to_memory(x, y)
    assert memory.current_size == 101
    assert memory.memory_size > 100

def test_memory_retrieval():
    """Test memory retrieval."""
    memory = Memory(
        initial_size=100,
        embedding_dim=10,
        compressed_dim=5,
        device='cpu'
    )
    
    # Add some entries
    for i in range(10):
        x = torch.randn(10)
        y = torch.randn(5)
        memory.add_to_memory(x, y)
    
    # Test retrieval
    query = torch.randn(1, 10)
    result = memory.get_memory(query, top_k=3)
    assert result.shape == (1, 3, 5)

def test_memory_compression():
    """Test memory compression."""
    memory = Memory(
        initial_size=100,
        embedding_dim=10,
        compressed_dim=5,
        device='cpu'
    )
    
    # Add enough entries to trigger compression
    for _ in range(10000):
        x = torch.randn(10)
        y = torch.randn(5)
        memory.add_to_memory(x, y)
    
    assert memory.compression_ratio < 1.0
    assert len(memory.compression_history) > 0

def test_memory_quantization():
    """Test memory quantization."""
    memory = Memory(
        initial_size=100,
        embedding_dim=10,
        compressed_dim=5,
        device='cpu',
        quantize=True,
        quantization_bits=8
    )
    
    # Add some entries
    for i in range(10):
        x = torch.randn(10)
        y = torch.randn(5)
        memory.add_to_memory(x, y)
    
    # Test quantization
    assert memory.quantized_memory is not None
    assert memory.quantization_scale is not None
    assert memory.quantization_zero_point is not None

def test_memory_statistics():
    """Test memory statistics."""
    memory = Memory(
        initial_size=100,
        embedding_dim=10,
        compressed_dim=5,
        device='cpu'
    )
    
    # Add some entries
    for i in range(10):
        x = torch.randn(10)
        y = torch.randn(5)
        memory.add_to_memory(x, y)
    
    stats = memory.get_memory_stats()
    assert stats['current_size'] == 10
    assert stats['memory_usage'] > 0
    assert stats['compression_ratio'] == 1.0
    assert stats['hit_rate'] == 0  # No retrieval has been done yet

def test_memory_distributed():
    """Test distributed memory functionality."""
    memory = Memory(
        initial_size=100,
        embedding_dim=10,
        compressed_dim=5,
        device='cpu',
        distributed=True,
        world_size=2,
        rank=0
    )
    
    # Add some entries
    for i in range(10):
        x = torch.randn(10)
        y = torch.randn(5)
        memory.add_to_memory(x, y)
    
    # Test synchronization
    memory._sync_memory()
    assert memory.distributed_memory_size > 0
    assert memory.world_size == 2
    assert memory.rank == 0

def test_memory_optimization():
    """Test memory optimization."""
    memory = Memory(
        initial_size=100,
        embedding_dim=10,
        compressed_dim=5,
        device='cpu',
        optimize_memory=True
    )
    
    # Add some entries
    for i in range(10):
        x = torch.randn(10)
        y = torch.randn(5)
        memory.add_to_memory(x, y)
    
    # Test optimization
    memory._optimize_tensors()
    assert memory.optimized_memory is not None
    assert memory.optimized_attention is not None
    assert memory.optimized_usage is not None

if __name__ == '__main__':
    pytest.main([__file__])
