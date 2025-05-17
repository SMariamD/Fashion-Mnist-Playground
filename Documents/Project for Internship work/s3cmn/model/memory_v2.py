import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import numpy as np
from typing import Optional, Tuple
import math

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Memory(nn.Module):
    """Memory system with attention-based retrieval, pruning, expansion, compression, parallel retrieval, distributed support, tensor optimization, and quantization."""
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
    ):
        super().__init__()
        self.initial_size = initial_size
        self.embedding_dim = embedding_dim
        self.compressed_dim = compressed_dim
        self.max_size = max_size or initial_size * 4
        self.growth_factor = growth_factor
        self.prune_threshold = prune_threshold
        self.compression_interval = compression_interval
        self.num_streams = num_streams
        self.device = device
        self.distributed = distributed
        self.world_size = world_size
        self.rank = rank
        self.optimize_memory = optimize_memory
        self.quantize = quantize
        self.quantization_bits = quantization_bits
        
        # Initialize CUDA streams if available
        self.streams = [torch.cuda.Stream(device=device) for _ in range(num_streams)] if device != 'cpu' else None
        
        # Initialize with initial size
        self.memory_size = initial_size
        self.memory = torch.zeros(self.memory_size, compressed_dim, device=device)
        self.current_size = 0
        
        # Attention mechanism with optimized parameters
        self.query_transform = nn.Linear(embedding_dim, compressed_dim, bias=False)
        self.key_transform = nn.Linear(embedding_dim, compressed_dim, bias=False)
        self.value_transform = nn.Linear(embedding_dim, compressed_dim, bias=False)
        
        # Attention weights and usage
        self.attention_weights = torch.zeros(self.memory_size, device=device)
        self.usage_counts = torch.zeros(self.memory_size, device=device)
        
        # Compression related
        self.compression_ratio = 1.0
        self.compression_history = []
        self.compression_threshold = 0.95
        
        # Expansion and pruning related
        self.prune_interval = 1000
        self.expand_interval = 5000
        self.query_count = 0
        
        # Distributed related
        self.distributed_memory = None
        self.distributed_memory_size = 0
        self.sync_interval = 1000
        
        # Optimization related
        self.optimized_memory = None
        self.optimized_attention = None
        self.optimized_usage = None
        self.optimization_interval = 1000
        
        # Quantization related
        self.quantized_memory = None
        self.quantization_scale = None
        self.quantization_zero_point = None
        self.quantization_interval = 1000
        
        self.to(device)
        
    def _quantize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize tensor to specified number of bits."""
        if not self.quantize:
            return x
            
        # Calculate quantization parameters
        min_val = x.min()
        max_val = x.max()
        scale = (max_val - min_val) / ((2 ** self.quantization_bits) - 1)
        zero_point = -min_val / scale
        
        # Quantize tensor
        quantized = torch.round(x / scale + zero_point).to(torch.int8)
        
        # Store quantization parameters
        self.quantization_scale = scale
        self.quantization_zero_point = zero_point
        
        return quantized
    
    def _dequantize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Dequantize tensor back to original scale."""
        if not self.quantize or self.quantization_scale is None:
            return x
            
        # Dequantize tensor
        dequantized = (x - self.quantization_zero_point) * self.quantization_scale
        return dequantized.to(torch.float32)
    
    def _compress(self, x: torch.Tensor) -> torch.Tensor:
        """Compress input tensor using adaptive compression with quantization."""
        # Flatten and normalize
        x = x.view(x.size(0), -1)
        x = F.normalize(x, dim=1)
        
        # Apply PCA with adaptive compression
        U, S, V = torch.pca_lowrank(x)
        
        # Calculate optimal compression dimension
        explained_variance = torch.cumsum(S ** 2, dim=0) / torch.sum(S ** 2)
        threshold_mask = (explained_variance >= self.compression_threshold).float()
        target_dim = torch.argmax(threshold_mask) + 1
        
        # Apply compression
        compressed = torch.matmul(x, V[:, :target_dim])
        
        # Ensure output matches expected shape
        if compressed.shape[1] != self.compressed_dim:
            compressed = torch.nn.functional.pad(
                compressed,
                (0, self.compressed_dim - compressed.shape[1]),
                mode='constant',
                value=0
            )
        
        # Reshape to match expected output shape
        compressed = compressed.view(-1)
        
        # Quantize if enabled
        if self.quantize:
            compressed = self._quantize_tensor(compressed)
            
        return compressed
    
    def _quantize_memory(self) -> None:
        """Quantize memory tensor."""
        if not self.quantize:
            return
            
        if self.query_count >= self.quantization_interval:
            # Quantize memory
            self.quantized_memory = self._quantize_tensor(self.memory[:self.current_size])
            
            # Reset query count
            self.query_count = 0
            
    def _get_memory_single(self, query: torch.Tensor, top_k: int = 5) -> torch.Tensor:
        """Get top-k memories for a single chunk with optimization and quantization."""
        if self.current_size == 0:
            return torch.zeros(query.size(0), top_k, self.compressed_dim).to(self.device)
            
        # Use optimized tensors if available
        memory = self.optimized_memory if self.optimized_memory is not None else self.memory[:self.current_size]
        
        # Use quantized memory if available
        if self.quantize and self.quantized_memory is not None:
            memory = self._dequantize_tensor(self.quantized_memory)
        
        attention = self.optimized_attention if self.optimized_attention is not None else self.attention_weights[:self.current_size]
        usage = self.optimized_usage if self.optimized_usage is not None else self.usage_counts[:self.current_size]
        
        # Transform query and memory
        query_transformed = self.query_transform(query)
        keys = self.key_transform(memory)
        values = self.value_transform(memory)
        
        # Calculate attention scores with optimization
        attention_scores = torch.matmul(
            query_transformed.unsqueeze(1),
            keys.transpose(-2, -1)
        ) / math.sqrt(self.compressed_dim)
        
        # Apply optimized attention weights
        attention_scores += attention.unsqueeze(0)
        
        # Get top-k indices
        top_k_indices = torch.topk(attention_scores, top_k, dim=-1).indices
        
        # Get top-k memories
        top_k_memories = []
        for i in range(query.size(0)):
            # Get top-k values
            memory_values = values[top_k_indices[i]]
            
            # Calculate attention distribution
            attention_dist = F.softmax(attention_scores[i, :top_k_indices[i].max() + 1], dim=-1)
            
            # Weighted sum of memories with optimization
            weighted_memory = torch.matmul(
                attention_dist.unsqueeze(1),
                memory_values
            ).squeeze(1)
            
            top_k_memories.append(weighted_memory)
            
            # Update usage counts for accessed memories
            usage[top_k_indices[i]] += 1
            
        # Update optimized usage counts
        if self.optimized_usage is not None:
            self.usage_counts[:self.current_size] = usage
            
        return torch.stack(top_k_memories, dim=0)
    
    def _should_expand(self) -> bool:
        """Check if memory should be expanded."""
        return self.current_size >= self.memory_size and self.growth_factor > 1.0

    def add_to_memory(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """Add new entry to memory with compression and quantization."""
        # Expand memory if needed
        if self._should_expand():
            new_size = min(int(self.memory_size * self.growth_factor), self.max_size)
            self._expand_memory(new_size)
            self.query_count = 0  # Reset query count after expansion
        
        if self.current_size < self.memory_size:
            # Add new entry
            compressed_x = self._compress(x)
            self.memory[self.current_size] = compressed_x
            self.current_size += 1
        else:
            # Prune memory if needed
            self._prune_memory()
            
            # Add new entry to available slot
            min_usage_idx = torch.argmin(self.usage_counts[:self.current_size])
            compressed_x = self._compress(x)
            self.memory[min_usage_idx] = compressed_x
            self.usage_counts[min_usage_idx] = 0
            self.attention_weights[min_usage_idx] = 0
        
        # Update attention weights and usage
        self.attention_weights[:self.current_size] = torch.ones(self.current_size, device=self.device)
        self.usage_counts[:self.current_size] += 1
        
        # Compress memory periodically
        self._compress_memory()
        
        # Quantize memory periodically
        self._quantize_memory()
        
        # Optimize tensors periodically
        self._optimize_tensors()
        
        # Synchronize memory if distributed
        self._sync_memory()
        
    def get_memory_stats(self) -> dict:
        """Get memory statistics with optimization and quantization metrics."""
        stats = {
            'hit_rate': (self.usage_counts[:self.current_size] > 0).float().mean().item(),
            'avg_similarity': self._calculate_avg_similarity(),
            'memory_usage': self.current_size / self.memory_size,
            'avg_attention': self.attention_weights[:self.current_size].mean().item(),
            'prune_ratio': (1 - self.current_size / self.memory_size) if self.current_size < self.memory_size else 0,
            'expansion_ratio': self.memory_size / self.initial_size,
            'compression_ratio': self.compression_ratio,
            'avg_compression_ratio': sum(self.compression_history) / len(self.compression_history) if self.compression_history else 1.0,
            'current_size': self.current_size,
            'memory_size': self.memory_size,
            'num_streams': self.num_streams if self.streams else 0,
            'optimization_enabled': self.optimize_memory,
            'optimized_memory_size': self.optimized_memory.size(0) if self.optimized_memory is not None else 0,
            'quantization_enabled': self.quantize,
            'quantization_bits': self.quantization_bits,
            'quantized_memory_size': self.quantized_memory.size(0) if self.quantized_memory is not None else 0
        }
        
        # Add distributed stats if available
        if self.distributed:
            stats.update({
                'distributed_memory_size': self.distributed_memory_size,
                'world_size': self.world_size,
                'rank': self.rank
            })
        
        return stats
    
    def reset_memory(self) -> None:
        """Reset memory to initial state."""
        self.memory.fill_(0)
        self.current_size = 0
        self.usage_counts.fill_(0)
        self.attention_weights.fill_(0)
        
        self.logger.debug("Memory reset")
    
    def get_stats(self) -> dict:
        """Get memory statistics."""
        return self.stats
