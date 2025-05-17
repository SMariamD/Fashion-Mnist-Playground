import torch
from typing import Optional, Tuple
import faiss
import numpy as np

class MemorySystem:
    """Memory system with FAISS-based retrieval."""
    def __init__(
        self,
        embedding_dim: int,
        memory_size: int,
        num_clusters: int = 128,
        max_retrieval_distance: float = 1.0,
        memory_compression: bool = True,
        device: str = "cpu"
    ):  # Add device parameter with default value
        self.embedding_dim = embedding_dim
        self.memory_size = memory_size
        self.embedding_dim = embedding_dim
        self.compressed_dim = compressed_dim
        self.device = device
        
        # Initialize memory tensor
        self.memory = nn.Parameter(torch.zeros(memory_size, compressed_dim), requires_grad=False)
        self.current_size = 0
        
        # Initialize stats
        self.stats = {
            'memory_usage': 0,
            'avg_similarity': 0
        }
        
        # Initialize logger
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def _calculate_avg_similarity(self):
        """Calculate average similarity in memory."""
        if self.current_size < 2:
            return 0
            
        # Get random pairs of memories
        indices = torch.randperm(self.current_size)[:100]
        pairs = torch.combinations(indices, 2)
        
        # Calculate similarities
        similarities = []
        for i, j in pairs:
            sim = F.cosine_similarity(
                self.memory[i].unsqueeze(0),
                self.memory[j].unsqueeze(0)
            )
            similarities.append(sim.item())
            
        return sum(similarities) / len(similarities) if similarities else 0
    
    def add_to_memory(self, values, keys=None):
        """Add values to memory."""
        if keys is None:
            keys = values
            
        batch_size = values.size(0)
        
        # Compress values
        compressed_values = self._compress(values)
        
        # Add to memory
        start_idx = self.current_size
        end_idx = min(self.current_size + batch_size, self.memory_size)
        
        # Store compressed values
        self.memory[start_idx:end_idx] = compressed_values
        self.current_size = end_idx
        
        # Update FAISS index
        if self.current_size >= self.num_clusters:
            self._update_faiss_index()

    def retrieve_from_memory(self, query: torch.Tensor, k: int = 1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve similar entries from memory.
        
        Args:
            query: Tensor of shape (batch_size, embedding_dim)
            k: Number of nearest neighbors to retrieve
        
        Returns:
            Tuple of (similarities, indices)
        """
        if self.current_size == 0:
            return torch.zeros(query.size(0), k), torch.zeros(query.size(0), k, dtype=torch.long)
        
        # Convert to numpy for FAISS
        query_np = query.numpy()
        
        # Search in FAISS
        similarities, indices = self.index.search(query_np, k)
        
        return torch.from_numpy(similarities), torch.from_numpy(indices)

    def get_memory_values(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Get memory values by indices.
        
        Args:
            indices: Tensor of shape (batch_size, k)
        
        Returns:
            Tensor of shape (batch_size, k, embedding_dim)
        """
        batch_size, k = indices.size()
        values = torch.zeros(batch_size, k, self.embedding_dim)
        
        for i in range(batch_size):
            for j in range(k):
                idx = indices[i, j]
                if idx < self.current_size:
                    values[i, j] = self.memory[idx]
        
        return values

    def get_memory_stats(self) -> dict:
        """Get memory statistics."""
        return self.stats

    def reset(self):
        """Reset memory system."""
        self.memory = torch.zeros(self.memory_size, self.compressed_dim, device=self.device)
        self.current_size = 0
        self.stats = {
            'hit_rate': 0.0,
            'avg_similarity': 0.0,
            'retrieval_time': 0.0,
            'memory_usage': 0.0,
            'spike_state': 0.0
        }
        print(f"Memory reset with shape: {self.memory.shape}")  # Debug

    def _compress_keys(self, keys: torch.Tensor) -> torch.Tensor:
        """Apply compression to keys using a linear transformation."""
        if not self.memory_compression:
            return keys
            
        # Get the number of components to keep
        num_components = min(keys.size(-1), 128)  # Limit to 128 components
        
        # Create a linear transformation matrix
        if not hasattr(self, '_compression_matrix') or self._compression_matrix.size(1) != keys.size(-1):
            self._compression_matrix = torch.nn.Parameter(
                torch.randn(keys.size(-1), num_components),
                requires_grad=False
            ).to(keys.device)
            
        # Apply linear transformation
        compressed = torch.matmul(keys, self._compression_matrix)
        
        # Normalize the output
        compressed = compressed / torch.norm(compressed, dim=-1, keepdim=True)
        
        return compressed

    def _compress_values(self, values: torch.Tensor) -> torch.Tensor:
        """Apply compression to values using a linear transformation."""
        if not self.memory_compression:
            return values
            
        # Get the number of components to keep
        num_components = min(values.size(-1), 128)  # Limit to 128 components
        
        # Create a linear transformation matrix
        if not hasattr(self, '_value_compression_matrix') or self._value_compression_matrix.size(1) != values.size(-1):
            self._value_compression_matrix = torch.nn.Parameter(
                torch.randn(values.size(-1), num_components),
                requires_grad=False
            ).to(values.device)
            
        # Apply linear transformation
        compressed = torch.matmul(values, self._value_compression_matrix)
        
        # Normalize the output
        compressed = compressed / torch.norm(compressed, dim=-1, keepdim=True)
        
        return compressed

    def _update_faiss_index(self):
        """Update FAISS index."""
        # Train the index
        self.index.train(self.memory[:self.current_size].numpy())
        
        # Add vectors
        self.index.add(self.memory[:self.current_size].numpy())
