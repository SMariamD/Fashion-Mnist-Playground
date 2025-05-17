import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from typing import Optional, Dict, Tuple
import logging
from .memory_v2 import Memory

logger = logging.getLogger(__name__)

class SpikingLayer(nn.Module):
    """Spiking activation layer with memory integration."""
    def __init__(
        self, 
        embedding_dim: int,
        memory_size: int,
        device: str = 'cpu'
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.memory_size = memory_size
        self.device = device
        
        # Initialize memory system
        self.memory = Memory(
            memory_size=memory_size,
            embedding_dim=embedding_dim,
            compressed_dim=128,
            device=device
        )
        
        # Spiking neuron
        self.spike = snn.Leaky(beta=0.9)
        self.beta = torch.tensor(0.9)
        
        # Adaptive threshold parameters
        self.base_threshold = 1.0
        self.threshold_range = 0.5  # Range for adaptive threshold
        self.target_spike_rate = 0.01  # Target spike rate
        self.membrane_potential = None
        self.spike_count = 0
        self.target_spike_rate = 0.02
        self.spike_rate_window = 100
        self.spike_rate_history = []
        
        # Layer-specific spiking parameters
        self.layer_threshold = threshold * (1 + layer_id * 0.1)  # Layer-specific threshold
        self.layer_beta = beta * (1 - layer_id * 0.05)  # Layer-specific leak rate
        
        self.to(device)
    
    def update_threshold(self, spikes: torch.Tensor) -> float:
        """Update threshold based on spike rate."""
        if isinstance(spikes, tuple):
            spikes = spikes[0]
        
        # Update spike rate history
        self.spike_rate_history.append(spikes.mean().item())
        if len(self.spike_rate_history) > self.spike_rate_window:
            self.spike_rate_history.pop(0)
        
        # Calculate current spike rate
        current_spike_rate = sum(self.spike_rate_history) / len(self.spike_rate_history)
        
        # Adjust threshold based on layer position
        adjustment_factor = 1.0 + (self.layer_id / self.num_layers) * 0.1
        
        # Adjust threshold
        if current_spike_rate > self.target_spike_rate:
            self.layer_threshold *= 1.01 * adjustment_factor
        else:
            self.layer_threshold *= 0.99 * adjustment_factor
        
        return current_spike_rate
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: Input tensor of shape (batch_size, embedding_dim)
        
        Returns:
            Tuple of (spikes, memory_state)
        """
        # Get input dimensions
        batch_size = x.size(0)
        embedding_dim = x.size(1)
        
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Initialize membrane potential if not set
        if self.membrane_potential is None:
            self.membrane_potential = torch.zeros(batch_size, embedding_dim, device=self.device)
        else:
            # Reset membrane potential for new batch
            self.membrane_potential = torch.zeros(batch_size, embedding_dim, device=self.device)
        
        # Query memory for relevant context
        memory_context = self.memory.get_memory(x, top_k=5)
        
        # Get attention weights
        attention_weights = self.memory.get_attention_weights()
        
        # Apply layer-specific attention-based fusion
        alpha = self.memory_alpha * (1 - self.layer_id / self.num_layers)  # Layer-specific fusion weight
        fused_input = alpha * memory_context + (1 - alpha) * x
        
        # Update attention weights based on memory usage
        self.memory.update_attention(
            torch.arange(batch_size, device=self.device),
            torch.ones(batch_size, device=self.device)
        )
        
        # Update membrane potential with layer-specific parameters
        self.membrane_potential = self.layer_beta * self.membrane_potential + fused_input
        
        # Spike generation with layer-specific threshold
        spikes = (self.membrane_potential > self.layer_threshold).float()
        
        # Update spike count
        if isinstance(spikes, tuple):
            spikes = spikes[0]  # Take the first element if it's a tuple
        self.spike_count += spikes.sum()
        
        # Update threshold based on spike rate
        current_spike_rate = self.update_threshold(spikes)
        
        # Memory operations with layer-specific parameters
        self.memory.add_to_memory(spikes, spikes)
        self.memory.add_to_memory(fused_input, fused_input)
        
        # Get memory stats
        memory_stats = self.memory.get_stats()
        
        # Update memory state with layer-specific information
        memory_state = torch.tensor([
            memory_stats['hit_rate'],
            memory_stats['avg_similarity'],
            memory_stats['memory_usage'],
            current_spike_rate,  # Add current spike rate to memory state
            attention_weights.mean().item(),  # Add average attention weight
            self.layer_id / self.num_layers  # Add layer position
        Args:
            x: Input tensor of shape (batch_size, embedding_dim)
        
        Returns:
            Tuple of (spikes, memory_state)
        """
        # Get input dimensions
        batch_size = x.size(0)
        embedding_dim = x.size(1)
        
        # Ensure input is on the correct device
        x = x.to(self.device)
        
        # Initialize membrane potential if not set
        if self.membrane_potential is None:
            self.membrane_potential = torch.zeros(batch_size, embedding_dim, device=self.device)
        else:
            # Reset membrane potential for new batch
            self.membrane_potential = torch.zeros(batch_size, embedding_dim, device=self.device)
        
        # Query memory for relevant context
        memory_context = self.memory.get_memory(x, top_k=5)
        
        # Fuse memory context with input using weighted sum
        alpha = 0.3  # Memory fusion weight
        fused_input = alpha * memory_context + (1 - alpha) * x
        
        # Update membrane potential with fused input
        self.membrane_potential = self.beta * self.membrane_potential + fused_input
        
        # Spike generation
        spikes = self.spike(self.membrane_potential)
        
        # Update spike count
        if isinstance(spikes, tuple):
            spikes = spikes[0]  # Take the first element if it's a tuple
        self.spike_count += spikes.sum()
        
        # Memory operations
        self.memory.add_to_memory(spikes, spikes)
        self.memory.add_to_memory(fused_input, fused_input)
        
        # Get memory stats
        memory_stats = self.memory.get_stats()
        
        # Update memory state
        memory_state = torch.tensor([
            memory_stats['hit_rate'],
            memory_stats['avg_similarity'],
            memory_stats['memory_usage']
        ], device=self.device)
        
        return spikes, memory_state

    def reset(self):
        """Reset the layer state."""
        self.membrane_potential = None
        self.spike_count = 0
        self.memory.reset()

    def set_beta(self, beta: torch.Tensor):
        """Set the beta value for the spiking layer."""
        self.beta = beta
        self.spike.beta = beta

    def set_threshold(self, threshold: float):
        """Set the spiking threshold."""
        self.threshold = threshold

    def get_state(self) -> Dict[str, float]:
        """Get the current state of the spiking layer."""
        return {
            'beta': float(self.beta),
            'threshold': self.threshold,
            'spike_count': float(self.spike_count),
            'membrane_potential': float(self.membrane_potential.mean()) if self.membrane_potential is not None else 0.0
        }
