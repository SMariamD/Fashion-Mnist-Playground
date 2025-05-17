import torch
from torch import nn
from typing import Optional, Tuple
import logging
from .memory_v2 import Memory
from .spiking import SpikingLayer

logger = logging.getLogger(__name__)

class S3CMN(nn.Module):
    """S³CMN (Spiking-State-Space Convolutional Memory Network) model."""
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 256,
        hidden_dim: int = 512,
        num_layers: int = 4,
        memory_size: int = 1024,
        compressed_dim: int = 128,
        device: str = "cpu",
    ):
        super().__init__()
        
        # Initialize embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )
        
        # Initialize memory system
        self.memory = Memory(
            memory_size=memory_size,
            embedding_dim=hidden_dim,
            compressed_dim=compressed_dim,
            device=device
        )
        
        # Initialize convolutional frontend
        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=embedding_dim,
                out_channels=hidden_dim,
                kernel_size=1,
                stride=1,
                padding=0,
            ),
            nn.GELU(),
        )
        
        # Initialize spiking layers with dimension scaling
        self.spiking_layers = nn.ModuleList([
            SpikingLayer(
                embedding_dim=hidden_dim,
                memory_size=memory_size,
                device=device
            )
            for _ in range(num_layers)
        ])
        
        # Dimension scaling parameters
        self.scaling_factors = nn.ParameterList([
            nn.Parameter(torch.tensor(1.0)) for _ in range(num_layers)
        ])
        
        # Initialize S4 layers
        self.ssm = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim) for _ in range(num_layers)
        ])
        
        # Initialize output layer
        self.output = nn.Linear(hidden_dim, vocab_size)
        
        # Initialize memory state
        self.register_buffer('memory_state', torch.zeros(1, hidden_dim))
        
        # WikiText-2 specific configurations
        self.wikitext_config = {
            'max_seq_length': 128,
            'batch_size': 32,
            'num_workers': 4
        }
        
        self.to(device)
    
    def scale_dimensions(self, scaling_factor: float):
        """Scale all dimensions by a factor."""
        for param in self.parameters():
            if param.requires_grad:
                param.data *= scaling_factor
        
    def get_scaled_model(self, scaling_factor: float):
        """Get a new model with scaled dimensions."""
        scaled_model = S3CMN(
            vocab_size=self.embedding.num_embeddings,
            embedding_dim=int(self.embedding.embedding_dim * scaling_factor),
            hidden_dim=int(self.embedding.embedding_dim * scaling_factor),
            num_layers=len(self.spiking_layers),
            memory_size=self.spiking_layers[0].memory_size,
            compressed_dim=self.memory.compressed_dim,
            device=self.device
        )
        return scaled_model
    
    def evaluate_wikitext(self, dataloader):
        """Evaluate model on WikiText-2 dataset."""
        self.eval()
        total_loss = 0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                # Forward pass
                logits, _ = self(input_ids)
                
                # Compute loss (ignoring padding tokens)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100
                )
                
                # Update metrics
                total_loss += loss.item()
                total_tokens += (labels != -100).sum().item()
        
        # Calculate perplexity
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        
        return {
            'perplexity': perplexity,
            'loss': avg_loss
        }
    
    def forward(
        self,
        input_ids: torch.Tensor,
        memory_state: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            input_ids: Input token IDs of shape (batch_size, seq_len)
            memory_state: Optional previous memory state
        Returns:
            Tuple of (logits, new_memory_state)
        """
        # Ensure proper shape
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        
        # Embedding
        x = self.embedding(input_ids)
        
        # Convolutional frontend
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        
        # Process through spiking layers with scaling
        memory_states = []
        for i, layer in enumerate(self.spiking_layers):
            # Apply scaling factor to input
            x = x * self.scaling_factors[i]
            
            # Process through layer
            spikes, memory_state = layer(x)
            x = spikes
            memory_states.append(memory_state)
        
        # S4 layers
        for layer in self.ssm:
            x = layer(x)
        
        # Output layer
        logits = self.output(x)
        
        return logits, memory_states[-1]

    def reset_memory(self):
        """Reset the memory state."""
        self.memory_state = torch.zeros(1, self.embedding.embedding_dim, device=self.memory_state.device)
        self.memory.reset()
        for layer in self.spiking_layers:
            layer.reset()
    

    def get_memory_state(self) -> torch.Tensor:
        """Get the current memory state."""
        return self.memory_state
