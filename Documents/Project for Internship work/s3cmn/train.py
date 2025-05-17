import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from model.model import S3CMN
from utils.tokenizer import S3CMNTokenizer
import logging
import random
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_length=50):
        self.tokenizer = tokenizer
        self.seq_length = seq_length
        self.text = text
        self.tokens = tokenizer.encode(text)
        
    def __len__(self):
        return len(self.tokens) - self.seq_length
    
    def __getitem__(self, idx):
        input_ids = self.tokens[idx:idx + self.seq_length]
        target_ids = self.tokens[idx + 1:idx + self.seq_length + 1]
        return torch.tensor(input_ids), torch.tensor(target_ids)

def train(model, train_loader, optimizer, criterion, device, scaler, epoch):
    model.train()
    total_loss = 0
    
    for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        with autocast():
            logits, _ = model(input_ids)
            
            # Compute loss
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # Log average loss
    avg_loss = total_loss / len(train_loader)
    logger.info(f"\nEpoch {epoch + 1} Average Loss: {avg_loss:.4f}")
    return avg_loss


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, path)
    logger.info(f"Checkpoint saved to {path}")


def main():
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seq_length = 50
    batch_size = 32
    learning_rate = 1e-4
    num_epochs = 15  # Increased epochs for dimension scaling
    
    # GPU benchmarking parameters
    batch_sizes = [32, 64, 128]
    seq_lengths = [64, 128, 256]
    current_bs_idx = 0
    current_seq_idx = 0
    
    # Dimension scaling parameters
    scaling_factors = [0.5, 1.0, 1.5]  # Scale by 50%, 100%, 150%
    current_scaling_idx = 0
    
    # Learning rate scheduling
    learning_rates = [1e-4, 5e-5, 1e-5]
    current_lr_idx = 0
    
    # Threshold tuning parameters
    threshold_ranges = [0.5, 1.0, 1.5]
    target_spike_rates = [0.01, 0.02, 0.05]
    current_threshold_idx = 0
    
    # Initialize model and tokenizer
    model = S3CMN(
        vocab_size=tokenizer.get_vocab_size(),
        embedding_dim=256,
        hidden_dim=512,
        num_layers=1,
        memory_size=1024,
        device=device
    ).to(device)
    tokenizer = S3CMNTokenizer()
    
    # Load data
    with open("data/train.txt", "r", encoding="utf-8") as f:
        text = f.read()
    
    # Create dataset and dataloader
    dataset = TextDataset(text, tokenizer, seq_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rates[current_lr_idx], weight_decay=0.01)
    scaler = GradScaler()
    
    # Create checkpoints directory
    os.makedirs("checkpoints", exist_ok=True)
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(num_epochs):
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Update batch size and sequence length for GPU benchmarking
        if epoch % 4 == 0 and current_bs_idx < len(batch_sizes) - 1:
            current_bs_idx += 1
            current_seq_idx = min(current_seq_idx + 1, len(seq_lengths) - 1)
            
            # Update dataloader
            train_loader = DataLoader(
                dataset,
                batch_size=batch_sizes[current_bs_idx],
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                drop_last=True
            )
            
            # Update model sequence length
            model.wikitext_config['max_seq_length'] = seq_lengths[current_seq_idx]
            logger.info(f"Updated batch size to {batch_sizes[current_bs_idx]}, seq length to {seq_lengths[current_seq_idx]}")
        
        # Update dimension scaling
        if epoch % 5 == 0 and current_scaling_idx < len(scaling_factors) - 1:
            current_scaling_idx += 1
            scaled_model = model.get_scaled_model(scaling_factors[current_scaling_idx])
            scaled_model.to(device)
            model = scaled_model
            
            # Update optimizer with new model parameters
            optimizer = torch.optim.AdamW(model.parameters(), 
                                        lr=learning_rates[current_lr_idx],
                                        weight_decay=0.01)
            logger.info(f"Updated model dimensions with scaling factor {scaling_factors[current_scaling_idx]}")
        
        # Update learning rate
        if epoch % 2 == 0 and current_lr_idx < len(learning_rates) - 1:
            current_lr_idx += 1
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rates[current_lr_idx]
            logger.info(f"Updated learning rate to {learning_rates[current_lr_idx]}")
        
        # Update threshold parameters
        if epoch % 3 == 0 and current_threshold_idx < len(threshold_ranges) - 1:
            current_threshold_idx += 1
            model.set_threshold_params(
                threshold_range=threshold_ranges[current_threshold_idx],
                target_spike_rate=target_spike_rates[current_threshold_idx]
            )
            logger.info(f"Updated threshold range to {threshold_ranges[current_threshold_idx]}")
        
        # Measure GPU memory usage
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated() / 1e9
            memory_reserved = torch.cuda.memory_reserved() / 1e9
            logger.info(f"GPU Memory: Allocated={memory_allocated:.2f}GB, Reserved={memory_reserved:.2f}GB")
        
        # Train for one epoch
        avg_loss = train(model, train_loader, optimizer, criterion, device, scaler, epoch)
        
        # Save checkpoint
        checkpoint_path = f"checkpoints/model_epoch_{epoch + 1}.pt"
        save_checkpoint(model, optimizer, epoch, avg_loss, checkpoint_path)
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = "checkpoints/best_model.pt"
            save_checkpoint(model, optimizer, epoch, avg_loss, best_path)

if __name__ == "__main__":
    main()
