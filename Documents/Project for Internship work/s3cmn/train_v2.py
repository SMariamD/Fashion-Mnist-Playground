import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from model.model import S3CMN
from utils.tokenizer import S3CMNTokenizer
import logging
import random

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

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (input_ids, target_ids) in enumerate(train_loader):
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits, memory_state = model(input_ids)
        
        # Compute loss
        loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))
        
        # Backward pass
        loss.backward(retain_graph=True)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:
            logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / len(train_loader)
    logger.info(f"Average loss: {avg_loss:.4f}")
    return avg_loss

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on device: {device}")
    
    # Initialize model and tokenizer
    tokenizer = S3CMNTokenizer()
    model = S3CMN(
        vocab_size=tokenizer.get_vocab_size(),
        embedding_dim=256,
        hidden_dim=512,
        num_layers=1,
        memory_size=1024,
        device=device
    ).to(device)
    
    # Training parameters
    batch_size = 32
    seq_length = 50
    num_epochs = 5
    learning_rate = 0.001
    
    # Create training data
    # Using a simple text for now - in practice you would use a larger corpus
    training_text = "".join([
        "The quick brown fox jumps over the lazy dog. ",
        "Once upon a time in a land far away, there was a brave knight. ",
        "In the year 2050, technology had advanced beyond imagination. ",
        "Artificial intelligence had become an integral part of daily life. ",
        "The future of humanity was uncertain but full of possibilities. ",
        "Exploration of space had opened new frontiers for civilization. ",
        "Renewable energy sources had solved the world's energy crisis. ",
        "Virtual reality had revolutionized entertainment and education. ",
        "Quantum computing had unlocked new possibilities in computation. ",
        "The world was a better place than it had ever been."
    ] * 100)  # Repeat text to create more training data
    
    # Create dataset and dataloader
    dataset = TextDataset(training_text, tokenizer, seq_length)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    for epoch in range(num_epochs):
        logger.info(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")
        avg_loss = train(model, train_loader, optimizer, criterion, device)
        logger.info(f"Average loss: {avg_loss:.4f}")
    
    # Save the trained model
    torch.save(model.state_dict(), "s3cmn_model.pth")
    logger.info("Model saved successfully!")

if __name__ == "__main__":
    main()
