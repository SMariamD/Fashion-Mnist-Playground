import torch
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from model.model import S3CMN
from utils.tokenizer import S3CMNTokenizer
import time
import torch.profiler
import logging
import os
import sys

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_perplexity(model, tokenizer, text, device="cpu"):
    """Calculate perplexity for a given model."""
    model.eval()
    
    # Tokenize input
    tokens = tokenizer.encode(text)
    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
    
    # Get logits
    with torch.no_grad():
        if isinstance(model, S3CMN):
            # Reset memory before each evaluation
            model.reset_memory()
            logits, _ = model(input_ids)
            
            # Get logits from the last layer
            logits = logits[:, -1, :]  # Take last token logits
        else:  # GPT-2
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]  # Take last token logits
            
    # Calculate probabilities
    probs = torch.softmax(logits, dim=-1)
    
    # Calculate log probabilities for the actual tokens
    log_probs = torch.gather(probs, -1, input_ids[:, -1].unsqueeze(-1)).squeeze(-1)
    
    # Calculate perplexity
    perplexity = torch.exp(-torch.mean(torch.log(log_probs))).item()
    return perplexity

def measure_speed_memory(model, tokenizer, text, device="cpu"):
    """Measure speed and memory usage."""
    model.eval()
    
    # Tokenize input
    tokens = tokenizer.encode(text)
    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
    
    # Warmup
    for _ in range(3):
        if isinstance(model, S3CMN):
            model.reset_memory()
        _ = model(input_ids)
    
    # Measure speed
    start_time = time.time()
    with torch.no_grad():
        for _ in range(100):  # Run multiple times for better measurement
            if isinstance(model, S3CMN):
                model.reset_memory()
            _ = model(input_ids)
    end_time = time.time()
    
    # Calculate ms/token
    total_tokens = len(tokens) * 100
    total_time = end_time - start_time
    ms_per_token = (total_time * 1000) / total_tokens
    
    # Measure memory
    if device == "cuda":
        peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # Convert to MB
    else:
        peak_memory = None
    
    return ms_per_token, peak_memory

def calculate_spike_rate(model, tokenizer, text, device="cpu"):
    """Calculate average spike rate."""
    model.eval()
    
    # Tokenize input
    tokens = tokenizer.encode(text)
    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
    
    # Get model output
    with torch.no_grad():
        logits, memory_state = model(input_ids)
    
    # Extract spike count from memory state
    spike_count = memory_state[0].item()
    total_neurons = model.spike.memory.embedding_dim  # Use memory's embedding_dim
    
    # Calculate spike rate
    spike_rate = spike_count / (len(tokens) * total_neurons)
    return spike_rate

def benchmark_s3cmn():
    """Run comprehensive benchmarking for S³CMN model."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running benchmarks on device: {device}")
    
    # Initialize models
    s3cmn_tokenizer = S3CMNTokenizer()
    s3cmn_model = S3CMN(
        vocab_size=s3cmn_tokenizer.get_vocab_size(),
        embedding_dim=256,
        hidden_dim=512,
        num_layers=1,
        memory_size=1024,
        device=device
    ).to(device)
    
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
    
    # Test text (using WikiText-2 validation set excerpt)
    test_text = """The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. 
    The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog."""
    
    # Perplexity comparison
    logger.info("\n=== Perplexity Comparison ===")
    s3cmn_ppx = calculate_perplexity(s3cmn_model, s3cmn_tokenizer, test_text, device)
    gpt2_ppx = calculate_perplexity(gpt2_model, gpt2_tokenizer, test_text, device)
    logger.info(f"S³CMN Perplexity: {s3cmn_ppx:.2f}")
    logger.info(f"GPT-2 Perplexity: {gpt2_ppx:.2f}")
    
    # Speed & Memory Comparison
    logger.info("\n=== Speed & Memory Comparison ===")
    s3cmn_speed, s3cmn_memory = measure_speed_memory(s3cmn_model, s3cmn_tokenizer, test_text, device)
    gpt2_speed, gpt2_memory = measure_speed_memory(gpt2_model, gpt2_tokenizer, test_text, device)
    logger.info(f"S³CMN Speed (ms/token): {s3cmn_speed:.2f}")
    logger.info(f"GPT-2 Speed (ms/token): {gpt2_speed:.2f}")
    if device == "cuda":
        logger.info(f"S³CMN Peak VRAM (MB): {s3cmn_memory:.2f}")
        logger.info(f"GPT-2 Peak VRAM (MB): {gpt2_memory:.2f}")
    
    # Sparsity Metrics
    logger.info("\n=== Sparsity Metrics ===")
    s3cmn_spike_rate = calculate_spike_rate(s3cmn_model, s3cmn_tokenizer, test_text, device)
    logger.info(f"S³CMN Spike Rate: {s3cmn_spike_rate:.4f}")

if __name__ == "__main__":
    benchmark_s3cmn()
