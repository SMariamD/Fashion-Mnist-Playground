import torch
import re
from model.model import S3CMN
from utils.tokenizer import S3CMNTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7, top_p=0.9, device="cpu"):
    """Generate text from the model with temperature and top-p control."""
    model.eval()
    
    # Tokenize input
    tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor(tokens).unsqueeze(0).to(device)
    
    # Generate text
    generated_tokens = []
    current_input = input_ids
    
    for _ in range(max_length):
        # Get model output
        with torch.no_grad():
            logits, memory_state = model(current_input)
        
        # Apply temperature
        logits = logits / temperature
        
        # Apply top-p (nucleus) sampling
        probs = torch.softmax(logits[0, -1], dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        
        # Remove tokens above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Get the indices to remove
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        probs[indices_to_remove] = 0
        
        # Normalize probabilities
        probs = probs / probs.sum()
        
        # Sample from probabilities
        next_token = torch.multinomial(probs, 1)
        
        # Add token to generated sequence
        generated_tokens.append(next_token.item())
        
        # Create new input
        current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1)
        
        # Reset memory for next token
        model.reset_memory()
    
    # Decode generated tokens
    generated_text = tokenizer.decode(generated_tokens)
    
    # Post-process text to make it more readable
    # Remove special tokens and clean up
    generated_text = generated_text.replace('[PAD]', '').replace('[CLS]', '').replace('[SEP]', '').strip()
    
    # Clean up BERT subwords
    generated_text = generated_text.replace(' ##', '')
    
    # Remove unused tokens
    generated_text = re.sub(r'\[unused\d+\]', '', generated_text)
    
    # Clean up any remaining special characters
    generated_text = re.sub(r'\[\w+\]', '', generated_text)
    
    # Remove extra whitespace
    generated_text = ' '.join(generated_text.split())
    
    return generated_text

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
    
    # Load trained model weights
    try:
        model.load_state_dict(torch.load("s3cmn_model.pth", map_location=device))
        logger.info("Loaded trained model weights")
    except:
        logger.warning("Could not load trained model weights. Using random initialization.")
    
    # Test different temperatures
    prompts = [
        {"prompt": "The quick brown fox jumps over the lazy dog.",
         "temperature": 0.5,
         "top_p": 0.9},
        
        {"prompt": "Once upon a time in a land far away,",
         "temperature": 0.7,
         "top_p": 0.85},
        
        {"prompt": "In the year 2050,",
         "temperature": 0.9,
         "top_p": 0.95}
    ]
    
    for params in prompts:
        logger.info(f"\n=== Generating from prompt: {params['prompt']} ===")
        logger.info(f"Temperature: {params['temperature']}, Top-p: {params['top_p']}")
        generated = generate_text(model, tokenizer, 
                                prompt=params['prompt'],
                                temperature=params['temperature'],
                                top_p=params['top_p'],
                                device=device)
        logger.info(f"Generated text: {generated}")

if __name__ == "__main__":
    main()
