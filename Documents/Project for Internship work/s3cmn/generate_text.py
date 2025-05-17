import torch
from model.model import S3CMN
from utils.tokenizer import S3CMNTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7, device="cpu"):
    """Generate text from the model with temperature control."""
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
        
        # Get probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Sample from probabilities
        next_token = torch.multinomial(probs[0, -1], 1)
        
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
    
    # Test different temperatures
    prompts = [
        {"prompt": "The quick brown fox jumps over the lazy dog.",
         "temperature": 0.5},
        
        {"prompt": "Once upon a time in a land far away,",
         "temperature": 0.7},
        
        {"prompt": "In the year 2050,",
         "temperature": 0.9}
    ]
    
    for params in prompts:
        logger.info(f"\n=== Generating from prompt: {params['prompt']} ===")
        logger.info(f"Temperature: {params['temperature']}")
        generated = generate_text(model, tokenizer, 
                                prompt=params['prompt'],
                                temperature=params['temperature'],
                                device=device)
        logger.info(f"Generated text: {generated}")

if __name__ == "__main__":
    main()

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Running on device: {device}")
    
    # Initialize model and tokenizer
    tokenizer = S3CMNTokenizer()
    model = S3CMN(
        vocab_size=tokenizer.get_vocab_size(),
        embedding_dim=256,
    ]
    
    for prompt in prompts:
        logger.info(f"\n=== Generating from prompt: {prompt} ===")
        generated = generate_text(model, tokenizer, prompt, max_length=50, temperature=0.7, device=device)
        logger.info(f"Generated text: {generated}")

if __name__ == "__main__":
    main()
