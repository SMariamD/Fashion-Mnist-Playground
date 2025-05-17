import torch
import logging
import numpy as np
from transformers import GPT2Tokenizer
from model.model import S3CMN
from model.memory_v2 import Memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_text(
    model_path: str,
    prompt: str,
    max_length: int = 50,
    temperature: float = 1.0,
    top_p: float = 0.9,
    device: str = "cpu"
):
    """Generate text using the trained model."""
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    
    # Load model
    from utils.tokenizer import S3CMNTokenizer
    tokenizer = S3CMNTokenizer()
    
    model = S3CMN(
        vocab_size=tokenizer.get_vocab_size(),
        embedding_dim=256,
        hidden_dim=512,
        num_layers=1,
        memory_size=1024,
        device=device
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Prepare input
    input_ids = tokenizer.encode(prompt).unsqueeze(0).to(device)  # Make it [1, seq_len]
    
    # Generate text
    generated = []
    with torch.no_grad():
        for _ in range(max_length):
            # Forward pass
            logits, _ = model(input_ids)
            
            # Get last token logits
            logits = logits[:, -1, :]
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply top-p sampling
            probs = torch.softmax(logits, dim=-1)
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = (cumulative_probs < top_p).sum(dim=-1)
            
            # Sample from filtered distribution
            indices_to_sample = sorted_indices[0, :cutoff.item()]
            filtered_probs = probs[0, indices_to_sample]
            
            # Sample from filtered distribution
            if filtered_probs.numel() == 0:
                # If no tokens pass top-p filtering, use the top token
                next_token = torch.argmax(probs[0])
            else:
                next_token = torch.multinomial(filtered_probs, 1)
                next_token = indices_to_sample[next_token]
            
            # Add to generated sequence
            generated.append(next_token.item())
            
            # Update input
            next_token = next_token.unsqueeze(0)  # Make it [1, 1]
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Keep only the last sequence length tokens
            if input_ids.size(1) > 50:
                input_ids = input_ids[:, -50:]
            
            # Stop if EOS token is generated
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    # Decode and return
    generated_text = tokenizer.decode(generated)
    return prompt + generated_text

def main():
    # Example usage
    model_path = "s3cmn_model.pth"
    prompt = "Once upon a time"
    
    logger.info("Generating text...")
    generated_text = generate_text(
        model_path=model_path,
        prompt=prompt,
        max_length=100,
        temperature=0.7,
        top_p=0.9
    )
    
    logger.info("\nGenerated text:")
    logger.info(generated_text)

if __name__ == "__main__":
    main()
