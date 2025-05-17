import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from ..model.spiking import SpikingLayer
from .evaluator import ModelEvaluator
import logging

logger = logging.getLogger(__name__)

def run_benchmarks():
    # Initialize logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Load GPT-2 small for comparison
    gpt2_tokenizer = AutoTokenizer.from_pretrained('gpt2')
    # Add special tokens for padding
    gpt2_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    gpt2_model = AutoModelForCausalLM.from_pretrained('gpt2')
    # Resize model embeddings to accommodate new tokens
    gpt2_model.resize_token_embeddings(len(gpt2_tokenizer))
    
    # Create spiking layer for evaluation
    spiking_layer = SpikingLayer(
        embedding_dim=768,  # Same as GPT-2 small
        memory_size=1000,
        device='cpu'
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator(gpt2_model, gpt2_tokenizer, device='cpu')
    
    # Example text for evaluation
    test_text = "The quick brown fox jumps over the lazy dog. " * 10
    
    # Run multiple iterations for better statistics
    num_iterations = 5
    for i in range(num_iterations):
        logger.info(f"\nIteration {i+1}/{num_iterations}")
        
        # Evaluate GPT-2
        logger.info("Evaluating GPT-2 small...")
        gpt2_results = evaluator.evaluate_model(test_text)
        evaluator.log_metrics(gpt2_results)
        
        # Evaluate Spiking model
        logger.info("\nEvaluating Spiking model...")
        spiking_results = evaluator.evaluate_model(test_text, layer=spiking_layer)
        evaluator.log_metrics(spiking_results)
    
    # Print summary statistics
    logger.info("\nSummary Statistics:")
    summary = evaluator.get_summary_stats()
    for key, value in summary.items():
        logger.info(f"{key}: {value:.4f}")
    
    # Print final comparison
    logger.info("\nFinal Comparison:")
    logger.info(f"GPT-2 Perplexity: {gpt2_results['perplexity']:.4f}")
    logger.info(f"Spiking Model Perplexity: {spiking_results['perplexity']:.4f}")
    logger.info(f"GPT-2 Processing Time: {gpt2_results['processing_time_ms_per_token']:.4f}ms/token")
    logger.info(f"Spiking Model Processing Time: {spiking_results['processing_time_ms_per_token']:.4f}ms/token")
    logger.info(f"Spiking Model Memory Usage: {spiking_results['memory_usage']:.2f}%")
    logger.info(f"Spiking Model Spike Rate: {spiking_results['spike_rate']:.4f}")
    logger.info(f"Spiking Model Sparsity: {spiking_results['sparsity']:.4f}")

if __name__ == "__main__":
    # Run benchmarks
    run_benchmarks()
    
    # Optionally run GPU benchmarks if available
    if torch.cuda.is_available():
        logger.info("\nRunning GPU benchmarks...")
        evaluator = ModelEvaluator(gpt2_model, gpt2_tokenizer, device='cuda')
        spiking_layer = SpikingLayer(
            embedding_dim=768,
            memory_size=1000,
            device='cuda'
        )
        
        # Run GPU benchmarks
        run_benchmarks()
