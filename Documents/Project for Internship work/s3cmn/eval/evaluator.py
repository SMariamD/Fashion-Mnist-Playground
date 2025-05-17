import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from s3cmn.model.spiking import SpikingLayer
import time
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, model, tokenizer, device='cpu'):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.to(device)
        
        # Initialize memory tracking
        self.memory_history = []
        self.spiking_history = []
        self.processing_times = []
        
    def log_metrics(self, metrics):
        """Log metrics for later analysis."""
        self.memory_history.append({
            'hit_rate': metrics.get('memory_hit_rate', 0),
            'avg_similarity': metrics.get('memory_avg_similarity', 0),
            'usage': metrics.get('memory_usage', 0)
        })
        
        if 'spike_rate' in metrics:
            self.spiking_history.append({
                'spike_rate': metrics['spike_rate'],
                'sparsity': metrics['sparsity']
            })
            
        self.processing_times.append(metrics.get('processing_time_ms_per_token', 0))
        
    def get_summary_stats(self):
        """Get summary statistics of all logged metrics."""
        return {
            'avg_memory_hit_rate': np.mean([m['hit_rate'] for m in self.memory_history]),
            'avg_memory_similarity': np.mean([m['avg_similarity'] for m in self.memory_history]),
            'avg_memory_usage': np.mean([m['usage'] for m in self.memory_history]),
            'avg_spike_rate': np.mean([s['spike_rate'] for s in self.spiking_history]),
            'avg_sparsity': np.mean([s['sparsity'] for s in self.spiking_history]),
            'avg_processing_time': np.mean(self.processing_times)
        }
        
    def calculate_perplexity(self, text):
        """Calculate perplexity of a given text."""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Calculate loss manually since we're using causal LM
            logits = outputs.logits
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs['input_ids'][..., 1:].contiguous()
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, logits.size(-1)), shift_labels.view(-1))
            perplexity = torch.exp(loss).item()
        
        return perplexity

    def measure_processing_time(self, text, batch_size=1, iterations=10):
        """Measure processing time per token."""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        times = []
        for _ in range(iterations):
            start_time = time.time()
            _ = self.model(**inputs)
            end_time = time.time()
            times.append(end_time - start_time)
        
        avg_time = np.mean(times)
        tokens = len(inputs['input_ids'][0])
        return avg_time / tokens

    def measure_memory_usage(self, text):
        """Measure peak memory usage."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        inputs = self.tokenizer(text, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            _ = self.model(**inputs)
            
        return torch.cuda.max_memory_allocated() / 1024**2  # Convert to MB

    def evaluate_sparsity(self, text, layer):
        """Evaluate spiking sparsity metrics."""
        inputs = self.tokenizer(text, return_tensors='pt', padding=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings from the model
        with torch.no_grad():
            embeddings = self.model.transformer.wte(inputs['input_ids'])
            
            # Process each token embedding through the spiking layer
            spikes = []
            for i in range(embeddings.size(1)):
                spike, _ = layer(embeddings[:, i, :])
                spikes.append(spike)
            
            # Stack all spikes
            spikes = torch.stack(spikes, dim=1)
        
        total_elements = spikes.numel()
        active_elements = (spikes > 0).sum().item()
        
        return {
            'spike_rate': active_elements / total_elements,
            'sparsity': 1 - (active_elements / total_elements)
        }

    def evaluate_model(self, text, layer=None):
        """Evaluate model with all metrics."""
        results = {}
        
        # Perplexity
        results['perplexity'] = self.calculate_perplexity(text)
        
        # Processing time
        results['processing_time_ms_per_token'] = self.measure_processing_time(text) * 1000
        
        # Memory usage
        if torch.cuda.is_available():
            results['peak_memory_mb'] = self.measure_memory_usage(text)
            results['device'] = 'GPU'
        else:
            results['device'] = 'CPU'
        
        # Sparsity metrics if layer is provided
        if layer is not None:
            sparsity_metrics = self.evaluate_sparsity(text, layer)
            results.update(sparsity_metrics)
            
            # Get detailed memory stats
            memory_stats = layer.memory.get_stats()
            results.update({
                'memory_hit_rate': memory_stats['hit_rate'],
                'memory_avg_similarity': memory_stats['avg_similarity'],
                'memory_usage': memory_stats['memory_usage'],
                'memory_capacity': layer.memory.memory_size,
                'memory_compression_ratio': layer.memory.compression_ratio
            })
        
        return results
