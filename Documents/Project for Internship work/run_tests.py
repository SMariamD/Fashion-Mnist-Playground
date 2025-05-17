import unittest
import torch
import numpy as np
from s3cmn.model.model import S3CMN
from s3cmn.model.memory import MemorySystem
from s3cmn.model.spiking import SpikingLayer
from s3cmn.utils.tokenizer import S3CMNTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TestS3CMNComponents(unittest.TestCase):
    def setUp(self):
        self.device = "cpu"
        self.tokenizer = S3CMNTokenizer()
        self.model = S3CMN(
            vocab_size=self.tokenizer.get_vocab_size(),
            embedding_dim=256,
            hidden_dim=512,
            num_layers=1,
            memory_size=1024,
            device=self.device
        )
        self.model.to(self.device)

    def test_tokenizer(self):
        """Test tokenizer functionality"""
        text = "The quick brown fox jumps over the lazy dog."
        tokens = self.tokenizer.encode(text)
        self.assertIsNotNone(tokens)
        self.assertTrue(len(tokens) > 0)
        
        decoded = self.tokenizer.decode(tokens)
        self.assertTrue(len(decoded) > 0)
        logger.info(f"Test Tokenizer: Original text: {text}, Decoded: {decoded}")

    def test_memory_system(self):
        """Test memory system functionality"""
        memory = MemorySystem(
            embedding_dim=256,
            memory_size=1024,
            num_clusters=128,
            max_retrieval_distance=1.0,
            memory_compression=True,
            device="cpu"
        )
        
        # Create test embeddings
        test_embeddings = torch.randn(5, 256)
        memory.add_to_memory(test_embeddings, test_embeddings)
        
        # Test retrieval
        query = test_embeddings[0] + 0.1
        query = memory._compress_keys(query.unsqueeze(0))  # Compress query
        similarities, indices = memory.retrieve_from_memory(query, k=2)
        
        self.assertEqual(len(similarities[0]), 2)
        self.assertEqual(len(indices[0]), 2)
        logger.info(f"Test Memory System: Retrieved similarities: {similarities}")

    def test_spiking_layer(self):
        """Test spiking layer functionality"""
        spiking = SpikingLayer(
            beta=0.9,
            embedding_dim=256,
            memory_size=1024,
            device=self.device
        )
        
        # Create test input
        x = torch.randn(1, 512, 256)
        spikes, memory_state = spiking(x)
        
        self.assertIsNotNone(spikes)
        self.assertIsNotNone(memory_state)
        logger.info(f"Test Spiking Layer: Spike count: {spiking.get_state()['spike_count']}")

    def test_model_forward(self):
        """Test complete model forward pass"""
        # Create test input
        text = "The quick brown fox jumps over the lazy dog."
        tokens = self.tokenizer.encode(text)
        
        # Forward pass
        logits, memory_state = self.model(
            input_ids=tokens,
        )
        
        self.assertIsNotNone(logits)
        self.assertIsNotNone(memory_state)
        
        # Check shapes
        self.assertEqual(logits.shape[0], 1)  # batch size
        self.assertEqual(logits.shape[1], len(tokens))  # sequence length
        self.assertEqual(logits.shape[2], self.tokenizer.get_vocab_size())  # vocab size
        
        logger.info(f"Test Model Forward: Logits shape: {logits.shape}")

    def test_memory_state_management(self):
        """Test memory state management"""
        # Reset memory
        self.model.reset_memory()
        
        # Process some input
        text = "The quick brown fox jumps over the lazy dog."
        tokens = self.tokenizer.encode(text)
        
        # Forward pass
        _, memory_state = self.model(
            input_ids=tokens,
        )
        
        # Check memory state
        self.assertIsNotNone(memory_state)
        logger.info(f"Test Memory State: Memory size: {self.model.memory.current_size}")

if __name__ == '__main__':
    unittest.main()
