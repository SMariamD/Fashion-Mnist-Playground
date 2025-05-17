import torch
from s3cmn.model.spiking import SpikingLayer

def test_spiking_layer():
    # Create a test input tensor
    batch_size = 2
    seq_len = 3
    embedding_dim = 4
    
    # Initialize the layer
    layer = SpikingLayer(
        embedding_dim=embedding_dim,
        memory_size=100,
        device='cpu'
    )
    
    # Create random input data
    input_data = torch.randn(batch_size, seq_len, embedding_dim)
    
    print("Input data:")
    print(input_data)
    
    # Forward pass
    spikes, memory_state = layer(input_data)
    
    print("\nSpikes:")
    print(spikes)
    
    print("\nMemory state:")
    print(memory_state)

if __name__ == "__main__":
    test_spiking_layer()
