import torch
from src.model import MNIST_CNN

def test_MNIST_CNN_output_shape():
    model = MNIST_CNN()
    input_tensor = torch.randn(32, 1, 28, 28)  # Batch of 32
    output = model(input_tensor)
    assert output.shape == (32, 10)
