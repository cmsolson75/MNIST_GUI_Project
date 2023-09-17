
import torch
from src.data_utils import load_mnist_data

def test_load_mnist_data():
    train_data, test_data = load_mnist_data("./data/mnist_train.csv", "./data/mnist_test.csv")
    assert len(train_data) > 0
    assert len(test_data) > 0
