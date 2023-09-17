import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd


def load_mnist_data(train_path, test_path):
    mnist_train = pd.read_csv(train_path)
    mnist_test = pd.read_csv(test_path)

    # Seperate the images and labels
    train_labels = mnist_train["label"].values
    train_images = mnist_train.drop(columns=["label"]).values
    test_labels = mnist_test["label"].values
    test_images = mnist_test.drop(columns=["label"]).values

    # Convert data to PyTorch tensors, reshape, and normalize
    train_images_tensor = (
        torch.tensor(train_images, dtype=torch.float32).reshape(-1, 1, 28, 28) / 255.0
    )
    train_labels_tensor = torch.tensor(train_labels, dtype=torch.int64)
    test_images_tensor = (
        torch.tensor(test_images, dtype=torch.float32).reshape(-1, 1, 28, 28) / 255.0
    )
    test_labels_tensor = torch.tensor(test_labels, dtype=torch.int64)

    # Create TensorDatasets for training and testing data
    train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
    test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)

    return train_dataset, test_dataset


def get_data_loaders(train_dataset, test_dataset, batch_size=64):
    # Create DataLoaders for training and testing data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader
