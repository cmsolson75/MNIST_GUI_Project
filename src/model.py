import torch.nn as nn
import torch.nn.functional as F


class MNIST_CNN(nn.Module):
    """Convolutional Neural Network (CNN) for MNIST digit classification.

    Attributes:
    - conv1 (nn.Conv2d): First convolutional layer.
    - conv2 (nn.Conv2d): Second convolutional layer.
    - fc1 (nn.Linear): First fully connected layer.
    - fc2 (nn.Linear): Second fully connected layer, the output layer.
    - dropout (nn.Dropout): Dropout layer for regularization.
    """

    def __init__(self):
        super(MNIST_CNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1
        )

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        """Forward pass of the neural network.

        Parameters:
        - x (torch.Tensor): Input tensor of shape (batch_size, 1, 28, 28).

        Returns:
        - x (torch.Tensor): Output tensor after passing through the network.
        """
        # First convolutional layer followed by pooling and activation
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        # Second convolutional layer followed by pooling and activation
        x = F.relu(F.max_pool2d(self.conv2(x), 2))

        # Flatten the output from convolutional layers
        x = x.view(-1, 64 * 7 * 7)

        # First fully connected layer with activation and dropout
        x = self.dropout(F.relu(self.fc1(x)))

        # Second fully connected layer (output layer)
        x = self.fc2(x)

        return x
