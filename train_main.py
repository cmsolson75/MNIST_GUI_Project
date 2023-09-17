
import argparse

from src.data_utils import load_mnist_data, get_data_loaders
from src.training import train_model
from src.visualization import visualize_training



def create_parser():
    """Creates ArgumentParser for user input"""
    parser = argparse.ArgumentParser(description="MNIST training script")
    parser.add_argument("-tr","--train_data", type=str, default="./data/mnist_train.csv", help="Path to training directory")
    parser.add_argument("-tst","--test_data", type=str, default="./data/mnist_test.csv", help="Path to test directory")
    parser.add_argument("-b","--batch_size", type=int, default=64, help='Batch size')
    parser.add_argument("-e","--epochs", type=int, default=10, help='Number of training epochs')
    parser.add_argument("-v","--visualize_train", type=bool, default='false', help='Visualize training, input true for visualization')

    return parser


def main(args):
    """Main function to execute the end-to-end workflow."""

    # Paths to the MNIST data files
    train_data_path = args.train_data
    test_data_path = args.test_data

    # Load data and get DataLoaders
    train_dataset, test_dataset = load_mnist_data(train_data_path, test_data_path)
    train_loader, val_loader = get_data_loaders(train_dataset, test_dataset, batch_size=args.batch_size)

    # Train the model
    train_losses, train_accuracies, val_losses, val_accuracies = train_model(train_loader, val_loader, num_epochs=args.epochs)


    # Visualize the training and validation metrics
    if args.visualize_train:
        visualize_training(train_losses, train_accuracies, val_losses, val_accuracies)

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)