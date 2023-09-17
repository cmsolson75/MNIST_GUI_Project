
# MNIST Hand-Drawn Digit Recognition

Welcome to my MNIST Hand-Drawn Digit Recognition project. This project focuses on training a deep learning model on the MNIST dataset and provides a user-friendly interface for real-time digit recognition.

## Project Structure

- **Jupyter Notebooks**:
  - `MNIST_Training_Notebook.ipynb`: This notebook contains the entire process of training the model on the MNIST dataset.
  - `MNIST_EDA_Notebook.ipynb`: Use this notebook for an in-depth exploratory data analysis of the MNIST dataset.
  
- **Python Scripts**:
  - `train_main.py`: The main script for training the model on the MNIST dataset.
  - `gui.py`: This script launches the tkinter-based GUI for real-time digit recognition and contains functions to process and recognize hand-drawn digits.
  - `model.py`: Defines the architecture and parameters of the neural network model.
  - `visualization.py`: Utilities for visualizing dataset samples, training results, and more.
  - `training.py`: Contains utilities and helper functions for model training.
  - `data_utils.py`: Helpful utilities for loading and processing the MNIST dataset.

## Getting Started

1. Ensure you have all the necessary libraries installed. 
    ```
    pip install requirements.txt
    ```
2. Ensure you have training data [MNIST](https://www.kaggle.com/datasets/hojjatk/mnist-dataset) in `./data` folder.

3. Run the `train_main.py` script to train the model:
   ```
   python train_main.py
   ```

# Running Inference with the tkinter App

The application provides an intuitive interface for users to draw digits and instantly get predictions from the trained model.

**Steps**:

1. Ensure the trained model is saved in the correct directory or specify the path to the trained model (e.g., `./saved_models/mnist_model.pth`).
   
2. Start the tkinter app by running the following command in your terminal or command prompt:
   ```
   python gui.py
   ```

3. Once the application window opens, use your mouse to draw a digit on the canvas.

4. Click on the "Predict" button. The application will process the drawn digit and display the predicted value.

5. To clear the canvas and draw a new digit, click on the "Clear" button.

---