import tkinter as tk
from tkinter import Canvas, Label
import tkinter.messagebox as messagebox
from PIL import Image, ImageDraw
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model import MNIST_CNN


class DigitRecognizerApp(tk.Tk):
    """Tkinter application for recognizing hand-drawn digits using a trained model."""

    def __init__(self, model_path):
        """Initialize the Tkinter app with the trained model and GUI components.
        Args:
            model_path (str): Path to the trained model's state_dict.
        """

        super().__init__()
        self.title("MNIST Digit Recognizer")

        try:
            # Load the trained model
            self.model = MNIST_CNN()
            self.model.load_state_dict(torch.load(model_path))
            self.model.eval()
        except Exception as e:
            tk.messagebox.showerror(
                "Error", f"Failed to load the model. Error: {str(e)}"
            )
            self.destroy()
            return

        self.initialize_canvas()
        self.canvas_width = 280
        self.canvas_height = 280

        # For saving the drawn content
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), "black")
        self.draw = ImageDraw.Draw(self.image)

        self.geometry("500x600")  # Setting fixed window size and initial position

        self.initialize_prediction_label()
        self.initialize_buttons()

    def initialize_canvas(self):
        """Initialize the drawing canvas and bind the painting function."""
        self.canvas = tk.Canvas(self, bg="black", width=280, height=280)
        self.canvas.pack(pady=20)

        # Bind the paint function to mouse drag on the canvas
        self.canvas.bind("<B1-Motion>", self.paint)

    def initialize_prediction_label(self):
        """Initialize the label to display digit prediction."""
        self.prediction_label = tk.Label(
            self, text="Draw a digit above and press 'Predict'", font=("Arial", 24)
        )
        self.prediction_label.pack(pady=20, padx=20)

    def initialize_buttons(self):
        """Initialize the 'Predict' and 'Clear' buttons."""
        predict_button = tk.Button(
            self, text="Predict", command=self.predict_digit, font=("Arial", 20)
        )
        predict_button.pack(side="left", pady=20, padx=20)

        clear_button = tk.Button(
            self, text="Clear", command=self.clear_canvas, font=("Arial", 20)
        )
        clear_button.pack(side="right", pady=20, padx=20)

    def paint(self, event):
        x, y = event.x, event.y
        r = 10
        self.canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", width=0)
        self.draw.ellipse([x - r, y - r, x + r, y + r], fill="white")

    def predict_digit(self):
        img = self.image.resize((28, 28))
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize to [0,1]
        img_tensor = (
            torch.tensor(img_array).unsqueeze(0).unsqueeze(0)
        )  # Shape it to [1,1,28,28]

        with torch.no_grad():
            outputs = self.model(img_tensor)

        predicted_class = torch.argmax(outputs).item()
        self.prediction_label.config(text=f"Predicted Digit: {predicted_class}")

    def clear_canvas(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, self.canvas_width, self.canvas_height], fill="black")
        self.prediction_label.config(text="")


if __name__ == "__main__":
    model_path = "models/best_model.pth"
    try:
        app = DigitRecognizerApp(model_path)
        app.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")
