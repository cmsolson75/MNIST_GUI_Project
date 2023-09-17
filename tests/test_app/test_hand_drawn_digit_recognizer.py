import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest
from unittest.mock import patch
from hand_drawn_digit_recognition import DigitRecognizerApp
from src.model import MNIST_CNN

# Mocking torch.load to avoid actual model loading
@patch("torch.load", return_value={})
def test_digit_recognizer_app_initialization(mocked_load):
    app = DigitRecognizerApp("dummy_path")
    assert isinstance(app, DigitRecognizerApp)
    assert app.title() == "MNIST Digit Recognizer"

# Mocking the torch model's prediction to always return 5 for testing purposes
@patch.object(MNIST_CNN, 'forward', return_value=torch.tensor([[0, 0, 0, 0, 0, 1, 0, 0, 0, 0]]))
def test_predict_digit(mocked_forward):
    app = DigitRecognizerApp("dummy_path")
    app.predict_digit()
    assert "Predicted Digit: 5" in app.prediction_label.cget("text")

def test_clear_canvas():
    app = DigitRecognizerApp("dummy_path")
    app.clear_canvas()
    assert app.prediction_label.cget("text") == ""

class MockEvent:
    def __init__(self, x, y):
        self.x = x
        self.y = y

        
def test_paint():
    app = DigitRecognizerApp("dummy_path")
    event = MockEvent(100, 100)
    app.paint(event)
    # Check if a white pixel is painted at the specified coordinate in the image
    assert app.image.getpixel((100, 100)) == 255

