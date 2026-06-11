import tkinter as tk
from PIL import Image, ImageDraw, ImageOps
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("digit_model.h5")

WIDTH = 280
HEIGHT = 280

root = tk.Tk()
root.title("Handwritten Digit Recognition Using CNN")
root.geometry("320x420")

canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="white")
canvas.pack(pady=10)

image = Image.new("L", (WIDTH, HEIGHT), 255)
draw = ImageDraw.Draw(image)

def paint(event):
    x1 = event.x - 8
    y1 = event.y - 8
    x2 = event.x + 8
    y2 = event.y + 8

    canvas.create_oval(
        x1, y1, x2, y2,
        fill="black",
        outline="black"
    )

    draw.ellipse(
        [x1, y1, x2, y2],
        fill=0
    )

canvas.bind("<B1-Motion>", paint)

result_label = tk.Label(
    root,
    text="Draw a Digit",
    font=("Arial", 18)
)
result_label.pack(pady=10)

def predict():
    img = image.resize((28, 28))
    img = ImageOps.invert(img)

    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = model.predict(img_array, verbose=0)

    digit = np.argmax(prediction)

    result_label.config(
        text=f"Predicted Digit: {digit}"
    )

def clear_canvas():
    canvas.delete("all")

    draw.rectangle(
        [0, 0, WIDTH, HEIGHT],
        fill=255
    )

    result_label.config(
        text="Draw a Digit"
    )

predict_button = tk.Button(
    root,
    text="Predict",
    command=predict,
    width=15
)
predict_button.pack(pady=5)

clear_button = tk.Button(
    root,
    text="Clear",
    command=clear_canvas,
    width=15
)
clear_button.pack(pady=5)

root.mainloop()