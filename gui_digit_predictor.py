
import tkinter as tk
from tkinter import Label, Button
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf

# Load trained model
model = tf.keras.models.load_model("mnist_cnn_model.keras")

# Main Window
root = tk.Tk()
root.title("MNIST Digit Recognizer")
root.geometry("420x520")
root.configure(bg="#F4F6F8")
root.resizable(False, False)

Label(root,
      text="MNIST Digit Recognizer",
      font=("Arial",18,"bold"),
      bg="#F4F6F8",
      fg="#1E3A8A").pack(pady=10)

Label(root,
      text="Draw a digit (0-9)",
      font=("Arial",12),
      bg="#F4F6F8").pack()

# Canvas
canvas = tk.Canvas(root, width=280, height=280,
                   bg="black", cursor="cross")
canvas.pack(pady=10)

# Blank Image
image = Image.new("L", (280,280), color=0)
draw = ImageDraw.Draw(image)

# Draw Function
def paint(event):
    x, y = event.x, event.y
    r = 10
    canvas.create_oval(x-r, y-r, x+r, y+r,
                       fill="white", outline="white")
    draw.ellipse([x-r, y-r, x+r, y+r], fill=255)

canvas.bind("<B1-Motion>", paint)

# Prediction Function
def predict_digit():
    img = image.resize((28,28))
    img_array = np.array(img)
    img_array = img_array.astype("float32") / 255.0
    img_array = img_array.reshape(1,28,28,1)

    prediction = model.predict(img_array, verbose=0)
    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    result.config(
        text=f"Prediction : {digit}
Confidence : {confidence:.2f}%"
    )

# Clear Canvas
def clear_canvas():
    canvas.delete("all")
    draw.rectangle([0,0,280,280], fill=0)
    result.config(text="")

# Predict Button
Button(root,
       text="Predict",
       command=predict_digit,
       bg="#16A34A",
       fg="white",
       font=("Arial",13,"bold"),
       width=15).pack(pady=8)

# Clear Button
Button(root,
       text="Clear",
       command=clear_canvas,
       bg="#DC2626",
       fg="white",
       font=("Arial",13,"bold"),
       width=15).pack()

# Result Label
result = Label(root,
               text="",
               font=("Arial",15,"bold"),
               bg="#F4F6F8",
               fg="#1E40AF")
result.pack(pady=20)

root.mainloop()
