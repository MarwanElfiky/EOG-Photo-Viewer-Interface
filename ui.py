import os
import time
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.ttk import Button, Label
import joblib
from PIL import Image, ImageTk
import numpy as np
import main


global photo

folder_path = "X:/AH/24_25_2/HCI/Project/HCI-PROJECT/images"
image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
current_index = 0

svm_model = joblib.load("svm_model.pkl")
label_encoder = joblib.load("label_encoder.pkl") if os.path.exists("label_encoder.pkl") else None

if svm_model:
    print("Model Is lock and loaded\n")


test_window = None
result_label = None


def show_image():
    global photo, current_index
    image_path = os.path.join(folder_path, image_files[current_index])
    img = Image.open(image_path)
    img = img.resize((900, 500))
    photo = ImageTk.PhotoImage(img)
    image_label.config(image=photo)
    status_label.config(text=f"{current_index + 1} / {len(image_files)}")


def show_next():
    global current_index
    if current_index < len(image_files) - 1:
        current_index += 1
        show_image()


def show_prev():
    global current_index
    if current_index > 0:
        current_index -= 1
        show_image()


def load_signal_file(file_path):
    with open(file_path, 'r') as f:
        signal = [int(line.strip()) for line in f]
    return signal


def test():
    h_signal = load_signal_file(filedialog.askopenfilename(title="Choose horizontal signal"))
    v_signal = load_signal_file(filedialog.askopenfilename(title="Choose vertical signal"))
    preprocessed_dict = main.preprocess_single_signal(h_signal, v_signal)
    # print((preprocessed_dict['horizontal']['temp_class']))
    extracted_features = main.extract_features_single_signal(preprocessed_dict)
    # print(len(extracted_features))
    X = np.array(extracted_features).reshape(1, -1)

    prediction = svm_model.predict(X)
    print(prediction)
    predicted_class = label_encoder.inverse_transform(prediction)[0]


    if predicted_class == "Right":
        messagebox.showinfo(title="Predicted Class", message=f"Predicted class is {predicted_class}\nAction : Next Photo")
        time.sleep(1)
        show_next()
    elif predicted_class == "Left":
        messagebox.showinfo(title="Predicted Class", message=f"Predicted class is {predicted_class}\nAction : Previous Photo")
        time.sleep(1)
        show_prev()
    else:
        messagebox.showinfo(title="Predicted Class", message=f"Predicted class is {predicted_class}\nNo Action")
    # print(f"Predicted movement: {predicted_class}")


root = tk.Tk()
root.title("Image Viewer")
root.geometry("1080x700")

image_label = Label(root)
image_label.pack()

status_label = Label(root, text="")
status_label.pack()

button_frame = tk.Frame(root)
button_frame.pack()

prev_button = Button(button_frame, text="<< Previous", command=show_prev)
prev_button.grid(row=0, column=0)

next_button = Button(button_frame, text="Next >>", command=show_next)
next_button.grid(row=0, column=1)

test_button = Button(root, text="Start Testing", command=test)
test_button.pack(padx=10, pady=10)

show_image()

root.mainloop()
